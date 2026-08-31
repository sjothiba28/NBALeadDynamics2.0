"""Per-game feature extraction for the lead-loss regressions.

Split out of regression.py so the feature definitions can be read, tested and
reviewed without the scraper's network machinery sitting on top of them.

Everything here is measured from the point of view of the team that reached the
peak lead. The lead path is signed and leader-relative -- positive while that
team is ahead, negative once it has been overtaken -- so the response
`Y_Absolute_Loss = L_max - L_final` is non-negative and unbounded above, and the
pre-peak rates describe that team's own play rather than both teams' pooled.

The pre-peak rates are emitted raw, as `G_*`, mirroring
`common.baselines.BASELINE_COLUMNS` one-for-one. The per-team, per-season
baseline is joined and subtracted downstream, not here: a game only knows which
team led, not what that team's season looked like.
"""

import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (_HERE, os.path.join(_HERE, "..")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import geometry
from common import pbp as pbp_common
from common.baselines import BASELINE_COLUMNS


def identify_leader(pbp):
    """(leader_tricode, trailer_tricode) at the peak lead, or None.

    The parsed feature frame carries a signed margin but no team scores, so the
    only place the leader can be named is the raw feed: the team whose score
    increases on a row owns that row. Same derivation as pca.extract_game_data.
    """
    if pbp is None or len(pbp) == 0:
        return None

    margin = pbp_common.score_margin(pbp)
    if margin.abs().max() == 0:
        return None
    leader_is_home = margin.loc[margin.abs().idxmax()] > 0

    home = pd.to_numeric(pbp.get("scoreHome"), errors="coerce").ffill().fillna(0)
    away = pd.to_numeric(pbp.get("scoreAway"), errors="coerce").ffill().fillna(0)
    # event_team, not the raw column: a team-charged turnover has a blank
    # tricode. It cannot score, so it cannot win either of these lookups, but
    # using the same attribution everywhere keeps the tricode vocabulary
    # identical to the one the pre-peak filter matches against.
    team = pbp_common.event_team(pbp)

    def first_scorer(diff):
        scored = team[diff > 0]
        scored = scored[scored.astype(str).str.strip() != ""]
        return next(iter(scored), None)

    home_tricode = first_scorer(home.diff())
    away_tricode = first_scorer(away.diff())
    if not home_tricode or not away_tricode or home_tricode == away_tricode:
        return None

    # Cross-check the pair against the whole game before trusting it. Each
    # tricode above rests on a SINGLE row -- the first on which that side's
    # score increases -- so one blank or mis-attributed opening score row is
    # enough to return a well-formed but wrong pair, and every consumer
    # downstream (the pre-peak leader filter, the baseline join, every D1_*)
    # would then measure the game from the trailing team's side without a word.
    # regression.py's leader assertion treats a missing leader as fatal-class;
    # a wrong leader must not be quieter than a missing one.
    #
    # A volume threshold, deliberately, not an equality on the tricode set:
    # blank teamTricode rows are normal in this feed (team-charged turnovers,
    # resolved from description text by event_team), so an unmapped nickname or
    # a stray row must not veto an otherwise unambiguous game. 1% of the
    # attributed rows is ~4-6 rows in a real game -- comfortably above any
    # plausible stray, and far below the ~45% share a genuinely displaced side
    # would carry. On all five fixture games the only non-blank event_team
    # values are the two real teams (216/223, 287/263, 197/185, 252/220,
    # 238/235), so this never fires on known-good data.
    attributed = team[team.astype(str).str.strip() != ""]
    if len(attributed):
        others = attributed.value_counts().drop(
            [home_tricode, away_tricode], errors="ignore"
        )
        if (others > 0.01 * len(attributed)).any():
            return None

    return (
        (home_tricode, away_tricode) if leader_is_home else (away_tricode, home_tricode)
    )


def window_rates(pbp_df, upto_time, focal_team):
    """G_* pre-window shot-profile/turnover/rebounding rates for `focal_team`,
    over events with game_time_sec <= upto_time. Generalizes the pre-peak
    block of extract_game_features to an arbitrary window endpoint and an
    arbitrary focal team, so the same rate definitions can be evaluated at
    any state-grid cell entry, not just at a game's single peak lead.
    """
    pre_window = pbp_df[pbp_df["game_time_sec"] <= upto_time]
    if focal_team is not None and "event_team" in pre_window.columns:
        pre_window = pre_window[pre_window["event_team"] == focal_team]
    all_shots = pre_window[pre_window["shot_zone"].notnull()]

    shots = all_shots[all_shots["shot_zone"] != "Backcourt"]
    total_shots = len(shots)
    features = {"Backcourt_FGA": len(all_shots) - total_shots}

    zone_counts = shots["shot_zone"].value_counts()
    zone_makes = shots[shots["made"] == 1.0]["shot_zone"].value_counts()

    def rate(z):
        return zone_counts.get(z, 0) / total_shots if total_shots else 0

    def pct(z):
        attempts = zone_counts.get(z, 0)
        return zone_makes.get(z, 0) / attempts if attempts else np.nan

    features["Possessions"] = int(pre_window["poss_start"].sum())

    n_orb = pre_window["is_orb"].sum()
    n_tov = pre_window["is_tov"].sum()
    n_ft = pre_window["is_ft"].sum()
    est_possessions = total_shots - n_orb + n_tov + 0.44 * n_ft
    possessions_usable = est_possessions >= 1
    orb_chances = n_orb + pre_window["opp_drb"].sum()

    features.update(
        {
            "G_r_Rim": rate("Rim"),
            "G_r_Paint": rate("Paint"),
            "G_r_Mid": rate("Mid"),
            "G_r_C3": rate("C3"),
            "G_r_ATB3": rate("ATB3"),
            "G_r_FT": n_ft / est_possessions if possessions_usable else np.nan,
            "G_p_Rim": pct("Rim"),
            "G_p_Paint": pct("Paint"),
            "G_p_Mid": pct("Mid"),
            "G_p_C3": pct("C3"),
            "G_p_ATB3": pct("ATB3"),
            "G_p_FT": (
                pre_window.loc[pre_window["is_ft"].astype(bool), "made"].mean()
                if n_ft > 0
                else np.nan
            ),
            "G_ORB_pct": n_orb / orb_chances if orb_chances else 0,
            "G_TOV_pct": n_tov / est_possessions if possessions_usable else np.nan,
            # Time-per-possession numerator is the window's own endpoint, not an
            # internal peak -- this generalizes to any elapsed_time, unlike the
            # single t_peak call site this was extracted from.
            "G_ATOP": upto_time / est_possessions if possessions_usable else np.nan,
        }
    )
    for key in ("G_r_Rim", "G_r_Paint", "G_r_Mid", "G_r_C3", "G_r_ATB3", "G_ORB_pct"):
        features[key] = float(np.nan_to_num(features[key]))
    for key in (
        "G_p_Rim",
        "G_p_Paint",
        "G_p_Mid",
        "G_p_C3",
        "G_p_ATB3",
        "G_p_FT",
        "G_r_FT",
        "G_TOV_pct",
        "G_ATOP",
    ):
        features[key] = float(features[key])

    features["parsed_ft_or_tov"] = int(n_ft > 0 or n_tov > 0)
    return features


def extract_game_features(pbp_df, game_id, season, season_type, leader=None):
    """One row of the feature table, or None for a game with no signal.

    `leader` is the tricode of the team that reached the peak lead. It is read
    off the `leader` column of the parsed frame when not passed explicitly;
    without it the pre-peak window cannot be restricted to one team and the
    rates would pool both.
    """
    if pbp_df is None or pbp_df.empty or pbp_df["signed_margin"].abs().max() == 0:
        return None

    if leader is None and "leader" in pbp_df.columns:
        first = pbp_df["leader"].iloc[0]
        leader = first if isinstance(first, str) and first else None

    signed = pbp_df["signed_margin"]
    peak_idx = signed.abs().idxmax()  # first occurrence: idxmax ties to the first
    sign = 1.0 if signed.loc[peak_idx] > 0 else -1.0

    # Leader-relative from here on. Positive while the team that reached L_max
    # is ahead, negative once they have been overtaken.
    lead = sign * signed
    l_max = float(lead.loc[peak_idx])
    t_peak = float(pbp_df.loc[peak_idx, "game_time_sec"])
    l_final = float(lead.iloc[-1])

    # Time left when the peak lead was reached. Measured against when the game
    # actually ended rather than a fixed 2880, so overtime games do not come out
    # negative.
    #
    # ENDOGENEITY: this makes R partly post-peak in overtime games. A game
    # reaches overtime precisely BECAUSE the lead was given back, so for those
    # games game_end -- and therefore R -- carries a component caused by the
    # response Y rather than known at t_peak. R is in all five models and is
    # Model 1's only non-L_max term, so this touches every number in
    # docs/results/models-1-4.md; see Caveat section 9 there for the size and
    # direction of the bias and for why the alternative (regulation end plus a
    # went_to_OT indicator) was not adopted.
    game_end = float(pbp_df["game_time_sec"].max())

    features = {
        "GAME_ID": game_id,
        "SEASON": season,
        "SEASON_TYPE": season_type,
        "LEADER": leader,
        "L_max": l_max,
        "R": game_end - t_peak,
        # Y is what the leading team gave back. Over the absolute margin this
        # was L_final - L_max, which is <= 0 by construction for every game and
        # scored an overtaken team on its opponent's final margin.
        "Y_Absolute_Loss": l_max - l_final,
        "Y_Fraction_Lost": (l_max - l_final) / l_max,
        "peak_lead_is_home": int(sign > 0),
        # A zero-crossing strictly after the peak. Anywhere-in-the-game fired on
        # 88% of games because an early 2-0 / 0-3 swap counts.
        "collapsed": int((lead[pbp_df["game_time_sec"] > t_peak] < 0).any()),
    }
    features.update(
        geometry.calculate_geometry(
            pbp_df["game_time_sec"].to_numpy(), lead.to_numpy(), t_peak, l_max
        )
    )

    # window_rates handles the leader-filtered pre-peak shot-profile/
    # turnover/rebounding rates -- see its docstring for the definitions.
    # extract_game_features' own contract (this call site) fixes the window
    # at t_peak and the focal team at leader; window_rates itself is
    # agnostic to both, generalized for state_grid's per-cell attach.
    features.update(window_rates(pbp_df, t_peak, leader))

    return features


def attach_baselines(df, baselines):
    """Join each game to the leading team's own season identity and difference.

    This replaces a hardcoded dict of league-wide averages that was written
    identically to every row, which made all 14 B_* columns constant: Model 3's
    design matrix came out rank 16 of 28 and Model 4 rank 18 of 30, so B
    contributed nothing beyond the intercept and D1 was a shifted game value.
    """
    dup_mask = baselines.duplicated(["SEASON", "LEADER"], keep=False)
    if dup_mask.any():
        dup_keys = baselines.loc[dup_mask, ["SEASON", "LEADER"]].drop_duplicates()
        n_dup_keys = len(dup_keys)
        example = tuple(dup_keys.iloc[0])
        raise ValueError(
            f"{n_dup_keys} duplicate (SEASON, LEADER) key"
            f"{'s' if n_dup_keys != 1 else ''} in baselines, e.g. "
            f"{example[0]}, {example[1]} -- an inner merge on this would fan "
            f"out instead of dropping, silently double-counting games."
        )

    before = len(df)
    df = df.merge(baselines, on=["SEASON", "LEADER"], how="inner")
    if len(df) < before:
        print(
            f"  [WARN] {before - len(df)} of {before} games dropped by the "
            "baseline merge (missing season baseline or unmapped team)."
        )
    elif len(df) > before:
        print(
            f"  [WARN] {len(df) - before} of {before} games gained by the "
            "baseline merge (duplicate season baseline keys)."
        )

    # fetch_season_baselines is called with its default season_type of
    # 'Regular Season' (regression.py:222) while the game finder pulls both
    # types, so every D1_* on a playoff row is a playoff deviation from a
    # regular-season identity. That is deliberate -- a per-team playoff
    # baseline would rest on a handful of games -- but it must not be silent.
    if "SEASON_TYPE" in df.columns:
        n_playoff = int((df["SEASON_TYPE"] != "Regular Season").sum())
        if n_playoff:
            # "non-regular-season", not "Playoff": the predicate is
            # != 'Regular Season', which also catches Play-In and would catch
            # PreSeason / All-Star if they ever entered the scrape.
            print(
                f"  [NOTE] {n_playoff} of {len(df)} games are "
                f"non-regular-season games differenced against Regular "
                f"Season baselines."
            )

    # Only the baseline columns actually present in `baselines` are iterated,
    # so this also works against test frames that carry a subset (e.g. just
    # 'b_r_Rim') rather than requiring all 15 BASELINE_COLUMNS every time.
    present = [col for col in BASELINE_COLUMNS if col in baselines.columns]
    for b_col in present:
        metric = b_col[2:]  # 'b_r_Rim' -> 'r_Rim'
        g_col = f"G_{metric}"
        df[f"B_{metric}"] = df[b_col]
        # An undefined game rate (no attempts in that zone) falls back to the
        # team's own baseline so the deviation is 0, rather than being filled
        # with 0 and scoring the game as a maximal fake deviation.
        df[f"D1_{metric}"] = df[g_col].fillna(df[b_col]) - df[b_col]

    return df.drop(columns=present)
