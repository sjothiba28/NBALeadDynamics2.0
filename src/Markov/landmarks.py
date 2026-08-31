"""Landmark detection and per-landmark feature extraction.

A landmark is the first regulation-time moment a team is ahead by AT LEAST
5, 10 or 15 points -- unlike features.py's peak lead, which is a global
argmax over the whole game. Fixing the threshold by design removes lead size
as a confound, so a landmark-based dataset can ask whether trajectory shape
(Control/Volatility) or recent behavior (D1_*) carries information beyond
lead size and time remaining -- the memorylessness test in
docs/results/landmark-memory-test.md.

This module deliberately duplicates ~two functions' worth of logic that is
structurally identical to features.py's (team-tricode resolution, pre-window
shot/possession rate computation) rather than importing or refactoring
features.py. That guarantees zero diff to the file the published peak-lead
results (docs/results/models-1-4.md) depend on. The cost is drift risk if
features.identify_leader's cross-check logic ever changes -- mitigated by
tests/test_landmarks.py::test_resolve_teams_agrees_with_identify_leader_on_all_fixtures,
which fails loudly if the two diverge.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (_HERE, os.path.join(_HERE, "..")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import geometry
from common import pbp as pbp_common

THRESHOLDS = (5, 10, 15)
REGULATION_END = pbp_common.REGULATION_END


def resolve_teams(df):
    """(team_a, team_b, sign_a) or None.

    sign_a is the multiplier such that sign_a * df['signed_margin'] is team
    a's own relative margin (positive while team_a is ahead). team_a/team_b
    are the two tricodes with the most event volume; a third tricode with
    real volume means the pair can't be trusted, mirroring
    features.identify_leader's cross-check (features.py:84-89).
    """
    if df is None or df.empty:
        return None

    team = df["event_team"].astype(str).str.strip()
    attributed = team[team != ""]
    if attributed.empty:
        return None

    counts = attributed.value_counts()
    if len(counts) < 2:
        return None
    team_a, team_b = counts.index[0], counts.index[1]

    others = counts.drop([team_a, team_b], errors="ignore")
    if (others > 0.01 * len(attributed)).any():
        return None

    scored = df[(team == team_a) & (df["made"] == 1.0)]
    if scored.empty:
        return None
    first_pos = df.index.get_loc(scored.index[0])

    margin = df["signed_margin"].astype(float)
    prev = margin.shift(1)
    prev.iloc[0] = 0.0  # the game was 0-0 before the first event
    diff_at_score = margin.iloc[first_pos] - prev.iloc[first_pos]

    if diff_at_score > 0:
        sign_a = 1.0
    elif diff_at_score < 0:
        sign_a = -1.0
    else:
        return None

    return (team_a, team_b, sign_a)


def find_landmark(df, team, sign, threshold):
    """Index of the first regulation-time row where `team`'s signed-relative
    margin reaches at least `threshold`, or None.

    `>=`, not `==`: a three-pointer can jump the margin from 3 to 6, skipping
    5 exactly. OT is excluded at detection time (game_time_sec <=
    REGULATION_END) so no predictor downstream ever depends on the game
    having gone to overtime.
    """
    if df is None or df.empty:
        return None

    team_margin = sign * df["signed_margin"].astype(float)
    in_regulation = df["game_time_sec"] <= REGULATION_END
    hit = (team_margin >= threshold) & in_regulation
    if not hit.any():
        return None
    return hit.idxmax()


def extract_landmark_features(
    df,
    game_id,
    season,
    season_type,
    team_a,
    team_b,
    focal_team,
    sign_focal,
    threshold,
    landmark_idx,
):
    """One landmark row: state at the landmark plus what happened after it.

    Mirrors features.extract_game_features (features.py:95-288) with
    t_peak -> t_landmark, l_max -> landmark_lead, leader -> focal_team. Unlike
    the peak, `final_team_relative_margin` is read from the TRUE game end
    (including overtime) -- only landmark SELECTION excludes overtime
    (find_landmark), not outcome measurement, so a game that goes to OT still
    has its actual final margin recorded.
    """
    team_margin = sign_focal * df["signed_margin"].astype(float)
    t_lm = float(df.loc[landmark_idx, "game_time_sec"])
    l_lm = float(team_margin.loc[landmark_idx])
    l_final = float(team_margin.iloc[-1])
    opponent = team_b if focal_team == team_a else team_a

    features = {
        "GAME_ID": game_id,
        "SEASON": season,
        "SEASON_TYPE": season_type,
        "LEADER": focal_team,
        "OPPONENT": opponent,
        "threshold": threshold,
        "landmark_lead": l_lm,
        "elapsed_time": t_lm,
        "period": int(t_lm // pbp_common.REGULATION_PERIOD) + 1,
        "time_remaining": REGULATION_END - t_lm,
        "final_team_relative_margin": l_final,
        "lead_loss": l_lm - l_final,
        # A zero-crossing strictly after the landmark, same "anywhere in the
        # game" false-positive reasoning as features.py:146-148's collapsed.
        "collapsed": int((team_margin[df["game_time_sec"] > t_lm] < 0).any()),
    }
    features.update(
        geometry.calculate_geometry(
            df["game_time_sec"].to_numpy(), team_margin.to_numpy(), t_lm, l_lm
        )
    )

    pre_landmark = df[(df["game_time_sec"] <= t_lm) & (df["event_team"] == focal_team)]
    features.update(_pre_window_rates(pre_landmark, t_lm))
    return features


def _pre_window_rates(pre_window, t_ref):
    """G_*/Possessions/Backcourt_FGA/parsed_ft_or_tov over a pre-landmark
    window already filtered to game_time_sec <= t_ref and event_team ==
    focal_team. Close copy of features.py:161-286's pre-peak block -- see
    that function's comments for why each NaN-vs-0 fallback and the estimator
    possession denominator (rather than the exact walk) are the way they are.
    """
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
            "G_ATOP": t_ref / est_possessions if possessions_usable else np.nan,
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


def build_landmark_rows(df, game_id, season, season_type):
    """0-6 row dicts: both teams x THRESHOLDS, only where actually reached.

    Both teams generate independent rows when both reach a threshold (e.g.
    team A hits +5, team B later erases it and hits +5 itself) -- the
    composite key (GAME_ID, LEADER, threshold) identifies a row, not GAME_ID
    alone.
    """
    resolved = resolve_teams(df)
    if resolved is None:
        return []
    team_a, team_b, sign_a = resolved

    rows = []
    for focal_team, sign_focal in ((team_a, sign_a), (team_b, -sign_a)):
        for threshold in THRESHOLDS:
            idx = find_landmark(df, focal_team, sign_focal, threshold)
            if idx is None:
                continue
            rows.append(
                extract_landmark_features(
                    df,
                    game_id,
                    season,
                    season_type,
                    team_a,
                    team_b,
                    focal_team,
                    sign_focal,
                    threshold,
                    idx,
                )
            )
    return rows
