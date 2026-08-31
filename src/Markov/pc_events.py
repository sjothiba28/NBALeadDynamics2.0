"""Event-level rows for the Peel & Clauset (2015) next-scorer models
(docs/plans/two-paper-synthesis-and-variable-test.md: PC-replication in
Step 2, and the H0/H1 primary test in Steps 3-5).

Two independent row builders, each producing one row per raw scoring
event, with a focal-relative margin BEFORE that event (`X_before`), which
side scored it (`scored_focal`), and which side scored the immediately
preceding event (`last_scorer_focal`, the anti-persistence feature). They
are NOT built from a shared helper, because their focal orientation is
genuinely different and the plan requires each to stand as its own
row set:

- `build_game_rows`: whole-game rows, every regulation scoring event in
  every game. Focal = the alphabetically-first of the game's two scoring
  teams (arbitrary but consistent -- the paper's r/b labeling carries no
  landmark, lead, or home/away meaning). PC-replication (Step 2) only.
- `build_landmark_rows`: POST-LANDMARK rows only (events strictly after a
  landmark's own elapsed_time), for games that reached a 5-point landmark.
  Focal = the landmark's LEADER (matches sde_transitions.
  build_transition_rows' sign_focal convention). z (D1_r_Mid/D1_r_Rim) is
  fixed at the landmark's own row and carried unchanged onto every event
  row for that landmark, per the plan's "z fixed at the landmark" rule.
  H0/H1 (Steps 3-5) only -- never mixed with build_game_rows' whole-game
  rows in the same fit.

Both builders derive `scored_focal` directly from `scoring_team ==
<focal team code>` -- events.py already collapses same-instant actions
into one row per distinct clock time, crediting it to one team, so there
is no need to infer the scorer from a margin delta's sign.
"""

import numpy as np
import pandas as pd

import sde_bt

_GAME_COLUMNS = [
    "GAME_ID",
    "SEASON",
    "SEASON_TYPE",
    "team_r",
    "team_b",
    "game_time_sec",
    "dt",
    "X_before",
    "scored_focal",
    "last_scorer_focal",
]

_LANDMARK_COLUMNS = [
    "landmark_row_id",
    "GAME_ID",
    "SEASON",
    "LEADER",
    "OPPONENT",
    "threshold",
    "game_time_sec",
    "s_since_landmark",
    "dt",
    "X_before",
    "scored_focal",
    "last_scorer_focal",
]


def build_game_rows(events_df, season_type="Regular Season"):
    """One row per regulation scoring event, every game, for
    PC-replication's whole-game season-based evaluation. A game is
    skipped if it does not have exactly two distinct scoring teams (same
    defensive skip as sde_bt._season_pairwise_counts). The very first
    scoring event of each game is dropped -- it has no preceding event,
    so `last_scorer_focal` is undefined for it (never guessed/defaulted;
    same drop-and-warn-free-but-documented convention as the rest of this
    study's event-resolution code)."""
    reg = events_df[
        (events_df["is_regulation"]) & (events_df["SEASON_TYPE"] == season_type)
    ].copy()
    reg["GAME_ID"] = reg["GAME_ID"].astype(str).str.strip().str.zfill(10)

    rows = []
    for game_id, g in reg.groupby("GAME_ID", sort=False):
        g = g.sort_values("game_time_sec")
        teams = sorted(t for t in g["scoring_team"].dropna().unique() if t)
        if len(teams) != 2:
            continue
        team_r, team_b = teams[0], teams[1]

        t = g["game_time_sec"].to_numpy()
        pts = g["points"].to_numpy(dtype=float)
        scorer = g["scoring_team"].to_numpy()
        season = g["SEASON"].iloc[0]
        season_type_val = g["SEASON_TYPE"].iloc[0]

        x = 0.0
        prev_scorer_focal = None
        for i in range(len(g)):
            scored_focal = bool(scorer[i] == team_r)
            if prev_scorer_focal is not None:
                rows.append(
                    {
                        "GAME_ID": game_id,
                        "SEASON": season,
                        "SEASON_TYPE": season_type_val,
                        "team_r": team_r,
                        "team_b": team_b,
                        "game_time_sec": t[i],
                        "dt": t[i] - t[i - 1],
                        "X_before": x,
                        "scored_focal": scored_focal,
                        "last_scorer_focal": prev_scorer_focal,
                    }
                )
            x = x + (pts[i] if scored_focal else -pts[i])
            prev_scorer_focal = scored_focal

    return pd.DataFrame(rows, columns=_GAME_COLUMNS)


def build_landmark_rows(landmarks_df, events_df):
    """One row per regulation scoring event STRICTLY AFTER a landmark's
    own elapsed_time, through end of regulation, for the games/landmarks
    in `landmarks_df`. Needs only GAME_ID/SEASON/LEADER/OPPONENT/
    threshold/landmark_lead/elapsed_time from `landmarks_df` -- unlike
    sde_transitions.build_transition_rows, this does NOT need
    'sign_focal' or a raw home_away_margin column, because focal-relative
    margin here is accumulated directly from each event's own
    scoring_team/points (scored_focal = scoring_team == LEADER), never
    from a home-minus-away sign flip.

    The first post-landmark row's `last_scorer_focal` is seeded True
    (focal/LEADER): a landmark is, by construction, the first time the
    leader's margin reaches `threshold` (landmarks.find_landmark), and
    margin only changes on a scoring event, so the event that produced
    the landmark itself was necessarily scored by LEADER."""
    events_reg = events_df[events_df["is_regulation"]].copy()
    events_reg["GAME_ID"] = events_reg["GAME_ID"].astype(str).str.strip().str.zfill(10)

    landmarks_df = landmarks_df.reset_index(drop=True).copy()
    landmarks_df["GAME_ID"] = (
        landmarks_df["GAME_ID"].astype(str).str.strip().str.zfill(10)
    )

    events_by_game = {
        game_id: g.sort_values("game_time_sec")
        for game_id, g in events_reg.groupby("GAME_ID")
    }

    rows = []
    for game_id, game_landmarks in landmarks_df.groupby("GAME_ID"):
        game_events = events_by_game.get(game_id)
        if game_events is None or game_events.empty:
            continue
        t = game_events["game_time_sec"].to_numpy()
        scorer = game_events["scoring_team"].to_numpy()
        pts = game_events["points"].to_numpy(dtype=float)

        for landmark_row_id, lm in game_landmarks.iterrows():
            after = t > lm["elapsed_time"]
            if not after.any():
                continue
            t_after = t[after]
            scorer_after = scorer[after]
            pts_after = pts[after]
            leader = lm["LEADER"]

            times = np.concatenate([[lm["elapsed_time"]], t_after])
            x = float(lm["landmark_lead"])
            prev_scorer_focal = (
                True  # seeded: LEADER scored the landmark-triggering event
            )
            for k in range(len(t_after)):
                scored_focal = bool(scorer_after[k] == leader)
                rows.append(
                    {
                        "landmark_row_id": landmark_row_id,
                        "GAME_ID": game_id,
                        "SEASON": lm["SEASON"],
                        "LEADER": lm["LEADER"],
                        "OPPONENT": lm["OPPONENT"],
                        "threshold": lm["threshold"],
                        "game_time_sec": times[k + 1],
                        "s_since_landmark": times[k + 1] - lm["elapsed_time"],
                        "dt": times[k + 1] - times[k],
                        "X_before": x,
                        "scored_focal": scored_focal,
                        "last_scorer_focal": prev_scorer_focal,
                    }
                )
                x = x + (pts_after[k] if scored_focal else -pts_after[k])
                prev_scorer_focal = scored_focal

    return pd.DataFrame(rows, columns=_LANDMARK_COLUMNS)


def attach_eta_bt(rows_df, events_df, season_type="Regular Season"):
    """rows_df (from build_game_rows) + 'eta_BT' = log(pi_r) - log(pi_b),
    the SAME leave-one-game-out strength fit attach_v_bt uses for the SDE
    study's v_BT (sde_bt.fit_season_strengths_leave_one_game_out), just
    converted to a log-odds offset (sde_bt.bt_log_odds) instead of a
    points/sec drift. `rows_df` must carry 'SEASON'/'GAME_ID'/'team_r'/
    'team_b' (build_game_rows' own output columns) -- this is the ONE
    place every PC-replication model (independent/restorative/
    independent-AP/restorative-AP) gets eta_BT from, so the offset can
    never silently diverge between them."""
    strengths_by_game = sde_bt.fit_season_strengths_leave_one_game_out(
        events_df, season_type=season_type
    )

    out = rows_df.reset_index(drop=True).copy()
    game_id_padded = out["GAME_ID"].astype(str).str.strip().str.zfill(10)

    eta = np.full(len(out), np.nan)
    for idx, (season, game_id, team_r, team_b) in enumerate(
        zip(out["SEASON"], game_id_padded, out["team_r"], out["team_b"], strict=True)
    ):
        strengths = strengths_by_game.get((season, game_id))
        if strengths is None or team_r not in strengths or team_b not in strengths:
            continue
        eta[idx] = float(sde_bt.bt_log_odds(strengths[team_r], strengths[team_b]))

    out["eta_BT"] = eta
    unmatched = np.isnan(eta)
    if unmatched.any():
        print(
            f"  [WARN] {int(unmatched.sum())} of {len(out)} PC event rows had no "
            "matching leave-one-game-out strength fit for their (SEASON, GAME_ID, "
            "team_r, team_b) -- eta_BT left NaN rather than defaulted to 0.0 (an "
            "even-strength guess). Likely cause: a (SEASON, GAME_ID) present in "
            "rows_df but absent from events_df's own regulation/season_type slice."
        )
    return out


def attach_eta_bt_from_strengths(rows_df, s_r, s_b):
    """Vectorized eta_BT attachment for callers that already have per-row
    S_focal/S_opponent arrays (e.g. H0/H1's landmark rows, which reuse
    sde_bt.attach_v_bt's own S_focal/S_opponent columns instead of
    re-fitting strengths here) -- kept separate from attach_eta_bt so a
    landmark-row caller never needs a `team_r`/`team_b` pair or a second
    strength fit."""
    out = rows_df.reset_index(drop=True).copy()
    out["eta_BT"] = sde_bt.bt_log_odds(
        np.asarray(s_r, dtype=float), np.asarray(s_b, dtype=float)
    )
    return out
