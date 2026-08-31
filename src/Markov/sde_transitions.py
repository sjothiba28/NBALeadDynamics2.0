"""Post-landmark transition rows: one row per consecutive pair of
regulation-time scoring events after a landmark, through end of regulation.

landmarks_df must already carry 'sign_focal' (sde_data.
attach_regulation_outcomes(..., keep_sign_column=True)) and 'v_BT'
(sde_bt.attach_v_bt) -- built by composing those two, never re-derived
here. Rows continue through regulation end regardless of a zero-crossing:
sde_simulate.simulate_paths never stops a path at erasure either (its
trajectory is left unperturbed for the rest of the horizon), so the data
sde_kappa.fit_kappa0 fits against must cover the same full horizon the
simulator integrates over.
"""

import numpy as np
import pandas as pd

_COLUMNS = [
    "landmark_row_id",
    "GAME_ID",
    "SEASON",
    "LEADER",
    "OPPONENT",
    "threshold",
    "s_prev",
    "s_next",
    "dt",
    "X_prev",
    "X_next",
    "v_BT",
]


def build_transition_rows(landmarks_df, events_df):
    events_reg = events_df[events_df["is_regulation"]].copy()
    events_reg["GAME_ID"] = events_reg["GAME_ID"].astype(str).str.strip().str.zfill(10)

    landmarks_df = landmarks_df.reset_index(drop=True).copy()
    landmarks_df["GAME_ID"] = (
        landmarks_df["GAME_ID"].astype(str).str.strip().str.zfill(10)
    )

    # Pre-group events ONCE (O(total_events)) instead of re-filtering the
    # full events_reg frame inside the per-landmark-game loop below
    # (O(num_games * total_events)) -- a dict of per-game frames, looked up
    # in O(1) per game, same output as the old per-iteration boolean mask.
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
        m = game_events["home_away_margin"].to_numpy()

        for landmark_row_id, lm in game_landmarks.iterrows():
            after = t > lm["elapsed_time"]
            t_after = t[after]
            m_after = m[after]
            if len(t_after) == 0:
                continue
            times = np.concatenate([[lm["elapsed_time"]], t_after])
            # k=0 uses landmark_lead directly -- it is ALREADY focal-relative
            # (positive by construction, landmarks.find_landmark's own
            # invariant); only the raw subsequent home_away_margin events
            # need sign_focal applied.
            margins_focal = np.concatenate(
                [[lm["landmark_lead"]], lm["sign_focal"] * m_after]
            )

            for k in range(len(times) - 1):
                rows.append(
                    {
                        "landmark_row_id": landmark_row_id,
                        "GAME_ID": game_id,
                        "SEASON": lm["SEASON"],
                        "LEADER": lm["LEADER"],
                        "OPPONENT": lm["OPPONENT"],
                        "threshold": lm["threshold"],
                        "s_prev": times[k] - lm["elapsed_time"],
                        "s_next": times[k + 1] - lm["elapsed_time"],
                        "dt": times[k + 1] - times[k],
                        "X_prev": margins_focal[k],
                        "X_next": margins_focal[k + 1],
                        "v_BT": lm["v_BT"],
                    }
                )

    return pd.DataFrame(rows, columns=_COLUMNS)
