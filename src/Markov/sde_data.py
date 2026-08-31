"""Regulation-only outcome columns for the SDE study.

landmarks.py's `collapsed`/`final_team_relative_margin` are measured to the
TRUE end of game (including overtime) -- correct for the peak-lead memory
test, wrong for this study, which compares against the Clauset paper's
regulation-only SDE. landmarks.py itself must stay untouched (its own tests
pin those two columns byte-for-byte), so `lead_erased`/
`final_regulation_margin`/`regulation_win` are built HERE, by joining the
landmark table against the event-level time series (events.py) rather than
by touching landmarks.py.

Sign recovery: `landmark_lead` is always positive by construction
(landmarks.find_landmark requires team_margin >= threshold > 0), so a
landmark row names no sign of its own -- the events table carries only the
raw HOME-MINUS-AWAY margin. The sign is recovered by looking up the events
table's home_away_margin at the landmark's own elapsed_time: because
landmark_lead > 0, that lookup value can never be exactly 0, so its sign
always resolves. A missing or duplicated match means the landmark and event
tables have silently diverged (different scrapes, a dropped-clock row) and
must be dropped with a warning, never guessed -- the same failure class
features.identify_leader's 1%-of-attributed-rows cross-check exists to
catch on the leader-tricode side.

regulation_win: 1.0/0.0 by sign of final_regulation_margin, 0.5 if the game
reaches regulation tied (final_regulation_margin == 0) -- the user-chosen
tie convention for building an empirical win-probability CALIBRATION target
only. It does not apply inside the continuous simulator (sde_simulate.py),
where a simulated path lands exactly on 0 with probability zero.
"""

import numpy as np


def attach_regulation_outcomes(
    landmarks_df, events_df, min_time_remaining=30.0, keep_sign_column=False
):
    """landmarks_df + three new columns: lead_erased (int 0/1),
    final_regulation_margin (float), regulation_win (float in {0.0, 0.5,
    1.0}) -- all measured only over regulation-time events. Rows with no
    matching event at their own elapsed_time, or with
    time_remaining < min_time_remaining, are DROPPED -- see module
    docstring for why.
    keep_sign_column=True additionally exposes the internally-computed focal
    sign as 'sign_focal' -- needed by sde_transitions.build_transition_rows
    to convert raw home_away_margin events into focal-relative margins after
    a landmark. Default False preserves the exact historical output schema.
    """
    landmarks_df = landmarks_df.copy()
    landmarks_df["GAME_ID"] = (
        landmarks_df["GAME_ID"].astype(str).str.strip().str.zfill(10)
    )
    events_reg = events_df[events_df["is_regulation"]].copy()
    events_reg["GAME_ID"] = events_reg["GAME_ID"].astype(str).str.strip().str.zfill(10)

    at_landmark = events_reg.rename(
        columns={"game_time_sec": "elapsed_time", "home_away_margin": "_m_at_landmark"}
    )[["GAME_ID", "elapsed_time", "_m_at_landmark"]]

    # A duplicated (GAME_ID, elapsed_time) key makes the lookup ambiguous --
    # any landmark row keyed to it must be dropped and warned about, same as
    # a missing match, never left to hit validate='m:1' and crash the merge.
    dup_keys = at_landmark.duplicated(subset=["GAME_ID", "elapsed_time"], keep=False)
    if dup_keys.any():
        n_dup = int(dup_keys.sum())
        print(
            f"  [WARN] {n_dup} events share a duplicated (GAME_ID, elapsed_time) "
            "key in the lookup table -- affected landmark rows dropped rather "
            "than sign-guessed. Likely cause: events.py failed to collapse "
            "same-clock-time scoring rows into one event, or the events table "
            "was hand-built/out of sync with the landmark table."
        )
    at_landmark = at_landmark[~dup_keys]

    merged = landmarks_df.merge(
        at_landmark, on=["GAME_ID", "elapsed_time"], how="left", validate="m:1"
    )

    missing = merged["_m_at_landmark"].isna()
    if missing.any():
        n_missing = int(missing.sum())
        print(
            f"  [WARN] {n_missing} of {len(merged)} landmark rows had no "
            "matching event at their own elapsed_time -- dropped rather "
            "than sign-guessed. Likely cause: the event and landmark "
            "tables were built from different scrapes/game sets."
        )
    merged = merged[~missing].copy()

    merged["_sign_focal"] = np.where(merged["_m_at_landmark"] > 0, 1.0, -1.0)

    last_reg = (
        events_reg.sort_values("game_time_sec")
        .groupby("GAME_ID", as_index=False)
        .last()[["GAME_ID", "game_time_sec", "home_away_margin"]]
        .rename(
            columns={"home_away_margin": "_m_last_reg", "game_time_sec": "_t_last_reg"}
        )
    )
    merged = merged.merge(last_reg, on="GAME_ID", how="left")
    merged["final_regulation_margin"] = np.where(
        merged["_t_last_reg"].notna(),
        merged["_sign_focal"] * merged["_m_last_reg"],
        0.0,
    )

    merged["lead_erased"] = _any_erased_after_landmark(merged, events_reg).astype(int)
    merged["regulation_win"] = np.select(
        [merged["final_regulation_margin"] > 0, merged["final_regulation_margin"] < 0],
        [1.0, 0.0],
        default=0.5,
    )

    merged = merged[merged["time_remaining"] >= min_time_remaining].copy()
    drop_cols = ["_m_at_landmark", "_t_last_reg", "_m_last_reg"]
    if keep_sign_column:
        return merged.drop(columns=drop_cols).rename(
            columns={"_sign_focal": "sign_focal"}
        )
    return merged.drop(columns=drop_cols + ["_sign_focal"])


def _any_erased_after_landmark(landmark_rows, events_reg):
    """Boolean aligned to landmark_rows.index: True if the focal team's
    signed margin is <= 0 at any regulation event strictly after the
    landmark's own elapsed_time, for that GAME_ID."""
    keyed = landmark_rows.reset_index().rename(columns={"index": "_row_id"})
    joined = keyed[["_row_id", "GAME_ID", "elapsed_time", "_sign_focal"]].merge(
        events_reg[["GAME_ID", "game_time_sec", "home_away_margin"]],
        on="GAME_ID",
        how="left",
    )
    after = joined[joined["game_time_sec"] > joined["elapsed_time"]]
    focal_margin = after["_sign_focal"] * after["home_away_margin"]
    erased_by_row = (focal_margin <= 0).groupby(after["_row_id"]).any()
    return landmark_rows.index.to_series().map(erased_by_row).fillna(False)
