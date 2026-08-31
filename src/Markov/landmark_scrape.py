"""Scrapes and caches the landmark dataset.

Mirrors regression.build_dataset (regression.py:197-287) exactly in
structure and durability contract -- same .partial checkpoint cadence, same
empty-frame guard before any write, same os.replace atomicity, same
post-write-strictly-after-replace diagnostic ordering. Only the per-game
extraction step differs: landmarks.build_landmark_rows in place of
extract_game_features, fanning out to 0-6 rows per game instead of 0-1.
"""

import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (_HERE, os.path.join(_HERE, "..")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import landmarks
import scrape_common as R
from common.baselines import fetch_season_baselines
from features import attach_baselines

DEFAULT_LANDMARK_DATASET = os.path.join(R.DATA_DIR, "nba_5_season_landmarks_v1.csv")


def build_landmark_dataset(filepath=None, limit=None):
    """Build (or load) the landmark table.

    Any future schema-incompatible change (new threshold set, changed
    detection semantics) must bump the filename to _v2.csv rather than
    overwrite this one in place -- like build_dataset, this treats any
    existing file at `filepath` as a complete, valid cache with no schema
    check, so an old-schema file would be silently loaded as "done".
    """
    filepath = filepath or DEFAULT_LANDMARK_DATASET
    if os.path.exists(filepath):
        print(f"Loading cached landmark dataset from {filepath}")
        return pd.read_csv(filepath)

    print(
        "No cache found. Initiating full 5-season landmark scrape "
        "(This will take a long time...)"
    )
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    partial = filepath + ".partial"

    game_ids = R.get_5_season_game_ids()
    if limit:
        game_ids = dict(list(game_ids.items())[:limit])
    print(f"Found {len(game_ids)} games. Processing...")

    seasons = sorted({season for season, _season_type in game_ids.values()})
    baselines = fetch_season_baselines(seasons)

    all_rows = []
    failures = 0

    for idx, (g_id, (season, season_type)) in enumerate(game_ids.items()):
        if idx > 0 and idx % 50 == 0:
            print(f"Processed {idx}/{len(game_ids)} games. Saving checkpoint...")
            pd.DataFrame(all_rows).to_csv(partial, index=False)

        # A single bad game must never take down a multi-hour run.
        try:
            pbp_df = R.fetch_and_parse_game(g_id)
            rows = (
                landmarks.build_landmark_rows(pbp_df, g_id, season, season_type)
                if pbp_df is not None and not pbp_df.empty
                else []
            )
        except Exception as e:
            print(f"  Skipping {g_id}: {type(e).__name__}: {e}")
            rows = []

        if rows:
            all_rows.extend(rows)
        else:
            failures += 1

    df = pd.DataFrame(all_rows)

    # An empty frame must never reach the cache path -- see
    # regression.build_dataset's identical guard for why.
    if df.empty:
        if os.path.exists(partial):
            os.remove(partial)
        raise ValueError(
            f"No landmark rows produced from any of {len(game_ids)} games. "
            f"Nothing written to {filepath} -- an empty cache there would be "
            "loaded as a finished scrape by every later run. The "
            "PlayByPlayV3 schema has probably changed; regenerate the test "
            "fixtures (tests/refresh_fixtures.py) to see how."
        )

    df = attach_baselines(df, baselines)

    df.to_csv(partial, index=False)
    os.replace(partial, filepath)

    # Checked only after the data is safely on disk -- raising before the
    # write would throw away a completed multi-hour scrape to report a
    # diagnostic. Mirrors regression.build_dataset:273-283.
    if not df.empty and df["parsed_ft_or_tov"].sum() == 0:
        raise ValueError(
            f"No free throws or turnovers parsed in any of {len(df)} landmark "
            f"rows (saved to {filepath} regardless) -- the PlayByPlayV3 "
            "actionType vocabulary has probably changed."
        )
    print(
        f"Complete. Saved {len(df)} landmark rows to {filepath} "
        f"({failures} of {len(game_ids)} games contributed none)."
    )
    return df


if __name__ == "__main__":
    landmark_df = build_landmark_dataset()
    if landmark_df.empty:
        print("Landmark dataset is empty. Ensure your API connection is stable.")
        raise SystemExit(1)
