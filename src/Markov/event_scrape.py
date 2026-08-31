"""Scrapes and caches the event-level dataset.

Mirrors landmark_scrape.build_landmark_dataset exactly in structure and
durability contract -- same .partial checkpoint cadence, same empty-frame
guard, same os.replace atomicity. Only the per-game extraction step differs:
events.extract_events in place of landmarks.build_landmark_rows, fanning out
to 0-many event rows per game.

Unlike the landmark dataset, this cache is NOT committed to git: it is new,
~6500 games x tens of events each, and no committed docs/results file
depends on it yet -- .gitignore's `src/Markov/data/*` rule already covers it
with no exception needed (contrast the explicit exceptions carved out there
for nba_5_season_features_v3.csv and nba_5_season_landmarks_v1.csv).
"""

import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (_HERE, os.path.join(_HERE, "..")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import events
import scrape_common as R

DEFAULT_EVENT_DATASET = os.path.join(R.DATA_DIR, "nba_5_season_events_v1.csv")


def build_event_dataset(filepath=None, limit=None):
    filepath = filepath or DEFAULT_EVENT_DATASET
    if os.path.exists(filepath):
        print(f"Loading cached event dataset from {filepath}")
        return pd.read_csv(filepath)

    print(
        "No cache found. Initiating full 5-season event scrape "
        "(This will take a long time...)"
    )
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    partial = filepath + ".partial"

    game_ids = R.get_5_season_game_ids()
    if limit:
        game_ids = dict(list(game_ids.items())[:limit])
    print(f"Found {len(game_ids)} games. Processing...")

    all_frames = []
    failures = 0

    for idx, (g_id, (season, season_type)) in enumerate(game_ids.items()):
        if idx > 0 and idx % 50 == 0:
            print(f"Processed {idx}/{len(game_ids)} games. Saving checkpoint...")
            checkpoint_df = (
                pd.concat(all_frames, ignore_index=True)
                if all_frames
                else pd.DataFrame()
            )
            if not checkpoint_df.empty:
                checkpoint_df.to_csv(partial, index=False)

        try:
            pbp_df = R.fetch_and_parse_game(g_id)
            frame = (
                events.extract_events(pbp_df, g_id, season, season_type)
                if pbp_df is not None and not pbp_df.empty
                else None
            )
        except Exception as e:
            print(f"  Skipping {g_id}: {type(e).__name__}: {e}")
            frame = None

        if frame is not None and not frame.empty:
            all_frames.append(frame)
        else:
            failures += 1

    df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

    if df.empty:
        if os.path.exists(partial):
            os.remove(partial)
        raise ValueError(
            f"No event rows produced from any of {len(game_ids)} games. "
            f"Nothing written to {filepath} -- an empty cache there would be "
            "loaded as a finished scrape by every later run. The "
            "PlayByPlayV3 schema has probably changed; regenerate the test "
            "fixtures (tests/refresh_fixtures.py) to see how."
        )

    df.to_csv(partial, index=False)
    os.replace(partial, filepath)

    print(
        f"Complete. Saved {len(df)} event rows to {filepath} "
        f"({failures} of {len(game_ids)} games contributed none)."
    )
    return df


if __name__ == "__main__":
    event_df = build_event_dataset()
    if event_df.empty:
        print("Event dataset is empty. Ensure your API connection is stable.")
        raise SystemExit(1)
