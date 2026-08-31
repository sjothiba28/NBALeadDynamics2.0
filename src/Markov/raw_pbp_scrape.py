"""Scrapes and caches raw per-event play-by-play (shot_zone/is_orb/is_tov/
is_ft/event_team), season-restricted, for the state-grid deviation-feature
pipeline. Mirrors event_scrape.build_event_dataset's durability contract
(.partial checkpoint, os.replace atomicity, empty-frame guard) with one
addition: resumes from an existing .partial on start, skipping GAME_IDs
already present, because this scrape's ~1-2.5 hour full-scope runtime makes
losing partial progress to a single interruption far costlier than
event_scrape's shorter, lighter-weight run.
"""

import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (_HERE, os.path.join(_HERE, "..")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import scrape_common as R


def build_raw_pbp_dataset(game_ids, filepath, limit=None):
    if os.path.exists(filepath):
        print(f"Loading cached raw pbp dataset from {filepath}")
        return pd.read_csv(filepath)

    if limit:
        game_ids = dict(list(game_ids.items())[:limit])

    partial = filepath + ".partial"
    all_frames = []
    done_ids = set()
    if os.path.exists(partial):
        cached = pd.read_csv(partial)
        if not cached.empty:
            all_frames.append(cached)
            done_ids = set(cached["GAME_ID"].astype(str).str.zfill(10).unique())
            print(f"Resuming from {partial}: {len(done_ids)} games already done.")

    remaining = {
        gid: v for gid, v in game_ids.items() if str(gid).zfill(10) not in done_ids
    }
    print(f"Found {len(game_ids)} games ({len(remaining)} remaining). Processing...")

    failures = 0
    for idx, (g_id, (season, season_type)) in enumerate(remaining.items()):
        if idx > 0 and idx % 50 == 0:
            print(
                f"Processed {idx}/{len(remaining)} remaining games. Saving checkpoint..."
            )
            checkpoint_df = (
                pd.concat(all_frames, ignore_index=True)
                if all_frames
                else pd.DataFrame()
            )
            if not checkpoint_df.empty:
                checkpoint_df.to_csv(partial, index=False)

        try:
            pbp_df = R.fetch_and_parse_game(g_id, require_leader=False)
        except Exception as e:
            print(f"  Skipping {g_id}: {type(e).__name__}: {e}")
            pbp_df = None

        if pbp_df is not None and not pbp_df.empty:
            pbp_df = pbp_df.copy()
            pbp_df["GAME_ID"] = g_id
            pbp_df["SEASON"] = season
            pbp_df["SEASON_TYPE"] = season_type
            all_frames.append(pbp_df)
        else:
            failures += 1

    df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

    if df.empty:
        if os.path.exists(partial):
            os.remove(partial)
        raise ValueError(
            f"No raw pbp rows produced from any of {len(game_ids)} games. "
            f"Nothing written to {filepath}."
        )

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    df.to_csv(partial, index=False)
    os.replace(partial, filepath)

    print(
        f"Complete. Saved {len(df)} raw pbp rows to {filepath} "
        f"({failures} of {len(remaining)} remaining games contributed none)."
    )
    return df
