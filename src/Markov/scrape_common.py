"""Scraper infrastructure for the NBA play-by-play datasets.

MIGRATION NOTE: extracted verbatim from the origin repo's
src/Markov/regression.py (lines 1-315) when this bundle was built.
That file held two unrelated halves under one name: this scraper
infrastructure, and the "Models 1-4" OLS ladder. Only the scraper half
migrated -- the models were deliberately excluded (see MIGRATION.md).
The single removed line was `from models import compare_models, get_vif,
prepare_model_frame`, which only regression.py's own __main__ analysis
block used. Everything below is otherwise unchanged.

The three scrapers (raw_pbp_scrape, event_scrape, landmark_scrape) import
this module as R and use DATA_DIR, fetch_and_parse_game and get_*.
"""

import pandas as pd
import numpy as np
import os
import sys
import time

from nba_api.stats.endpoints import leaguegamefinder, playbyplayv3
from requests.exceptions import RequestException, ReadTimeout, ConnectionError

_HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (_HERE, os.path.join(_HERE, "..")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from common import pbp as pbp_common
from common.baselines import fetch_season_baselines
from features import extract_game_features, identify_leader, attach_baselines

# ==========================================
# 0. RETRY WRAPPER
# ==========================================

# This module used to pass a hand-rolled "browser spoofing" header dict to every
# endpoint. nba_api does not merge those with its own defaults, it replaces them,
# which dropped Accept-Encoding, Cache-Control, Pragma, Sec-Ch-Ua and
# Sec-Ch-Ua-Mobile. stats.nba.com then accepted the connection but never
# responded, so every request burned the full 60s timeout and all three retries
# before the game was dropped. The library's own headers are maintained and work,
# so nothing is passed now.


def safe_api_call(api_func, *args, max_retries=5, initial_delay=2, **kwargs):
    """
    Wraps NBA API calls with exponential backoff to handle
    ConnectionResetError (10054), timeouts, and server drops.
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return api_func(*args, **kwargs)
        except (
            RequestException,
            ReadTimeout,
            ConnectionError,
            ConnectionResetError,
            OSError,
        ) as e:
            if attempt == max_retries - 1:
                print(f"Max retries reached. Final Error: {e}")
                raise e
            print(
                f"Connection dropped ({e}). Retrying in {delay}s (Attempt {attempt + 1}/{max_retries})..."
            )
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"Unexpected error ({e}). Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2
    return None


# ==========================================
# 1. API EXTRACTION & CACHING (LAST 5 SEASONS)
# ==========================================


def get_game_ids(seasons, season_types=("Regular Season", "Playoffs")):
    """{game_id: (season, season_type)} for the given seasons, with auto-retry.

    The season has to travel with the id: every game row is joined to its
    leading team's per-season baseline downstream, so a bare id list cannot be
    merged against anything.
    """
    games_by_id = {}

    print(f"Fetching Game IDs for {seasons}...")
    for season in seasons:
        for s_type in season_types:
            # Bound as defaults, not read from the enclosing scope: safe_api_call
            # happens to invoke this in the same iteration, but a closure over the
            # loop variables would silently fetch the last season for every
            # request the moment that call is deferred or parallelised.
            def fetch_gf(season=season, s_type=s_type):
                gamefinder = leaguegamefinder.LeagueGameFinder(
                    season_nullable=season,
                    season_type_nullable=s_type,
                    league_id_nullable="00",
                    timeout=60,
                )
                return gamefinder.get_data_frames()[0]

            try:
                games = safe_api_call(fetch_gf, max_retries=5, initial_delay=3)
                for gid in games["GAME_ID"].astype(str).str.zfill(10).unique():
                    games_by_id.setdefault(gid, (season, s_type))
                time.sleep(1.0)
            except Exception as e:
                print(f"Error fetching {season} {s_type}: {e}")

    if not games_by_id:
        raise ValueError(
            "Failed to fetch any games. The NBA API is blocking connections."
        )

    return games_by_id


def get_5_season_game_ids():
    """{game_id: (season, season_type)} for the last 5 seasons -- see
    get_game_ids for the retry/merge contract."""
    return get_game_ids(["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"])


def fetch_and_parse_game(game_id, require_leader=True):
    """
    Pulls Play-by-Play for a single game and classifies spatial zones and game
    events with auto-retry.

    Everything is derived from the V3 feed alone. An earlier version also pulled
    ShotChartDetail and merged it on the event id, which had two problems:
    the endpoint returns nothing at all for playoff games, and actionNumber is
    not unique on the play-by-play side (a block shares the action number of the
    shot it blocked), so the merge attached shot data to companion rows and
    inflated attempts by around 5%.
    """
    try:

        def fetch_pbp():
            return playbyplayv3.PlayByPlayV3(
                game_id=game_id, timeout=60
            ).get_data_frames()[0]

        pbp = safe_api_call(fetch_pbp, max_retries=3, initial_delay=2)
        time.sleep(0.8)

        if pbp is None or pbp.empty:
            return None

        # V3 exposes no score-margin column; derive it from the running score.
        margin = pbp_common.score_margin(pbp)
        pbp["game_time_sec"] = pbp_common.game_time_seconds(pbp)

        # Rows with an unparseable clock are dropped from the parsed frame
        # below, so the leader must be picked from the surviving rows only.
        # Naming the leader off the full feed while extract_game_features takes
        # the sign of the margin off the filtered one lets the two disagree
        # whenever a peak-margin row loses its clock -- and a disagreement there
        # is silent: the pre-peak window would be filtered to the TRAILING team.
        # Safe to filter for this one call because identify_leader reads only
        # scores and tricodes; the possession walk below cannot be sliced.
        timed = pbp["game_time_sec"].notna()
        leaders = identify_leader(pbp[timed])
        leader = None
        if leaders is not None:
            leader, _trailer = leaders
        elif require_leader:
            # No team ever led, or the tricodes could not be resolved. Either
            # way there is no leading team to measure, so the game carries no
            # signal rather than being measured against the wrong side.
            # require_leader=False callers (state_grid's raw pbp scrape)
            # resolve LEADER independently from the events cache and don't
            # need this internal resolution to succeed.
            return None

        is_orb, is_drb = pbp_common.classify_rebounds(pbp)
        is_fta, is_ftm = pbp_common.free_throw_results(pbp)
        zone = pbp_common.classify_shot_zones(pbp)

        made = pd.Series(np.nan, index=pbp.index, dtype=float)
        made[zone.notna()] = pbp_common.is_made_shot(pbp)[zone.notna()].astype(float)
        # Free throws carry no shot zone, so their result has to be filled in
        # separately or the FT percentage below is computed over an empty slice.
        made[is_fta] = is_ftm[is_fta].astype(float)

        df = pd.DataFrame(
            {
                "EVENTNUM": pbp["actionNumber"],
                "game_time_sec": pbp["game_time_sec"],
                "lead": margin.abs(),
                "signed_margin": margin,
                "is_ft": is_fta,
                "is_tov": pbp_common.is_turnover(pbp),
                "is_orb": is_orb,
                "opp_drb": is_drb,
                "shot_zone": zone,
                "made": made,
                # Not the raw teamTricode: team-charged turnovers (shot clock,
                # excess timeout) arrive blank and name the team only in the
                # description, so matching on the raw column drops them from the
                # leading team's window entirely.
                "event_team": pbp_common.event_team(pbp),
                # Constant per game. Carried on the frame so the parsed output is
                # self-contained -- the leader can only be named from the raw feed's
                # team scores, which do not survive into this table.
                "leader": leader,
                # Marked here, on the whole game, because the possession walk depends
                # on a running per-player rebound tally and cannot be re-run on a
                # slice. Summing this column over a window is safe.
                "poss_start": pbp_common.possession_starts(pbp),
            }
        )

        df = df.dropna(subset=["game_time_sec"])
        return df

    except Exception as e:
        print(f"Failed to process Game {game_id} after retries: {e}")
        return None


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_DATASET = os.path.join(DATA_DIR, "nba_5_season_features_v3.csv")


def build_dataset(filepath=None, limit=None):
    """Build (or load) the per-game feature table.

    Checkpoints go to a .partial file and are renamed on completion, so a run
    that dies midway cannot leave a truncated file that the next run mistakes
    for a finished cache.
    """
    filepath = filepath or DEFAULT_DATASET
    if os.path.exists(filepath):
        print(f"Loading cached dataset from {filepath}")
        return pd.read_csv(filepath)

    print(
        "No cache found. Initiating full 5-season download (This will take a long time...)"
    )
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    partial = filepath + ".partial"

    game_ids = get_5_season_game_ids()
    if limit:
        game_ids = dict(list(game_ids.items())[:limit])
    print(f"Found {len(game_ids)} games. Processing...")

    # Fetched once per run rather than per game: each row's B_/D1_ block only
    # needs its own team's SEASON/LEADER identity, which never changes
    # mid-scrape. Regular Season only, on
    # purpose: game_ids spans both season types, so playoff rows are
    # differenced against their team's regular-season identity. See the
    # [NOTE] line attach_baselines prints and docs/results/models-1-4.md.
    seasons = sorted({season for season, _season_type in game_ids.values()})
    baselines = fetch_season_baselines(seasons)

    all_game_features = []
    failures = 0

    for idx, (g_id, (season, season_type)) in enumerate(game_ids.items()):
        if idx > 0 and idx % 50 == 0:
            print(f"Processed {idx}/{len(game_ids)} games. Saving checkpoint...")
            pd.DataFrame(all_game_features).to_csv(partial, index=False)

        # A single bad game must never take down a multi-hour run.
        try:
            pbp_df = fetch_and_parse_game(g_id)
            features = (
                extract_game_features(pbp_df, g_id, season, season_type)
                if pbp_df is not None and not pbp_df.empty
                else None
            )
        except Exception as e:
            print(f"  Skipping {g_id}: {type(e).__name__}: {e}")
            features = None

        if features:
            all_game_features.append(features)
        else:
            failures += 1

    df = pd.DataFrame(all_game_features)

    # An empty frame must never reach the cache path. Games were found, so this
    # is not "no data upstream" -- every one of them failed to parse, which in
    # practice means a PlayByPlayV3 schema change. Writing it and os.replace-ing
    # it on would poison the cache permanently: every later run loads the empty
    # file as a finished scrape and exits with "Ensure your API connection is
    # stable", with nothing pointing at the cache as the culprit.
    if df.empty:
        if os.path.exists(partial):
            os.remove(partial)
        raise ValueError(
            f"All {len(game_ids)} games failed to parse, so the run produced no "
            f"rows. Nothing written to {filepath} -- an empty cache there would "
            "be loaded as a finished scrape by every later run. The "
            "PlayByPlayV3 schema has probably changed; regenerate the test "
            "fixtures (tests/refresh_fixtures.py) to see how."
        )

    df = attach_baselines(df, baselines)

    df.to_csv(partial, index=False)
    os.replace(partial, filepath)

    # Checked only after the data is safely on disk. The per-game flag is
    # meaningless individually -- a game can legitimately have no pre-peak free
    # throw or turnover -- but a whole run without one means the actionType
    # vocabulary changed. Raising before the write would throw away a completed
    # multi-hour scrape to report a diagnostic.
    if not df.empty and df["parsed_ft_or_tov"].sum() == 0:
        raise ValueError(
            f"No free throws or turnovers parsed in any of {len(df)} games "
            f"(saved to {filepath} regardless) -- the PlayByPlayV3 actionType "
            "vocabulary has probably changed."
        )
    # Report drops rather than silently analysing a non-random subset.
    print(
        f"Complete. Saved {len(df)} games to {filepath} "
        f"({failures} of {len(game_ids)} dropped)."
    )
    return df
