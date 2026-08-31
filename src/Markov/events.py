"""Event-level signed-margin time series, one row per distinct scoring event.

Landmark rows (landmarks.py) know only the state AT a landmark; testing the
Clauset SDE against real history needs the score margin's full trajectory
between landmarks, at Clauset's own event resolution (one row per distinct
clock time a score changes) rather than per play-by-play row. Multiple
same-clock-time actions (a make plus a same-instant technical free throw)
are the paper's own convention: they are one scoring event, not two, so this
module collapses them before anything downstream (fit_clauset's p/s/dt, or
the sde_data.py outcome join) ever sees a row.

`home_away_margin` here is common.pbp.score_margin()'s home-minus-away
convention -- NOT landmark-focal-relative -- because a single event table
serves every landmark row in a game, home and away alike; the sign is
recovered per-landmark downstream (sde_data.py), not baked in here.
"""

import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (_HERE, os.path.join(_HERE, "..")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from common import pbp as pbp_common

_COLUMNS = [
    "GAME_ID",
    "SEASON",
    "SEASON_TYPE",
    "game_time_sec",
    "home_away_margin",
    "scoring_team",
    "points",
    "is_regulation",
]


def _empty_events():
    return pd.DataFrame(columns=_COLUMNS)


def extract_events(pbp_df, game_id, season, season_type):
    if pbp_df is None or pbp_df.empty:
        return _empty_events()

    df = pbp_df.reset_index(drop=True)
    margin = df["signed_margin"].astype(float)
    team = df["event_team"].astype(str).str.strip()
    time = df["game_time_sec"].astype(float)

    prev_margin = margin.shift(1).fillna(0.0)
    scored = margin.ne(prev_margin)
    if not scored.any():
        return _empty_events()

    raw = pd.DataFrame(
        {
            "game_time_sec": time[scored].to_numpy(),
            "margin": margin[scored].to_numpy(),
            "team": team[scored].to_numpy(),
        }
    )

    # Same-clock-time scoring rows collapse into ONE event at that clock
    # time's FINAL cumulative margin. groupby(sort=True) sorts by
    # game_time_sec but is stable within each group, so .last() picks the
    # temporally last row at that clock time -- the Clauset paper's
    # same-instant convention. `points` is the net change across the whole
    # cluster, already correct because margin is cumulative.
    last = raw.groupby("game_time_sec", sort=True, as_index=False).last()
    prev = last["margin"].shift(1).fillna(0.0)
    points = (last["margin"] - prev).abs()

    return pd.DataFrame(
        {
            "GAME_ID": game_id,
            "SEASON": season,
            "SEASON_TYPE": season_type,
            "game_time_sec": last["game_time_sec"],
            "home_away_margin": last["margin"],
            "scoring_team": last["team"].replace("", None),
            "points": points,
            "is_regulation": last["game_time_sec"] <= pbp_common.REGULATION_END,
        }
    )
