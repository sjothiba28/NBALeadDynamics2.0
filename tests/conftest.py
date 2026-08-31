"""Shared fixtures.

Tests run offline against play-by-play captured in tests/fixtures, so they are
deterministic and do not depend on stats.nba.com being reachable or on the
current season's data. Regenerate with tests/refresh_fixtures.py.
"""

import os
import sys

import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

for path in (SRC, os.path.join(SRC, "Markov"), os.path.join(SRC, "PCA")):
    if path not in sys.path:
        sys.path.insert(0, path)

# Each fixture covers a case that has previously broken this code.
GAMES = {
    "0022300061": "regular season",
    "0022300476": "double overtime (6 periods)",
    "0042200401": "playoff game (ShotChartDetail returns nothing)",
    "0022100001": "contains a flagrant foul",
    "0022400500": "2024-25 season",
}


def load_pbp(game_id):
    """Play-by-play exactly as PlayByPlayV3 returns it.

    keep_default_na=False matters: several columns are empty strings in the live
    feed (teamTricode on team rebounds, shotResult on free throws) and the code
    under test distinguishes '' from NaN.
    """
    return pd.read_csv(
        os.path.join(FIXTURES, f"pbp_{game_id}.csv"),
        keep_default_na=False,
        low_memory=False,
    )


def load_box(game_id):
    return pd.read_csv(os.path.join(FIXTURES, f"box_{game_id}.csv"))


@pytest.fixture(scope="session")
def pbp_regular():
    return load_pbp("0022300061")


@pytest.fixture(scope="session")
def pbp_overtime():
    return load_pbp("0022300476")


@pytest.fixture(scope="session")
def pbp_playoff():
    return load_pbp("0042200401")


@pytest.fixture(params=sorted(GAMES), ids=lambda g: f"{g}-{GAMES[g].split()[0]}")
def game_id(request):
    """Every fixture game, one test invocation each."""
    return request.param


CLOCK = "PT10M00.00S"
_COLS = [
    "period",
    "teamTricode",
    "actionType",
    "description",
    "isFieldGoal",
    "personId",
    "shotResult",
    "subType",
    "clock",
    "xLegacy",
    "yLegacy",
    "scoreHome",
    "scoreAway",
]


def make_pbp(rows):
    """Build a minimal V3-shaped frame from partial row dicts."""
    default = {
        "period": 1,
        "teamTricode": "DEN",
        "actionType": "",
        "description": "",
        "isFieldGoal": 0,
        "personId": 1,
        "shotResult": "",
        "subType": "",
        "clock": CLOCK,
        "xLegacy": 0,
        "yLegacy": 0,
        "scoreHome": "",
        "scoreAway": "",
    }
    return pd.DataFrame([{**default, **r} for r in rows], columns=_COLS)


def shot(team="DEN", made=False, x=0, y=0, desc=None, **kw):
    return {
        "teamTricode": team,
        "isFieldGoal": 1,
        "actionType": "Made Shot" if made else "Missed Shot",
        "description": desc
        if desc is not None
        else ("Jokic 5ft Layup" if made else "MISS Jokic 5ft Layup"),
        "xLegacy": x,
        "yLegacy": y,
        **kw,
    }


def rebound(team, person, off, dfn, **kw):
    return {
        "teamTricode": team,
        "actionType": "Rebound",
        "personId": person,
        "description": f"Player REBOUND (Off:{off} Def:{dfn})",
        **kw,
    }


def free_throw(team="DEN", made=True, n="1 of 2", person=1, **kw):
    return {
        "teamTricode": team,
        "actionType": "Free Throw",
        "personId": person,
        "subType": f"Free Throw {n}",
        "description": (
            f"Jokic Free Throw {n} (2 PTS)" if made else f"MISS Jokic Free Throw {n}"
        ),
        **kw,
    }


def turnover(team="DEN", **kw):
    return {
        "teamTricode": team,
        "actionType": "Turnover",
        "description": "Bad Pass Turnover",
        **kw,
    }
