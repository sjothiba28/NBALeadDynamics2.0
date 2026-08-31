"""Unit tests for src/common/pbp.py against hand-built play-by-play.

Each test names the real defect it guards against, because most of these
functions replaced code that failed silently rather than raising.
"""

import pandas as pd
import pytest

from common import pbp
from conftest import free_throw, make_pbp, rebound, shot, turnover


# ---------------------------------------------------------------------------
# Clock parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "clock,expected",
    [
        ("PT11M56.00S", 716.0),
        ("PT12M00.00S", 720.0),
        ("PT00M34.00S", 34.0),
        ("PT05M00.00S", 300.0),
        ("PT00M00.00S", 0.0),
        ("PT00M00.60S", 0.6),
        ("11:56", 716.0),
    ],
)
def test_clock_remaining(clock, expected):
    """A pattern of M(\\d+) matches the minutes marker before the SECONDS, so
    PT11M56.00S used to parse as 56 minutes -> 3416s instead of 716s."""
    assert pbp.clock_remaining(clock) == pytest.approx(expected)


@pytest.mark.parametrize("bad", [None, "", "garbage", float("nan")])
def test_clock_remaining_handles_junk(bad):
    assert pbp.clock_remaining(bad) == 0.0


def test_game_time_regulation_periods():
    df = make_pbp([{"period": p, "clock": "PT00M00.00S"} for p in (1, 2, 3, 4)])
    assert list(pbp.game_time_seconds(df)) == [720, 1440, 2160, 2880]


def test_game_time_overtime_periods_are_five_minutes():
    """Treating OT as 12 minutes pushed OT past the end of regulation and made
    any 'time remaining' figure derived from a 2880s game negative."""
    df = make_pbp([{"period": p, "clock": "PT00M00.00S"} for p in (5, 6, 7)])
    assert list(pbp.game_time_seconds(df)) == [3180, 3480, 3780]


# ---------------------------------------------------------------------------
# Score margin
# ---------------------------------------------------------------------------


def test_score_margin_forward_fills():
    """PlayByPlayV3 has no scoreMargin column at all, and scores are only
    populated on scoring events."""
    df = make_pbp(
        [
            {"scoreHome": "2", "scoreAway": "0"},
            {},  # non-scoring event
            {"scoreHome": "2", "scoreAway": "3"},
        ]
    )
    assert list(pbp.score_margin(df)) == [2, 2, -1]


def test_score_margin_starts_at_zero():
    df = make_pbp([{}, {"scoreHome": "3", "scoreAway": "0"}])
    assert list(pbp.score_margin(df)) == [0, 3]


# ---------------------------------------------------------------------------
# Rebounds
# ---------------------------------------------------------------------------


def test_rebound_is_never_both_offensive_and_defensive():
    """Descriptions carry a running per-player tally '(Off:1 Def:0)', so testing
    for the substrings 'off:' and 'def:' matched EVERY rebound as both."""
    df = make_pbp([rebound("DEN", 1, 1, 0), rebound("LAL", 9, 0, 1)])
    orb, drb = pbp.classify_rebounds(df)
    assert not (orb & drb).any()
    assert list(orb) == [True, False]
    assert list(drb) == [False, True]


def test_rebound_tally_diffed_per_player():
    """A player's second rebound must be read as a delta, not a fresh total."""
    df = make_pbp(
        [
            rebound("DEN", 1, 1, 0),  # first: offensive
            rebound("DEN", 1, 1, 1),  # same player, defensive now
            rebound("DEN", 2, 0, 1),  # different player, defensive
        ]
    )
    orb, drb = pbp.classify_rebounds(df)
    assert list(orb) == [True, False, False]
    assert list(drb) == [False, True, True]


def test_team_rebounds_excluded():
    """Team rebounds carry no per-player tally and are not in the boxscore."""
    df = make_pbp(
        [
            {
                "teamTricode": "",
                "actionType": "Rebound",
                "description": "NUGGETS Rebound",
                "personId": 0,
            }
        ]
    )
    orb, drb = pbp.classify_rebounds(df)
    assert not orb.any() and not drb.any()


def test_classify_rebounds_on_empty_frame():
    orb, drb = pbp.classify_rebounds(make_pbp([]))
    assert len(orb) == 0 and len(drb) == 0


# ---------------------------------------------------------------------------
# Free throws
# ---------------------------------------------------------------------------


def test_free_throw_made_and_missed():
    """shotResult is empty for free throws; only the description says."""
    df = make_pbp([free_throw(made=True), free_throw(made=False)])
    fta, ftm = pbp.free_throw_results(df)
    assert list(fta) == [True, True]
    assert list(ftm) == [True, False]


def test_free_throw_not_matched_by_actiontype_substring():
    """actionType is 'Free Throw' with a space -- a 'freethrow' substring test
    silently matched nothing and only the description fallback fired."""
    df = make_pbp([free_throw()])
    assert pbp._text(df, "actionType").iloc[0] == "Free Throw"
    assert pbp.free_throw_results(df)[0].all()


# ---------------------------------------------------------------------------
# Shot zones
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x,y,desc,expected",
    [
        (0, 20, None, "Rim"),  # restricted area
        (0, 39, None, "Rim"),
        (70, 130, None, "Paint"),  # inside the lane, outside RA
        (0, 200, None, "Mid"),  # long two
        (230, 50, "Murray 24' 3PT Jump Shot", "C3"),
        (0, 260, "Murray 26' 3PT Jump Shot", "ATB3"),
        (-172, 503, "Murray 53' 3PT Jump Shot", "Backcourt"),
    ],
)
def test_shot_zone_geometry(x, y, desc, expected):
    """Rim is the restricted area ONLY. Folding non-RA paint into Rim while the
    season baseline splits them biased the rim and mid deviations oppositely."""
    df = make_pbp([shot(x=x, y=y, desc=desc or "Jokic Layup")])
    assert pbp.classify_shot_zones(df).iloc[0] == expected


def test_non_shot_rows_get_no_zone():
    df = make_pbp([turnover(), rebound("DEN", 1, 1, 0)])
    assert pbp.classify_shot_zones(df).isna().all()


def test_zones_partition_every_field_goal(pbp_regular):
    zone = pbp.classify_shot_zones(pbp_regular)
    n_fg = int(pbp.is_field_goal(pbp_regular).sum())
    assert int(zone.notna().sum()) == n_fg
    # and no zone leaks onto a non-shot row
    assert not (zone.notna() & ~pbp.is_field_goal(pbp_regular)).any()


# ---------------------------------------------------------------------------
# Possessions
# ---------------------------------------------------------------------------


def test_offensive_rebound_continues_the_possession():
    df = make_pbp([shot(), rebound("DEN", 2, 1, 0), shot(made=True)])
    assert pbp.count_possessions(df) == {"DEN": 1}


def test_defensive_rebound_flips_the_possession():
    df = make_pbp([shot("DEN"), rebound("LAL", 9, 0, 1), shot("LAL", made=True)])
    assert pbp.count_possessions(df) == {"DEN": 1, "LAL": 1}


def test_and_one_is_a_single_possession():
    df = make_pbp([shot(made=True), free_throw(n="1 of 1")])
    assert pbp.count_possessions(df) == {"DEN": 1}


def test_multi_shot_free_throw_trip_is_a_single_possession():
    df = make_pbp(
        [free_throw(made=False, n="1 of 2"), free_throw(made=True, n="2 of 2")]
    )
    assert pbp.count_possessions(df) == {"DEN": 1}


def test_technical_free_throw_does_not_open_a_possession():
    """A technical is shot by whichever team was awarded it and does not
    transfer the ball, so counting it invents a possession and splits the real
    one in two."""
    df = make_pbp(
        [
            shot("DEN"),
            free_throw("LAL", n="Technical 1 of 1", person=9),
            shot("DEN", made=True),
        ]
    )
    assert pbp.count_possessions(df) == {"DEN": 1}


def test_period_boundary_opens_a_new_possession():
    df = make_pbp([shot("DEN", made=True), shot("DEN", made=True, period=2)])
    assert pbp.count_possessions(df) == {"DEN": 2}


def test_turnover_ends_the_possession():
    df = make_pbp(
        [shot("DEN"), rebound("LAL", 9, 0, 1), turnover("LAL"), shot("DEN", made=True)]
    )
    assert pbp.count_possessions(df) == {"DEN": 2, "LAL": 1}


def test_possession_apis_agree(pbp_regular):
    total = sum(pbp.count_possessions(pbp_regular).values())
    assert int(pbp.possession_starts(pbp_regular).sum()) == total
    assert len(pbp.possession_events(pbp_regular)) == total


def test_possession_starts_survives_a_non_default_index(pbp_regular):
    """Callers attach this mask to a frame and sum it over slices, so it must be
    label-aligned rather than positional."""
    shuffled = pbp_regular.copy()
    shuffled.index = range(1000, 1000 + len(shuffled))
    assert int(pbp.possession_starts(shuffled).sum()) == int(
        pbp.possession_starts(pbp_regular).sum()
    )


def test_count_possessions_on_empty_frame():
    assert pbp.count_possessions(make_pbp([])) == {}


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def test_missing_column_yields_empty_strings_not_an_error():
    """_text is deliberately forgiving, which is why callers must assert on
    counts -- a missing column produces all-False classifications, not a raise."""
    assert (pbp._text(pd.DataFrame(index=[0, 1]), "nope") == "").all()


def test_is_turnover_and_is_made_shot():
    df = make_pbp([turnover(), shot(made=True), shot(made=False)])
    assert list(pbp.is_turnover(df)) == [True, False, False]
    assert list(pbp.is_made_shot(df)) == [False, True, False]
