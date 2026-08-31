"""Event-level signed-margin time series for src/Markov/events.py -- the
Clauset-paper event resolution (one row per distinct scoring instant,
same-clock-time events combined), built from regression.fetch_and_parse_game's
tidy frame, no network."""

import pandas as pd
import pytest

import events


def make_tidy_frame(times, signed_margin, event_team):
    """Mirrors landmarks.py's tests' make_landmark_frame -- a tidy frame
    shaped like fetch_and_parse_game's output, minimal columns needed by
    extract_events."""
    return pd.DataFrame(
        {
            "game_time_sec": times,
            "signed_margin": signed_margin,
            "event_team": event_team,
        }
    )


def test_extract_events_keeps_only_rows_where_margin_actually_changes():
    df = make_tidy_frame(
        times=[10, 20, 30], signed_margin=[2, 2, 5], event_team=["DEN", "DEN", "LAL"]
    )
    # t=20 is a non-scoring row (margin unchanged from t=10) -- a rebound or
    # missed shot at the same margin -- and must not appear as an event.
    out = events.extract_events(df, "G1", "2023-24", "Regular Season")
    assert list(out["game_time_sec"]) == [10, 30]
    assert list(out["points"]) == [2, 3]


def test_same_clock_time_scoring_events_collapse_into_one_row():
    """A made shot immediately followed by a same-clock-time technical free
    throw (e.g. a make plus a delay-of-game T) must produce ONE event row at
    that clock time, at the net cumulative margin, with points summing both
    -- the Clauset paper's same-instant convention, not two rows that would
    double-count the antipersistence/rate statistics fit_clauset computes."""
    df = make_tidy_frame(
        times=[10, 500, 500, 900],
        signed_margin=[
            2,
            4,
            5,
            5,
        ],  # DEN 2pt make; then DEN make+FT at t=500; t=900 no change
        event_team=["DEN", "DEN", "DEN", "LAL"],
    )
    out = events.extract_events(df, "G1", "2023-24", "Regular Season")
    assert list(out["game_time_sec"]) == [10, 500]
    assert list(out["home_away_margin"]) == [2, 5]
    assert list(out["points"]) == [2, 3]  # 3 = (5-2), the net across the t=500 cluster
    assert list(out["scoring_team"]) == ["DEN", "DEN"]


def test_is_regulation_flag_matches_the_2880s_cutoff():
    df = make_tidy_frame(
        times=[100, 2900], signed_margin=[2, 4], event_team=["DEN", "LAL"]
    )
    out = events.extract_events(df, "G1", "2023-24", "Regular Season")
    assert list(out["is_regulation"]) == [True, False]


def test_empty_or_none_input_returns_an_empty_typed_frame():
    for df in (
        None,
        pd.DataFrame(columns=["game_time_sec", "signed_margin", "event_team"]),
    ):
        out = events.extract_events(df, "G1", "2023-24", "Regular Season")
        assert out.empty
        assert set(out.columns) == {
            "GAME_ID",
            "SEASON",
            "SEASON_TYPE",
            "game_time_sec",
            "home_away_margin",
            "scoring_team",
            "points",
            "is_regulation",
        }


def test_a_game_with_no_scoring_at_all_returns_empty():
    df = make_tidy_frame(
        times=[10, 20], signed_margin=[0, 0], event_team=["DEN", "LAL"]
    )
    out = events.extract_events(df, "G1", "2023-24", "Regular Season")
    assert out.empty


@pytest.mark.parametrize("game_id", ["0022300061", "0022300476"])
def test_extract_events_on_real_fixtures_produces_a_monotone_nonzero_stream(game_id):
    """Integration case a synthetic frame can't fully exercise: real feed
    noise (duplicate actionNumbers, forward-filled scores) interacting with
    the collapsing logic on a genuine game."""
    from conftest import load_pbp
    from common import pbp as pbp_common

    raw = load_pbp(game_id)
    tidy = pd.DataFrame(
        {
            "game_time_sec": pbp_common.game_time_seconds(raw),
            "signed_margin": pbp_common.score_margin(raw),
            "event_team": pbp_common.event_team(raw),
        }
    ).dropna(subset=["game_time_sec"])
    out = events.extract_events(tidy, game_id, "2023-24", "Regular Season")
    assert not out.empty
    assert out["game_time_sec"].is_monotonic_increasing
    assert (out["points"] > 0).all()
