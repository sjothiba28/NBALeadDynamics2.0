"""Validate the classifiers against official NBA boxscores.

These are the tests that actually prove correctness: every classification
decision rolls up to a number the NBA publishes independently. Run offline
against cached fixtures.
"""

import pandas as pd

from common import pbp
from conftest import load_box, load_pbp


def team_rows(game_id):
    return load_box(game_id).set_index("teamTricode")


def test_rebounds_match_boxscore_exactly(game_id):
    """The previous implementation flagged every rebound as both offensive and
    defensive, so phase-1 ORB averaged 31.5/game against a true ~10."""
    df = load_pbp(game_id)
    box = load_box(game_id)
    orb, drb = pbp.classify_rebounds(df)

    for _, row in box.iterrows():
        mask = df["teamId"] == row["teamId"]
        assert int((orb & mask).sum()) == int(row["reboundsOffensive"]), (
            f"{row['teamTricode']} offensive rebounds"
        )
        assert int((drb & mask).sum()) == int(row["reboundsDefensive"]), (
            f"{row['teamTricode']} defensive rebounds"
        )


def test_field_goals_match_boxscore(game_id):
    df = load_pbp(game_id)
    box = load_box(game_id)
    zone = pbp.classify_shot_zones(df)
    made = pbp.is_made_shot(df)

    for _, row in box.iterrows():
        mask = df["teamId"] == row["teamId"]
        assert int((zone.notna() & mask).sum()) == int(row["fieldGoalsAttempted"])
        assert int((zone.notna() & mask & made).sum()) == int(row["fieldGoalsMade"])


def test_free_throws_match_boxscore(game_id):
    df = load_pbp(game_id)
    box = load_box(game_id)
    fta, ftm = pbp.free_throw_results(df)

    for _, row in box.iterrows():
        mask = df["teamId"] == row["teamId"]
        assert int((fta & mask).sum()) == int(row["freeThrowsAttempted"])
        assert int((ftm & mask).sum()) == int(row["freeThrowsMade"])


def test_possessions_are_physically_plausible(game_id):
    """Possessions alternate, so the two teams' totals cannot diverge much, and
    pace has a narrow real-world range."""
    df = load_pbp(game_id)
    counts = pbp.count_possessions(df)
    assert len(counts) == 2

    periods = pd.to_numeric(df["period"], errors="coerce").max()
    minutes = 48 + max(0, periods - 4) * 5
    values = sorted(counts.values())

    assert values[1] - values[0] <= 3, f"implausible team asymmetry: {counts}"
    for team, n in counts.items():
        pace = n / minutes * 48
        assert 80 <= pace <= 115, f"{team} pace {pace:.1f} outside plausible range"


def test_possessions_alternate_within_each_period(game_id):
    """Within a period possessions strictly alternate, so per-team counts differ
    by at most one. Classification must happen on the whole game -- slicing to a
    period corrupts the running rebound tally.

    Flagrant fouls award free throws AND retain the ball, which this walk scores
    as one possession rather than two, so allow a single period to be off by one
    extra in a game containing one.
    """
    df = load_pbp(game_id)
    events = pbp.possession_events(df)
    has_flagrant = df["subType"].astype(str).str.contains("Flagrant", case=False).any()
    tolerance = 2 if has_flagrant else 1

    for period in sorted({p for p, _, _ in events}):
        teams = [t for p, _, t in events if p == period]
        counts = sorted(teams.count(t) for t in set(teams))
        if len(counts) == 2:
            assert counts[1] - counts[0] <= tolerance, (
                f"period {period} asymmetry {counts} in {game_id}"
            )


def test_shot_zones_sum_to_total_attempts(game_id):
    """No shot may be dropped or counted twice across the zone partition."""
    df = load_pbp(game_id)
    zone = pbp.classify_shot_zones(df)
    counts = zone.value_counts()
    assert counts.sum() == int(pbp.is_field_goal(df).sum())
    assert set(counts.index) <= {"Rim", "Paint", "Mid", "C3", "ATB3", "Backcourt"}


def test_turnovers_are_close_to_boxscore(game_id):
    """PBP counts team turnovers (shot-clock violations) that the player boxscore
    does not, so this is a bounded check rather than an equality."""
    df = load_pbp(game_id)
    box = load_box(game_id)
    tov = pbp.is_turnover(df)
    for _, row in box.iterrows():
        mask = df["teamId"] == row["teamId"]
        diff = int((tov & mask).sum()) - int(row["turnovers"])
        assert 0 <= diff <= 6, f"{row['teamTricode']} turnover gap {diff}"
