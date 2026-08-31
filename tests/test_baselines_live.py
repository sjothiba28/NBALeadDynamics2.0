"""Baseline tests that hit stats.nba.com.

Deselected by default because they are slow and depend on the API being up:

    pytest                     # offline suite only
    pytest -m network          # just these
    pytest -m ''               # everything
"""

import pytest

from common import baselines


@pytest.mark.network
def test_baselines_are_per_team_not_league_constants():
    """The whole point: B must vary across teams, or Models 3 and 4 are
    rank-deficient and the B block collapses into the intercept."""
    df = baselines.fetch_season_baselines(["2023-24"])
    assert len(df) == 30
    for col in baselines.BASELINE_COLUMNS:
        assert df[col].nunique() > 1, f"{col} is constant across all 30 teams"


@pytest.mark.network
def test_zone_rates_partition_attempts():
    df = baselines.fetch_season_baselines(["2023-24"])
    total = sum(df[f"b_r_{z}"] for z in ["Rim", "Paint", "Mid", "C3", "ATB3"])
    assert total.between(0.97, 1.005).all()


@pytest.fixture(scope="module")
def two_season_baselines():
    return baselines.fetch_season_baselines(["2021-22", "2023-24"])


@pytest.mark.network
def test_one_row_per_team_per_season(two_season_baselines):
    assert set(two_season_baselines["SEASON"]) == {"2021-22", "2023-24"}
    assert two_season_baselines.groupby("SEASON").size().eq(30).all()


@pytest.mark.network
def test_corner_three_not_double_counted(two_season_baselines):
    """The endpoint publishes 'Left Corner 3', 'Right Corner 3' AND a combined
    'Corner 3' column, so a substring regex summed all three and returned exactly
    twice the true corner-3 volume."""
    assert two_season_baselines["b_r_C3"].mean() < 0.15


@pytest.mark.network
def test_baselines_differ_by_season(two_season_baselines):
    """A pooled baseline reused across years would make these identical."""
    pivot = two_season_baselines.pivot_table(
        index="LEADER", columns="SEASON", values="b_r_ATB3"
    )
    assert not pivot["2021-22"].equals(pivot["2023-24"])


@pytest.mark.network
def test_shooting_percentages_are_plausible(two_season_baselines):
    assert two_season_baselines["b_p_Rim"].between(0.55, 0.80).all()
    assert two_season_baselines["b_p_C3"].between(0.30, 0.50).all()
    assert two_season_baselines["b_p_ATB3"].between(0.28, 0.45).all()


@pytest.mark.network
def test_every_season_in_config_resolves():
    """A season whose shot-location columns are missing now raises rather than
    silently dropping the whole season from the inner merge."""
    seasons = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    result = baselines.fetch_season_baselines(seasons)
    assert set(result["SEASON"]) == set(seasons)


@pytest.mark.network
def test_game_joins_its_own_season_baseline(two_season_baselines):
    import pandas as pd

    games = pd.DataFrame(
        [{"SEASON": "2021-22", "LEADER": "BOS"}, {"SEASON": "2023-24", "LEADER": "BOS"}]
    )
    merged = games.merge(
        two_season_baselines[["SEASON", "LEADER", "b_r_ATB3"]],
        on=["SEASON", "LEADER"],
        how="inner",
    )
    assert len(merged) == 2
    assert merged["b_r_ATB3"].nunique() == 2
