"""Driver durability tests for src/Markov/landmark_scrape.py -- monkeypatched,
no network. Mirrors test_audit_regressions.py's build_dataset durability
tests (297-366): same .partial checkpoint / os.replace / non-poisoning
contract, reused verbatim by the driver under test.
"""

import pandas as pd
import pytest

import landmark_scrape as LS
import landmarks
import scrape_common as R


def _row(game_id="x", season="2023-24", leader="DEN", parsed_ft_or_tov=0):
    return {
        "GAME_ID": game_id,
        "SEASON": season,
        "SEASON_TYPE": "Regular Season",
        "LEADER": leader,
        "OPPONENT": "LAL",
        "threshold": 5,
        "landmark_lead": 5.0,
        "elapsed_time": 100.0,
        "period": 1,
        "time_remaining": 2780.0,
        "final_team_relative_margin": 2.0,
        "lead_loss": 3.0,
        "collapsed": 0,
        "C": 0.5,
        "V_var": 0.1,
        "V_qv": 0.1,
        "Backcourt_FGA": 0,
        "Possessions": 5,
        "G_r_Rim": 0.5,
        "G_r_Paint": 0.0,
        "G_r_Mid": 0.0,
        "G_r_C3": 0.0,
        "G_r_ATB3": 0.5,
        "G_r_FT": 0.0,
        "G_p_Rim": 1.0,
        "G_p_Paint": 0.0,
        "G_p_Mid": 0.0,
        "G_p_C3": 0.0,
        "G_p_ATB3": 1.0,
        "G_p_FT": 0.0,
        "G_ORB_pct": 0.0,
        "G_TOV_pct": 0.0,
        "G_ATOP": 20.0,
        "parsed_ft_or_tov": parsed_ft_or_tov,
    }


def _baselines(seasons=("2023-24",), leader="DEN"):
    return pd.DataFrame({"SEASON": list(seasons), "LEADER": [leader] * len(seasons)})


def test_diagnostic_check_runs_after_the_data_is_saved(tmp_path, monkeypatch):
    """Same non-negotiable ordering as scrape_common.build_dataset: the schema
    canary must never discard a completed scrape."""
    target = tmp_path / "out.csv"
    monkeypatch.setattr(
        R,
        "get_5_season_game_ids",
        lambda: {
            "x": ("2023-24", "Regular Season"),
            "y": ("2023-24", "Regular Season"),
        },
    )
    monkeypatch.setattr(R, "fetch_and_parse_game", lambda g: pd.DataFrame({"a": [1]}))
    monkeypatch.setattr(
        landmarks,
        "build_landmark_rows",
        lambda df, g, s, t: [_row(game_id=g, season=s, parsed_ft_or_tov=0)],
    )
    monkeypatch.setattr(
        LS, "fetch_season_baselines", lambda seasons: _baselines(seasons)
    )

    with pytest.raises(ValueError, match="vocabulary"):
        LS.build_landmark_dataset(filepath=str(target))

    assert target.exists(), "scraped data was discarded by the diagnostic"
    assert len(pd.read_csv(target)) == 2
    assert not (tmp_path / "out.csv.partial").exists()


def test_a_run_that_produces_no_rows_does_not_poison_the_cache(tmp_path, monkeypatch):
    target = tmp_path / "out.csv"
    monkeypatch.setattr(
        R, "get_5_season_game_ids", lambda: {"x": ("2023-24", "Regular Season")}
    )
    monkeypatch.setattr(R, "fetch_and_parse_game", lambda g: None)
    monkeypatch.setattr(landmarks, "build_landmark_rows", lambda df, g, s, t: [])
    monkeypatch.setattr(
        LS, "fetch_season_baselines", lambda seasons: _baselines(seasons)
    )

    with pytest.raises(ValueError, match=r"(?i)no landmark rows"):
        LS.build_landmark_dataset(filepath=str(target))

    assert not target.exists(), "an empty frame reached the cache path"
    assert not (tmp_path / "out.csv.partial").exists()


def test_one_bad_game_does_not_abort_the_run(tmp_path, monkeypatch):
    target = tmp_path / "out.csv"
    monkeypatch.setattr(
        R,
        "get_5_season_game_ids",
        lambda: {g: ("2023-24", "Regular Season") for g in ["good", "bad", "good2"]},
    )
    monkeypatch.setattr(R, "fetch_and_parse_game", lambda g: pd.DataFrame({"a": [1]}))

    def flaky(df, game_id, season, season_type):
        flaky.n += 1
        if flaky.n == 2:
            raise RuntimeError("boom")
        return [_row(game_id=game_id, season=season, parsed_ft_or_tov=1)]

    flaky.n = 0
    monkeypatch.setattr(landmarks, "build_landmark_rows", flaky)
    monkeypatch.setattr(
        LS, "fetch_season_baselines", lambda seasons: _baselines(seasons)
    )

    out = LS.build_landmark_dataset(filepath=str(target))
    assert len(out) == 2, "the surviving games should still be written"


def test_default_dataset_path_is_versioned_separately_from_the_peak_dataset():
    assert LS.DEFAULT_LANDMARK_DATASET != R.DEFAULT_DATASET
    assert "landmarks" in LS.DEFAULT_LANDMARK_DATASET


def test_cached_file_is_loaded_without_rescraping(tmp_path, monkeypatch):
    target = tmp_path / "out.csv"
    pd.DataFrame([_row()]).to_csv(target, index=False)

    def boom():
        raise AssertionError("should not scrape when a cache exists")

    monkeypatch.setattr(R, "get_5_season_game_ids", boom)

    out = LS.build_landmark_dataset(filepath=str(target))
    assert len(out) == 1
