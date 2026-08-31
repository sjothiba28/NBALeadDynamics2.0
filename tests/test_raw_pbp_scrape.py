"""Durability-contract tests for src/Markov/raw_pbp_scrape.py -- monkeypatched,
no network. Mirrors tests/test_event_scrape.py's checkpoint/os.replace/
non-poisoning contract, plus this module's resume-from-.partial addition."""

import pandas as pd
import pytest

import raw_pbp_scrape as RPS
import scrape_common as R


def _pbp_frame():
    return pd.DataFrame(
        {
            "EVENTNUM": [1],
            "game_time_sec": [10.0],
            "shot_zone": ["Rim"],
            "made": [1.0],
            "is_orb": [False],
            "is_tov": [False],
            "is_ft": [False],
            "opp_drb": [False],
            "event_team": ["A"],
            "poss_start": [1],
        }
    )


def test_build_raw_pbp_dataset_loads_existing_cache(tmp_path, monkeypatch):
    filepath = tmp_path / "cache.csv"
    pd.DataFrame({"GAME_ID": ["G1"], "game_time_sec": [10.0]}).to_csv(
        filepath, index=False
    )

    def fail_fetch(*a, **kw):
        raise AssertionError("should not fetch when cache exists")

    monkeypatch.setattr(R, "fetch_and_parse_game", fail_fetch)

    result = RPS.build_raw_pbp_dataset(
        {"G1": ("2024-25", "Regular Season")}, filepath=str(filepath)
    )
    assert len(result) == 1


def test_build_raw_pbp_dataset_one_bad_game_does_not_abort_run(tmp_path, monkeypatch):
    filepath = tmp_path / "cache.csv"

    def fetch(game_id, require_leader=False):
        if game_id == "BAD":
            raise RuntimeError("boom")
        return _pbp_frame()

    monkeypatch.setattr(R, "fetch_and_parse_game", fetch)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)

    game_ids = {
        "BAD": ("2024-25", "Regular Season"),
        "G2": ("2024-25", "Regular Season"),
    }
    result = RPS.build_raw_pbp_dataset(game_ids, filepath=str(filepath))
    assert set(result["GAME_ID"].unique()) == {"G2"}


def test_build_raw_pbp_dataset_all_fail_raises_and_leaves_no_cache(
    tmp_path, monkeypatch
):
    filepath = tmp_path / "cache.csv"
    monkeypatch.setattr(R, "fetch_and_parse_game", lambda *a, **kw: None)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)

    with pytest.raises(ValueError):
        RPS.build_raw_pbp_dataset(
            {"BAD": ("2024-25", "Regular Season")}, filepath=str(filepath)
        )
    assert not filepath.exists()
    assert not (tmp_path / "cache.csv.partial").exists()


def test_build_raw_pbp_dataset_resumes_from_partial_skips_completed_games(
    tmp_path, monkeypatch
):
    filepath = tmp_path / "cache.csv"
    partial = str(filepath) + ".partial"
    pd.DataFrame(
        {
            "GAME_ID": ["G1"],
            "SEASON": ["2024-25"],
            "SEASON_TYPE": ["Regular Season"],
            "game_time_sec": [10.0],
        }
    ).to_csv(partial, index=False)

    fetched = []

    def fetch(game_id, require_leader=False):
        fetched.append(game_id)
        return _pbp_frame()

    monkeypatch.setattr(R, "fetch_and_parse_game", fetch)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)

    game_ids = {
        "G1": ("2024-25", "Regular Season"),
        "G2": ("2024-25", "Regular Season"),
    }
    RPS.build_raw_pbp_dataset(game_ids, filepath=str(filepath))
    assert fetched == ["G2"]


def test_build_raw_pbp_dataset_stamps_game_season_season_type(tmp_path, monkeypatch):
    filepath = tmp_path / "cache.csv"
    monkeypatch.setattr(
        R, "fetch_and_parse_game", lambda g, require_leader=False: _pbp_frame()
    )
    monkeypatch.setattr(R.time, "sleep", lambda s: None)

    result = RPS.build_raw_pbp_dataset(
        {"G1": ("2024-25", "Playoffs")}, filepath=str(filepath)
    )
    assert result.loc[0, "GAME_ID"] == "G1"
    assert result.loc[0, "SEASON"] == "2024-25"
    assert result.loc[0, "SEASON_TYPE"] == "Playoffs"


def test_build_raw_pbp_dataset_respects_limit(tmp_path, monkeypatch):
    filepath = tmp_path / "cache.csv"
    fetched = []

    def fetch(game_id, require_leader=False):
        fetched.append(game_id)
        return _pbp_frame()

    monkeypatch.setattr(R, "fetch_and_parse_game", fetch)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)

    game_ids = {f"G{i}": ("2024-25", "Regular Season") for i in range(5)}
    RPS.build_raw_pbp_dataset(game_ids, filepath=str(filepath), limit=2)
    assert len(fetched) == 2
