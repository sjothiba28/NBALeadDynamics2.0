"""Driver durability tests for src/Markov/event_scrape.py -- monkeypatched,
no network. Mirrors tests/test_landmark_scrape.py's .partial checkpoint /
os.replace / non-poisoning contract, reused verbatim by the driver under
test."""

import pandas as pd
import pytest

import event_scrape as ES
import events
import scrape_common as R


def _frame(game_id="x", season="2023-24"):
    return pd.DataFrame(
        {
            "GAME_ID": [game_id],
            "SEASON": [season],
            "SEASON_TYPE": ["Regular Season"],
            "game_time_sec": [10.0],
            "home_away_margin": [2.0],
            "scoring_team": ["DEN"],
            "points": [2.0],
            "is_regulation": [True],
        }
    )


def test_a_run_that_produces_no_rows_does_not_poison_the_cache(tmp_path, monkeypatch):
    target = tmp_path / "out.csv"
    monkeypatch.setattr(
        R, "get_5_season_game_ids", lambda: {"x": ("2023-24", "Regular Season")}
    )
    monkeypatch.setattr(R, "fetch_and_parse_game", lambda g: None)
    monkeypatch.setattr(events, "extract_events", lambda df, g, s, t: pd.DataFrame())

    with pytest.raises(ValueError, match=r"(?i)no event rows"):
        ES.build_event_dataset(filepath=str(target))

    assert not target.exists()
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
        return _frame(game_id=game_id, season=season)

    flaky.n = 0
    monkeypatch.setattr(events, "extract_events", flaky)

    out = ES.build_event_dataset(filepath=str(target))
    assert len(out) == 2


def test_cached_file_is_loaded_without_rescraping(tmp_path, monkeypatch):
    target = tmp_path / "out.csv"
    _frame().to_csv(target, index=False)

    def boom():
        raise AssertionError("should not scrape when a cache exists")

    monkeypatch.setattr(R, "get_5_season_game_ids", boom)

    out = ES.build_event_dataset(filepath=str(target))
    assert len(out) == 1


def test_checkpoint_does_not_crash_when_all_frames_empty_across_boundary(
    tmp_path, monkeypatch
):
    """Regression test for checkpoint line crash: if the first 50+ games all
    fail to produce events, pd.concat(all_frames) on empty list raises ValueError.
    The checkpoint guard must handle empty all_frames gracefully."""
    target = tmp_path / "out.csv"
    # Create enough games to cross the idx % 50 == 0 boundary while still empty
    game_ids = {f"game_{i}": ("2023-24", "Regular Season") for i in range(55)}

    monkeypatch.setattr(R, "get_5_season_game_ids", lambda: game_ids)
    monkeypatch.setattr(R, "fetch_and_parse_game", lambda g: pd.DataFrame({"a": [1]}))

    def always_empty(df, game_id, season, season_type):
        # All games up to game_50 produce no events; game_51 onward produce events
        if int(game_id.split("_")[1]) <= 50:
            return pd.DataFrame()
        return _frame(game_id=game_id, season=season)

    monkeypatch.setattr(events, "extract_events", always_empty)

    # This should NOT raise ValueError at the checkpoint line (idx==50) even though
    # all_frames is empty; it should only raise if NO events are produced at the end
    out = ES.build_event_dataset(filepath=str(target))
    # Should have frames from game_51, game_52, game_53, game_54
    assert len(out) == 4


def test_default_dataset_path_is_versioned_separately():
    assert ES.DEFAULT_EVENT_DATASET != R.DEFAULT_DATASET
    assert "events" in ES.DEFAULT_EVENT_DATASET
