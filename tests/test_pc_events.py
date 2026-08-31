"""Event-level rows for the Peel-Clauset next-scorer models
(src/Markov/pc_events.py)."""

import pandas as pd
import pytest

import pc_events as PC


def _event(
    game_id,
    t,
    team,
    points,
    season="2019-20",
    season_type="Regular Season",
    is_reg=True,
):
    return {
        "GAME_ID": game_id,
        "SEASON": season,
        "SEASON_TYPE": season_type,
        "game_time_sec": t,
        "scoring_team": team,
        "points": points,
        "is_regulation": is_reg,
    }


# ---- build_game_rows -------------------------------------------------


def test_game_rows_focal_is_alphabetically_first_team():
    events_df = pd.DataFrame(
        [
            _event("G1", 10.0, "LAL", 2),
            _event("G1", 40.0, "DEN", 3),
            _event("G1", 70.0, "LAL", 2),
        ]
    )
    rows = PC.build_game_rows(events_df)
    assert set(rows["team_r"]) == {"DEN"}
    assert set(rows["team_b"]) == {"LAL"}


def test_game_rows_first_event_dropped_no_preceding_scorer():
    events_df = pd.DataFrame(
        [
            _event("G1", 10.0, "LAL", 2),
            _event("G1", 40.0, "DEN", 3),
        ]
    )
    rows = PC.build_game_rows(events_df)
    assert len(rows) == 1  # only the second event gets a row


def test_game_rows_x_before_accumulates_focal_relative_margin():
    events_df = pd.DataFrame(
        [
            _event(
                "G1", 10.0, "DEN", 2
            ),  # focal (r=DEN) scores +2 -> x=2 (not itself a row)
            _event(
                "G1", 40.0, "LAL", 3
            ),  # row: X_before=2, scored_focal=False -> x=2-3=-1
            _event(
                "G1", 70.0, "DEN", 2
            ),  # row: X_before=-1, scored_focal=True -> x=-1+2=1
        ]
    )
    rows = (
        PC.build_game_rows(events_df)
        .sort_values("game_time_sec")
        .reset_index(drop=True)
    )
    assert rows["X_before"].tolist() == pytest.approx([2.0, -1.0])
    assert rows["scored_focal"].tolist() == [False, True]


def test_game_rows_last_scorer_focal_tracks_previous_event():
    events_df = pd.DataFrame(
        [
            _event("G1", 10.0, "DEN", 2),
            _event("G1", 40.0, "DEN", 2),  # last_scorer_focal True (DEN scored event 1)
            _event("G1", 70.0, "LAL", 3),  # last_scorer_focal True (DEN scored event 2)
            _event(
                "G1", 100.0, "LAL", 2
            ),  # last_scorer_focal False (LAL scored event 3)
        ]
    )
    rows = (
        PC.build_game_rows(events_df)
        .sort_values("game_time_sec")
        .reset_index(drop=True)
    )
    assert rows["last_scorer_focal"].tolist() == [True, True, False]


def test_game_rows_skips_games_without_exactly_two_scoring_teams():
    events_df = pd.DataFrame(
        [
            _event("G1", 10.0, "DEN", 2),
            _event("G1", 40.0, "DEN", 2),  # only one team scored all game
        ]
    )
    rows = PC.build_game_rows(events_df)
    assert rows.empty


def test_game_rows_excludes_overtime_and_other_season_types():
    events_df = pd.DataFrame(
        [
            _event("G1", 10.0, "DEN", 2),
            _event("G1", 40.0, "LAL", 3),
            _event("G1", 2800.0, "DEN", 2, is_reg=False),
            _event("G1", 2900.0, "LAL", 2, season_type="Playoffs"),
        ]
    )
    rows = PC.build_game_rows(events_df)
    assert len(rows) == 1


# ---- build_landmark_rows ----------------------------------------------


def _landmark_row(
    game_id="G1",
    season="2019-20",
    leader="DEN",
    opponent="LAL",
    threshold=5,
    landmark_lead=5.0,
    elapsed_time=100.0,
):
    return {
        "GAME_ID": game_id,
        "SEASON": season,
        "LEADER": leader,
        "OPPONENT": opponent,
        "threshold": threshold,
        "landmark_lead": landmark_lead,
        "elapsed_time": elapsed_time,
    }


def test_landmark_rows_first_row_last_scorer_is_seeded_true():
    landmarks_df = pd.DataFrame([_landmark_row()])
    events_df = pd.DataFrame(
        [
            _event("G1", 100.0, "DEN", 2),  # the landmark-triggering event itself
            _event("G1", 130.0, "LAL", 3),
        ]
    )
    rows = PC.build_landmark_rows(landmarks_df, events_df)
    assert len(rows) == 1
    first = rows.iloc[0]
    assert bool(first["last_scorer_focal"]) == True
    assert first["X_before"] == pytest.approx(5.0)  # landmark_lead itself
    assert (
        bool(first["scored_focal"]) == False
    )  # LAL scored this (only post-landmark) event


def test_landmark_rows_x_before_accumulates_after_landmark():
    landmarks_df = pd.DataFrame([_landmark_row(landmark_lead=5.0)])
    events_df = pd.DataFrame(
        [
            _event("G1", 100.0, "DEN", 2),
            _event("G1", 130.0, "LAL", 3),  # X_before=5.0 -> after: 5-3=2
            _event("G1", 160.0, "DEN", 2),  # X_before=2.0 -> after: 2+2=4
        ]
    )
    rows = PC.build_landmark_rows(landmarks_df, events_df).sort_values("game_time_sec")
    assert rows["X_before"].tolist() == pytest.approx([5.0, 2.0])
    assert rows["scored_focal"].tolist() == [False, True]
    assert rows["last_scorer_focal"].tolist() == [True, False]


def test_landmark_rows_only_events_strictly_after_elapsed_time():
    landmarks_df = pd.DataFrame([_landmark_row(elapsed_time=100.0)])
    events_df = pd.DataFrame(
        [
            _event("G1", 100.0, "DEN", 2),  # AT the landmark, not after -- excluded
            _event("G1", 130.0, "LAL", 3),
        ]
    )
    rows = PC.build_landmark_rows(landmarks_df, events_df)
    assert len(rows) == 1
    assert rows.iloc[0]["game_time_sec"] == pytest.approx(130.0)


def test_landmark_rows_overtime_events_excluded():
    landmarks_df = pd.DataFrame([_landmark_row(elapsed_time=100.0, landmark_lead=5.0)])
    events_df = pd.DataFrame(
        [
            _event("G1", 100.0, "DEN", 2),
            _event("G1", 2800.0, "LAL", 3),
            _event("G1", 3100.0, "DEN", 2, is_reg=False),
        ]
    )
    rows = PC.build_landmark_rows(landmarks_df, events_df)
    assert len(rows) == 1


def test_landmark_rows_a_landmark_with_no_subsequent_events_produces_no_rows():
    landmarks_df = pd.DataFrame([_landmark_row(elapsed_time=2880.0, landmark_lead=5.0)])
    events_df = pd.DataFrame([_event("G1", 2880.0, "DEN", 2)])
    rows = PC.build_landmark_rows(landmarks_df, events_df)
    assert rows.empty


def test_attach_eta_bt_matches_bt_log_odds_of_leave_one_game_out_strengths():
    events = []
    for g in range(5):
        for k in range(60):
            scorer = "AAA" if k % 2 == 0 else "BBB"
            events.append(_event(f"G{g}", float(k * 30), scorer, 2))
    events_df = pd.DataFrame(events)
    game_rows = PC.build_game_rows(events_df)
    out = PC.attach_eta_bt(game_rows, events_df)
    assert "eta_BT" in out.columns
    assert not out["eta_BT"].isna().any()

    import sde_bt

    strengths_by_game = sde_bt.fit_season_strengths_leave_one_game_out(events_df)
    row = out.iloc[0]
    game_id_padded = str(row["GAME_ID"]).zfill(10)
    strengths = strengths_by_game[(row["SEASON"], game_id_padded)]
    expected = sde_bt.bt_log_odds(strengths[row["team_r"]], strengths[row["team_b"]])
    assert row["eta_BT"] == pytest.approx(expected)


def test_attach_eta_bt_warns_and_leaves_nan_on_unmatched_game(capsys):
    events_df = pd.DataFrame(
        [
            _event("G1", 10.0, "DEN", 2),
            _event("G1", 40.0, "LAL", 3),
        ]
    )
    game_rows = PC.build_game_rows(events_df)
    # Corrupt team_r so it no longer matches any fitted strength.
    game_rows["team_r"] = "ZZZ"
    out = PC.attach_eta_bt(game_rows, events_df)
    assert out["eta_BT"].isna().all()
    captured = capsys.readouterr()
    assert "[WARN]" in captured.out


def test_attach_eta_bt_from_strengths_matches_bt_log_odds():
    rows_df = pd.DataFrame({"x": [1, 2, 3]})
    s_r = [2.0, 1.0, 0.5]
    s_b = [1.0, 1.0, 1.0]
    out = PC.attach_eta_bt_from_strengths(rows_df, s_r, s_b)
    import numpy as np

    assert out["eta_BT"].to_numpy() == pytest.approx(
        np.log(np.array(s_r)) - np.log(np.array(s_b))
    )


def test_landmark_rows_landmark_row_id_links_back_to_the_landmark_tables_own_index():
    landmarks_df = pd.DataFrame(
        [_landmark_row(game_id="G1"), _landmark_row(game_id="G2")]
    )
    events_df = pd.DataFrame(
        [
            _event("G1", 100.0, "DEN", 2),
            _event("G1", 130.0, "LAL", 2),
            _event("G2", 100.0, "DEN", 2),
            _event("G2", 150.0, "LAL", 2),
        ]
    )
    rows = PC.build_landmark_rows(landmarks_df, events_df)
    assert set(rows["landmark_row_id"].unique()) == {0, 1}
