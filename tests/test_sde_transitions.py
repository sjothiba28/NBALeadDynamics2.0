"""Post-landmark transition rows (src/Markov/sde_transitions.py): one row
per consecutive pair of regulation-time scoring events after a landmark,
through end of regulation -- the empirical data sde_kappa.fit_kappa0 fits
kappa_0 against. NOT truncated at a zero-crossing: simulate_paths never
stops a path at erasure either (its trajectory is left unperturbed for the
rest of the horizon), so the fitting data must cover the same full horizon."""

import pandas as pd
import pytest

import sde_transitions as T


def _landmark_row(
    game_id="G1",
    season="2019-20",
    leader="DEN",
    opponent="LAL",
    threshold=5,
    landmark_lead=6.0,
    elapsed_time=100.0,
    sign_focal=1.0,
    v_bt=0.005,
):
    return {
        "GAME_ID": game_id,
        "SEASON": season,
        "LEADER": leader,
        "OPPONENT": opponent,
        "threshold": threshold,
        "landmark_lead": landmark_lead,
        "elapsed_time": elapsed_time,
        "sign_focal": sign_focal,
        "v_BT": v_bt,
    }


def _event(game_id, t, margin, is_reg=True):
    return {
        "GAME_ID": game_id,
        "game_time_sec": t,
        "home_away_margin": margin,
        "is_regulation": is_reg,
    }


def test_first_row_starts_at_the_landmarks_own_state():
    landmarks_df = pd.DataFrame([_landmark_row()])
    events_df = pd.DataFrame(
        [
            _event("G1", 100.0, 6.0),
            _event("G1", 130.0, 8.0),
            _event("G1", 200.0, 5.0),
        ]
    )
    rows = T.build_transition_rows(landmarks_df, events_df)
    first = rows.iloc[0]
    assert first["X_prev"] == pytest.approx(6.0)  # landmark_lead itself
    assert first["s_prev"] == pytest.approx(0.0)
    assert first["dt"] == pytest.approx(30.0)  # 130 - 100
    assert first["X_next"] == pytest.approx(8.0)  # sign_focal(+1) * margin


def test_transitions_continue_through_regulation_end_regardless_of_sign_crossing():
    """The focal margin crosses zero mid-sequence (8 -> -2 -> 3) -- rows
    must keep being emitted afterward, not truncate at the crossing."""
    landmarks_df = pd.DataFrame([_landmark_row(landmark_lead=6.0)])
    events_df = pd.DataFrame(
        [
            _event("G1", 100.0, 6.0),
            _event("G1", 130.0, 8.0),
            _event("G1", 400.0, -2.0),
            _event("G1", 900.0, 3.0),
        ]
    )
    rows = T.build_transition_rows(landmarks_df, events_df)
    assert len(rows) == 3  # (100->130), (130->400), (400->900)
    assert rows["X_next"].iloc[1] == pytest.approx(-2.0)
    assert rows["X_next"].iloc[2] == pytest.approx(3.0)


def test_away_focal_team_sign_is_applied_to_subsequent_events():
    """sign_focal=-1: raw home_away_margin events must be NEGATED to become
    focal-relative, same convention as sde_data.py's _sign_focal."""
    landmarks_df = pd.DataFrame([_landmark_row(landmark_lead=6.0, sign_focal=-1.0)])
    events_df = pd.DataFrame(
        [
            _event("G1", 100.0, -6.0),
            _event("G1", 200.0, -10.0),
        ]
    )
    rows = T.build_transition_rows(landmarks_df, events_df)
    assert rows["X_next"].iloc[0] == pytest.approx(10.0)


def test_v_bt_is_carried_unchanged_onto_every_row_of_a_landmark():
    landmarks_df = pd.DataFrame([_landmark_row(v_bt=0.0123)])
    events_df = pd.DataFrame(
        [
            _event("G1", 100.0, 6.0),
            _event("G1", 200.0, 8.0),
            _event("G1", 300.0, 9.0),
        ]
    )
    rows = T.build_transition_rows(landmarks_df, events_df)
    # pd.Series.__eq__ against pytest.approx() no longer broadcasts correctly
    # under pandas 3.0 (always False even for exact matches) -- .to_numpy()
    # sidesteps that and reduces the comparison to a single bool.
    assert rows["v_BT"].to_numpy() == pytest.approx(0.0123)


def test_landmark_row_id_links_back_to_the_landmark_tables_own_index():
    landmarks_df = pd.DataFrame(
        [_landmark_row(game_id="G1"), _landmark_row(game_id="G2")]
    )
    events_df = pd.DataFrame(
        [
            _event("G1", 100.0, 6.0),
            _event("G1", 200.0, 8.0),
            _event("G2", 150.0, 6.0),
            _event("G2", 250.0, 7.0),
        ]
    )
    rows = T.build_transition_rows(landmarks_df, events_df)
    assert set(rows["landmark_row_id"].unique()) == {0, 1}


def test_overtime_events_are_excluded():
    landmarks_df = pd.DataFrame([_landmark_row(landmark_lead=6.0)])
    events_df = pd.DataFrame(
        [
            _event("G1", 100.0, 6.0),
            _event("G1", 2800.0, 10.0),
            _event("G1", 3100.0, -2.0, is_reg=False),
        ]
    )
    rows = T.build_transition_rows(landmarks_df, events_df)
    assert len(rows) == 1
    assert rows["X_next"].iloc[0] == pytest.approx(10.0)


def test_a_landmark_with_no_subsequent_events_produces_no_rows():
    landmarks_df = pd.DataFrame([_landmark_row(elapsed_time=2880.0, landmark_lead=6.0)])
    events_df = pd.DataFrame([_event("G1", 2880.0, 6.0)])
    rows = T.build_transition_rows(landmarks_df, events_df)
    assert rows.empty
