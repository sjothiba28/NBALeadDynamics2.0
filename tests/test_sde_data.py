"""Regulation-only outcome columns for the SDE study
(src/Markov/sde_data.py). landmarks.py's collapsed/final_team_relative_margin
are measured to true game end (including OT) -- correct for the peak-lead
memory test, wrong here. This module builds lead_erased/
final_regulation_margin/regulation_win by joining the landmark table against
the event-level time series (events.py), without touching landmarks.py."""

import pandas as pd
import pytest

import sde_data


def _landmark_row(
    game_id="G1",
    leader="DEN",
    threshold=5,
    landmark_lead=6.0,
    elapsed_time=100.0,
    time_remaining=2780.0,
):
    return {
        "GAME_ID": game_id,
        "LEADER": leader,
        "threshold": threshold,
        "landmark_lead": landmark_lead,
        "elapsed_time": elapsed_time,
        "time_remaining": time_remaining,
    }


def _event(game_id, t, margin, team="DEN", is_reg=True):
    return {
        "GAME_ID": game_id,
        "game_time_sec": t,
        "home_away_margin": margin,
        "scoring_team": team,
        "points": 2,
        "is_regulation": is_reg,
    }


def test_home_focal_team_lead_erased_is_false_when_lead_holds_to_regulation_end():
    landmarks_df = pd.DataFrame([_landmark_row()])
    events_df = pd.DataFrame(
        [
            _event("G1", 100.0, 6.0),
            _event("G1", 2800.0, 10.0),
        ]
    )
    out = sde_data.attach_regulation_outcomes(landmarks_df, events_df)
    assert len(out) == 1
    assert out["lead_erased"].iloc[0] == 0
    assert out["final_regulation_margin"].iloc[0] == pytest.approx(10.0)
    assert out["regulation_win"].iloc[0] == pytest.approx(1.0)


def test_overtime_events_are_excluded_from_every_outcome():
    """The game the true final margin says DEN LOST (via an OT collapse)
    must not surface here -- OT is out of scope for lead_erased/
    final_regulation_margin/regulation_win even though landmarks.collapsed
    WOULD flag it."""
    landmarks_df = pd.DataFrame([_landmark_row()])
    events_df = pd.DataFrame(
        [
            _event("G1", 100.0, 6.0),
            _event("G1", 2800.0, 10.0),  # last regulation event
            _event(
                "G1", 3100.0, -2.0, team="LAL", is_reg=False
            ),  # OT collapse, out of scope
        ]
    )
    out = sde_data.attach_regulation_outcomes(landmarks_df, events_df)
    assert out["lead_erased"].iloc[0] == 0
    assert out["final_regulation_margin"].iloc[0] == pytest.approx(10.0)


def test_lead_erased_true_when_margin_reaches_zero_or_negative_in_regulation():
    landmarks_df = pd.DataFrame([_landmark_row(landmark_lead=6.0)])
    events_df = pd.DataFrame(
        [
            _event("G1", 100.0, 6.0),
            _event("G1", 1500.0, 0.0, team="LAL"),  # tie -- must count as erased
        ]
    )
    out = sde_data.attach_regulation_outcomes(landmarks_df, events_df)
    assert out["lead_erased"].iloc[0] == 1


def test_regulation_win_is_a_half_when_the_game_reaches_regulation_tied():
    """The user-chosen tie convention: a real game that hits 0 at the last
    regulation event (genuinely resolved in OT) is a coin-flip target, not
    a forced 0 or 1."""
    landmarks_df = pd.DataFrame([_landmark_row(landmark_lead=6.0)])
    events_df = pd.DataFrame(
        [
            _event("G1", 100.0, 6.0),
            _event("G1", 2800.0, 0.0, team="LAL"),  # regulation ends tied
        ]
    )
    out = sde_data.attach_regulation_outcomes(landmarks_df, events_df)
    assert out["regulation_win"].iloc[0] == pytest.approx(0.5)
    assert out["lead_erased"].iloc[0] == 1  # a tie IS an erased lead


def test_away_focal_team_sign_is_recovered_correctly():
    """LAL is the focal team and is on the NEGATIVE side of home_away_margin
    (they are away and leading) -- the sign-recovery trick must invert
    correctly, not assume the focal team is always home."""
    landmarks_df = pd.DataFrame([_landmark_row(leader="LAL")])
    events_df = pd.DataFrame(
        [
            _event("G1", 100.0, -6.0, team="LAL"),  # LAL (away) leads by 6
            _event("G1", 2800.0, -10.0, team="LAL"),  # extends to 10
        ]
    )
    out = sde_data.attach_regulation_outcomes(landmarks_df, events_df)
    assert out["lead_erased"].iloc[0] == 0
    assert out["final_regulation_margin"].iloc[0] == pytest.approx(10.0)


def test_a_landmark_at_the_regulation_buzzer_is_dropped():
    """time_remaining == 0 is reachable (find_landmark's own bound is <=
    REGULATION_END) and must not reach fit_drift's y = .../time_remaining,
    which would produce inf. min_time_remaining=30 (default) drops it."""
    landmarks_df = pd.DataFrame(
        [_landmark_row(elapsed_time=2880.0, time_remaining=0.0)]
    )
    events_df = pd.DataFrame([_event("G1", 2880.0, 5.0)])
    out = sde_data.attach_regulation_outcomes(landmarks_df, events_df)
    assert out.empty


def test_landmarks_with_no_matching_event_at_elapsed_time_are_dropped_and_reported(
    capsys,
):
    """A missing match must be dropped with a warning, never silently
    sign-guessed -- the same failure class features.identify_leader's
    cross-check exists to catch."""
    landmarks_df = pd.DataFrame([_landmark_row(elapsed_time=999.0)])
    events_df = pd.DataFrame([_event("G1", 100.0, 6.0)])  # no event at t=999
    out = sde_data.attach_regulation_outcomes(landmarks_df, events_df)
    assert out.empty
    assert "dropped rather than sign-guessed" in capsys.readouterr().out


def test_duplicate_events_at_the_same_elapsed_time_are_dropped_with_a_warning_not_a_crash(
    capsys,
):
    """Two events sharing (GAME_ID, elapsed_time) make the landmark's own
    lookup key ambiguous -- must be dropped and warned about, matching the
    'never sign-guessed' rule, not a validate='m:1' MergeError crash."""
    landmarks_df = pd.DataFrame([_landmark_row()])
    events_df = pd.DataFrame(
        [
            _event("G1", 100.0, 6.0),
            _event(
                "G1", 100.0, 5.0, team="LAL"
            ),  # duplicate (GAME_ID, elapsed_time) key
        ]
    )
    out = sde_data.attach_regulation_outcomes(landmarks_df, events_df)
    assert out.empty
    assert "dropped rather than sign-guessed" in capsys.readouterr().out


def test_game_id_dtype_mismatch_landmarks_int_events_zero_padded_str_still_merges():
    """Regression test for the cold-run crash: landmarks_df's GAME_ID comes
    back from pd.read_csv as int64 (leading zeros lost, e.g. 22300061
    instead of '0022300061'), while a freshly-scraped events_df keeps the
    zero-padded str straight from regression.get_5_season_game_ids. Both
    sides must be normalized to a zero-padded 10-char string before the
    merge, or pandas raises ValueError: merge on int64 and str columns."""
    landmarks_df = pd.DataFrame(
        [_landmark_row(game_id=22300061)]
    )  # int, no leading zeros
    events_df = pd.DataFrame(
        [
            _event("0022300061", 100.0, 6.0),  # str, zero-padded
            _event("0022300061", 2800.0, 10.0),
        ]
    )
    out = sde_data.attach_regulation_outcomes(landmarks_df, events_df)
    assert len(out) == 1
    assert out["lead_erased"].iloc[0] == 0
    assert out["final_regulation_margin"].iloc[0] == pytest.approx(10.0)
    assert out["regulation_win"].iloc[0] == pytest.approx(1.0)


def test_game_id_dtype_mismatch_landmarks_zero_padded_str_events_int_still_merges():
    """The reverse dtype combination: landmarks_df already holds a
    zero-padded str GAME_ID (e.g. built directly in-process, never
    round-tripped through CSV) while events_df's GAME_ID is int-typed. The
    fix must normalize both sides regardless of which one arrives padded."""
    landmarks_df = pd.DataFrame(
        [_landmark_row(game_id="0022300061")]
    )  # str, zero-padded
    events_df = pd.DataFrame(
        [
            _event(22300061, 100.0, 6.0),  # int, no leading zeros
            _event(22300061, 2800.0, 10.0),
        ]
    )
    out = sde_data.attach_regulation_outcomes(landmarks_df, events_df)
    assert len(out) == 1
    assert out["lead_erased"].iloc[0] == 0
    assert out["final_regulation_margin"].iloc[0] == pytest.approx(10.0)
    assert out["regulation_win"].iloc[0] == pytest.approx(1.0)


def test_all_rows_dropped_via_missing_match_still_has_all_three_columns():
    """The returned schema must not depend on WHICH drop path emptied the
    frame -- downstream code does out['lead_erased'] unconditionally, so an
    empty result from the missing-match path must carry the same three
    columns as an empty result from the min_time_remaining floor."""
    landmarks_df = pd.DataFrame([_landmark_row(elapsed_time=999.0)])
    events_df = pd.DataFrame([_event("G1", 100.0, 6.0)])  # no event at t=999
    out = sde_data.attach_regulation_outcomes(landmarks_df, events_df)
    assert out.empty
    for col in ("lead_erased", "final_regulation_margin", "regulation_win"):
        assert col in out.columns


def test_keep_sign_column_exposes_sign_focal_without_changing_default_output():
    """Additive-only change: keep_sign_column=False (the default) must
    produce byte-identical output to every existing call in this file --
    checked here by re-running one of them and comparing frames -- while
    keep_sign_column=True exposes the sign this module already computes
    internally as _sign_focal, needed by sde_transitions.py to convert raw
    home_away_margin into focal-relative margins after a landmark."""
    landmarks_df = pd.DataFrame([_landmark_row(leader="LAL")])
    events_df = pd.DataFrame(
        [
            _event("G1", 100.0, -6.0, team="LAL"),
            _event("G1", 2800.0, -10.0, team="LAL"),
        ]
    )
    default = sde_data.attach_regulation_outcomes(landmarks_df, events_df)
    assert "sign_focal" not in default.columns

    with_sign = sde_data.attach_regulation_outcomes(
        landmarks_df, events_df, keep_sign_column=True
    )
    assert with_sign["sign_focal"].iloc[0] == pytest.approx(-1.0)
    pd.testing.assert_frame_equal(default, with_sign.drop(columns=["sign_focal"]))
