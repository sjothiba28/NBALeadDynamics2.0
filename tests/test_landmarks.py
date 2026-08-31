"""Landmark detection and per-landmark feature extraction for src/Markov/landmarks.py.

Landmarks are the first regulation-time moment a team reaches a lead of at
least 5/10/15 points -- as opposed to features.py's peak lead, which is a
global argmax over the whole game. See landmarks.py's module docstring for
why this file duplicates rather than imports features.py's tricode
resolution and pre-window rate logic.
"""

import numpy as np
import pandas as pd
import pytest

import features as F
import landmarks as L
from common import pbp as pbp_common
from conftest import GAMES, load_pbp


def _tidy_from_raw(raw):
    """Minimal tidy frame from a raw PlayByPlayV3 fixture, built only from
    common.pbp's pure functions -- no network, no dependency on
    regression.fetch_and_parse_game. Enough for resolve_teams/find_landmark,
    which only need game_time_sec, signed_margin, event_team and made."""
    game_time_sec = pbp_common.game_time_seconds(raw)
    margin = pbp_common.score_margin(raw)
    is_fta, is_ftm = pbp_common.free_throw_results(raw)
    zone = pbp_common.classify_shot_zones(raw)
    made = pd.Series(np.nan, index=raw.index, dtype=float)
    made[zone.notna()] = pbp_common.is_made_shot(raw)[zone.notna()].astype(float)
    made[is_fta] = is_ftm[is_fta].astype(float)
    return pd.DataFrame(
        {
            "game_time_sec": game_time_sec,
            "signed_margin": margin,
            "event_team": pbp_common.event_team(raw),
            "made": made,
        }
    ).dropna(subset=["game_time_sec"])


def _full_tidy_from_raw(raw):
    """Full tidy frame mirroring regression.fetch_and_parse_game's
    construction (regression.py:128-186), built directly from common.pbp's
    pure functions -- no network. Used only for real-fixture integration
    tests (OT exclusion, time_remaining correctness against a genuine
    6-period game) that a synthetic frame can't exercise, since it needs
    every column build_landmark_rows/extract_landmark_features touch, not
    just the subset _tidy_from_raw above provides."""
    margin = pbp_common.score_margin(raw)
    game_time_sec = pbp_common.game_time_seconds(raw)
    is_orb, is_drb = pbp_common.classify_rebounds(raw)
    is_fta, is_ftm = pbp_common.free_throw_results(raw)
    zone = pbp_common.classify_shot_zones(raw)
    made = pd.Series(np.nan, index=raw.index, dtype=float)
    made[zone.notna()] = pbp_common.is_made_shot(raw)[zone.notna()].astype(float)
    made[is_fta] = is_ftm[is_fta].astype(float)
    df = pd.DataFrame(
        {
            "game_time_sec": game_time_sec,
            "signed_margin": margin,
            "is_ft": is_fta,
            "is_tov": pbp_common.is_turnover(raw),
            "is_orb": is_orb,
            "opp_drb": is_drb,
            "shot_zone": zone,
            "made": made,
            "event_team": pbp_common.event_team(raw),
            "poss_start": pbp_common.possession_starts(raw),
        }
    )
    return df.dropna(subset=["game_time_sec"])


def make_landmark_frame(signed, event_team, made=None, times=None):
    """A tidy frame shaped like fetch_and_parse_game's output, minimal
    columns needed by resolve_teams/find_landmark/extract_landmark_features."""
    n = len(signed)
    if times is None:
        times = list(range(10, 10 * n + 1, 10))
    if made is None:
        made = [1.0] * n
    return pd.DataFrame(
        {
            "game_time_sec": times,
            "signed_margin": signed,
            "event_team": event_team,
            "made": made,
        }
    )


# ---------------------------------------------------------------------------
# resolve_teams
# ---------------------------------------------------------------------------


def test_resolve_teams_finds_the_dominant_pair_and_the_scoring_teams_sign():
    """team_a is whichever tricode has the most event volume; sign_a is +1
    when team_a's first make coincides with signed_margin rising (i.e. team_a
    is the 'home' side of the signed margin). Each row's signed_margin is the
    cumulative score AFTER that row's own event, so DEN's opening make at
    row 0 must already show up as +2, not 0."""
    df = make_landmark_frame(
        signed=[2, 4, 2, 4], event_team=["DEN", "DEN", "LAL", "DEN"]
    )
    result = L.resolve_teams(df)
    assert result is not None
    team_a, team_b, sign_a = result
    assert {team_a, team_b} == {"DEN", "LAL"}
    assert team_a == "DEN"
    assert sign_a == 1.0


def test_resolve_teams_gives_the_away_team_a_negative_sign():
    """DEN's first make coincides with signed_margin FALLING, so DEN is on
    the away/negative side of signed_margin."""
    df = make_landmark_frame(
        signed=[-2, -4, -2, -4], event_team=["DEN", "DEN", "LAL", "DEN"]
    )
    team_a, _team_b, sign_a = L.resolve_teams(df)
    assert team_a == "DEN"
    assert sign_a == -1.0


def test_resolve_teams_rejects_a_third_tricode_with_real_volume():
    """Mirrors features.identify_leader's >1%-of-attributed-rows guard: a
    genuinely displaced third tricode means the pair cannot be trusted."""
    rows = ["DEN"] * 40 + ["LAL"] * 40 + ["XXX"] * 30
    df = make_landmark_frame(signed=list(range(len(rows))), event_team=rows)
    assert L.resolve_teams(df) is None


def test_resolve_teams_returns_none_on_an_empty_frame():
    assert (
        L.resolve_teams(
            pd.DataFrame(
                columns=["game_time_sec", "signed_margin", "event_team", "made"]
            )
        )
        is None
    )


@pytest.mark.parametrize("game_id", sorted(GAMES))
def test_resolve_teams_agrees_with_identify_leader_on_all_fixtures(game_id):
    """Drift detector: if features.identify_leader's cross-check logic ever
    changes, this fails loudly rather than the two silently diverging, which
    is the risk taken on by duplicating the logic instead of sharing it."""
    raw = load_pbp(game_id)
    tidy = _tidy_from_raw(raw)
    resolved = L.resolve_teams(tidy)
    assert resolved is not None
    team_a, team_b, _sign_a = resolved
    leader, trailer = F.identify_leader(raw)
    assert {team_a, team_b} == {leader, trailer}


# ---------------------------------------------------------------------------
# find_landmark
# ---------------------------------------------------------------------------


def test_find_landmark_returns_the_first_row_meeting_the_threshold():
    df = make_landmark_frame(signed=[2, 4, 6, 8, 10], event_team=["DEN"] * 5)
    idx = L.find_landmark(df, team="DEN", sign=1.0, threshold=5)
    assert idx == 2
    assert df.loc[idx, "signed_margin"] == 6


def test_landmark_detection_handles_a_jump_over_the_exact_threshold():
    """A 3-pointer can jump the margin from 3 straight to 6, skipping exactly
    5. The landmark still fires -- at the true margin of 6 -- rather than
    silently contributing no row for that team-game-threshold."""
    df = make_landmark_frame(signed=[3, 6, 9], event_team=["DEN"] * 3)
    idx = L.find_landmark(df, team="DEN", sign=1.0, threshold=5)
    assert idx == 1
    assert df.loc[idx, "signed_margin"] == 6


def test_find_landmark_returns_none_when_the_team_never_reaches_the_threshold():
    df = make_landmark_frame(signed=[1, 2, 3], event_team=["DEN"] * 3)
    assert L.find_landmark(df, team="DEN", sign=1.0, threshold=5) is None


def test_find_landmark_uses_the_signed_team_relative_margin():
    """sign=-1 for the away team: a signed_margin of -6 is a 6-point lead for
    the away team, not a 6-point deficit."""
    df = make_landmark_frame(signed=[-2, -4, -6], event_team=["LAL"] * 3)
    idx = L.find_landmark(df, team="LAL", sign=-1.0, threshold=5)
    assert idx == 2


def test_find_landmark_excludes_rows_reached_in_overtime():
    """Regulation ends at 2880s (common.pbp.REGULATION_END). A threshold only
    reached after that must not be selected -- no predictor may depend on a
    game having gone to overtime."""
    df = make_landmark_frame(
        signed=[2, 4, 10], event_team=["DEN"] * 3, times=[1000, 2000, 2900]
    )  # last row is in overtime
    assert L.find_landmark(df, team="DEN", sign=1.0, threshold=5) is None


def test_ot_landmarks_are_excluded_using_the_double_ot_fixture(pbp_overtime):
    """Integration case a synthetic frame can't exercise: real overtime-period
    length handling (5-minute OT periods, common.pbp.game_time_seconds)
    interacting with the 2880s regulation cutoff, on a genuine 6-period
    (double-OT) game -- the fixture conftest.GAMES picks specifically for
    this class of bug."""
    df = _full_tidy_from_raw(pbp_overtime)
    assert df["game_time_sec"].max() > L.REGULATION_END, (
        "fixture must actually reach overtime for this test to mean anything"
    )

    rows = L.build_landmark_rows(df, "0022300476", "2023-24", "Regular Season")
    assert rows, "a real, closely-contested double-OT game should have landmarks"
    for r in rows:
        assert r["elapsed_time"] <= L.REGULATION_END
        assert r["time_remaining"] == pytest.approx(
            L.REGULATION_END - r["elapsed_time"]
        )
        assert r["time_remaining"] >= 0


def test_find_landmark_returns_at_most_one_row_even_if_the_lead_is_regained():
    """A team that reaches +5, gives it back, and regains +5 later must still
    only ever produce the FIRST crossing -- find_landmark returns a single
    index by construction (idxmax on a boolean mask stops at the first True)."""
    df = make_landmark_frame(signed=[5, 0, -5, 0, 5], event_team=["DEN"] * 5)
    idx = L.find_landmark(df, team="DEN", sign=1.0, threshold=5)
    assert idx == 0


# ---------------------------------------------------------------------------
# extract_landmark_features / build_landmark_rows
# ---------------------------------------------------------------------------


def row(
    t,
    signed,
    team,
    made=np.nan,
    is_ft=False,
    is_tov=False,
    is_orb=False,
    opp_drb=False,
    shot_zone=None,
    poss_start=True,
):
    return dict(
        game_time_sec=t,
        signed_margin=signed,
        event_team=team,
        made=made,
        is_ft=is_ft,
        is_tov=is_tov,
        is_orb=is_orb,
        opp_drb=opp_drb,
        shot_zone=shot_zone,
        poss_start=poss_start,
    )


def frame(rows):
    return pd.DataFrame(rows)


def test_lead_loss_and_collapse_are_focal_team_relative():
    """Mirrors test_regression_features.py's
    test_peak_belonging_to_the_away_team_is_measured_from_their_side: the
    focal team here is the AWAY side of signed_margin (sign_focal=-1), and
    every outcome must be measured from their side, not home's."""
    df = frame(
        [
            row(100, -5, "LAL", made=1.0, shot_zone="Rim"),
            row(2000, 4, "DEN", made=1.0, shot_zone="Rim"),
        ]
    )
    idx = L.find_landmark(df, "LAL", -1.0, 5)
    f = L.extract_landmark_features(
        df,
        "G1",
        "2023-24",
        "Regular Season",
        team_a="DEN",
        team_b="LAL",
        focal_team="LAL",
        sign_focal=-1.0,
        threshold=5,
        landmark_idx=idx,
    )
    assert f["landmark_lead"] == 5
    assert f["final_team_relative_margin"] == -4
    assert f["lead_loss"] == 9
    assert f["collapsed"] == 1
    assert f["OPPONENT"] == "DEN"
    assert f["LEADER"] == "LAL"


def test_geometry_and_rates_exclude_events_after_the_landmark():
    """Mirrors test_regression_features.py's
    test_g_rates_exclude_the_leader_own_post_peak_shots: peaks on the SECOND
    shot, then takes a third in a zone none of the pre-landmark shots used, so
    a mutant that drops the `game_time_sec <= t_landmark` filter is visible in
    the rates rather than hidden by symmetry."""
    df = frame(
        [
            row(10, 5, "DEN", made=1.0, shot_zone="Rim"),
            row(20, 10, "DEN", made=1.0, shot_zone="Rim"),
            row(30, 3, "DEN", made=1.0, shot_zone="ATB3"),
        ]
    )
    idx = L.find_landmark(df, "DEN", 1.0, 10)
    f = L.extract_landmark_features(
        df,
        "G1",
        "2023-24",
        "Regular Season",
        team_a="DEN",
        team_b="LAL",
        focal_team="DEN",
        sign_focal=1.0,
        threshold=10,
        landmark_idx=idx,
    )
    assert f["time_remaining"] == pytest.approx(2880 - 20)
    assert f["G_r_Rim"] == pytest.approx(1.0)
    assert f["G_r_ATB3"] == pytest.approx(0.0)
    assert np.isnan(f["G_p_ATB3"])


def test_no_post_landmark_leakage_into_possessions_or_diagnostics():
    """A turnover that occurs only after the landmark must not flip
    parsed_ft_or_tov, and its possession-opening row must not inflate
    Possessions -- both computed only over the pre-landmark window."""
    df = frame(
        [
            row(10, 5, "DEN", made=1.0, shot_zone="Rim"),
            row(20, 10, "DEN", made=1.0, shot_zone="Rim"),
            row(30, 10, "DEN", is_tov=True, poss_start=True),
        ]
    )
    idx = L.find_landmark(df, "DEN", 1.0, 10)
    f = L.extract_landmark_features(
        df,
        "G1",
        "2023-24",
        "Regular Season",
        team_a="DEN",
        team_b="LAL",
        focal_team="DEN",
        sign_focal=1.0,
        threshold=10,
        landmark_idx=idx,
    )
    assert f["Possessions"] == 2
    assert f["parsed_ft_or_tov"] == 0


def test_at_most_one_row_per_team_and_threshold():
    """DEN reaches +5, gives it back, and regains +5 later -- build_landmark_rows
    must still emit exactly one (DEN, 5) row, not two."""
    df = frame(
        [
            row(10, 5, "DEN", made=1.0, shot_zone="Rim"),
            row(20, 0, "LAL", made=1.0, shot_zone="Rim"),
            row(1500, 5, "DEN", made=1.0, shot_zone="Rim"),
        ]
    )
    rows = L.build_landmark_rows(df, "G1", "2023-24", "Regular Season")
    den_5_rows = [r for r in rows if r["LEADER"] == "DEN" and r["threshold"] == 5]
    assert len(den_5_rows) == 1
    assert den_5_rows[0]["elapsed_time"] == 10


def test_build_landmark_rows_emits_an_independent_row_per_team():
    """Both teams reaching the same threshold at different times produces two
    rows, keyed by (GAME_ID, LEADER, threshold), not one per game."""
    df = frame(
        [
            row(10, 5, "DEN", made=1.0, shot_zone="Rim"),
            row(1500, -5, "LAL", made=1.0, shot_zone="Rim"),
            row(1600, -5, "DEN", made=1.0, shot_zone="Mid"),
        ]
    )
    rows = L.build_landmark_rows(df, "G1", "2023-24", "Regular Season")
    five_pt_rows = [r for r in rows if r["threshold"] == 5]
    assert {r["LEADER"] for r in five_pt_rows} == {"DEN", "LAL"}
    assert len(five_pt_rows) == 2


def test_build_landmark_rows_on_a_game_with_no_qualifying_landmark_is_empty():
    df = frame([row(10, 1, "DEN", made=1.0, shot_zone="Rim")])
    assert L.build_landmark_rows(df, "G1", "2023-24", "Regular Season") == []
