import numpy as np
import pytest

import geometry
from geometry import calculate_geometry


def test_constant_lead_has_control_one_and_no_volatility():
    """Led by exactly L_max from the opening tip: maximally controlled, and a
    flat path has neither variance nor roughness."""
    t = np.arange(0, 1201, 10)
    lead = np.full(len(t), 10.0)
    g = calculate_geometry(t, lead, t_peak=1200, l_max=10)
    assert g["C"] == pytest.approx(1.0, abs=1e-6)
    assert g["V_var"] == pytest.approx(0.0, abs=1e-9)
    assert g["V_qv"] == pytest.approx(0.0, abs=1e-9)


def test_linear_buildup_has_control_one_half():
    """A lead growing linearly to L_max spends half the area under a flat
    L_max path, so C = 1/2 regardless of scale."""
    t = np.arange(0, 1201, 10)
    lead = 10.0 * t / 1200
    g = calculate_geometry(t, lead, t_peak=1200, l_max=10)
    assert g["C"] == pytest.approx(0.5, abs=0.02)


def test_control_is_scale_free():
    """Doubling both the lead and L_max must leave C unchanged."""
    t = np.arange(0, 1201, 10)
    lead = 10.0 * t / 1200
    a = calculate_geometry(t, lead, 1200, 10)
    b = calculate_geometry(t, 2 * lead, 1200, 20)
    assert a["C"] == pytest.approx(b["C"])


def test_sawtooth_is_rougher_than_monotone_at_equal_variance():
    """The distinguishing property of V_qv. Both paths swing over the same
    range, but one oscillates and one does not; only V_qv separates them."""
    t = np.arange(0, 1201, 10)
    smooth = 10.0 * t / 1200
    saw = 5.0 + 5.0 * np.sign(np.sin(t / 40.0))
    g_smooth = calculate_geometry(t, smooth, 1200, 10)
    g_saw = calculate_geometry(t, saw, 1200, 10)
    assert g_saw["V_qv"] > 10 * g_smooth["V_qv"]


def test_negative_lead_contributes_negative_area():
    """Lead is signed and leader-relative, so a team that trailed early must
    score lower than one that was level."""
    t = np.arange(0, 1201, 10)
    level = np.where(t < 600, 0.0, 10.0)
    trailed = np.where(t < 600, -10.0, 10.0)
    assert (
        calculate_geometry(t, trailed, 1200, 10)["C"]
        < calculate_geometry(t, level, 1200, 10)["C"]
    )


def test_v_qv_ignores_non_scoring_rows():
    """Forward-filled rows contribute nothing: diff == 0 on a repeated score.

    pbp.score_margin() forward-fills across non-scoring events, so row
    density alone cannot move V_qv. Pinning this stops anyone from
    reintroducing the belief that stoppages inflate it.
    """
    t_sparse = np.array([0.0, 400.0, 800.0, 1200.0])
    l_sparse = np.array([8.0, 12.0, 8.0, 12.0])
    t_dense = np.arange(0, 1201, 10.0)
    idx = np.searchsorted(t_sparse, t_dense, side="right") - 1
    l_dense = l_sparse[idx]

    sparse = geometry.calculate_geometry(t_sparse, l_sparse, 1200.0, 12.0)
    dense = geometry.calculate_geometry(t_dense, l_dense, 1200.0, 12.0)

    assert dense["V_qv"] == pytest.approx(sparse["V_qv"], abs=1e-15)


def test_v_qv_falls_when_the_same_run_is_split_into_smaller_scores():
    """Squares are subadditive across a split: 4**2 > 2**2 + 2**2.

    The same net lead change scores lower when it arrives as more, smaller
    scoring events. So V_qv is sensitive to scoring granularity -- three
    pointers vs free throws -- not only to the shape of the path.
    """
    t = np.array([0.0, 600.0, 1200.0])
    one_big = geometry.calculate_geometry(t, np.array([0.0, 4.0, 4.0]), 1200.0, 4.0)
    t_split = np.array([0.0, 400.0, 800.0, 1200.0])
    two_small = geometry.calculate_geometry(
        t_split, np.array([0.0, 2.0, 4.0, 4.0]), 1200.0, 4.0
    )

    assert two_small["V_qv"] < one_big["V_qv"]


def test_control_one_when_window_does_not_start_at_zero():
    """A constant lead of l_max over [t[0], t_peak] with t[0] > 0 must give
    C == 1.0: the averaging window is [t[0], t_peak], not [0, t_peak].

    Under the old t_peak-normalized code this would come out at
    (t_peak - t[0]) / t_peak instead of 1.0.
    """
    t = np.arange(300, 1201, 10)
    lead = np.full(len(t), 10.0)
    g = calculate_geometry(t, lead, t_peak=1200, l_max=10)
    assert g["C"] == pytest.approx(1.0, abs=1e-6)


def test_samples_after_t_peak_are_excluded_from_the_window():
    """The build-up window is [t[0], t_peak]; what happened afterwards is the
    response, not a descriptor of the build-up.

    Every other case in this file passes an array whose last sample IS t_peak,
    so dropping the `times <= t_peak` mask entirely changes nothing and the
    suite stays green. Here the path collapses from +l_max to -l_max after the
    peak: if the trailing samples leak in, C rises to ~1.0167 and V_qv
    stops being 0.
    """
    t = np.arange(0, 2401, 10.0)
    lead = np.where(t <= 1200, 10.0, -10.0)
    g = calculate_geometry(t, lead, t_peak=1200, l_max=10)
    # Identical to the constant-lead case above: the collapse is invisible.
    assert g["C"] == pytest.approx(1.0, abs=1e-6)
    assert g["V_var"] == pytest.approx(0.0, abs=1e-9)
    assert g["V_qv"] == pytest.approx(0.0, abs=1e-9)


def test_degenerate_inputs_return_zeros():
    g = calculate_geometry([0], [0], t_peak=0, l_max=0)
    assert g == {"C": 0.0, "V_var": 0.0, "V_qv": 0.0}
