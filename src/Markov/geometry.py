"""Geometric descriptors of the lead trajectory up to the peak lead.

eoeo.pdf section 2.2 names Control and Volatility but defers their definitions
("Precise mathematical definitions will be finalized separately"). These are
those definitions.

The lead path passed in is signed and leader-relative -- positive when the team
that eventually held L_max is ahead -- so a team that trailed early is scored
below one that was level, which an absolute margin cannot express.
"""

import numpy as np

_ZERO = {"C": 0.0, "V_var": 0.0, "V_qv": 0.0}


def calculate_geometry(times, lead, t_peak, l_max):
    """C, V_var and V_qv over the build-up window [t[0], t_peak].

    C     normalized time-averaged lead, 1.0 if the team led by l_max
          throughout and 0.0 if the lead appeared only at t_peak.
    V_var normalized time-weighted variance of the path about its own mean.
    V_qv is the normalized quadratic variation, i.e. path roughness, which
    distinguishes a monotone ramp from an oscillation over the same range --
    V_var cannot, because both have the same spread. Because lead is
    forward-filled (pbp.score_margin()), non-scoring rows contribute
    diff == 0, so V_qv is invariant to stoppages and row density -- it is a
    per-scoring-event quadratic variation, not per-row. But because it sums
    squares, it is sensitive to scoring granularity: splitting one run into
    smaller scores lowers it (a 4-point run contributes 16; the same run as
    two 2-point baskets contributes 8), so a lead built from three-pointers
    scores higher than the same lead built from free throws. See
    tests/test_geometry.py::test_v_qv_ignores_non_scoring_rows and
    ::test_v_qv_falls_when_the_same_run_is_split_into_smaller_scores.
    """
    if t_peak <= 0 or l_max <= 0:
        return dict(_ZERO)

    times = np.asarray(times, dtype=float)
    lead = np.asarray(lead, dtype=float)

    window = times <= t_peak
    t, L = times[window], lead[window]
    if len(t) < 2:
        return dict(_ZERO)

    # Each sample holds until the next one; the last holds until t_peak. A
    # centred or trailing difference would drop the final segment, which is the
    # one nearest the peak and so the most heavily weighted in C.
    dt = np.diff(np.append(t, t_peak))
    total_time = dt.sum()
    if total_time <= 0:
        return dict(_ZERO)

    area = float(np.sum(L * dt))
    mean_lead = area / total_time

    # Normalising by total_time * l_max (the same time base as mean_lead,
    # i.e. the window [t[0], t_peak] rather than an implicit [0, t_peak])
    # makes C scale-free and directly comparable across games of different
    # length and lead size.
    c = area / (total_time * l_max)
    v_var = float(np.sum(((L - mean_lead) ** 2) * dt)) / (total_time * l_max**2)
    # Normalized by total_time * l_max**2 -- the same [t[0], t_peak] window
    # used above -- which makes V_qv scale-free but also strongly
    # anti-correlated with L_max (-0.748 on the real dataset). See
    # models.CONTROL_VARS for the control model that tests whether the
    # geometry block adds anything beyond nonlinear L_max.
    v_qv = float(np.sum(np.diff(L) ** 2)) / (total_time * l_max**2)

    return {"C": c, "V_var": v_var, "V_qv": v_qv}
