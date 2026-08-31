"""Discretized Brownian-motion-with-drift simulation of the post-landmark
score-differential path through end of regulation, with a Brownian-bridge
correction for zero-crossings a coarse discrete step would otherwise miss.

X_{k+1} = X_k + mu*dt + sqrt(2*D*dt)*Z, Z~N(0,1). A discrete step landing on
opposite signs (or hitting exactly 0) is an unambiguous crossing. A step
that stays on the same side despite the true continuous path being free to
have dipped through zero and back needs the barrier/bridge correction
(Clauset et al.): P(cross inside the step) = exp(-2*X_k*X_{k+1}/(2*D*dt)).
Far from the barrier this underflows to exactly 0.0 -- expected, not a bug,
do not clamp it. Once a path is flagged erased its OWN trajectory is left
unperturbed for the rest of the horizon: only the erasure flag/time is
affected, never the marginal distribution of X, which is what makes this
correction statistically valid rather than an ad hoc patch.

`mu` may be a scalar float (constant drift), a one-argument callable
`mu(s) -> float` evaluated once per step at that step's elapsed-time cursor
(piecewise-constant approximation to a time-varying drift), or a
two-argument callable `mu(X, s) -> array` evaluated once per step on the
FULL chunk array X (a state-dependent / restoring-force drift, frozen at
each step's starting X -- ordinary Euler-Maruyama discretization of a
continuously state-dependent drift). The bridge-correction formula below is
unchanged by this: a Brownian bridge conditioned on both of a step's
endpoints is drift-invariant in law for ANY per-step-constant drift
(scalar, mu(s), or mu(X,s) frozen at the step's start alike), so the only
extra error a state-dependent mu introduces is ordinary discretization
error -- bounded by the dt-convergence tests, not a crossing-formula defect.
"""

import inspect

import numpy as np
from scipy.special import erfc


def eq17_survival_probability(L, tau, D, v):
    """Closed-form Eq. 17 (Clauset, Kogan & Redner 2015, "Safe Leads and
    Lead Changes in Competitive Team Sports"): the probability a lead of
    size L survives (never touches zero) over a remaining time tau under
    CONSTANT bias velocity v. At v=0 this collapses to Eq. 15's erf(z).
    Vectorized over v (or L/tau) via numpy broadcasting -- used by
    sde_bt.paper_bias_averaged_safe_lead to average analytically over many
    strength draws instead of running simulate_paths per draw.
    """
    L = np.asarray(L, dtype=float)
    tau = np.asarray(tau, dtype=float)
    v = np.asarray(v, dtype=float)
    z = L / np.sqrt(4 * D * tau)
    Pe = v * L / (2 * D)
    return 1 - 0.5 * (np.exp(-2 * Pe) * erfc(z - Pe / (2 * z)) + erfc(z + Pe / (2 * z)))


def simulate_paths(
    L0,
    tau,
    D,
    mu,
    n_paths=2000,
    dt=30.0,
    chunk_size=500,
    apply_bridge_correction=True,
    rng=None,
):
    """Simulate `n_paths` trajectories from t=0 (X_0=L0) to t=tau, in steps
    of `dt` seconds (the final step is shortened to land exactly on tau).

    Chunked over paths so memory is bounded regardless of n_paths; only
    per-chunk running sums and the final-margin array (length n_paths, not
    n_paths*n_steps) are retained.
    """
    rng = np.random.default_rng(rng)
    if tau <= 0:
        raise ValueError(
            "tau must be > 0 -- callers must apply sde_data."
            "attach_regulation_outcomes' min_time_remaining floor before simulating"
        )

    n_full_steps = int(tau // dt)
    step_sizes = [dt] * n_full_steps
    remainder = tau - n_full_steps * dt
    if remainder > 1e-9:
        step_sizes.append(remainder)
    step_sizes = np.asarray(step_sizes, dtype=float)

    mu_is_state_dependent = callable(mu) and len(inspect.signature(mu).parameters) == 2

    sum_final = sumsq_final = sum_erasure_time = sum_max_loss = 0.0
    n_erased = n_win = 0
    all_finals = []

    n_done = 0
    while n_done < n_paths:
        chunk_n = min(chunk_size, n_paths - n_done)
        X = np.full(chunk_n, L0, dtype=float)
        erased = np.zeros(chunk_n, dtype=bool)
        erasure_time = np.full(chunk_n, np.nan)
        running_min = X.copy()
        t_cursor = 0.0

        for step_dt in step_sizes:
            Z = rng.standard_normal(chunk_n)
            if callable(mu):
                step_mu = mu(X, t_cursor) if mu_is_state_dependent else mu(t_cursor)
            else:
                step_mu = mu
            X_next = X + step_mu * step_dt + np.sqrt(2 * D * step_dt) * Z

            discrete_cross = (X * X_next <= 0) & ~erased
            if discrete_cross.any():
                denom = X - X_next
                frac = np.where(denom != 0, X / denom, 0.5)
                frac = np.clip(frac, 0.0, 1.0)
                erasure_time[discrete_cross] = t_cursor + frac[discrete_cross] * step_dt
                erased[discrete_cross] = True

            if apply_bridge_correction:
                u = rng.random(
                    chunk_n
                )  # FIXED size -- stream position independent of mu
                same_sign = (X * X_next > 0) & ~erased
                if same_sign.any():
                    exponent = -2 * X[same_sign] * X_next[same_sign] / (2 * D * step_dt)
                    p_cross = np.exp(exponent)
                    bridge_hit = u[same_sign] < p_cross
                    idx = np.flatnonzero(same_sign)[bridge_hit]
                    erasure_time[idx] = (
                        t_cursor + step_dt / 2.0
                    )  # documented approximation
                    erased[idx] = True

            X = X_next
            t_cursor += step_dt
            running_min = np.minimum(running_min, X)

        sum_final += X.sum()
        sumsq_final += (X**2).sum()
        n_erased += int(erased.sum())
        sum_erasure_time += float(np.nansum(erasure_time))
        sum_max_loss += float((L0 - running_min).sum())
        n_win += int((X > 0).sum())
        all_finals.append(X.copy())
        n_done += chunk_n

    finals = np.concatenate(all_finals)
    mean_final = sum_final / n_paths
    var_final = max(sumsq_final / n_paths - mean_final**2, 0.0)
    frac_erased = n_erased / n_paths

    return {
        "p_lead_erased": frac_erased,
        "p_win": n_win / n_paths,
        "final_margin_mean": float(mean_final),
        "final_margin_std": float(np.sqrt(var_final)),
        "final_margin_quantiles": {
            q: float(np.quantile(finals, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "max_lead_loss_mean": sum_max_loss / n_paths,
        "first_erasure_time_mean": (sum_erasure_time / n_erased)
        if n_erased
        else np.nan,
        "frac_erased": frac_erased,
    }


def simulate_landmark_row(
    row,
    D,
    mu,
    n_paths=2000,
    dt=30.0,
    chunk_size=500,
    apply_bridge_correction=True,
    rng=None,
):
    """Convenience wrapper: pulls L0=row['landmark_lead'],
    tau=row['time_remaining'] off a landmark row and calls simulate_paths."""
    return simulate_paths(
        L0=float(row["landmark_lead"]),
        tau=float(row["time_remaining"]),
        D=D,
        mu=mu,
        n_paths=n_paths,
        dt=dt,
        chunk_size=chunk_size,
        apply_bridge_correction=apply_bridge_correction,
        rng=rng,
    )
