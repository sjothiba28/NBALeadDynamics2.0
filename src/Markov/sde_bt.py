"""Bradley-Terry team-strength constructions for the Clauset SDE study.

Two clearly separate constructions (per the project spec,
docs/plans/claude-sde-repair-and-implementation-guide.md):
1. Exact paper replication (R0): Gaussian(mean=1, std=0.09) strength
   distribution, S1/(S1+S2) scoring rate, bias-averaged over many draws --
   an AGGREGATE benchmark, not a fitted rating for any observed game.
2. Empirical Clauset-form control: for each focal game, team-season
   strengths fit on that season's OTHER games (leave-one-game-out), same
   S1/(S1+S2) likelihood, normalized to mean-zero log strength. This is the
   `v_BT` every Stage-1 model (R0's BT-only check, M0) reads identically.

Both share bt_scoring_prob/v_from_bt_prob -- the ONE place the
scoring-probability-to-drift conversion is defined, so R0 and the empirical
control can never silently diverge in how q becomes v.
"""

import numpy as np

import sde_simulate

_PAPER_STRENGTH_MEAN = 1.0
_PAPER_STRENGTH_STD = 0.09


def bt_scoring_prob(s1, s2):
    """P(team 1 scores the next event) = S1/(S1+S2), the paper's own
    Bradley-Terry competition-model form (Section III)."""
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    return s1 / (s1 + s2)


def v_from_bt_prob(q, event_rate, mean_s):
    """Convert a Bradley-Terry scoring probability q into a drift velocity
    (points/sec, focal-team-signed): rate_focal = event_rate*q,
    rate_opponent = event_rate*(1-q), v = (rate_focal-rate_opponent)*mean_s
    = event_rate*mean_s*(2*q-1). NOT given explicitly by the paper -- this
    is the documented derivation (see docs/plans/
    claude-sde-repair-and-implementation-guide.md's team-strength control
    section). event_rate/mean_s should come from sde_clauset.fit_clauset's
    return dict, shared by every caller so this conversion never gets
    redefined with different constants in two places."""
    q = np.asarray(q, dtype=float)
    return event_rate * mean_s * (2 * q - 1)


def bt_log_odds(s_r, s_b):
    """eta_BT = log(pi_r) - log(pi_b) = logit(bt_scoring_prob(s_r, s_b)) --
    the log-odds offset the Peel-Clauset event model (docs/plans/
    two-paper-synthesis-and-variable-test.md) needs for its sigmoid, as
    opposed to v_from_bt_prob's points/sec drift conversion (the SDE
    model's own object). Kept as a separate function, not a v_from_bt_prob
    call plus an inverse-sigmoid, so the two conversions can never be
    confused for each other at a call site."""
    s_r = np.asarray(s_r, dtype=float)
    s_b = np.asarray(s_b, dtype=float)
    return np.log(s_r) - np.log(s_b)


def paper_bias_averaged_safe_lead(
    L,
    tau,
    D,
    event_rate,
    mean_s,
    n_draws=20000,
    strength_mean=_PAPER_STRENGTH_MEAN,
    strength_std=_PAPER_STRENGTH_STD,
    rng=None,
):
    """R0: the paper's own aggregate-benchmark bias averaging (Section III).
    Draw n_draws i.i.d. (S1,S2) ~ Gaussian(strength_mean, strength_std),
    convert to a scoring probability (bt_scoring_prob) then a drift
    (v_from_bt_prob), and average the CLOSED-FORM Eq.17 survival
    probability over the draws -- analytic averaging, not nested Monte
    Carlo path simulation, since this is an aggregate benchmark over the
    strength distribution, not a per-landmark fitted rating (contrast with
    attach_v_bt, Task 7). Returns the bias-averaged safe-lead probability
    (directly comparable to Eq.15's erf(z) -- NOT an erasure probability).
    """
    rng = np.random.default_rng(rng)
    s1 = np.clip(rng.normal(strength_mean, strength_std, n_draws), 1e-6, None)
    s2 = np.clip(rng.normal(strength_mean, strength_std, n_draws), 1e-6, None)
    q = bt_scoring_prob(s1, s2)
    v = v_from_bt_prob(q, event_rate, mean_s)
    Q = sde_simulate.eq17_survival_probability(L, tau, D, v)
    return float(np.mean(Q))


def _season_pairwise_counts(season_events):
    """Aggregate {frozenset({team_i,team_j}): {'n': int, 'w': {team: int}}}
    ONCE per season, plus a per-GAME_ID breakdown (game_contribution) so a
    single game's contribution can be SUBTRACTED in O(1) (_exclude_game)
    instead of re-scanning the season's events for every excluded game.
    n = total scoring events between the pair; w[team] = events scored by
    that team. Games with other than exactly 2 distinct scoring teams are
    skipped (should not occur post events.py parsing; defensive guard)."""
    pair_totals = {}
    game_contribution = {}
    for game_id, g in season_events.groupby("GAME_ID", sort=False):
        teams = [t for t in g["scoring_team"].dropna().unique() if t]
        if len(teams) != 2:
            continue
        key = frozenset(teams)
        counts = g["scoring_team"].value_counts().to_dict()
        n = len(g)
        game_contribution[game_id] = (key, n, counts)
        entry = pair_totals.setdefault(key, {"n": 0, "w": {}})
        entry["n"] += n
        for team, c in counts.items():
            entry["w"][team] = entry["w"].get(team, 0) + int(c)
    return pair_totals, game_contribution


def _exclude_game(pair_totals, game_contribution, game_id):
    """A NEW dict with `game_id`'s own contribution subtracted from its
    matchup pair only -- every other pair is the SAME reference (cheap:
    O(num_pairs) dict copy, O(1) per-pair mutation), not a full rescan."""
    if game_id not in game_contribution:
        return pair_totals
    key, n, counts = game_contribution[game_id]
    excluded = dict(pair_totals)
    entry = excluded[key]
    new_w = dict(entry["w"])
    for team, c in counts.items():
        new_w[team] = new_w.get(team, 0) - c
    excluded[key] = {"n": entry["n"] - n, "w": new_w}
    return excluded


def fit_bt_strengths(pair_totals, teams, max_iter=200, tol=1e-8):
    """Zermelo/minorization-maximization fixed point for Bradley-Terry
    strengths from aggregated pairwise scoring-event counts:
    S_i <- W_i / sum_{j!=i}[n_ij / (S_i + S_j)], W_i = total events scored
    by team i across all its games. Converges to the MLE of the paper's own
    S1/(S1+S2) likelihood. Normalized so mean(log(S)) == 0 (guide's
    explicit 'normalize log strengths to mean zero') -- absolute strength
    is immaterial, only ratios matter, matching the paper's own framing."""
    S = {t: 1.0 for t in teams}
    W = {t: 0 for t in teams}
    neighbors = {t: [] for t in teams}
    for key, entry in pair_totals.items():
        i, j = tuple(key)
        if i not in W or j not in W:
            continue
        neighbors[i].append((j, entry["n"]))
        neighbors[j].append((i, entry["n"]))
        W[i] += entry["w"].get(i, 0)
        W[j] += entry["w"].get(j, 0)

    for _ in range(max_iter):
        max_delta = 0.0
        S_new = {}
        for t in teams:
            denom = sum(n / (S[t] + S[j]) for j, n in neighbors[t])
            S_new[t] = (W[t] / denom) if denom > 0 else S[t]
            max_delta = max(max_delta, abs(S_new[t] - S[t]))
        S = S_new
        if max_delta < tol:
            break

    log_s = {t: np.log(max(v, 1e-12)) for t, v in S.items()}
    mean_log = np.mean(list(log_s.values()))
    return {t: float(np.exp(v - mean_log)) for t, v in log_s.items()}


def fit_season_strengths_leave_one_game_out(events_df, season_type="Regular Season"):
    """{(SEASON, GAME_ID): {team: strength}} -- leave-one-game-out
    Bradley-Terry team strengths for every (SEASON, GAME_ID) pair present in
    `events_df`'s regulation-time events of `season_type`. Extracted out of
    attach_v_bt (which drives it with a landmark-row loop) so a whole-game,
    non-landmark-anchored caller -- pc_events.py's PC-replication event
    rows, which need a strength fit for every scoring event's own game, not
    just landmark games -- can reuse the identical leave-one-game-out fit
    without duplicating _season_pairwise_counts/_exclude_game/
    fit_bt_strengths' three-function dance. attach_v_bt itself is
    unchanged (still drives its own loop keyed off landmark rows), but both
    now go through fit_bt_strengths with an _exclude_game'd pair_totals, so
    the strength fit for a given (SEASON, GAME_ID) can never silently
    diverge between the two callers.
    """
    reg = events_df[
        (events_df["is_regulation"]) & (events_df["SEASON_TYPE"] == season_type)
    ].copy()
    reg["GAME_ID"] = reg["GAME_ID"].astype(str).str.strip().str.zfill(10)

    out = {}
    for season, season_events in reg.groupby("SEASON"):
        teams = sorted(t for t in season_events["scoring_team"].dropna().unique() if t)
        pair_totals, game_contribution = _season_pairwise_counts(season_events)
        for game_id in game_contribution:
            excluded = _exclude_game(pair_totals, game_contribution, game_id)
            out[(season, game_id)] = fit_bt_strengths(excluded, teams)
    return out


def attach_v_bt(
    landmarks_df, events_df, event_rate, mean_s, season_type="Regular Season"
):
    """landmarks_df + v_BT/q_focal/S_focal/S_opponent columns -- the SINGLE
    stored v_BT every Stage-1 model (R0's BT-only check, M0) reads, computed
    ONCE here and never re-derived downstream.

    v_BT is a SEASON-LEVEL, leave-one-GAME-out control, independent of any
    later leave-one-SEASON-out fold split -- exactly like this codebase's
    existing B_*/Delta_* season-baseline controls (sde_drift.
    attach_opponent_baselines), which are also computed once, before any
    fold split, so a held-out season's own landmarks still have a control
    to be scored against. event_rate/mean_s should come from sde_clauset.
    fit_clauset's GLOBAL (all-season) values -- v_BT must not itself vary
    by which fold is asking; only D varies per fold (see sde_kappa.py).

    Grouped by (SEASON, GAME_ID) rather than per landmark row: every
    landmark row from the SAME game excludes the SAME game and therefore
    shares ONE strength fit -- refit once per game, not once per landmark
    row (a game can produce up to 6 landmark rows: both teams x 3
    thresholds).
    """
    reg = events_df[
        (events_df["is_regulation"]) & (events_df["SEASON_TYPE"] == season_type)
    ].copy()
    reg["GAME_ID"] = reg["GAME_ID"].astype(str).str.strip().str.zfill(10)

    out = landmarks_df.reset_index(drop=True).copy()
    out["GAME_ID"] = out["GAME_ID"].astype(str).str.strip().str.zfill(10)

    v_bt = np.full(len(out), np.nan)
    q_focal = np.full(len(out), np.nan)
    s_focal_arr = np.full(len(out), np.nan)
    s_opp_arr = np.full(len(out), np.nan)

    for season, season_events in reg.groupby("SEASON"):
        teams = sorted(t for t in season_events["scoring_team"].dropna().unique() if t)
        pair_totals, game_contribution = _season_pairwise_counts(season_events)
        season_rows = out[out["SEASON"] == season]
        for game_id, game_rows in season_rows.groupby("GAME_ID"):
            excluded = _exclude_game(pair_totals, game_contribution, game_id)
            strengths = fit_bt_strengths(excluded, teams)
            for idx, row in game_rows.iterrows():
                # A team code genuinely absent from this season's fitted
                # strengths (should not happen with real NBA data -- every
                # team scores in its own game) must NOT silently default to
                # a neutral strength (that would produce a fake, plausible-
                # looking v_BT=0.0 "even matchup" instead of surfacing the
                # bug). Leave this row NaN, same as the season-mismatch case
                # below, so the existing unmatched-row [WARN] catches it too
                # -- one detection path for both failure modes, not two.
                if row["LEADER"] not in strengths or row["OPPONENT"] not in strengths:
                    continue
                s_l = strengths[row["LEADER"]]
                s_o = strengths[row["OPPONENT"]]
                q = float(bt_scoring_prob(s_l, s_o))
                v_bt[idx] = float(v_from_bt_prob(q, event_rate, mean_s))
                q_focal[idx] = q
                s_focal_arr[idx] = s_l
                s_opp_arr[idx] = s_o

    out["v_BT"] = v_bt
    out["q_focal"] = q_focal
    out["S_focal"] = s_focal_arr
    out["S_opponent"] = s_opp_arr

    unmatched = np.isnan(v_bt)
    if unmatched.any():
        n_unmatched = int(unmatched.sum())
        print(
            f"  [WARN] {n_unmatched} of {len(out)} landmark rows had no "
            "matching (SEASON, GAME_ID) in events_df, or a LEADER/OPPONENT "
            "team code absent from that season's fitted strengths -- "
            "v_BT/q_focal/S_focal/S_opponent left NaN rather than "
            "sign-guessed or defaulted to a neutral strength. Likely "
            "cause: a SEASON label mismatch between landmarks_df and "
            "events_df, a season whose regulation events were entirely "
            "filtered out by season_type/is_regulation, or a team code "
            "that never scored in that season's regulation events."
        )

    return out
