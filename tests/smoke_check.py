"""Hand-checkable verification: prints numbers you can eyeball against reality.

    python tests/smoke_check.py            # offline, uses cached fixtures
    python tests/smoke_check.py --live     # also pulls one game from the API

Every number printed has an expected value or range next to it, so you can judge
the pipeline without reading the test suite.
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
for path in (SRC, os.path.join(SRC, "Markov"), HERE):
    sys.path.insert(0, path)

from common import pbp
from conftest import GAMES, load_box, load_pbp

OK, BAD = "  OK  ", " FAIL "
failures = []
skipped = []


def check(label, passed, detail):
    print(f"[{OK if passed else BAD}] {label:<46} {detail}")
    if not passed:
        failures.append(label)


def rule(title):
    print(f"\n{title}\n" + "-" * 78)


# ---------------------------------------------------------------------------
rule("1. CLASSIFICATION vs THE OFFICIAL BOXSCORE  (must match exactly)")
print("   Every rebound, shot and free throw is classified from the play-by-play")
print("   text, then totalled per team and compared to what the NBA publishes.\n")

for gid in sorted(GAMES):
    df, box = load_pbp(gid), load_box(gid)
    orb, drb = pbp.classify_rebounds(df)
    zone, made = pbp.classify_shot_zones(df), pbp.is_made_shot(df)
    fta, ftm = pbp.free_throw_results(df)

    for _, row in box.iterrows():
        m = df["teamId"] == row["teamId"]
        got = (
            int((orb & m).sum()),
            int((drb & m).sum()),
            int((zone.notna() & m).sum()),
            int((zone.notna() & m & made).sum()),
            int((fta & m).sum()),
            int((ftm & m).sum()),
        )
        want = (
            int(row["reboundsOffensive"]),
            int(row["reboundsDefensive"]),
            int(row["fieldGoalsAttempted"]),
            int(row["fieldGoalsMade"]),
            int(row["freeThrowsAttempted"]),
            int(row["freeThrowsMade"]),
        )
        check(
            f"{gid} {row['teamTricode']}  ORB/DRB/FGA/FGM/FTA/FTM",
            got == want,
            f"{got}  vs official {want}",
        )

# ---------------------------------------------------------------------------
rule("2. POSSESSIONS  (NBA teams run 90-110 per game; the two teams must agree)")
print("   Counted by walking the event stream. An offensive rebound continues the")
print("   possession instead of starting a new one.\n")

for gid in sorted(GAMES):
    df = load_pbp(gid)
    counts = pbp.count_possessions(df)
    periods = int(pd.to_numeric(df["period"], errors="coerce").max())
    minutes = 48 + max(0, periods - 4) * 5
    lo, hi = sorted(counts.values())
    pace = [round(n / minutes * 48, 1) for n in counts.values()]
    check(
        f"{gid} ({periods} periods)",
        hi - lo <= 3 and all(80 <= p <= 115 for p in pace),
        f"{counts}  pace {pace}  gap {hi - lo}",
    )

# ---------------------------------------------------------------------------
rule("3. SHOT ZONES  (five zones must account for every attempt, no double count)")
for gid in sorted(GAMES):
    df = load_pbp(gid)
    counts = pbp.classify_shot_zones(df).value_counts()
    check(
        f"{gid} zone totals == field goal attempts",
        counts.sum() == int(pbp.is_field_goal(df).sum()),
        f"{counts.sum()} zoned of {int(pbp.is_field_goal(df).sum())} attempts",
    )

# ---------------------------------------------------------------------------
# MIGRATION NOTE: section 4 (GAME VECTOR) was removed when this bundle was
# built -- it exercised src/PCA/pca.py, the abandoned PCA line, which is not
# part of this migration. Sections are left numbered as in the origin repo.

# ---------------------------------------------------------------------------
rule("5. GUARDS  (things that should refuse to run)")

try:
    pbp.count_possessions(pd.concat([load_pbp("0022300061")] * 2))
    check("duplicated index is rejected", False, "no error raised")
except ValueError:
    check("duplicated index is rejected", True, "ValueError, as intended")

try:
    pbp.count_possessions(load_pbp("0022300061").drop(columns=["period"]))
    check("missing period column is rejected", False, "no error raised")
except KeyError:
    check("missing period column is rejected", True, "KeyError, as intended")

# ---------------------------------------------------------------------------
rule("6. V3 FEATURE TABLE  (regenerated dataset; requires the full scrape to exist)")

import scrape_common as regression

v3_path = regression.DEFAULT_DATASET
if not os.path.exists(v3_path):
    print(f"   Skipped: {v3_path} does not exist yet (run the full scrape first).")
    skipped.append("section 6 (v3 feature table) skipped: v3 dataset not present")
else:
    v3 = pd.read_csv(v3_path)

    y_min = v3["Y_Absolute_Loss"].min()
    check(
        "Y_Absolute_Loss min",
        y_min >= 0,
        f"{y_min}     (expect >= 0; was <= 0 for all games before)",
    )

    b_rim_nunique = v3["B_r_Rim"].nunique()
    check(
        "B_r_Rim distinct values",
        b_rim_nunique > 1,
        f"{b_rim_nunique}     (expect ~150, 30 teams x 5 seasons, not 1)",
    )

    collapsed_frac = v3["collapsed"].mean()
    check(
        "collapsed fraction",
        collapsed_frac < 0.5,
        f"{collapsed_frac:.3f}   (expect ~0.338, well below the old 0.88 that "
        "lead_changed_hands reported)",
    )

    c_mean = v3["C"].mean()
    check("C mean", 0 <= c_mean <= 1, f"{c_mean:.3f}   (expect ~0.292, within [0, 1])")

    if "SEASON_TYPE" in v3.columns:
        n_playoff = int((v3["SEASON_TYPE"] != "Regular Season").sum())
        check(
            "v3 playoff rows are a small minority",
            0 < n_playoff < 0.15 * len(v3),
            f"{n_playoff} of {len(v3)} playoff rows (expect ~422 of 6572)",
        )

# ---------------------------------------------------------------------------
if "--live" in sys.argv:
    rule("7. LIVE API  (confirms the endpoints still return what we expect)")
    from nba_api.stats.endpoints import playbyplayv3

    gid = "0022300061"
    live = playbyplayv3.PlayByPlayV3(game_id=gid, timeout=60).get_data_frames()[0]
    cached = load_pbp(gid)
    check(
        "live play-by-play matches the cached fixture shape",
        list(live.columns) == list(cached.columns) and len(live) == len(cached),
        f"{live.shape} vs cached {cached.shape}",
    )
    check(
        "no scoreMargin column (must be derived)",
        "scoreMargin" not in live.columns,
        "absent, as expected",
    )

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
if failures:
    print(f"{len(failures)} CHECK(S) FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
if skipped:
    print("ALL CHECKS PASSED (" + "; ".join(skipped) + ")")
else:
    print("ALL CHECKS PASSED")
