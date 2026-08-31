"""Investigation 2 of docs/superpowers/specs/2026-08-07-markov-violation-diagnostics-design.md:
builds the "spell" (lead-holding run) and person-period hazard dataset that
duration_hazard.py tests for duration-dependence on.

A spell is a maximal run of regulation-time scoring events (events.py's event stream)
during which the leading side (sign of home_away_margin) does not change. This module
turns that into a person-period table: one row per event a spell "owns", carrying how long
the spell has run so far (duration, in seconds and in event count) and whether the spell
ends at the very next event -- the input a discrete-time hazard regression needs.

Two tie-handling conventions, both surfaced rather than one silently picked -- the exact
convention gap sde-memory-test.md's Clauset baseline validation already found moves
lead-change counts by ~65% (6.57 strict vs. 10.83 tie-inclusive, per-game):

  tie_inclusive=False (PRIMARY): a margin==0 touch does not end the spell. The row at that
  touch has margin=0 but its leader stays whoever led into it.
  tie_inclusive=True (SENSITIVITY CHECK): a margin==0 touch ends the spell and is its own
  degenerate, immediately-closed leader='tied' spell. Whatever leader resumes afterward --
  even the same team as before the touch -- starts a genuinely new spell.

Regulation-only, grouped by GAME_ID -- game boundaries never leak into duration or margin,
the same discipline sde_clauset.fit_clauset/per_game_diagnostics already apply.
"""

import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (_HERE, os.path.join(_HERE, "..")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from common import pbp as pbp_common

_COLUMNS = [
    "GAME_ID",
    "SEASON",
    "spell_id",
    "leader",
    "spell_start_time",
    "duration",
    "n_events_since_spell_start",
    "margin",
    "time_remaining",
    "spell_ends",
]


def build_person_period_table(events_df, tie_inclusive=False):
    """One row per regulation-time scoring event a spell owns. See module docstring for
    the tie_inclusive contract and column meanings. spell_id is a table-wide unique integer
    (not reset per game) so a caller can group by it directly.
    """
    reg = events_df[events_df["is_regulation"]].sort_values(
        ["GAME_ID", "game_time_sec"]
    )
    rows = []
    spell_counter = 0

    for game_id, g in reg.groupby("GAME_ID", sort=False):
        t = g["game_time_sec"].to_numpy()
        m = g["home_away_margin"].to_numpy()
        season = g["SEASON"].iloc[0]

        current_spell_id = None
        current_leader = None
        current_spell_start = None
        spell_event_count = 0
        last_row_idx = None  # index into `rows` of the current spell's most recent row

        for i in range(len(t)):
            margin_i = float(m[i])
            raw_sign = "home" if margin_i > 0 else ("away" if margin_i < 0 else "tied")

            if raw_sign == "tied":
                if tie_inclusive:
                    if last_row_idx is not None:
                        rows[last_row_idx]["spell_ends"] = 1
                    spell_counter += 1
                    rows.append(
                        {
                            "GAME_ID": game_id,
                            "SEASON": season,
                            "spell_id": spell_counter,
                            "leader": "tied",
                            "spell_start_time": float(t[i]),
                            "duration": 0.0,
                            "n_events_since_spell_start": 0,
                            "margin": 0.0,
                            "time_remaining": pbp_common.REGULATION_END - float(t[i]),
                            "spell_ends": 1,
                        }
                    )
                    current_spell_id = None
                    current_leader = None
                    current_spell_start = None
                    spell_event_count = 0
                    last_row_idx = None
                    continue
                if current_leader is None:
                    # No lead has been established yet -- this touch belongs to no spell.
                    continue
                effective_leader = current_leader
            else:
                effective_leader = raw_sign

            if effective_leader != current_leader:
                if last_row_idx is not None:
                    rows[last_row_idx]["spell_ends"] = 1
                spell_counter += 1
                current_spell_id = spell_counter
                current_leader = effective_leader
                current_spell_start = float(t[i])
                spell_event_count = 0

            rows.append(
                {
                    "GAME_ID": game_id,
                    "SEASON": season,
                    "spell_id": current_spell_id,
                    "leader": current_leader,
                    "spell_start_time": current_spell_start,
                    "duration": float(t[i]) - current_spell_start,
                    "n_events_since_spell_start": spell_event_count,
                    "margin": abs(margin_i),
                    "time_remaining": pbp_common.REGULATION_END - float(t[i]),
                    "spell_ends": 0,
                }
            )
            last_row_idx = len(rows) - 1
            spell_event_count += 1

        # A spell that survives to the game's last regulation event is right-censored:
        # its last row's spell_ends stays 0 (the default) -- no extra row is emitted.

    if not rows:
        return pd.DataFrame(columns=_COLUMNS)
    return pd.DataFrame(rows, columns=_COLUMNS)


def spells_per_game(person_period_df):
    """GAME_ID -> number of distinct spells. Cross-check against
    sde_clauset.per_game_diagnostics' n_lead_changes: in strict mode (tie_inclusive=False),
    n_spells == n_lead_changes + 1 exactly on the same input, since both are counting runs
    between the same sign-flip boundaries."""
    return person_period_df.groupby("GAME_ID")["spell_id"].nunique()
