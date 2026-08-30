"""Shared PlayByPlayV3 parsing helpers.

Both the PCA and Markov pipelines classify the same raw play-by-play feed, so the
zone and rebound logic lives here rather than being reimplemented per script.

Verified against nba_api 1.11.4. The V3 feed exposes these columns:

    gameId, actionNumber, clock, period, teamId, teamTricode, personId,
    playerName, playerNameI, xLegacy, yLegacy, shotDistance, shotResult,
    isFieldGoal, scoreHome, scoreAway, pointsTotal, location, description,
    actionType, subType, videoAvailable, actionId

Note there is no score-margin column; use score_margin() to derive it.
"""

import re

import numpy as np
import pandas as pd

# The full closed vocabulary of actionType, confirmed against live data.
# Matching these exactly is safer than substring tests against the prose in
# `description`, which is what the earlier versions of both scripts relied on.
ACTION_REBOUND = "Rebound"
ACTION_FREE_THROW = "Free Throw"
ACTION_TURNOVER = "Turnover"
ACTION_MADE_SHOT = "Made Shot"
ACTION_MISSED_SHOT = "Missed Shot"

# Court geometry in the legacy coordinate system (tenths of a foot), hoop at
# the origin.
RIM_RADIUS = 40.0  # 4 ft, matching the Restricted Area
LANE_HALF_WIDTH = 80.0  # 16 ft lane
FREE_THROW_LINE = 142.5
CORNER_X = 220.0  # corner 3 sits at 22 ft from the hoop
CORNER_Y = 92.5  # above this the arc begins
HALF_COURT = 422.5  # 47 ft from the baseline, less the 4.75 ft hoop inset

# Rebound descriptions carry a running per-player tally, e.g.
#   "Reaves REBOUND (Off:1 Def:0)"
# Both substrings appear on every rebound, so testing for their presence
# classifies every rebound as both offensive and defensive. The counts have to
# be diffed instead.
_REBOUND_TALLY = re.compile(r"Off:(\d+)\s*Def:(\d+)")

# Team-charged turnovers name the team only in the description, by nickname:
#   "Nets Turnover: Shot Clock (T#2)"   "BUCKS Turnover: Shot Clock (T#8)"
_TEAM_EVENT = re.compile(r"^([A-Za-z][A-Za-z0-9 ]*?)\s+Turnover")
_NICKNAMES: dict = {}


def _load_nicknames():
    """Nickname -> tricode, upper-cased because the feed varies its casing."""
    if _NICKNAMES:
        return _NICKNAMES
    try:
        from nba_api.stats.static import teams as _teams

        for t in _teams.get_teams():
            _NICKNAMES[t["nickname"].upper()] = t["abbreviation"]
            _NICKNAMES[t["full_name"].upper()] = t["abbreviation"]
    except Exception:
        pass
    # The feed uses short forms the static table does not carry.
    _NICKNAMES.setdefault("BLAZERS", "POR")
    _NICKNAMES.setdefault("CAVS", "CLE")
    _NICKNAMES.setdefault("SIXERS", "PHI")
    return _NICKNAMES


def _text(df, col):
    """Column as a plain string Series, or empty strings if absent."""
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype=object)
    return df[col].astype(str).fillna("")


def score_margin(df):
    """Signed home-away margin per event.

    PlayByPlayV3 has no scoreMargin column. Scores are only populated on scoring
    events, so they are forward-filled across the intervening rows.
    """
    home = pd.to_numeric(_text(df, "scoreHome"), errors="coerce").ffill().fillna(0)
    away = pd.to_numeric(_text(df, "scoreAway"), errors="coerce").ffill().fillna(0)
    return (home - away).astype(int)


REGULATION_PERIOD = 720  # 12 minutes
OVERTIME_PERIOD = 300  # 5 minutes
REGULATION_END = 4 * REGULATION_PERIOD

_ISO_CLOCK = re.compile(r"PT(\d+)M([\d.]+)S")
_MMSS_CLOCK = re.compile(r"(\d+):([\d.]+)")


def clock_remaining(clock_str):
    """Seconds left in the period.

    The V3 feed uses ISO 8601 durations ("PT11M56.00S"). Matching M(\\d+) against
    that string finds the minutes marker sitting in front of the *seconds* and
    returns 56 minutes, so the pattern has to be anchored on PT.
    """
    if pd.isna(clock_str):
        return 0.0
    s = str(clock_str)
    m = _ISO_CLOCK.search(s) or _MMSS_CLOCK.search(s)
    if not m:
        return 0.0
    return int(m.group(1)) * 60 + float(m.group(2))


def game_time_seconds(df):
    """Elapsed game time per event, measured from tip-off.

    Overtime periods are 5 minutes, not 12. Treating them as 12 pushes OT events
    past the end of regulation and makes any "time remaining" figure derived
    from a 2880-second game negative.
    """
    period = pd.to_numeric(_text(df, "period"), errors="coerce").fillna(1)
    remaining = df["clock"].map(clock_remaining) if "clock" in df.columns else 0.0

    is_reg = period <= 4
    period_len = np.where(is_reg, REGULATION_PERIOD, OVERTIME_PERIOD)
    prior = np.where(
        is_reg,
        (period - 1) * REGULATION_PERIOD,
        REGULATION_END + (period - 5) * OVERTIME_PERIOD,
    )
    return pd.Series(prior + (period_len - remaining), index=df.index)


def classify_rebounds(df):
    """Split rebounds into offensive and defensive.

    Returns (is_orb, is_drb) as boolean Series aligned to df.index.

    Works by diffing the running Off:/Def: tally within each player's own
    sequence of rebounds. The diff has to be taken over the rebound rows only —
    grouping across the whole frame misaligns the shift, because the tally is
    NaN on every non-rebound row and the shift then picks up a NaN neighbour.

    Team rebounds (blank teamTricode, no tally in the description) are excluded.
    This matches the official boxscore, which counts only player rebounds.
    """
    is_orb = pd.Series(False, index=df.index)
    is_drb = pd.Series(False, index=df.index)

    reb = _text(df, "actionType").eq(ACTION_REBOUND)
    if not reb.any():
        return is_orb, is_drb

    tally = _text(df, "description").str.extract(_REBOUND_TALLY).astype(float)
    scored = reb & tally[0].notna()
    if not scored.any():
        return is_orb, is_drb

    r = pd.DataFrame(
        {
            "person": df.loc[scored, "personId"],
            "off": tally.loc[scored, 0],
            "def": tally.loc[scored, 1],
        }
    )

    off_delta = r["off"] - r.groupby("person")["off"].shift(1).fillna(0)
    def_delta = r["def"] - r.groupby("person")["def"].shift(1).fillna(0)

    is_orb.loc[scored] = (off_delta > 0).values
    is_drb.loc[scored] = (def_delta > 0).values
    return is_orb, is_drb


def classify_shot_zones(df):
    """Assign each field goal attempt to exactly one zone.

    Returns a Series over df.index holding one of 'Rim', 'Paint', 'Mid', 'C3',
    'ATB3', or None for non-shot rows.

    The three two-point zones mirror how the season baseline endpoint splits
    them (Restricted Area / In The Paint (Non-RA) / Mid-Range), so game-level
    rates and baseline rates measure the same regions. An earlier version
    compared a 4 ft "Rim" against a baseline that folded all paint shots
    together, which biased the rim and mid deviations in opposite directions.
    """
    is_fg = pd.to_numeric(_text(df, "isFieldGoal"), errors="coerce").fillna(0) == 1

    desc = _text(df, "description")
    is_3pt = is_fg & desc.str.contains("3PT", case=False, regex=False)
    is_2pt = is_fg & ~is_3pt

    x = pd.to_numeric(_text(df, "xLegacy"), errors="coerce").abs()
    y = pd.to_numeric(_text(df, "yLegacy"), errors="coerce")
    dist = np.sqrt(x**2 + y**2)

    # NaN coordinates would silently fall through to the catch-all zones, which
    # would read as a behavioural signal rather than as missing data.
    missing = is_fg & (x.isna() | y.isna())

    at_rim = dist <= RIM_RADIUS
    in_paint = (x <= LANE_HALF_WIDTH) & (y <= FREE_THROW_LINE)
    in_corner = (x >= CORNER_X) & (y <= CORNER_Y)
    # Heaves from beyond half court are their own zone in the baseline endpoint
    # (`Backcourt FGA`), so folding them into ATB3 would overstate the game-level
    # above-the-break rate against a baseline that excludes them.
    backcourt = y > HALF_COURT

    zone = pd.Series(None, index=df.index, dtype=object)
    zone[is_2pt & at_rim] = "Rim"
    zone[is_2pt & ~at_rim & in_paint] = "Paint"
    zone[is_2pt & ~at_rim & ~in_paint] = "Mid"
    zone[is_3pt & in_corner] = "C3"
    zone[is_3pt & ~in_corner] = "ATB3"
    zone[is_fg & backcourt] = "Backcourt"
    zone[missing] = None
    return zone


def free_throw_results(df):
    """(is_fta, is_ftm) boolean Series.

    Free throws carry an empty shotResult, so the make/miss has to come from the
    description: made attempts read "Davis Free Throw 1 of 2 (3 PTS)" while
    misses are prefixed "MISS".
    """
    is_fta = _text(df, "actionType").eq(ACTION_FREE_THROW)
    missed = _text(df, "description").str.upper().str.lstrip().str.startswith("MISS")
    return is_fta, is_fta & ~missed


def is_turnover(df):
    return _text(df, "actionType").eq(ACTION_TURNOVER)


def is_made_shot(df):
    """Made field goals. Does not cover free throws — see free_throw_results."""
    return _text(df, "actionType").eq(ACTION_MADE_SHOT)


def is_field_goal(df):
    return pd.to_numeric(_text(df, "isFieldGoal"), errors="coerce").fillna(0) == 1


def count_possessions(df):
    """Exact possession count per team, walking the event stream.

    Returns {teamTricode: possessions}.

    MUST be called on a whole game. It depends on classify_rebounds, which diffs
    a running per-player tally, so on a sliced frame the first rebound of each
    player reads as both offensive and defensive. To count possessions inside a
    window, use possession_events() on the full game and filter the result.

    This replaces the usual `FGA - OREB + TOV + 0.44*FTA` estimator. Because the
    full event stream is already available there is no need to estimate: a
    possession runs until the ball actually changes hands.

      - an offensive rebound CONTINUES the possession; the same team stays on
        offense and no new possession is opened
      - a defensive rebound ends it
      - a team rebound ends it only if the rebounding team is not the team that
        just shot (otherwise it is a ball retained out of bounds)
      - a period boundary resets possession
      - consecutive live events by the same team collapse into one possession,
        so an and-1 and a two-shot foul each count once

    Sanity check for callers: within a single period the two teams' counts should
    differ by at most one, since possessions alternate. Across a whole game a
    larger gap is legitimate -- a team that both ends one period and starts the
    next genuinely gains a possession.

    Known limitation: flagrant fouls award free throws and also retain the ball
    for the fouled team, which this walk scores as one possession rather than
    two. Measured across 28 games that is about 0.18 possessions per game.

    The `possessions` field in BoxScoreAdvancedV3 is itself a formula estimate,
    not ground truth. Measured over 56 team-games it reads 1.4 higher than the
    exact walk on average (0.6 after run-out rebounds were counted), and in 3 of
    28 games it reported the two teams differing by more than one, which cannot
    physically happen. It is a useful sanity anchor -- far closer than
    FGA - OREB + TOV + 0.44*FTA, which runs 3.9 high -- but do not calibrate
    against it.

    Note this counts possessions that reach a shot, free throw, turnover, or
    defensive rebound. A possession that produces none of those is invisible.
    """
    team = event_team(df)
    counts = {}
    for i in _possession_indices(df):
        t = team.iloc[i]
        counts[t] = counts.get(t, 0) + 1
    return counts


def possession_starts(df):
    """Boolean Series marking the row that opens each possession.

    Same constraint as count_possessions -- pass a whole game. Callers that need
    possessions inside a window should attach this to the frame once and then
    sum it over the slice, rather than re-running the walk on the slice.
    """
    mask = pd.Series(False, index=df.index)
    positions = _possession_indices(df)
    if positions:
        mask.iloc[positions] = True
    return mask


def possession_events(df):
    """One entry per possession: (period, game_time_sec, teamTricode)."""
    period = pd.to_numeric(_text(df, "period"), errors="coerce")
    elapsed = game_time_seconds(df)
    team = event_team(df)
    return [
        (period.iloc[i], elapsed.iloc[i], team.iloc[i]) for i in _possession_indices(df)
    ]


def event_team(df):
    """Team credited with each event, filling in the blanks the feed leaves.

    Turnovers charged to the team rather than a player -- shot-clock and excess
    timeout violations -- arrive with teamTricode '' and teamId 0, naming the
    team only in the description ("Nets Turnover: Shot Clock (T#2)"). Left blank
    they are skipped by the possession walk, and a possession whose only live
    event is a shot-clock violation disappears entirely.
    """
    team = _text(df, "teamTricode").str.strip()
    blank = team.eq("") & is_turnover(df)
    if not blank.any():
        return team
    nick = _text(df, "description").str.extract(_TEAM_EVENT)[0]
    resolved = nick.str.upper().map(_load_nicknames()).fillna("")
    return team.mask(blank, resolved)


def _possession_indices(df):
    """Positions (not labels) of rows that open a possession.

    Positional so a frame with a duplicated index -- two games concatenated, say
    -- cannot silently resolve one label to several rows.
    """
    if "period" not in df.columns:
        raise KeyError("'period' is required to count possessions")
    if not df.index.is_unique:
        raise ValueError(
            "possession counting needs a unique index; a duplicated one (two "
            "games concatenated, or an actionNumber index -- action numbers are "
            "shared between a shot and the block on it) would resolve one label "
            "to several rows."
        )

    team = event_team(df)
    # Without a forward fill a NaN period compares unequal to itself and resets
    # possession on every single row, silently multiplying the count.
    period = pd.to_numeric(_text(df, "period"), errors="coerce").ffill().fillna(1)
    # Offensive rebounds need no explicit branch: a rebound is not a live event,
    # so it is already skipped below and the possession simply continues.
    _, drb = classify_rebounds(df)

    fta, _ = free_throw_results(df)
    # Technical free throws are shot by whichever team was awarded them and do
    # not transfer the ball, so treating them as a live possession event invents
    # a possession for the shooter and splits the real one in two.
    sub = _text(df, "subType").str.lower()
    fta = fta & ~sub.str.contains("technical", na=False)
    live = is_field_goal(df) | fta | is_turnover(df)

    out = []
    current = None
    current_period = None

    for i in range(len(df)):
        p = period.iloc[i]
        t = team.iloc[i]

        if p != current_period:
            current = None
            current_period = p

        if drb.iloc[i]:
            # The rebound itself opens the possession rather than merely ending
            # the previous one. Waiting for the next shot loses every possession
            # that runs the clock out without one -- a rebound with five seconds
            # left, dribbled out -- which is about 1.7 per team per game.
            if t:
                out.append(i)
                current = t
            else:
                current = None
            continue

        if not live.iloc[i] or not t:
            # Team rebounds land here: blank tricode, so defer to the next live
            # event, which resolves retained-ball and change-of-possession alike.
            continue

        if t != current:
            out.append(i)
            current = t

    return out
