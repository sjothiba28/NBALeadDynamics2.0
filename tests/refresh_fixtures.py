"""Regenerate the cached play-by-play and boxscore fixtures.

    python tests/refresh_fixtures.py

Only needed when adding a game or when the V3 schema changes. The offline suite
runs against whatever is committed here, so re-running this is what makes a
schema change visible as a test failure rather than a silent behaviour change.
"""

import os
import time

from nba_api.stats.endpoints import boxscoretraditionalv3, playbyplayv3

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")

# Each game covers a case that has previously broken this code.
GAMES = {
    "0022300061": "regular season",
    "0022300476": "double overtime, 6 periods",
    "0042200401": "playoff game; ShotChartDetail returns nothing for these",
    "0022100001": "contains a flagrant foul",
    "0022400500": "2024-25 season",
}

BOX_COLUMNS = [
    "fieldGoalsAttempted",
    "fieldGoalsMade",
    "reboundsOffensive",
    "reboundsDefensive",
    "freeThrowsAttempted",
    "freeThrowsMade",
    "turnovers",
]


def main():
    os.makedirs(FIXTURES, exist_ok=True)
    for game_id, note in GAMES.items():
        pbp = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=60).get_data_frames()[
            0
        ]
        pbp.to_csv(os.path.join(FIXTURES, f"pbp_{game_id}.csv"), index=False)

        # Frame 1 is a starters/bench split, two rows per team, so it has to be
        # grouped to get team totals.
        box = boxscoretraditionalv3.BoxScoreTraditionalV3(
            game_id=game_id, timeout=60
        ).get_data_frames()[1]
        (
            box.groupby(["teamId", "teamTricode"], as_index=False)[BOX_COLUMNS]
            .sum()
            .to_csv(os.path.join(FIXTURES, f"box_{game_id}.csv"), index=False)
        )

        print(f"{game_id}  {pbp.shape[0]:>4} events  ({note})")
        time.sleep(0.6)


if __name__ == "__main__":
    main()
