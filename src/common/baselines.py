import time

import numpy as np
import pandas as pd

from nba_api.stats.endpoints import leaguedashteamstats, leaguedashteamshotlocations
from nba_api.stats.static import teams

BASELINE_COLUMNS = [
    "b_r_Rim",
    "b_r_Paint",
    "b_r_Mid",
    "b_r_C3",
    "b_r_ATB3",
    "b_r_FT",
    "b_p_Rim",
    "b_p_Paint",
    "b_p_Mid",
    "b_p_C3",
    "b_p_ATB3",
    "b_p_FT",
    "b_ORB_pct",
    "b_TOV_pct",
    "b_ATOP",
]


# ==========================================
# 2. SEASONS BASELINE (The Control Group)
# ==========================================
def fetch_season_baselines(seasons, season_type="Regular Season"):
    print("\n--- 1. Compiling True Season Baselines ---")
    all_stats = []
    nba_teams = teams.get_teams()
    team_abbr_map = {team["id"]: team["abbreviation"] for team in nba_teams}

    for season in seasons:
        try:
            stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season, season_type_all_star=season_type
            ).get_data_frames()[0]
            stats["SEASON"] = season
            stats["LEADER"] = stats["TEAM_ID"].map(team_abbr_map)

            zones = leaguedashteamshotlocations.LeagueDashTeamShotLocations(
                season=season,
                season_type_all_star=season_type,
                distance_range="By Zone",
            ).get_data_frames()[0]
            if isinstance(zones.columns, pd.MultiIndex):
                zones.columns = [" ".join(col).strip() for col in zones.columns.values]
            stats = pd.merge(stats, zones, on="TEAM_ID", how="inner")

            def sum_zone(df, *names):
                """Sum named zone columns, matching each name exactly.

                Substring matching is unsafe here: the endpoint publishes
                'Left Corner 3', 'Right Corner 3' AND a combined 'Corner 3'
                column, so a regex like 'Corner 3.*FGA' matched all three and
                returned exactly double the true corner-3 volume.
                """
                cols = [c for c in names if c in df.columns]
                missing = [c for c in names if c not in df.columns]
                if missing:
                    raise KeyError(f"Shot-location columns missing: {missing}")
                return df[cols].sum(axis=1)

            # Zones mirror common.pbp.classify_shot_zones so game-level rates and
            # baseline rates describe the same regions. Rim is the restricted
            # area only; non-RA paint is its own zone rather than folded in.
            stats["rim_fga"] = sum_zone(stats, "Restricted Area FGA")
            stats["rim_fgm"] = sum_zone(stats, "Restricted Area FGM")
            stats["paint_fga"] = sum_zone(stats, "In The Paint (Non-RA) FGA")
            stats["paint_fgm"] = sum_zone(stats, "In The Paint (Non-RA) FGM")
            stats["mid_fga"] = sum_zone(stats, "Mid-Range FGA")
            stats["mid_fgm"] = sum_zone(stats, "Mid-Range FGM")
            stats["c3_fga"] = sum_zone(stats, "Left Corner 3 FGA", "Right Corner 3 FGA")
            stats["c3_fgm"] = sum_zone(stats, "Left Corner 3 FGM", "Right Corner 3 FGM")
            stats["atb3_fga"] = sum_zone(stats, "Above the Break 3 FGA")
            stats["atb3_fgm"] = sum_zone(stats, "Above the Break 3 FGM")

            stats["possessions"] = (
                (stats["FGA"] - stats["OREB"]) + stats["TOV"] + (0.44 * stats["FTA"])
            )

            # Baseline Data stored as pure ratios/percentages
            stats["b_r_Rim"] = stats["rim_fga"] / stats["FGA"]
            stats["b_r_Paint"] = stats["paint_fga"] / stats["FGA"]
            stats["b_r_Mid"] = stats["mid_fga"] / stats["FGA"]
            stats["b_r_C3"] = stats["c3_fga"] / stats["FGA"]
            stats["b_r_ATB3"] = stats["atb3_fga"] / stats["FGA"]
            stats["b_r_FT"] = stats["FTA"] / stats["possessions"]

            stats["b_p_Rim"] = stats["rim_fgm"] / stats["rim_fga"].replace(0, np.nan)
            stats["b_p_Paint"] = stats["paint_fgm"] / stats["paint_fga"].replace(
                0, np.nan
            )
            stats["b_p_Mid"] = stats["mid_fgm"] / stats["mid_fga"].replace(0, np.nan)
            stats["b_p_C3"] = stats["c3_fgm"] / stats["c3_fga"].replace(0, np.nan)
            stats["b_p_ATB3"] = stats["atb3_fgm"] / stats["atb3_fga"].replace(0, np.nan)
            stats["b_p_FT"] = stats["FT_PCT"]

            stats["b_ORB_pct"] = stats["OREB"] / (stats["OREB"] + stats["DREB"])
            stats["b_TOV_pct"] = stats["TOV"] / stats["possessions"]
            # Seconds per possession, matching the game-side ATOP so the differenced
            # pair shares units. The raw possessions/GP is a pace count, not a
            # duration, and differencing it against a seconds/possession game value is
            # incoherent.
            stats["b_ATOP"] = (48 * 60) / (stats["possessions"] / stats["GP"])

            # The zone rates partition all field goal attempts, so they must sum
            # to ~1 (backcourt heaves are the only excluded zone). This is the
            # check that would have caught the doubled corner-3 volume.
            zone_sum = (
                stats["b_r_Rim"]
                + stats["b_r_Paint"]
                + stats["b_r_Mid"]
                + stats["b_r_C3"]
                + stats["b_r_ATB3"]
            )
            if not zone_sum.between(0.97, 1.005).all():
                raise ValueError(
                    f"{season}: zone rates sum to {zone_sum.min():.3f}-{zone_sum.max():.3f}, "
                    "expected ~1.0 -- shot-location zones overlap or are missing."
                )

            for col in ["b_p_Rim", "b_p_Paint", "b_p_Mid", "b_p_C3", "b_p_ATB3"]:
                stats[col] = stats[col].fillna(0)

            all_stats.append(stats)
            print(f"  [OK] Profile Locked for Season: {season}")
            time.sleep(1)
        except Exception as e:
            # Swallowing this silently dropped an entire season from the inner
            # merge later with no indication anything was wrong.
            print(f"  [FAIL] Baseline for {season}: {type(e).__name__}: {e}")

    if not all_stats:
        raise RuntimeError("No season baselines could be fetched.")
    if len(all_stats) != len(seasons):
        print(
            f"  [WARN] Only {len(all_stats)}/{len(seasons)} season baselines fetched; "
            "games in the missing seasons will be dropped by the merge."
        )
    return pd.concat(all_stats)[["SEASON", "LEADER"] + BASELINE_COLUMNS]
