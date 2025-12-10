"""
upcoming_match_projection.py – Football Model v1

Build player-level projections for upcoming EPL fixtures using
existing per-90 player baselines, team baselines, substitution profiles
and team style.

Inputs (all local CSVs, no live lineup / stats):
    data/history/fixtures_epl_2025_26_upcoming.csv
    data/history/team_baselines_epl_2025_26_overall.csv
    data/history/player_baselines_epl_2025_26_overall.csv
    data/history/substitution_profiles_epl_2025_26.csv
    data/history/team_style_epl_2025_26.csv

Output:
    data/history/match_projections_epl_2025_26_upcoming.csv

Notes:
    - We only project fixtures where we have player baselines for BOTH teams.
    - This is an approximate "pre-lineups" projection:
        * minutes from substitution profiles / avg minutes
        * shots from team baselines + player per-90
        * fouls/cards adjusted by team style
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


# ----------------------------------------------------------------------
# Path helpers
# ----------------------------------------------------------------------


def get_history_dir() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    history_dir = project_root / "data" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir


def load_upcoming_inputs() -> Dict[str, pd.DataFrame]:
    """
    Load all CSV inputs required for upcoming projections.
    """
    history_dir = get_history_dir()

    fixtures_path = history_dir / "fixtures_epl_2025_26_upcoming.csv"
    team_baselines_path = history_dir / "team_baselines_epl_2025_26_overall.csv"
    player_baselines_path = history_dir / "player_baselines_epl_2025_26_overall.csv"
    sub_profiles_path = history_dir / "substitution_profiles_epl_2025_26.csv"
    team_style_path = history_dir / "team_style_epl_2025_26.csv"

    fixtures_df = pd.read_csv(fixtures_path)
    team_baselines_df = pd.read_csv(team_baselines_path)
    player_baselines_df = pd.read_csv(player_baselines_path)
    sub_profiles_df = pd.read_csv(sub_profiles_path)
    team_style_df = pd.read_csv(team_style_path)

    return {
        "fixtures": fixtures_df,
        "team_baselines": team_baselines_df,
        "player_baselines": player_baselines_df,
        "sub_profiles": sub_profiles_df,
        "team_style": team_style_df,
    }


# ----------------------------------------------------------------------
# Core math helpers (mostly copied from match_projection.py)
# ----------------------------------------------------------------------


def _safe_scale(total_target: float, total_raw: float) -> float:
    if total_raw and total_raw > 0:
        return float(total_target) / float(total_raw)
    return 1.0


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _get_style_row(team_style: pd.DataFrame, team_id: int) -> pd.Series:
    row = team_style[team_style["team_id"] == team_id]
    if row.empty:
        return pd.Series(
            {
                "team_id": team_id,
                "foul_rate": 0.0,
                "card_rate": 0.0,
                "shots_allowed_rate": 0.0,
                "press_intensity": 0.0,
                "foul_z": 0.0,
                "card_z": 0.0,
                "shots_allowed_z": 0.0,
                "press_z": 0.0,
                "style_tag": "balanced",
            }
        )
    return row.iloc[0]


def _choose_expected_minutes_upcoming(row: pd.Series) -> float:
    """
    Expected minutes for upcoming matches (no fixture-specific minutes).

    Uses:
        - starts
        - sub_on_appearances
        - sub_off_appearances
        - avg_sub_on_minute
        - avg_sub_off_minute
        - avg_minutes
    """
    starts = row.get("starts", 0) or 0
    sub_on_apps = row.get("sub_on_appearances", 0) or 0
    sub_off_apps = row.get("sub_off_appearances", 0) or 0
    avg_sub_on = row.get("avg_sub_on_minute")
    avg_sub_off = row.get("avg_sub_off_minute")
    avg_min = row.get("avg_minutes")

    # Main starter
    if starts > 0 and sub_on_apps == 0:
        if pd.notna(avg_sub_off) and avg_sub_off > 0:
            return float(_clamp(avg_sub_off + 2.0, 60.0, 90.0))
        if pd.notna(avg_min) and avg_min > 0:
            return float(_clamp(avg_min, 75.0, 90.0))
        return 90.0

    # Pure sub
    if starts == 0 and sub_on_apps > 0:
        if pd.notna(avg_sub_on) and avg_sub_on > 0:
            return float(max(15.0, 90.0 - avg_sub_on))
        return 20.0

    # Mixed / fallback: use avg_minutes if present
    if pd.notna(avg_min) and avg_min > 0:
        return float(_clamp(avg_min, 30.0, 90.0))

    # Last resort
    return 30.0


# ----------------------------------------------------------------------
# Team-level projection for an upcoming fixture
# ----------------------------------------------------------------------


def _project_team_upcoming(
    fixture_row: pd.Series,
    team_id: int,
    player_baselines: pd.DataFrame,
    team_baselines: pd.DataFrame,
    sub_profiles: pd.DataFrame,
    team_style: pd.DataFrame,
) -> pd.DataFrame | None:
    """
    Build projections for all players of a single team
    in a single upcoming fixture.

    Unlike history-based projections, this does NOT use per-fixture
    player stats. It relies only on season baselines and substitution patterns.
    """
    fixture_id = fixture_row["fixture_id"]
    home_team_id = fixture_row["home_team_id"]
    away_team_id = fixture_row["away_team_id"]
    opponent_team_id = away_team_id if team_id == home_team_id else home_team_id

    # Team baseline (shots, fouls, etc.)
    team_base_row = team_baselines[team_baselines["team_id"] == team_id]
    if team_base_row.empty:
        print(
            f"[SKIP] No team baseline for team_id={team_id} in upcoming fixture "
            f"{fixture_id}. You probably need more history for this team."
        )
        return None
    team_base_row = team_base_row.iloc[0]
    team_expected_shots = team_base_row["shots_total_per_match"]

    # Player baselines for this team
    team_players = player_baselines[player_baselines["team_id"] == team_id].copy()
    if team_players.empty:
        print(
            f"[SKIP] No player baselines for team_id={team_id} in upcoming fixture "
            f"{fixture_id}. You probably need more history for this team."
        )
        return None

    # Join substitution profiles
    sub_small = sub_profiles[
        [
            "player_id",
            "team_id",
            "matches_played",
            "starts",
            "sub_on_appearances",
            "sub_off_appearances",
            "avg_sub_on_minute",
            "avg_sub_off_minute",
        ]
    ].copy()

    merged = team_players.merge(
        sub_small,
        on=["player_id", "team_id"],
        how="left",
        suffixes=("", "_sub"),
    )

    # Style info
    own_style = _get_style_row(team_style, team_id)
    opp_style = _get_style_row(team_style, opponent_team_id)

    own_foul_z = float(own_style.get("foul_z", 0.0))
    own_card_z = float(own_style.get("card_z", 0.0))
    opp_foul_z = float(opp_style.get("foul_z", 0.0))
    opp_card_z = float(opp_style.get("card_z", 0.0))

    # Expected minutes
    merged["expected_minutes"] = merged.apply(
        _choose_expected_minutes_upcoming, axis=1
    )
    merged["minutes_ratio"] = merged["expected_minutes"] / 90.0

    # Helper for raw stats from per-90 columns
    def per_player_raw(col_per90: str) -> pd.Series:
        if col_per90 not in merged.columns:
            return pd.Series([0.0] * len(merged), index=merged.index)
        return merged[col_per90].fillna(0.0) * merged["minutes_ratio"]

    merged["raw_shots_total"] = per_player_raw("shots_total_per90")
    merged["raw_shots_on"] = per_player_raw("shots_on_per90")
    merged["raw_fouls_committed"] = per_player_raw("fouls_committed_per90")
    merged["raw_fouls_drawn"] = per_player_raw("fouls_drawn_per90")
    merged["raw_yellow_cards"] = per_player_raw("yellow_cards_per90")

    # Style-based adjustments
    fouls_drawn_factor = _clamp(1.0 + 0.15 * opp_foul_z, 0.7, 1.3)
    fouls_comm_factor = _clamp(1.0 + 0.15 * own_foul_z, 0.7, 1.3)
    card_index = 0.5 * (own_card_z + opp_card_z)
    cards_factor = _clamp(1.0 + 0.2 * card_index, 0.7, 1.4)

    merged["raw_fouls_committed"] *= fouls_comm_factor
    merged["raw_fouls_drawn"] *= fouls_drawn_factor
    merged["raw_yellow_cards"] *= cards_factor

    # Scale shots to team expectation
    total_raw_shots = merged["raw_shots_total"].sum()
    shots_scale = _safe_scale(team_expected_shots, total_raw_shots)

    merged["proj_shots_total"] = merged["raw_shots_total"] * shots_scale
    merged["proj_shots_on"] = merged["raw_shots_on"] * shots_scale
    merged["proj_fouls_committed"] = merged["raw_fouls_committed"]
    merged["proj_fouls_drawn"] = merged["raw_fouls_drawn"]
    merged["proj_yellow_cards"] = merged["raw_yellow_cards"]

    merged["fixture_id"] = fixture_id
    merged["opponent_team_id"] = opponent_team_id
    merged["is_home"] = team_id == home_team_id
    merged["team_expected_shots_total"] = team_expected_shots

    proj_cols = [
        "fixture_id",
        "team_id",
        "opponent_team_id",
        "is_home",
        "player_id",
        "player_name",
        "expected_minutes",
        "proj_shots_total",
        "proj_shots_on",
        "proj_fouls_committed",
        "proj_fouls_drawn",
        "proj_yellow_cards",
    ]
    for col in proj_cols:
        if col not in merged.columns:
            merged[col] = None

    return merged[proj_cols].copy()


def project_upcoming_fixtures(
    fixtures_df: pd.DataFrame,
    team_baselines: pd.DataFrame,
    player_baselines: pd.DataFrame,
    sub_profiles: pd.DataFrame,
    team_style: pd.DataFrame,
) -> pd.DataFrame:
    """
    Project all upcoming fixtures where we have enough data.
    """
    all_rows: List[pd.DataFrame] = []

    for _, fx in fixtures_df.iterrows():
        fixture_id = fx["fixture_id"]
        home_team_id = fx["home_team_id"]
        away_team_id = fx["away_team_id"]

        print(
            f"Projecting upcoming fixture_id={fixture_id}, "
            f"home_team_id={home_team_id}, away_team_id={away_team_id}"
        )

        home_proj = _project_team_upcoming(
            fixture_row=fx,
            team_id=home_team_id,
            player_baselines=player_baselines,
            team_baselines=team_baselines,
            sub_profiles=sub_profiles,
            team_style=team_style,
        )

        away_proj = _project_team_upcoming(
            fixture_row=fx,
            team_id=away_team_id,
            player_baselines=player_baselines,
            team_baselines=team_baselines,
            sub_profiles=sub_profiles,
            team_style=team_style,
        )

        if home_proj is None or away_proj is None:
            print(
                f"[SKIP FIXTURE] fixture_id={fixture_id} – missing data "
                f"for home or away team."
            )
            continue

        all_rows.append(home_proj)
        all_rows.append(away_proj)

    if not all_rows:
        return pd.DataFrame(
            columns=[
                "fixture_id",
                "team_id",
                "opponent_team_id",
                "is_home",
                "player_id",
                "player_name",
                "expected_minutes",
                "proj_shots_total",
                "proj_shots_on",
                "proj_fouls_committed",
                "proj_fouls_drawn",
                "proj_yellow_cards",
            ]
        )

    return pd.concat(all_rows, ignore_index=True)


# ----------------------------------------------------------------------
# Script entrypoint
# ----------------------------------------------------------------------


if __name__ == "__main__":
    data = load_upcoming_inputs()
    fixtures_df = data["fixtures"]
    team_baselines_df = data["team_baselines"]
    player_baselines_df = data["player_baselines"]
    sub_profiles_df = data["sub_profiles"]
    team_style_df = data["team_style"]

    print("Loaded upcoming inputs:")
    print("  Fixtures:", fixtures_df.shape)
    print("  Team baselines:", team_baselines_df.shape)
    print("  Player baselines:", player_baselines_df.shape)
    print("  Substitution profiles:", sub_profiles_df.shape)
    print("  Team style:", team_style_df.shape)

    upcoming_proj_df = project_upcoming_fixtures(
        fixtures_df=fixtures_df,
        team_baselines=team_baselines_df,
        player_baselines=player_baselines_df,
        sub_profiles=sub_profiles_df,
        team_style=team_style_df,
    )

    print("\nUpcoming match projections shape:", upcoming_proj_df.shape)
    if upcoming_proj_df.empty:
        print("No upcoming fixtures could be projected (likely missing baselines).")
    else:
        print("Sample rows:")
        print(upcoming_proj_df.head().to_string(index=False))

    history_dir = get_history_dir()
    out_path = history_dir / "match_projections_epl_2025_26_upcoming.csv"
    upcoming_proj_df.to_csv(out_path, index=False)

    print("\nSaved upcoming match projections to:")
    print("  ", out_path)