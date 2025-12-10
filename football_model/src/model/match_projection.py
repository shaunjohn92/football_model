"""
match_projection.py – Football Model v1

Match projection module (v3 – with substitution-aware minutes + team style).

Inputs (all local CSVs, no API calls):

    data/history/fixtures_epl_2025_26_sample.csv
    data/history/team_baselines_epl_2025_26_overall.csv
    data/history/player_baselines_epl_2025_26_overall.csv
    data/history/player_match_stats_epl_2025_26_sample_enriched.csv
    data/history/substitution_profiles_epl_2025_26.csv
    data/history/team_style_epl_2025_26.csv

Outputs:

    data/history/match_projections_epl_2025_26_sample.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


# ----------------------------------------------------------------------
# Path + input loaders
# ----------------------------------------------------------------------


def get_history_dir() -> Path:
    """
    Locate the data/history directory relative to this file.
    """
    project_root = Path(__file__).resolve().parents[2]
    history_dir = project_root / "data" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir


def load_projection_inputs() -> Dict[str, pd.DataFrame]:
    """
    Load all the CSVs we need for projections.
    """
    history_dir = get_history_dir()

    fixtures_path = history_dir / "fixtures_epl_2025_26_sample.csv"
    team_baselines_path = history_dir / "team_baselines_epl_2025_26_overall.csv"
    player_baselines_path = history_dir / "player_baselines_epl_2025_26_overall.csv"
    players_enriched_path = (
        history_dir / "player_match_stats_epl_2025_26_sample_enriched.csv"
    )
    sub_profiles_path = history_dir / "substitution_profiles_epl_2025_26.csv"
    team_style_path = history_dir / "team_style_epl_2025_26.csv"

    fixtures_df = pd.read_csv(fixtures_path)
    team_baselines_df = pd.read_csv(team_baselines_path)
    player_baselines_df = pd.read_csv(player_baselines_path)
    players_enriched_df = pd.read_csv(players_enriched_path)
    sub_profiles_df = pd.read_csv(sub_profiles_path)
    team_style_df = pd.read_csv(team_style_path)

    return {
        "fixtures": fixtures_df,
        "team_baselines": team_baselines_df,
        "player_baselines": player_baselines_df,
        "players_enriched": players_enriched_df,
        "sub_profiles": sub_profiles_df,
        "team_style": team_style_df,
    }


# ----------------------------------------------------------------------
# Core projection logic
# ----------------------------------------------------------------------


def _safe_scale(total_target: float, total_raw: float) -> float:
    """
    Return a multiplicative factor so that:
        raw * factor ≈ target

    If total_raw is 0, we return 1.0 (no scaling).
    """
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
    """
    Safely fetch a style row for a team_id, with sensible defaults if missing.
    """
    row = team_style[team_style["team_id"] == team_id]
    if row.empty:
        # Build a dummy row with zeros / balanced
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


def _project_team_for_fixture(
    fixture_row: pd.Series,
    team_id: int,
    players_enriched: pd.DataFrame,
    team_baselines: pd.DataFrame,
    player_baselines: pd.DataFrame,
    sub_profiles: pd.DataFrame,
    team_style: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build projections for all players of a single team in a single fixture.

    Uses:
        - team_baselines: expected team shots
        - player_baselines: per-90 stats
        - sub_profiles: expected minutes
        - team_style: fouls/cards style to adjust foul & card props
    """
    fixture_id = fixture_row["fixture_id"]
    home_team_id = fixture_row["home_team_id"]
    away_team_id = fixture_row["away_team_id"]
    opponent_team_id = away_team_id if team_id == home_team_id else home_team_id

    # 1) Team baseline row
    team_base_row = team_baselines[team_baselines["team_id"] == team_id]
    if team_base_row.empty:
        raise ValueError(f"No team baseline found for team_id={team_id}")

    team_base_row = team_base_row.iloc[0]
    team_expected_shots = team_base_row["shots_total_per_match"]

    # 2) Players for this fixture + team
    team_players = players_enriched[
        (players_enriched["fixture_id"] == fixture_id)
        & (players_enriched["team_id"] == team_id)
    ].copy()

    if team_players.empty:
        raise ValueError(
            f"No player records found for team_id={team_id} in fixture_id={fixture_id}"
        )

    # 3a) Join in player baselines (per-90) on (player_id, team_id, player_name)
    player_base_small = player_baselines[
        [
            "player_id",
            "team_id",
            "player_name",
            "minutes_total",
            "games",
            "avg_minutes",
            "shots_total_per90",
            "shots_on_per90",
            "fouls_committed_per90",
            "fouls_drawn_per90",
            "yellow_cards_per90",
        ]
    ].copy()

    merged = team_players.merge(
        player_base_small,
        on=["player_id", "team_id", "player_name"],
        how="left",
        suffixes=("", "_base"),
    )

    # 3b) Join in substitution profiles (how they are usually used)
    sub_profiles_small = sub_profiles[
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

    merged = merged.merge(
        sub_profiles_small,
        on=["player_id", "team_id"],
        how="left",
        suffixes=("", "_sub"),
    )

    # 3c) Fetch style rows for own team and opponent
    own_style = _get_style_row(team_style, team_id)
    opp_style = _get_style_row(team_style, opponent_team_id)

    own_foul_z = float(own_style.get("foul_z", 0.0))
    own_card_z = float(own_style.get("card_z", 0.0))
    opp_foul_z = float(opp_style.get("foul_z", 0.0))
    opp_card_z = float(opp_style.get("card_z", 0.0))

    # 4) Choose expected minutes
    def choose_expected_minutes(row: pd.Series) -> float:
        """
        Heuristic rules:

        - If he has starts and rarely comes on as a sub:
            -> treat him as a main starter.
        - If he only comes on as a sub (no starts):
            -> use 90 - avg_sub_on_minute (a typical sub stint).
        - If mixed profile or missing sub data:
            -> fall back to avg_minutes, then this fixture's minutes.
        """
        starts = row.get("starts", 0) or 0
        sub_on_apps = row.get("sub_on_appearances", 0) or 0
        sub_off_apps = row.get("sub_off_appearances", 0) or 0
        avg_sub_on = row.get("avg_sub_on_minute")
        avg_sub_off = row.get("avg_sub_off_minute")
        avg_min = row.get("avg_minutes")
        fixture_minutes = row.get("minutes")

        # Main starter: has starts, rarely comes on from the bench
        if starts > 0 and sub_on_apps == 0:
            # If he often gets subbed off, use that as a guide
            if pd.notna(avg_sub_off) and avg_sub_off > 0:
                # e.g. subbed around 70' -> expect ~72 minutes
                return float(_clamp(avg_sub_off + 2.0, 60.0, 90.0))
            # Otherwise assume close to full 90, but at least 75
            if pd.notna(avg_min) and avg_min > 0:
                return float(_clamp(avg_min, 75.0, 90.0))
            return 90.0

        # Pure sub: no starts but does come on as a sub
        if starts == 0 and sub_on_apps > 0:
            if pd.notna(avg_sub_on) and avg_sub_on > 0:
                # e.g. average sub on at 70' -> 20 minutes
                return float(max(15.0, 90.0 - avg_sub_on))
            # If we don't know the minute, assume ~20 mins
            return 20.0

        # Mixed usage or missing sub data – use avg_minutes if available
        if pd.notna(avg_min) and avg_min > 0:
            return float(_clamp(avg_min, 30.0, 90.0))

        # Fallback to this fixture's minutes if we have them
        if pd.notna(fixture_minutes) and fixture_minutes > 0:
            return float(fixture_minutes)

        # Last resort: assume 30 minutes
        return 30.0

    merged["expected_minutes"] = merged.apply(choose_expected_minutes, axis=1)
    merged["minutes_ratio"] = merged["expected_minutes"] / 90.0

    # 5) Compute raw per-player stats from per-90 baselines
    def per_player_raw(col_per90: str) -> pd.Series:
        if col_per90 not in merged.columns:
            return pd.Series([0.0] * len(merged), index=merged.index)
        return merged[col_per90].fillna(0.0) * merged["minutes_ratio"]

    merged["raw_shots_total"] = per_player_raw("shots_total_per90")
    merged["raw_shots_on"] = per_player_raw("shots_on_per90")
    merged["raw_fouls_committed"] = per_player_raw("fouls_committed_per90")
    merged["raw_fouls_drawn"] = per_player_raw("fouls_drawn_per90")
    merged["raw_yellow_cards"] = per_player_raw("yellow_cards_per90")

    # 6) Style-based adjustments for fouls/cards
    # Opponent fouliness boosts fouls drawn for attackers
    fouls_drawn_factor = _clamp(1.0 + 0.15 * opp_foul_z, 0.7, 1.3)
    # Own fouliness boosts fouls committed
    fouls_comm_factor = _clamp(1.0 + 0.15 * own_foul_z, 0.7, 1.3)
    # Cards – combine both sides' cardiness
    card_index = 0.5 * (own_card_z + opp_card_z)
    cards_factor = _clamp(1.0 + 0.2 * card_index, 0.7, 1.4)

    merged["raw_fouls_committed"] *= fouls_comm_factor
    merged["raw_fouls_drawn"] *= fouls_drawn_factor
    merged["raw_yellow_cards"] *= cards_factor

    # 7) Scale shots so that team total matches team_expected_shots
    total_raw_shots = merged["raw_shots_total"].sum()
    shots_scale = _safe_scale(team_expected_shots, total_raw_shots)

    merged["proj_shots_total"] = merged["raw_shots_total"] * shots_scale
    merged["proj_shots_on"] = merged["raw_shots_on"] * shots_scale

    # For fouls/cards we now use style-adjusted raw values
    merged["proj_fouls_committed"] = merged["raw_fouls_committed"]
    merged["proj_fouls_drawn"] = merged["raw_fouls_drawn"]
    merged["proj_yellow_cards"] = merged["raw_yellow_cards"]

    # Add some context columns from fixture + team baseline
    merged["fixture_id"] = fixture_id
    merged["opponent_team_id"] = opponent_team_id
    merged["is_home"] = team_players["team_id"].iloc[0] == home_team_id
    merged["team_expected_shots_total"] = team_expected_shots

    # Select only the columns we care about for projections
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

    # Ensure all columns exist
    for col in proj_cols:
        if col not in merged.columns:
            merged[col] = None

    projections = merged[proj_cols].copy()

    return projections


def project_all_fixtures(
    fixtures_df: pd.DataFrame,
    team_baselines: pd.DataFrame,
    player_baselines: pd.DataFrame,
    players_enriched: pd.DataFrame,
    sub_profiles: pd.DataFrame,
    team_style: pd.DataFrame,
) -> pd.DataFrame:
    """
    Project ALL fixtures in fixtures_df.
    """
    if fixtures_df.empty:
        raise ValueError("No fixtures available to project.")

    all_projections = []

    for _, fixture_row in fixtures_df.iterrows():
        fixture_id = fixture_row["fixture_id"]
        home_team_id = fixture_row["home_team_id"]
        away_team_id = fixture_row["away_team_id"]

        print(
            f"Projecting fixture_id={fixture_id}, "
            f"home_team_id={home_team_id}, away_team_id={away_team_id}"
        )

        # Project for home team
        home_proj = _project_team_for_fixture(
            fixture_row=fixture_row,
            team_id=home_team_id,
            players_enriched=players_enriched,
            team_baselines=team_baselines,
            player_baselines=player_baselines,
            sub_profiles=sub_profiles,
            team_style=team_style,
        )

        # Project for away team
        away_proj = _project_team_for_fixture(
            fixture_row=fixture_row,
            team_id=away_team_id,
            players_enriched=players_enriched,
            team_baselines=team_baselines,
            player_baselines=player_baselines,
            sub_profiles=sub_profiles,
            team_style=team_style,
        )

        fixture_proj = pd.concat([home_proj, away_proj], ignore_index=True)
        all_projections.append(fixture_proj)

    result = pd.concat(all_projections, ignore_index=True)
    return result


# ----------------------------------------------------------------------
# Script entrypoint
# ----------------------------------------------------------------------


if __name__ == "__main__":
    data = load_projection_inputs()
    fixtures_df = data["fixtures"]
    team_baselines_df = data["team_baselines"]
    player_baselines_df = data["player_baselines"]
    players_enriched_df = data["players_enriched"]
    sub_profiles_df = data["sub_profiles"]
    team_style_df = data["team_style"]

    print("Loaded inputs for projections:")
    print("  Fixtures:", fixtures_df.shape)
    print("  Team baselines:", team_baselines_df.shape)
    print("  Player baselines:", player_baselines_df.shape)
    print("  Enriched player stats:", players_enriched_df.shape)
    print("  Substitution profiles:", sub_profiles_df.shape)
    print("  Team style:", team_style_df.shape)

    projections_df = project_all_fixtures(
        fixtures_df=fixtures_df,
        team_baselines=team_baselines_df,
        player_baselines=player_baselines_df,
        players_enriched=players_enriched_df,
        sub_profiles=sub_profiles_df,
        team_style=team_style_df,
    )

    print("\nMatch projections shape:", projections_df.shape)
    print("First few rows:")
    print(projections_df.head(10))

    history_dir = get_history_dir()
    proj_path = history_dir / "match_projections_epl_2025_26_sample.csv"
    projections_df.to_csv(proj_path, index=False)

    print("\nSaved match projections to:")
    print("  ", proj_path)