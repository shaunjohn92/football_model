"""
team_expectations.py – Football Model v1

Build team-level match stats and season baselines from local history.

Inputs (from previous steps):
    data/history/fixtures_epl_2025_26_sample.csv
    data/history/player_match_stats_epl_2025_26_sample_enriched.csv

Outputs:
    data/history/team_match_stats_epl_2025_26_sample.csv
    data/history/team_baselines_epl_2025_26_overall.csv

This module does NOT call the API – it only works on local CSVs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


# ----------------------------------------------------------------------
# Path helpers
# ----------------------------------------------------------------------


def get_history_dir() -> Path:
    """
    Locate the data/history directory relative to this file.
    """
    project_root = Path(__file__).resolve().parents[2]
    history_dir = project_root / "data" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir


def load_team_inputs() -> Dict[str, pd.DataFrame]:
    """
    Load the fixture and enriched player stats CSVs.
    """
    history_dir = get_history_dir()

    fixtures_path = history_dir / "fixtures_epl_2025_26_sample.csv"
    players_path = history_dir / "player_match_stats_epl_2025_26_sample_enriched.csv"

    fixtures_df = pd.read_csv(fixtures_path)
    players_df = pd.read_csv(players_path)

    return {
        "fixtures": fixtures_df,
        "player_match_stats": players_df,
    }


# ----------------------------------------------------------------------
# Team match stats per fixture
# ----------------------------------------------------------------------


def build_team_match_stats(
    fixtures_df: pd.DataFrame, players_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate player_match_stats into team-level stats per fixture.

    Result schema (for now):
        fixture_id
        league_id
        season
        date
        team_id
        opponent_team_id
        is_home
        goals_for
        goals_against
        minutes_total
        shots_total
        shots_on
        fouls_committed
        fouls_drawn
        yellow_cards
        red_cards
        duels_total
        duels_won
        tackles
        dribbles
    """
    players = players_df.copy()
    players["minutes"] = players["minutes"].fillna(0)

    # Aggregate from player → team per fixture
    group_cols = ["fixture_id", "team_id", "opponent_team_id"]
    agg_cols = {
        "minutes": "sum",
        "shots_total": "sum",
        "shots_on": "sum",
        "fouls_committed": "sum",
        "fouls_drawn": "sum",
        "yellow_cards": "sum",
        "red_cards": "sum",
        "duels_total": "sum",
        "duels_won": "sum",
        "tackles": "sum",
        "dribbles": "sum",
    }

    team_stats = players.groupby(group_cols, dropna=False).agg(agg_cols).reset_index()

    team_stats.rename(columns={"minutes": "minutes_total"}, inplace=True)

    # Bring in fixture context: league_id, season, date, home/away, goals
    fixtures_small = fixtures_df[
        [
            "fixture_id",
            "league_id",
            "season",
            "date",
            "home_team_id",
            "away_team_id",
            "home_goals",
            "away_goals",
        ]
    ].copy()

    merged = team_stats.merge(fixtures_small, on="fixture_id", how="left")

    # Compute is_home + goals_for/goals_against
    def compute_is_home(row):
        return row["team_id"] == row["home_team_id"]

    merged["is_home"] = merged.apply(compute_is_home, axis=1)

    def compute_goals_for(row):
        if row["is_home"]:
            return row["home_goals"]
        return row["away_goals"]

    def compute_goals_against(row):
        if row["is_home"]:
            return row["away_goals"]
        return row["home_goals"]

    merged["goals_for"] = merged.apply(compute_goals_for, axis=1)
    merged["goals_against"] = merged.apply(compute_goals_against, axis=1)

    # Drop helper columns we no longer need
    merged = merged.drop(
        columns=["home_team_id", "away_team_id", "home_goals", "away_goals"]
    )

    # Order columns nicely
    cols_order = [
        "fixture_id",
        "league_id",
        "season",
        "date",
        "team_id",
        "opponent_team_id",
        "is_home",
        "goals_for",
        "goals_against",
        "minutes_total",
        "shots_total",
        "shots_on",
        "fouls_committed",
        "fouls_drawn",
        "yellow_cards",
        "red_cards",
        "duels_total",
        "duels_won",
        "tackles",
        "dribbles",
    ]

    # Ensure all expected columns exist
    for col in cols_order:
        if col not in merged.columns:
            merged[col] = None

    merged = merged[cols_order].copy()

    return merged


# ----------------------------------------------------------------------
# Season baselines per team
# ----------------------------------------------------------------------


def build_team_baselines(team_match_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Build season-level team expectations from team_match_stats.

    For each team_id, compute:
        - matches
        - home_matches, away_matches
        - goals_for_per_match, goals_against_per_match
        - shots_total_per_match, shots_on_per_match
        - fouls_committed_per_match, fouls_drawn_per_match
        - yellow_cards_per_match, red_cards_per_match
        - tackles_per_match, dribbles_per_match
    """
    df = team_match_stats.copy()

    # Treat each row as one match for that team
    df["match"] = 1
    df["home_match"] = df["is_home"].astype(int)
    df["away_match"] = (~df["is_home"]).astype(int)

    group_cols = ["team_id"]
    agg_cols = {
        "match": "sum",
        "home_match": "sum",
        "away_match": "sum",
        "goals_for": "sum",
        "goals_against": "sum",
        "shots_total": "sum",
        "shots_on": "sum",
        "fouls_committed": "sum",
        "fouls_drawn": "sum",
        "yellow_cards": "sum",
        "red_cards": "sum",
        "tackles": "sum",
        "dribbles": "sum",
    }

    grouped = df.groupby(group_cols, dropna=False).agg(agg_cols).reset_index()
    grouped.rename(
        columns={
            "match": "matches",
            "home_match": "home_matches",
            "away_match": "away_matches",
        },
        inplace=True,
    )

    def safe_div(total, count):
        if count and count > 0:
            return float(total) / float(count)
        return 0.0

    # Per-match rates
    grouped["goals_for_per_match"] = grouped.apply(
        lambda row: safe_div(row["goals_for"], row["matches"]), axis=1
    )
    grouped["goals_against_per_match"] = grouped.apply(
        lambda row: safe_div(row["goals_against"], row["matches"]), axis=1
    )
    grouped["shots_total_per_match"] = grouped.apply(
        lambda row: safe_div(row["shots_total"], row["matches"]), axis=1
    )
    grouped["shots_on_per_match"] = grouped.apply(
        lambda row: safe_div(row["shots_on"], row["matches"]), axis=1
    )
    grouped["fouls_committed_per_match"] = grouped.apply(
        lambda row: safe_div(row["fouls_committed"], row["matches"]), axis=1
    )
    grouped["fouls_drawn_per_match"] = grouped.apply(
        lambda row: safe_div(row["fouls_drawn"], row["matches"]), axis=1
    )
    grouped["yellow_cards_per_match"] = grouped.apply(
        lambda row: safe_div(row["yellow_cards"], row["matches"]), axis=1
    )
    grouped["red_cards_per_match"] = grouped.apply(
        lambda row: safe_div(row["red_cards"], row["matches"]), axis=1
    )
    grouped["tackles_per_match"] = grouped.apply(
        lambda row: safe_div(row["tackles"], row["matches"]), axis=1
    )
    grouped["dribbles_per_match"] = grouped.apply(
        lambda row: safe_div(row["dribbles"], row["matches"]), axis=1
    )

    # Sort for readability
    result = grouped.sort_values("team_id").reset_index(drop=True)
    return result


# ----------------------------------------------------------------------
# Script entrypoint
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # 1) Load fixtures + enriched player stats
    data = load_team_inputs()
    fixtures_df = data["fixtures"]
    players_df = data["player_match_stats"]

    print("Loaded inputs:")
    print("  Fixtures:", fixtures_df.shape)
    print("  Player match stats (enriched):", players_df.shape)

    # 2) Build team match stats
    team_match_stats = build_team_match_stats(fixtures_df, players_df)
    print("Team match stats shape:", team_match_stats.shape)

    # 3) Build season baselines per team
    team_baselines = build_team_baselines(team_match_stats)
    print("Team baselines shape:", team_baselines.shape)

    # 4) Save outputs
    history_dir = get_history_dir()
    match_stats_path = history_dir / "team_match_stats_epl_2025_26_sample.csv"
    baselines_path = history_dir / "team_baselines_epl_2025_26_overall.csv"

    team_match_stats.to_csv(match_stats_path, index=False)
    team_baselines.to_csv(baselines_path, index=False)

    print("\nSaved team-level CSVs to:")
    print("  ", match_stats_path)
    print("  ", baselines_path)