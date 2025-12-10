"""
substitution_patterns.py – Football Model v1

Learn substitution dynamics from local history:

Inputs:
    data/history/subs_epl_2025_26_sample_enriched.csv
    data/history/player_match_stats_epl_2025_26_sample_enriched.csv

Outputs:
    data/history/substitution_pairs_epl_2025_26.csv
        - one row per (team, player_out, player_in) combination
        - frequency and average minute of the substitution
        - roles for out/in players

    data/history/substitution_profiles_epl_2025_26.csv
        - one row per player per team
        - how often they start / come on as sub / go off
        - average minutes when subbed on / subbed off
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


# ----------------------------------------------------------------------
# Path helpers and loaders
# ----------------------------------------------------------------------


def get_history_dir() -> Path:
    """
    Locate the data/history directory relative to this file.
    """
    project_root = Path(__file__).resolve().parents[2]
    history_dir = project_root / "data" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir


def load_substitution_inputs() -> Dict[str, pd.DataFrame]:
    """
    Load enriched subs + player stats.
    """
    history_dir = get_history_dir()

    subs_path = history_dir / "subs_epl_2025_26_sample_enriched.csv"
    players_path = history_dir / "player_match_stats_epl_2025_26_sample_enriched.csv"

    subs_df = pd.read_csv(subs_path)
    players_df = pd.read_csv(players_path)

    return {"subs": subs_df, "players": players_df}


# ----------------------------------------------------------------------
# 1) Substitution pairs: who replaces who, and when
# ----------------------------------------------------------------------


def build_substitution_pairs(subs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a table describing (player_out -> player_in) patterns.

    Output columns:
        team_id
        player_out_id
        player_in_id
        times
        avg_minute
        min_minute
        max_minute
        role_out
        role_in
    """
    df = subs_df.copy()

    # Some subs might have missing minute – fill with 0 just so aggregation works
    df["minute"] = df["minute"].fillna(0)

    # Make sure we have role_out / role_in columns, even if they're missing
    if "role_out" not in df.columns:
        df["role_out"] = None
    if "role_in" not in df.columns:
        df["role_in"] = None

    group_cols = ["team_id", "player_out_id", "player_in_id"]

    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            times=("minute", "count"),
            avg_minute=("minute", "mean"),
            min_minute=("minute", "min"),
            max_minute=("minute", "max"),
            # For roles, we just take the most common (mode) if present
            role_out=(
                "role_out",
                lambda s: s.mode().iloc[0] if not s.mode().empty else None,
            ),
            role_in=(
                "role_in",
                lambda s: s.mode().iloc[0] if not s.mode().empty else None,
            ),
        )
        .reset_index()
    )

    # Sort for readability: frequent patterns first
    agg = agg.sort_values(
        ["times", "team_id", "avg_minute"], ascending=[False, True, True]
    ).reset_index(drop=True)

    return agg

    # Sort for readability: frequent patterns first
    agg = agg.sort_values(
        ["times", "team_id", "avg_minute"], ascending=[False, True, True]
    ).reset_index(drop=True)

    return agg


# ----------------------------------------------------------------------
# 2) Substitution profiles per player
# ----------------------------------------------------------------------


def build_substitution_profiles(
    subs_df: pd.DataFrame, players_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a per-player substitution profile.

    For each (player_id, team_id) we compute:
        - matches_played
        - starts
        - sub_on_appearances
        - sub_off_appearances
        - avg_sub_on_minute
        - avg_sub_off_minute
        - typical_role (mode)
    """
    players = players_df.copy()
    subs = subs_df.copy()

    # Ensure booleans exist & fill
    players["minutes"] = players["minutes"].fillna(0)
    players["appearance"] = players["minutes"] > 0
    players["subbed_on"] = players["subbed_on"].fillna(False)
    players["subbed_off"] = players["subbed_off"].fillna(False)

    # Basic counts per player+team
    group_cols = ["player_id", "team_id"]
    base = (
        players.groupby(group_cols, dropna=False)
        .agg(
            player_name=("player_name", "first"),
            matches_played=("appearance", "sum"),
            starts=("subbed_on", lambda s: int((~s).sum())),  # rough: appearances where not subbed_on
            sub_on_appearances=("subbed_on", "sum"),
            sub_off_appearances=("subbed_off", "sum"),
        )
        .reset_index()
    )

    # Average sub-on / sub-off minutes from subs table
    subs["minute"] = subs["minute"].fillna(0)

    # For player coming ON
    sub_on_agg = (
        subs.groupby(["team_id", "player_in_id"], dropna=False)
        .agg(avg_sub_on_minute=("minute", "mean"))
        .reset_index()
        .rename(columns={"player_in_id": "player_id"})
    )

    # For player going OFF
    sub_off_agg = (
        subs.groupby(["team_id", "player_out_id"], dropna=False)
        .agg(avg_sub_off_minute=("minute", "mean"))
        .reset_index()
        .rename(columns={"player_out_id": "player_id"})
    )

    profiles = base.merge(
        sub_on_agg, on=["player_id", "team_id"], how="left"
    ).merge(sub_off_agg, on=["player_id", "team_id"], how="left")

    # Typical role from players_df
    def most_frequent(series: pd.Series):
        if series.empty:
            return None
        mode = series.mode()
        if len(mode) == 0:
            return None
        return mode.iloc[0]

    roles = (
        players.groupby(group_cols, dropna=False)
        .agg(typical_role=("role", most_frequent))
        .reset_index()
    )

    profiles = profiles.merge(roles, on=["player_id", "team_id"], how="left")

    # Sort nicely
    profiles = profiles.sort_values(
        ["team_id", "player_name"]
    ).reset_index(drop=True)

    return profiles


# ----------------------------------------------------------------------
# Script entrypoint
# ----------------------------------------------------------------------


if __name__ == "__main__":
    data = load_substitution_inputs()
    subs_df = data["subs"]
    players_df = data["players"]

    print("Loaded substitution inputs:")
    print("  Subs enriched:", subs_df.shape)
    print("  Player stats enriched:", players_df.shape)

    sub_pairs = build_substitution_pairs(subs_df)
    print("Substitution pairs shape:", sub_pairs.shape)

    sub_profiles = build_substitution_profiles(subs_df, players_df)
    print("Substitution profiles shape:", sub_profiles.shape)

    history_dir = get_history_dir()
    pairs_path = history_dir / "substitution_pairs_epl_2025_26.csv"
    profiles_path = history_dir / "substitution_profiles_epl_2025_26.csv"

    sub_pairs.to_csv(pairs_path, index=False)
    sub_profiles.to_csv(profiles_path, index=False)

    print("\nSaved substitution patterns to:")
    print("  ", pairs_path)
    print("  ", profiles_path)