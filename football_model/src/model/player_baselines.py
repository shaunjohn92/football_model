"""
player_baselines.py – Football Model v1

Build per-90 baselines for players from the enriched history CSVs.

Input:
    data/history/player_match_stats_epl_2025_26_sample_enriched.csv

Outputs:
    data/history/player_baselines_epl_2025_26_overall.csv
    data/history/player_baselines_epl_2025_26_by_role.csv

This is a pure offline step – no API calls.
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


def load_enriched_player_stats() -> pd.DataFrame:
    """
    Load the enriched player_match_stats CSV produced by history_joiner.py.
    """
    history_dir = get_history_dir()
    path = history_dir / "player_match_stats_epl_2025_26_sample_enriched.csv"
    df = pd.read_csv(path)
    return df


# ----------------------------------------------------------------------
# Core per-90 baseline logic
# ----------------------------------------------------------------------


def _safe_per90(total: float, minutes: float) -> float:
    """
    Compute per-90 rate safely. Returns 0.0 if minutes == 0.
    """
    if minutes and minutes > 0:
        return 90.0 * float(total) / float(minutes)
    return 0.0


def build_player_baselines_overall(players: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per player+team (overall) and compute:

        - games played
        - starts
        - sub appearances
        - total minutes
        - average minutes per game
        - per-90 stats for key metrics
        - "typical" role/side/band (most frequent value)
    """
    df = players.copy()

    # Treat any row with minutes > 0 as an appearance
    df["minutes"] = df["minutes"].fillna(0)
    df["appearance"] = df["minutes"] > 0

    # A simple heuristic for "started": appeared and not subbed_on
    df["started"] = df["appearance"] & (~df["subbed_on"].fillna(False))
    df["came_on"] = df["appearance"] & df["subbed_on"].fillna(False)

    # Group by player + team (and keep name for readability)
    group_cols = ["player_id", "team_id"]
    agg_cols = {
        "player_name": "first",
        "minutes": "sum",
        "appearance": "sum",
        "started": "sum",
        "came_on": "sum",
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

    grouped = df.groupby(group_cols, dropna=False).agg(agg_cols).reset_index()

    grouped.rename(
        columns={
            "minutes": "minutes_total",
            "appearance": "games",
            "started": "starts",
            "came_on": "sub_apps",
        },
        inplace=True,
    )

    # Average minutes per appearance
    grouped["avg_minutes"] = grouped.apply(
        lambda row: (row["minutes_total"] / row["games"])
        if row["games"] and row["games"] > 0
        else 0.0,
        axis=1,
    )

    # Compute per-90 stats
    grouped["shots_total_per90"] = grouped.apply(
        lambda row: _safe_per90(row["shots_total"], row["minutes_total"]), axis=1
    )
    grouped["shots_on_per90"] = grouped.apply(
        lambda row: _safe_per90(row["shots_on"], row["minutes_total"]), axis=1
    )
    grouped["fouls_committed_per90"] = grouped.apply(
        lambda row: _safe_per90(row["fouls_committed"], row["minutes_total"]), axis=1
    )
    grouped["fouls_drawn_per90"] = grouped.apply(
        lambda row: _safe_per90(row["fouls_drawn"], row["minutes_total"]), axis=1
    )
    grouped["yellow_cards_per90"] = grouped.apply(
        lambda row: _safe_per90(row["yellow_cards"], row["minutes_total"]), axis=1
    )
    grouped["red_cards_per90"] = grouped.apply(
        lambda row: _safe_per90(row["red_cards"], row["minutes_total"]), axis=1
    )
    grouped["duels_total_per90"] = grouped.apply(
        lambda row: _safe_per90(row["duels_total"], row["minutes_total"]), axis=1
    )
    grouped["duels_won_per90"] = grouped.apply(
        lambda row: _safe_per90(row["duels_won"], row["minutes_total"]), axis=1
    )
    grouped["tackles_per90"] = grouped.apply(
        lambda row: _safe_per90(row["tackles"], row["minutes_total"]), axis=1
    )
    grouped["dribbles_per90"] = grouped.apply(
        lambda row: _safe_per90(row["dribbles"], row["minutes_total"]), axis=1
    )

    # Typical role/side/band = most frequent value across appearances
    def most_frequent(series: pd.Series):
        if series.empty:
            return None
        mode = series.mode()
        if len(mode) == 0:
            return None
        return mode.iloc[0]

    # Compute typical role/side/band separately and merge in
    roles = (
        df.groupby(group_cols)
        .agg(
            typical_role=("role", most_frequent),
            typical_side=("side", most_frequent),
            typical_band=("band", most_frequent),
        )
        .reset_index()
    )

    result = grouped.merge(roles, on=group_cols, how="left")

    # Sort for readability: by team and then player name
    result = result.sort_values(["team_id", "player_name"]).reset_index(drop=True)

    return result


def build_player_baselines_by_role(players: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per player+team+role, so we can see how a player behaves
    when used specifically as CB vs FB vs W vs ST, etc.
    """
    df = players.copy()
    df["minutes"] = df["minutes"].fillna(0)
    df["appearance"] = df["minutes"] > 0

    group_cols = ["player_id", "team_id", "role"]
    agg_cols = {
        "player_name": "first",
        "minutes": "sum",
        "appearance": "sum",
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

    grouped = df.groupby(group_cols, dropna=False).agg(agg_cols).reset_index()

    grouped.rename(
        columns={
            "minutes": "minutes_total",
            "appearance": "games",
        },
        inplace=True,
    )

    grouped["avg_minutes"] = grouped.apply(
        lambda row: (row["minutes_total"] / row["games"])
        if row["games"] and row["games"] > 0
        else 0.0,
        axis=1,
    )

    # Per-90 stats (same metrics as overall)
    grouped["shots_total_per90"] = grouped.apply(
        lambda row: _safe_per90(row["shots_total"], row["minutes_total"]), axis=1
    )
    grouped["shots_on_per90"] = grouped.apply(
        lambda row: _safe_per90(row["shots_on"], row["minutes_total"]), axis=1
    )
    grouped["fouls_committed_per90"] = grouped.apply(
        lambda row: _safe_per90(row["fouls_committed"], row["minutes_total"]), axis=1
    )
    grouped["fouls_drawn_per90"] = grouped.apply(
        lambda row: _safe_per90(row["fouls_drawn"], row["minutes_total"]), axis=1
    )
    grouped["yellow_cards_per90"] = grouped.apply(
        lambda row: _safe_per90(row["yellow_cards"], row["minutes_total"]), axis=1
    )
    grouped["red_cards_per90"] = grouped.apply(
        lambda row: _safe_per90(row["red_cards"], row["minutes_total"]), axis=1
    )
    grouped["duels_total_per90"] = grouped.apply(
        lambda row: _safe_per90(row["duels_total"], row["minutes_total"]), axis=1
    )
    grouped["duels_won_per90"] = grouped.apply(
        lambda row: _safe_per90(row["duels_won"], row["minutes_total"]), axis=1
    )
    grouped["tackles_per90"] = grouped.apply(
        lambda row: _safe_per90(row["tackles"], row["minutes_total"]), axis=1
    )
    grouped["dribbles_per90"] = grouped.apply(
        lambda row: _safe_per90(row["dribbles"], row["minutes_total"]), axis=1
    )

    result = grouped.sort_values(["team_id", "player_name", "role"]).reset_index(
        drop=True
    )
    return result


# ----------------------------------------------------------------------
# Script entrypoint
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # 1) Load enriched player match stats
    players = load_enriched_player_stats()
    print("Loaded enriched player stats:", players.shape)

    # 2) Build overall per-player baselines
    baselines_overall = build_player_baselines_overall(players)
    print("Overall baselines shape:", baselines_overall.shape)

    # 3) Build per-role baselines
    baselines_by_role = build_player_baselines_by_role(players)
    print("By-role baselines shape:", baselines_by_role.shape)

    # 4) Save to CSV
    history_dir = get_history_dir()
    overall_path = history_dir / "player_baselines_epl_2025_26_overall.csv"
    by_role_path = history_dir / "player_baselines_epl_2025_26_by_role.csv"

    baselines_overall.to_csv(overall_path, index=False)
    baselines_by_role.to_csv(by_role_path, index=False)

    print("\nSaved baselines to:")
    print("  ", overall_path)
    print("  ", by_role_path)