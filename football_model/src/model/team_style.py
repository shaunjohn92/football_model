"""
team_style.py – Football Model v1

Derive team style metrics from local history.

Input:
    data/history/team_match_stats_epl_2025_26_sample.csv

Output:
    data/history/team_style_epl_2025_26.csv

For each team we compute:

    - matches
    - goals_for_per_match
    - goals_against_per_match
    - shots_for_per_match
    - shots_allowed_per_match
    - fouls_committed_per_match
    - fouls_drawn_per_match
    - yellow_cards_per_match
    - red_cards_per_match
    - tackles_per_match
    - dribbles_per_match

Then we build simple style indices:

    - foul_rate = fouls_committed_per_match
    - card_rate = yellow_cards_per_match + 2 * red_cards_per_match
    - shots_allowed_rate = shots_allowed_per_match
    - press_intensity (proxy) = tackles + 0.5 * dribbles + 0.25 * fouls_committed

We also compute z-scores vs league average and assign style_tag:

    - foul-heavy          (foul_z >= +0.5)
    - card-prone          (card_z >= +0.5)
    - high-press          (press_z >= +0.5)
    - shot-allowing       (shots_allowed_z >= +0.5)
    - defensively-solid   (shots_allowed_z <= -0.5)
    - balanced            (if no other tag)
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


def load_team_style_inputs() -> pd.DataFrame:
    """
    Load team_match_stats CSV produced by team_expectations.py.
    """
    history_dir = get_history_dir()
    path = history_dir / "team_match_stats_epl_2025_26_sample.csv"
    df = pd.read_csv(path)
    return df


# ----------------------------------------------------------------------
# Core style logic
# ----------------------------------------------------------------------


def _add_shots_allowed(team_match_stats: pd.DataFrame) -> pd.DataFrame:
    """
    For each team+fixture row, add a 'shots_allowed' column
    based on the opponent's shots_total in that fixture.
    """
    df = team_match_stats.copy()

    # Extract (fixture_id, team_id, shots_total) as the opponent view
    opp = df[["fixture_id", "team_id", "shots_total"]].copy()
    opp = opp.rename(
        columns={
            "team_id": "opponent_team_id",
            "shots_total": "opp_shots_total",
        }
    )

    # Merge so each row sees its opponent's shots
    df = df.merge(
        opp,
        on=["fixture_id", "opponent_team_id"],
        how="left",
    )

    df["shots_allowed"] = df["opp_shots_total"].fillna(0)
    df.drop(columns=["opp_shots_total"], inplace=True)

    return df


def build_team_style(team_match_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Build team style metrics and tags from team_match_stats.

    Returns one row per team_id with:
        - per-match metrics
        - style indices
        - z-scores
        - style_tag
    """
    df = team_match_stats.copy()

    # Add shots_allowed per match by looking at opponents' shots
    df = _add_shots_allowed(df)

    # One row per team per fixture -> treat each as a "match"
    df["match"] = 1

    group_cols = ["team_id"]
    agg_cols = {
        "match": "sum",
        "goals_for": "sum",
        "goals_against": "sum",
        "shots_total": "sum",
        "shots_allowed": "sum",
        "fouls_committed": "sum",
        "fouls_drawn": "sum",
        "yellow_cards": "sum",
        "red_cards": "sum",
        "tackles": "sum",
        "dribbles": "sum",
    }

    grouped = df.groupby(group_cols, dropna=False).agg(agg_cols).reset_index()
    grouped.rename(columns={"match": "matches"}, inplace=True)

    def safe_div(total, count):
        if count and count > 0:
            return float(total) / float(count)
        return 0.0

    # Per-match metrics
    grouped["goals_for_per_match"] = grouped.apply(
        lambda row: safe_div(row["goals_for"], row["matches"]), axis=1
    )
    grouped["goals_against_per_match"] = grouped.apply(
        lambda row: safe_div(row["goals_against"], row["matches"]), axis=1
    )
    grouped["shots_for_per_match"] = grouped.apply(
        lambda row: safe_div(row["shots_total"], row["matches"]), axis=1
    )
    grouped["shots_allowed_per_match"] = grouped.apply(
        lambda row: safe_div(row["shots_allowed"], row["matches"]), axis=1
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

    # Style indices
    grouped["foul_rate"] = grouped["fouls_committed_per_match"]
    grouped["card_rate"] = (
        grouped["yellow_cards_per_match"] + 2.0 * grouped["red_cards_per_match"]
    )
    grouped["shots_allowed_rate"] = grouped["shots_allowed_per_match"]
    grouped["press_intensity"] = (
        grouped["tackles_per_match"]
        + 0.5 * grouped["dribbles_per_match"]
        + 0.25 * grouped["fouls_committed_per_match"]
    )

    # Z-score helpers
    def add_z_score(df: pd.DataFrame, col: str, z_col: str) -> None:
        mean = df[col].mean()
        std = df[col].std()
        if std and std > 0:
            df[z_col] = (df[col] - mean) / std
        else:
            df[z_col] = 0.0

    add_z_score(grouped, "foul_rate", "foul_z")
    add_z_score(grouped, "card_rate", "card_z")
    add_z_score(grouped, "shots_allowed_rate", "shots_allowed_z")
    add_z_score(grouped, "press_intensity", "press_z")

    # Style tag assignment
    style_tags = []

    for _, row in grouped.iterrows():
        tags = []

        if row["foul_z"] >= 0.5:
            tags.append("foul-heavy")
        if row["card_z"] >= 0.5:
            tags.append("card-prone")
        if row["press_z"] >= 0.5:
            tags.append("high-press")
        if row["shots_allowed_z"] >= 0.5:
            tags.append("shot-allowing")
        if row["shots_allowed_z"] <= -0.5:
            tags.append("defensively-solid")

        if not tags:
            tags.append("balanced")

        style_tags.append(",".join(tags))

    grouped["style_tag"] = style_tags

    # Sort nicely
    grouped = grouped.sort_values("team_id").reset_index(drop=True)

    return grouped


# ----------------------------------------------------------------------
# Script entrypoint
# ----------------------------------------------------------------------


if __name__ == "__main__":
    team_match_stats = load_team_style_inputs()
    print("Loaded team_match_stats:", team_match_stats.shape)

    team_style_df = build_team_style(team_match_stats)
    print("Team style shape:", team_style_df.shape)

    print("\nSample rows:")
    cols_to_show = [
        "team_id",
        "matches",
        "shots_for_per_match",
        "shots_allowed_per_match",
        "fouls_committed_per_match",
        "yellow_cards_per_match",
        "press_intensity",
        "foul_rate",
        "card_rate",
        "shots_allowed_rate",
        "press_z",
        "style_tag",
    ]
    print(team_style_df[cols_to_show].head().to_string(index=False))

    # Save to CSV
    history_dir = get_history_dir()
    out_path = history_dir / "team_style_epl_2025_26.csv"
    team_style_df.to_csv(out_path, index=False)

    print("\nSaved team style metrics to:")
    print("  ", out_path)