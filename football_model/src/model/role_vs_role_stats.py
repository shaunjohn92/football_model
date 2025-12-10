"""
role_vs_role_stats.py – Football Model v1

Build aggregated role-vs-role matchup stats from local history.

Input:
    data/history/player_match_stats_epl_2025_26_sample_enriched.csv

Output:
    data/history/role_vs_role_epl_2025_26.csv

Each row in the output describes a pairing:

    attacker_role
    defender_role
    side_match          (same_side / opp_side / central_or_unknown)
    pair_appearances    (# of attacker-defender "pair rows" across all matches)
    shots_allowed       (attacker shots_total)
    fouls_drawn         (attacker fouls_drawn)
    fouls_committed     (defender fouls_committed)
    cards_committed     (defender yellow_cards)

This is a coarse first version:
    - For each fixture, every attacking player is paired with every opponent player.
    - side_match is based on attacker.side vs defender.side.
"""

from __future__ import annotations

from pathlib import Path

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


def load_role_inputs() -> pd.DataFrame:
    """
    Load enriched player match stats CSV.
    """
    history_dir = get_history_dir()
    path = history_dir / "player_match_stats_epl_2025_26_sample_enriched.csv"
    df = pd.read_csv(path)
    return df


# ----------------------------------------------------------------------
# Core logic
# ----------------------------------------------------------------------


def _side_match(att_side: str, def_side: str) -> str:
    """
    Determine if the sides match:

        same_side            – both non-central and equal (L vs L, R vs R)
        opp_side             – both non-central and different (L vs R, R vs L)
        central_or_unknown   – if either side is 'C' or missing/other
    """
    att_side = (att_side or "").upper()
    def_side = (def_side or "").upper()

    if att_side in {"L", "R"} and def_side in {"L", "R"}:
        if att_side == def_side:
            return "same_side"
        else:
            return "opp_side"
    return "central_or_unknown"


def build_role_vs_role_stats(players_enriched: pd.DataFrame) -> pd.DataFrame:
    """
    Build aggregated role-vs-role stats.

    Steps:
        1) Take player rows as attackers & defenders.
        2) Self-join on fixture_id with opposite teams.
        3) Compute side_match category.
        4) Aggregate attacker shots/fouls_drawn and defender fouls/cards.
    """
    df = players_enriched.copy()

    # Basic safety: fill missing role/side with placeholders
    df["role"] = df["role"].fillna("UNK")
    df["side"] = df["side"].fillna("C")

    # Build attacker view
    att_cols = [
        "fixture_id",
        "team_id",
        "player_id",
        "player_name",
        "role",
        "side",
        "shots_total",
        "fouls_drawn",
    ]
    attackers = df[att_cols].copy()
    attackers = attackers.rename(
        columns={
            "team_id": "att_team_id",
            "player_id": "att_player_id",
            "player_name": "att_player_name",
            "role": "att_role",
            "side": "att_side",
            "shots_total": "att_shots_total",
            "fouls_drawn": "att_fouls_drawn",
        }
    )

    # Defender view (note team_id + opponent_team_id)
    def_cols = [
        "fixture_id",
        "team_id",
        "opponent_team_id",
        "player_id",
        "player_name",
        "role",
        "side",
        "fouls_committed",
        "yellow_cards",
    ]
    defenders = df[def_cols].copy()
    defenders = defenders.rename(
        columns={
            "team_id": "def_team_id",
            "opponent_team_id": "def_opponent_team_id",
            "player_id": "def_player_id",
            "player_name": "def_player_name",
            "role": "def_role",
            "side": "def_side",
            "fouls_committed": "def_fouls_committed",
            "yellow_cards": "def_yellow_cards",
        }
    )

    # Join attackers with defenders from the *opposite* team in the same fixture.
    # Condition: same fixture_id, and attacker team_id == defender.opponent_team_id
    pairs = attackers.merge(
        defenders,
        left_on=["fixture_id", "att_team_id"],
        right_on=["fixture_id", "def_opponent_team_id"],
        how="inner",
    )

    if pairs.empty:
        # Return an empty shell with proper columns
        columns = [
            "attacker_role",
            "defender_role",
            "side_match",
            "pair_appearances",
            "shots_allowed",
            "fouls_drawn",
            "fouls_committed",
            "cards_committed",
        ]
        return pd.DataFrame(columns=columns)

    # Determine side_match
    pairs["side_match"] = pairs.apply(
        lambda row: _side_match(row["att_side"], row["def_side"]), axis=1
    )

    # Aggregate
    group_cols = ["att_role", "def_role", "side_match"]

    agg = (
        pairs.groupby(group_cols, dropna=False)
        .agg(
            pair_appearances=("att_player_id", "count"),
            shots_allowed=("att_shots_total", "sum"),
            fouls_drawn=("att_fouls_drawn", "sum"),
            fouls_committed=("def_fouls_committed", "sum"),
            cards_committed=("def_yellow_cards", "sum"),
        )
        .reset_index()
    )

    # Rename columns to match schema
    agg = agg.rename(
        columns={
            "att_role": "attacker_role",
            "def_role": "defender_role",
        }
    )

    # Sort for readability
    agg = agg.sort_values(
        ["shots_allowed", "fouls_drawn", "fouls_committed"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return agg


# ----------------------------------------------------------------------
# Script entrypoint
# ----------------------------------------------------------------------


if __name__ == "__main__":
    players_enriched_df = load_role_inputs()
    print("Loaded enriched player stats:", players_enriched_df.shape)

    role_stats_df = build_role_vs_role_stats(players_enriched_df)
    print("Role-vs-role stats shape:", role_stats_df.shape)

    print("\nSample rows:")
    print(
        role_stats_df.head(10).to_string(index=False)
        if not role_stats_df.empty
        else "  (no data – check your input CSV)"
    )

    history_dir = get_history_dir()
    out_path = history_dir / "role_vs_role_epl_2025_26.csv"
    role_stats_df.to_csv(out_path, index=False)

    print("\nSaved role-vs-role stats to:")
    print("  ", out_path)