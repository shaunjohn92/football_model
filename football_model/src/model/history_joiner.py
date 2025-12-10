"""
history_joiner.py – Football Model v1

Offline step that works on the CSVs saved by history_loader.py.

Goals:
- Load fixtures, player_match_stats, subs, role_assignments from data/history.
- Join role_assignments into player_match_stats so each player row has role/side/band.
- Join roles into subs (role_out, role_in).
- Fill basic sub_on_min / sub_off_min and subbed_on / subbed_off flags from subs.

This does NOT call the API at all – it only reads/writes local CSV files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


# ----------------------------------------------------------------------
# Paths and filenames
# ----------------------------------------------------------------------


def get_history_dir() -> Path:
    """
    Locate the data/history directory relative to this file.
    We assume project structure:

        project_root/
            data/
                history/
            src/
                model/
                    history_loader.py
                    history_joiner.py
    """
    project_root = Path(__file__).resolve().parents[2]
    history_dir = project_root / "data" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir


def load_history_sample() -> Dict[str, pd.DataFrame]:
    """
    Load the 4 sample CSVs we just generated with history_loader.py.
    """
    history_dir = get_history_dir()

    fixtures_path = history_dir / "fixtures_epl_2025_26_sample.csv"
    players_path = history_dir / "player_match_stats_epl_2025_26_sample.csv"
    subs_path = history_dir / "subs_epl_2025_26_sample.csv"
    roles_path = history_dir / "role_assignments_epl_2025_26_history.csv"

    fixtures_df = pd.read_csv(fixtures_path)
    players_df = pd.read_csv(players_path)
    subs_df = pd.read_csv(subs_path)
    roles_df = pd.read_csv(roles_path)

    return {
        "fixtures": fixtures_df,
        "player_match_stats": players_df,
        "subs": subs_df,
        "role_assignments": roles_df,
    }


# ----------------------------------------------------------------------
# Joining roles into player_match_stats
# ----------------------------------------------------------------------


def add_roles_to_player_stats(
    player_match_stats: pd.DataFrame, role_assignments: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge role_assignments into player_match_stats so that each row has:

        role, side, band

    We join on (fixture_id, team_id, player_id).
    """
    players = player_match_stats.copy()
    roles = role_assignments.copy()

    # Keep only the join keys + role info from roles
    roles_small = roles[
        ["fixture_id", "team_id", "player_id", "role", "side", "band"]
    ].copy()

    # Sometimes player_match_stats already has empty role/side/band columns.
    # Drop them to avoid confusion.
    for col in ["role", "side", "band"]:
        if col in players.columns:
            players = players.drop(columns=[col])

    merged = players.merge(
        roles_small,
        on=["fixture_id", "team_id", "player_id"],
        how="left",
    )

    return merged


# ----------------------------------------------------------------------
# Joining roles into subs
# ----------------------------------------------------------------------


def add_roles_to_subs(subs: pd.DataFrame, role_assignments: pd.DataFrame) -> pd.DataFrame:
    """
    Add role_out and role_in columns to subs based on role_assignments.

    We use two joins:
        - one for player_out_id
        - one for player_in_id
    """
    subs_df = subs.copy()
    roles = role_assignments.copy()

    # For player who goes off
    out_roles = roles[["fixture_id", "team_id", "player_id", "role"]].rename(
        columns={"player_id": "player_out_id", "role": "role_out"}
    )
    subs_df = subs_df.merge(
        out_roles, on=["fixture_id", "team_id", "player_out_id"], how="left"
    )

    # For player who comes on
    in_roles = roles[["fixture_id", "team_id", "player_id", "role"]].rename(
        columns={"player_id": "player_in_id", "role": "role_in"}
    )
    subs_df = subs_df.merge(
        in_roles, on=["fixture_id", "team_id", "player_in_id"], how="left"
    )

    return subs_df


# ----------------------------------------------------------------------
# Filling sub_on_min / sub_off_min in player_match_stats
# ----------------------------------------------------------------------


def apply_sub_minutes(
    player_match_stats_with_roles: pd.DataFrame, subs_with_roles: pd.DataFrame
) -> pd.DataFrame:
    """
    Use the subs table to fill:
        - subbed_on / sub_on_min
        - subbed_off / sub_off_min
    in player_match_stats.

    For each substitution event at 'minute':
        player_out_id → we set subbed_off=True, sub_off_min = minute
        player_in_id  → we set subbed_on=True,  sub_on_min  = minute
    """
    players = player_match_stats_with_roles.copy()
    subs = subs_with_roles.copy()

    # Ensure the boolean/number columns exist
    if "subbed_on" not in players.columns:
        players["subbed_on"] = False
    else:
        players["subbed_on"] = players["subbed_on"].fillna(False)

    if "subbed_off" not in players.columns:
        players["subbed_off"] = False
    else:
        players["subbed_off"] = players["subbed_off"].fillna(False)

    if "sub_on_min" not in players.columns:
        players["sub_on_min"] = None
    if "sub_off_min" not in players.columns:
        players["sub_off_min"] = None

    # Loop over each substitution and update the relevant players
    for _, row in subs.iterrows():
        fixture_id = row.get("fixture_id")
        team_id = row.get("team_id")
        minute = row.get("minute")
        player_out_id = row.get("player_out_id")
        player_in_id = row.get("player_in_id")

        # Player going off
        mask_out = (
            (players["fixture_id"] == fixture_id)
            & (players["team_id"] == team_id)
            & (players["player_id"] == player_out_id)
        )
        if mask_out.any():
            players.loc[mask_out, "subbed_off"] = True
            players.loc[mask_out, "sub_off_min"] = minute

        # Player coming on
        mask_in = (
            (players["fixture_id"] == fixture_id)
            & (players["team_id"] == team_id)
            & (players["player_id"] == player_in_id)
        )
        if mask_in.any():
            players.loc[mask_in, "subbed_on"] = True
            players.loc[mask_in, "sub_on_min"] = minute

    return players


# ----------------------------------------------------------------------
# Main script entrypoint
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # 1) Load the sample CSVs
    data = load_history_sample()
    fixtures_df = data["fixtures"]
    players_df = data["player_match_stats"]
    subs_df = data["subs"]
    roles_df = data["role_assignments"]

    print("Loaded sample CSVs:")
    print("  Fixtures:", fixtures_df.shape)
    print("  Player match stats:", players_df.shape)
    print("  Subs:", subs_df.shape)
    print("  Role assignments:", roles_df.shape)

    # 2) Add roles into player stats
    players_with_roles = add_roles_to_player_stats(players_df, roles_df)
    print("After adding roles to player stats:", players_with_roles.shape)

    # 3) Add roles into subs
    subs_with_roles = add_roles_to_subs(subs_df, roles_df)
    print("After adding roles to subs:", subs_with_roles.shape)

    # 4) Fill sub_on_min / sub_off_min in player stats
    players_with_subs = apply_sub_minutes(players_with_roles, subs_with_roles)

    # 5) Save the enriched tables back to data/history
    history_dir = get_history_dir()
    players_out_path = history_dir / "player_match_stats_epl_2025_26_sample_enriched.csv"
    subs_out_path = history_dir / "subs_epl_2025_26_sample_enriched.csv"

    players_with_subs.to_csv(players_out_path, index=False)
    subs_with_roles.to_csv(subs_out_path, index=False)

    print("\nSaved enriched CSVs to:")
    print("  ", players_out_path)
    print("  ", subs_out_path)
