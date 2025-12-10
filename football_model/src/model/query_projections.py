"""
query_projections.py – Football Model v1

Helper script to make it easier to inspect projections.

Inputs (all local CSVs):

    data/history/fixtures_epl_2025_26_sample.csv
    data/history/match_projections_epl_2025_26_sample_with_explanations.csv

What it does:

    1) Prints a simple list of fixtures:
           [fixture_id]  [date]  [away_team_id] @ [home_team_id]

    2) For each fixture_id, creates a separate CSV:

           data/history/match_projections_fixture_<fixture_id>.csv

       containing ONLY the players and explanations for that single match.
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


def load_query_inputs() -> Dict[str, pd.DataFrame]:
    """
    Load fixtures + projections-with-explanations.
    """
    history_dir = get_history_dir()

    fixtures_path = history_dir / "fixtures_epl_2025_26_sample.csv"
    proj_with_expl_path = (
        history_dir / "match_projections_epl_2025_26_sample_with_explanations.csv"
    )

    fixtures_df = pd.read_csv(fixtures_path)
    projections_df = pd.read_csv(proj_with_expl_path)

    return {"fixtures": fixtures_df, "projections": projections_df}


# ----------------------------------------------------------------------
# Main query/export logic
# ----------------------------------------------------------------------


def list_fixtures(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a small table of fixtures with:

        fixture_id, date, home_team_id, away_team_id
    """
    cols = ["fixture_id", "date", "home_team_id", "away_team_id"]
    for col in cols:
        if col not in fixtures_df.columns:
            raise KeyError(f"Expected column '{col}' not found in fixtures CSV.")

    listing = fixtures_df[cols].copy().sort_values("date").reset_index(drop=True)
    return listing


def export_per_fixture_projections(
    fixtures_df: pd.DataFrame, projections_df: pd.DataFrame
) -> None:
    """
    For each fixture_id in fixtures_df, filter projections_df and
    save a separate CSV:

        match_projections_fixture_<fixture_id>.csv
    """
    history_dir = get_history_dir()
    unique_fixtures = fixtures_df["fixture_id"].unique()

    for fixture_id in unique_fixtures:
        mask = projections_df["fixture_id"] == fixture_id
        match_proj = projections_df[mask].copy()

        if match_proj.empty:
            # No projections for this fixture_id – skip
            continue

        out_path = history_dir / f"match_projections_fixture_{fixture_id}.csv"
        match_proj.to_csv(out_path, index=False)
        print(f"  Saved fixture {fixture_id} projections to: {out_path}")


# ----------------------------------------------------------------------
# Script entrypoint
# ----------------------------------------------------------------------


if __name__ == "__main__":
    data = load_query_inputs()
    fixtures_df = data["fixtures"]
    projections_df = data["projections"]

    print("Loaded query inputs:")
    print("  Fixtures:", fixtures_df.shape)
    print("  Projections:", projections_df.shape)

    # 1) List fixtures
    listing = list_fixtures(fixtures_df)

    print("\nAvailable fixtures in history (EPL 2025–26 sample):")
    print("  fixture_id    date        away_team_id @ home_team_id")
    for _, row in listing.iterrows():
        print(
            f"  {int(row['fixture_id'])}    {row['date']}    "
            f"{int(row['away_team_id'])} @ {int(row['home_team_id'])}"
        )

    # 2) Export per-fixture projections
    print("\nExporting per-fixture projection CSVs...")
    export_per_fixture_projections(fixtures_df, projections_df)

    print("\nDone. Open any of the 'match_projections_fixture_<fixture_id>.csv' files in:")
    print("  ", get_history_dir())