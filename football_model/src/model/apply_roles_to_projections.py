"""
apply_roles_to_projections.py

Merges player_roles_epl_2025_26.csv into:
    - match_projections_epl_2025_26_sample_with_explanations.csv
    - match_projections_epl_2025_26_upcoming.csv

Replaces the 'role' column with 'primary_role' from the roles file
(where available).
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def get_history_dir() -> Path:
    # File is in src/model, so:
    # parents[0] = src/model
    # parents[1] = src
    # parents[2] = project root (where app.py lives)
    root = Path(__file__).resolve().parents[2]
    hist = root / "data" / "history"
    return hist


def apply_roles_to_file(proj_filename: str) -> None:
    hist_dir = get_history_dir()

    proj_path = hist_dir / proj_filename
    roles_path = hist_dir / "player_roles_epl_2025_26.csv"

    if not proj_path.exists():
        print(f"[APPLY_ROLES] Projection file not found: {proj_path}")
        return
    if not roles_path.exists():
        raise FileNotFoundError(
            f"[APPLY_ROLES] Roles file not found: {roles_path}. "
            "Run role_assignment.build_player_roles_cli() first."
        )

    proj_df = pd.read_csv(proj_path)
    roles_df = pd.read_csv(roles_path)

    # Merge on player_id + team_id (most robust)
    merged = proj_df.merge(
        roles_df[["player_id", "team_id", "primary_role"]],
        on=["player_id", "team_id"],
        how="left",
    )

    # If there was already a 'role' column, keep it for debugging
    if "role" in merged.columns:
        merged["role_old"] = merged["role"]

    # Overwrite 'role' with primary_role where available
    merged["role"] = merged["primary_role"].fillna(merged.get("role", "UNK"))

    # Drop helper column
    merged = merged.drop(columns=["primary_role"])

    # Save back to the same path (in-place update)
    merged.to_csv(proj_path, index=False)
    print(f"[APPLY_ROLES] Updated roles in {proj_path}")


def main():
    apply_roles_to_file("match_projections_epl_2025_26_sample_with_explanations.csv")
    apply_roles_to_file("match_projections_epl_2025_26_upcoming.csv")


if __name__ == "__main__":
    main()