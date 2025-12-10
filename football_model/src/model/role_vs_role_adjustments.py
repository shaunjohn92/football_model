"""
role_vs_role_adjustments.py

Infrastructure for role-vs-role modifiers in Football Model v1.

It does two things:

1) build_role_vs_role_template(...)
   - Reads player_roles_epl_2025_26.csv
   - Finds all distinct attacking roles
   - Builds a template CSV:
         data/history/role_vs_role_modifiers_epl_2025_26.csv
     with rows:
         attacking_role, defending_role, shots_total_factor, shots_on_factor,
         fouls_committed_factor, fouls_drawn_factor, yellow_cards_factor
     All factors start at 1.0 (no effect) – you can edit this in Excel.

2) apply_role_vs_role_to_projections(...)
   - For each projections CSV:
         match_projections_epl_2025_26_sample_with_explanations.csv
         match_projections_epl_2025_26_upcoming.csv
     it:
       * maps each player's attacking role -> a defensive archetype
       * looks up the matching row in the modifiers CSV
       * multiplies projection columns by the chosen factors
       * keeps original values in *_raw columns for debugging
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------


def get_project_root() -> Path:
    # This file is in src/model/, so:
    # parents[0] = src/model
    # parents[1] = src
    # parents[2] = project root (where app.py lives)
    return Path(__file__).resolve().parents[2]


def get_history_dir() -> Path:
    root = get_project_root()
    hist = root / "data" / "history"
    hist.mkdir(parents=True, exist_ok=True)
    return hist


# ---------------------------------------------------------------------
# Step 1: Build template CSV of modifiers
# ---------------------------------------------------------------------


def build_role_vs_role_template(overwrite: bool = False) -> pd.DataFrame:
    """
    Build a template modifiers CSV based on roles we have in
    player_roles_epl_2025_26.csv.

    Output file:
        data/history/role_vs_role_modifiers_epl_2025_26.csv

    Columns:
        attacking_role, defending_role,
        shots_total_factor, shots_on_factor,
        fouls_committed_factor, fouls_drawn_factor,
        yellow_cards_factor
    """
    hist_dir = get_history_dir()
    roles_path = hist_dir / "player_roles_epl_2025_26.csv"
    out_path = hist_dir / "role_vs_role_modifiers_epl_2025_26.csv"

    if out_path.exists() and not overwrite:
        print(f"[RVR] Modifiers file already exists, not overwriting: {out_path}")
        return pd.read_csv(out_path)

    if not roles_path.exists():
        raise FileNotFoundError(
            f"[RVR] player roles file not found: {roles_path}. "
            "Make sure you ran role_assignment first."
        )

    roles_df = pd.read_csv(roles_path)
    attacking_roles = (
        roles_df["primary_role"].dropna().astype(str).unique().tolist()
    )
    attacking_roles = sorted(set(attacking_roles))

    # Defensive archetypes we care about initially – you can expand this later.
    defending_roles: List[str] = [
        "CB",   # centre-backs
        "FB",   # full-backs / wingbacks
        "DM",   # defensive mids
        "CM",   # central mids
        "W",    # wide forwards/wingers defending
        "ST",   # strikers defending
        "GK",   # goalkeeper
        "ALL",  # generic fallback
    ]

    rows = []
    for att in attacking_roles:
        for deff in defending_roles:
            rows.append(
                {
                    "attacking_role": att,
                    "defending_role": deff,
                    "shots_total_factor": 1.0,
                    "shots_on_factor": 1.0,
                    "fouls_committed_factor": 1.0,
                    "fouls_drawn_factor": 1.0,
                    "yellow_cards_factor": 1.0,
                }
            )

    mods_df = pd.DataFrame(rows)
    mods_df.to_csv(out_path, index=False)
    print(f"[RVR] Wrote template modifiers to {out_path}")
    print(
        "[RVR] All factors are 1.0 by default (no change). "
        "You can open this CSV in Excel and tweak specific combinations."
    )
    return mods_df


# ---------------------------------------------------------------------
# Step 2: Apply modifiers to projections
# ---------------------------------------------------------------------


def _estimate_defensive_archetype(att_role: str) -> str:
    """
    Given an attacking role string like 'ST', 'W_L', 'FB_R', map it to a
    defensive archetype label that lines up with defending_role in the
    modifiers CSV.

    This is a heuristic – later we can make this smarter.
    """
    if not isinstance(att_role, str):
        return "ALL"

    att = att_role.upper()

    # Normalise some prefixes
    if att.startswith("ST"):
        return "CB"       # strikers mainly face centre-backs
    if att.startswith("W"):
        return "FB"       # wingers mainly face full-backs
    if att.startswith("FB"):
        return "W"        # full-backs mainly deal with wingers
    if att == "AM":
        return "DM"       # attacking mids vs defensive mids
    if att == "CM":
        return "CM"
    if att == "DM":
        return "CM"
    if att == "CB":
        return "ST"       # CBs mainly face opposing strikers
    if att == "GK":
        return "ALL"

    return "ALL"


def apply_role_vs_role_to_file(proj_filename: str) -> None:
    hist_dir = get_history_dir()
    proj_path = hist_dir / proj_filename
    mods_path = hist_dir / "role_vs_role_modifiers_epl_2025_26.csv"

    if not proj_path.exists():
        print(f"[RVR] Projection file not found: {proj_path}")
        return
    if not mods_path.exists():
        raise FileNotFoundError(
            f"[RVR] Modifiers file not found: {mods_path}. "
            "Run build_role_vs_role_template() / learn_from_history() first."
        )

    proj_df = pd.read_csv(proj_path)
    mods_df = pd.read_csv(mods_path)

    # If there's no role column, nothing to do
    if "role" not in proj_df.columns:
        print(f"[RVR] No 'role' column in {proj_path}, skipping.")
        return

    # --- IMPORTANT: drop any existing factor columns from previous runs ---
    factor_cols = [
        "shots_total_factor",
        "shots_on_factor",
        "fouls_committed_factor",
        "fouls_drawn_factor",
        "yellow_cards_factor",
    ]
    existing_factor_cols = [c for c in factor_cols if c in proj_df.columns]
    if existing_factor_cols:
        proj_df = proj_df.drop(columns=existing_factor_cols)

    # Compute defensive archetype for each row
    proj_df["def_role_mod"] = proj_df["role"].astype(str).apply(
        _estimate_defensive_archetype
    )

    # Merge modifiers on (attacking_role, defending_role)
    merged = proj_df.merge(
        mods_df,
        left_on=["role", "def_role_mod"],
        right_on=["attacking_role", "defending_role"],
        how="left",
    )

    # Ensure all factor columns exist and fill missing with 1.0 (no change)
    for col in factor_cols:
        if col not in merged.columns:
            merged[col] = 1.0
        else:
            merged[col] = merged[col].fillna(1.0)

    # Map factor columns to projection columns
    mapping = {
        "proj_shots_total": "shots_total_factor",
        "proj_shots_on": "shots_on_factor",
        "proj_fouls_committed": "fouls_committed_factor",
        "proj_fouls_drawn": "fouls_drawn_factor",
        "proj_yellow_cards": "yellow_cards_factor",
    }

    for proj_col, factor_col in mapping.items():
        if proj_col in merged.columns:
            raw_col = proj_col + "_raw"

            # Idempotent behaviour:
            # - If *_raw exists, treat it as the baseline.
            # - Otherwise, create *_raw from current proj_col.
            if raw_col in merged.columns:
                base_values = merged[raw_col]
            else:
                base_values = merged[proj_col]
                merged[raw_col] = base_values

            merged[proj_col] = base_values * merged[factor_col]

    # Drop helper columns we don't need in the final CSV
    for col in ["attacking_role", "defending_role", "def_role_mod"]:
        if col in merged.columns:
            merged = merged.drop(columns=[col])

    merged.to_csv(proj_path, index=False)
    print(f"[RVR] Applied role-vs-role modifiers to {proj_path}")


def apply_to_all_projections() -> None:
    apply_role_vs_role_to_file("match_projections_epl_2025_26_sample_with_explanations.csv")
    apply_role_vs_role_to_file("match_projections_epl_2025_26_upcoming.csv")

# ---------------------------------------------------------------------
# Step 3: Learn per-role factors from history data
# ---------------------------------------------------------------------


def learn_role_factors_from_history(
    history_stats_filename: str = "player_match_stats_epl_2025_26_history.csv",
    history_proj_filename: str = "match_projections_epl_2025_26_history_with_explanations.csv",
    min_samples: int = 30,
    shrink_n: int = 100,
) -> pd.DataFrame:
    """
    Learn per-attacking-role multipliers from history:

        factor = average( actual / projected )  [with some smoothing]

    For now we ignore the defending_role and only fill the rows where
    defending_role == 'ALL' in role_vs_role_modifiers_epl_2025_26.csv.

    Params:
        history_stats_filename: CSV with actual per-player match stats.
        history_proj_filename: CSV with historical projections.
        min_samples: minimum rows needed before we trust the raw factor.
        shrink_n: stronger shrinkage towards 1.0 when sample size is small.

    Output:
        Updated modifiers DataFrame (also saved to CSV).
    """
    hist_dir = get_history_dir()

    stats_path = hist_dir / history_stats_filename
    proj_path = hist_dir / history_proj_filename
    mods_path = hist_dir / "role_vs_role_modifiers_epl_2025_26.csv"

    if not stats_path.exists():
        raise FileNotFoundError(
        f"[RVR-LEARN] History stats file not found: {stats_path}\n"
        "If your file has a different name, update 'history_stats_filename' "
        "in learn_role_factors_from_history()."
        )

    if not proj_path.exists():
        raise FileNotFoundError(
        f"[RVR-LEARN] History projections file not found: {proj_path}"
        )

    if not mods_path.exists():
        raise FileNotFoundError(
        f"[RVR-LEARN] Modifiers file not found: {mods_path}. "
        "Run build_role_vs_role_template() first."
        )

    # Load data
    stats_df = pd.read_csv(stats_path)
    proj_df = pd.read_csv(proj_path)
    mods_df = pd.read_csv(mods_path)

    # Merge projections with actual stats on (fixture_id, team_id, player_id)
    # so we have: role, projected stats, actual stats in one table.
    merge_keys = ["fixture_id", "team_id", "player_id"]
    for k in merge_keys:
        if k not in stats_df.columns or k not in proj_df.columns:
            raise KeyError(
                f"[RVR-LEARN] Key column '{k}' missing in stats or projections CSV."
            )

    merged = proj_df.merge(stats_df, on=merge_keys, how="inner", suffixes=("_proj", "_act"))

    # Make sure we have a 'role' column – try a few possible names
    possible_role_cols = ["role", "primary_role", "player_role", "role_old"]
    role_col_found = None
    for col in possible_role_cols:
        if col in merged.columns:
            role_col_found = col
            break

    if role_col_found is None:
        raise KeyError(
            "[RVR-LEARN] Could not find a role column in the merged data.\n"
            f"Available columns: {list(merged.columns)}\n"
            "If your role column has a different name, add it to 'possible_role_cols'."
        )

    # Standardise: create a 'role' column for the rest of the function to use
    merged["role"] = merged[role_col_found].astype(str)

    # Define which projection/actual column pairs to use
    pairs = [
        ("proj_shots_total", "shots_total"),
        ("proj_shots_on", "shots_on"),
        ("proj_fouls_committed", "fouls_committed"),
        ("proj_fouls_drawn", "fouls_drawn"),
        ("proj_yellow_cards", "yellow_cards"),
    ]

    # We'll collect per-role ratios
    records = []

    for proj_col, act_col in pairs:
        if proj_col not in merged.columns or act_col not in merged.columns:
            print(f"[RVR-LEARN] Skipping {proj_col}/{act_col}, column missing.")
            continue

        df_pair = merged[["role", proj_col, act_col]].copy()
        df_pair = df_pair.dropna(subset=[proj_col, act_col])

        # Only consider rows with positive projection to avoid division by zero
        df_pair = df_pair[df_pair[proj_col] > 0]

        if df_pair.empty:
            continue

        # Compute ratio = actual / projected
        df_pair["ratio"] = df_pair[act_col] / df_pair[proj_col]

        # Clip extreme ratios (very noisy)
        df_pair["ratio"] = df_pair["ratio"].clip(lower=0.2, upper=2.5)

        grouped = df_pair.groupby("role")["ratio"].agg(["mean", "count"]).reset_index()
        grouped.rename(columns={"mean": "raw_factor", "count": "n"}, inplace=True)

        grouped["metric"] = proj_col

        records.append(grouped)

    if not records:
        raise RuntimeError("[RVR-LEARN] No metric pairs could be processed.")

    all_factors = pd.concat(records, ignore_index=True)

    # Apply simple shrinkage towards 1.0, so low-sample roles don't overfit
    def _shrink(row):
        n = row["n"]
        raw = row["raw_factor"]
        if n <= 0:
            return 1.0
        weight = n / (n + shrink_n)
        return 1.0 + (raw - 1.0) * weight

    all_factors["factor"] = all_factors.apply(_shrink, axis=1)

    # Now we have per-role, per-metric factors for the projection column names,
    # e.g. proj_shots_total, proj_shots_on, etc.
    # We want to write them into mods_df rows where defending_role == 'ALL'.

    metric_to_factor_col = {
        "proj_shots_total": "shots_total_factor",
        "proj_shots_on": "shots_on_factor",
        "proj_fouls_committed": "fouls_committed_factor",
        "proj_fouls_drawn": "fouls_drawn_factor",
        "proj_yellow_cards": "yellow_cards_factor",
    }

    # Work on a copy of modifiers
    mods_new = mods_df.copy()

    for metric, factor_col in metric_to_factor_col.items():
        subset = all_factors[all_factors["metric"] == metric]

        for _, row in subset.iterrows():
            role = row["role"]
            factor = row["factor"]
            n = row["n"]

            if n < min_samples:
                # Not enough data; skip updating for this role/metric
                continue

            # Identify rows in mods_new to update:
            # attacking_role == role AND defending_role == 'ALL'
            mask = (mods_new["attacking_role"] == role) & (
                mods_new["defending_role"] == "ALL"
            )

            if mask.any():
                mods_new.loc[mask, factor_col] = factor

    # Save updated modifiers
    mods_new.to_csv(mods_path, index=False)
    print(f"[RVR-LEARN] Updated modifiers written to {mods_path}")

    return mods_new


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main():
    import sys

    args = sys.argv[1:]

    if args and args[0] == "build_template":
        # Only build template (no changes to projections)
        build_role_vs_role_template(overwrite=False)
    elif args and args[0] == "learn_from_history":
        # Learn per-role factors and update modifiers CSV
        learn_role_factors_from_history()
    else:
        # Default: apply modifiers to all projections
        apply_to_all_projections()


if __name__ == "__main__":
    main()