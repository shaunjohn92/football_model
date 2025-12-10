"""
run_history_projections.py

Driver to run the full Football Model v1 pipeline over *history* fixtures
and produce a big projections file for calibration:

    data/history/match_projections_epl_2025_26_history_with_explanations.csv

This is a FIRST VERSION. Some function names imported from the other
modules may need to be adjusted to match your actual code. If you get an
AttributeError or similar, keep this file and just tell ChatGPT the exact
error line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

# Import your existing model modules
from . import (
    player_baselines,
    team_expectations,
    substitution_patterns,
    team_style,
    role_vs_role_stats,
    match_projection,
    projection_explanations,
    weighted_player_baselines,
)

EPL_LEAGUE_ID = 39
SEASON_2025_26 = 2025  # project convention


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------


def get_project_root() -> Path:
    # this file lives in src/model, so parents[2] is the project root
    return Path(__file__).resolve().parents[2]


def get_history_dir() -> Path:
    root = get_project_root()
    hist = root / "data" / "history"
    hist.mkdir(parents=True, exist_ok=True)
    return hist


# ----------------------------------------------------------------------
# Loading base history tables (fixtures + stats + roles + subs)
# ----------------------------------------------------------------------


def load_history_tables() -> Dict[str, pd.DataFrame]:
    """
    Load the full-season history CSVs we generated with history_loader.

    Expects:
        fixtures_epl_2025_26_history.csv
        player_match_stats_epl_2025_26_history.csv
        subs_epl_2025_26_history.csv
        role_assignments_epl_2025_26_history.csv
    """
    hist_dir = get_history_dir()

    fixtures_path = hist_dir / "fixtures_epl_2025_26_history.csv"
    players_path = hist_dir / "player_match_stats_epl_2025_26_history.csv"
    subs_path = hist_dir / "subs_epl_2025_26_history.csv"
    roles_path = hist_dir / "role_assignments_epl_2025_26_history.csv"

    fixtures_df = pd.read_csv(fixtures_path)
    player_stats_df = pd.read_csv(players_path)
    subs_df = pd.read_csv(subs_path)
    roles_df = pd.read_csv(roles_path)

    print("[RUN_HISTORY] Loaded history tables:")
    print("  Fixtures:", fixtures_df.shape)
    print("  Player stats:", player_stats_df.shape)
    print("  Subs:", subs_df.shape)
    print("  Role assignments:", roles_df.shape)

    return {
        "fixtures": fixtures_df,
        "player_match_stats": player_stats_df,
        "subs": subs_df,
        "role_assignments": roles_df,
    }


# ----------------------------------------------------------------------
# Simple join of stats + roles (local history_joiner)
# ----------------------------------------------------------------------


def build_history_dataset_simple(history_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Minimal 'history joiner' built directly in this file.

    Starts from player_match_stats and left-joins role_assignments to ensure
    we have role/side/band for each player, per fixture & team.
    """
    player_stats = history_tables["player_match_stats"].copy()
    roles = history_tables["role_assignments"].copy()

    # Make sure key columns exist
    key_cols = ["fixture_id", "team_id", "player_id"]
    for col in key_cols:
        if col not in player_stats.columns or col not in roles.columns:
            raise KeyError(
                f"[RUN_HISTORY] Expected key column '{col}' in both "
                "player_match_stats and role_assignments."
            )

    # Keep only the role-related columns from roles
    role_cols = [
        "fixture_id",
        "team_id",
        "player_id",
        "role",
        "side",
        "band",
        "role_profile",
    ]
    role_cols = [c for c in role_cols if c in roles.columns]
    roles = roles[role_cols].copy()

    # Suffix role columns to avoid clashes
    suffix = "_from_lineups"
    roles = roles.add_suffix(suffix)
    # restore key names after suffixing
    roles.rename(
        columns={
            f"fixture_id{suffix}": "fixture_id",
            f"team_id{suffix}": "team_id",
            f"player_id{suffix}": "player_id",
        },
        inplace=True,
    )

    # Merge
    merged = player_stats.merge(
        roles,
        on=["fixture_id", "team_id", "player_id"],
        how="left",
    )

    # For each of role/side/band, if base column is missing or null,
    # fill from the lineups version.
    for col in ["role", "side", "band"]:
        base_col = col
        lineup_col = base_col + suffix
        if lineup_col in merged.columns:
            if base_col not in merged.columns:
                merged[base_col] = merged[lineup_col]
            else:
                merged[base_col] = merged[base_col].fillna(merged[lineup_col])

    # We don't strictly need to keep the *_from_lineups columns
    drop_cols = [c for c in merged.columns if c.endswith(suffix)]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)

        # For each of role/side/band, if base column is missing or null,
    # fill from the lineups version.
    for col in ["role", "side", "band"]:
        base_col = col
        lineup_col = base_col + suffix
        if lineup_col in merged.columns:
            if base_col not in merged.columns:
                merged[base_col] = merged[lineup_col]
            else:
                merged[base_col] = merged[base_col].fillna(merged[lineup_col])

    # We don't strictly need to keep the *_from_lineups columns
    drop_cols = [c for c in merged.columns if c.endswith(suffix)]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)

    # ⬇⬇⬇ NEW BLOCK: bring in fixture dates for recency-weighted baselines
    fixtures = history_tables.get("fixtures")
    if fixtures is not None and "date" in fixtures.columns:
        fixtures_dates = fixtures[["fixture_id", "date"]].drop_duplicates()
        merged = merged.merge(fixtures_dates, on="fixture_id", how="left")
    else:
        print(
            "[RUN_HISTORY] WARNING: fixtures table or 'date' column missing – "
            "recency-weighted baselines will not work correctly."
        )
    # ⬆⬆⬆ END NEW BLOCK

    print("[RUN_HISTORY] Simple joined history shape:", merged.shape)
    return merged



# ----------------------------------------------------------------------
# Build global baselines / team style / role-vs-role from history
# ----------------------------------------------------------------------


def build_global_inputs(history_tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Build all global inputs from full history:

        • joined_history (players + roles)
        • player baselines
        • team match stats
        • team baselines (team expectations)
        • substitution profiles
        • team style table
        • role-vs-role stats
    """

    # -------------------------------------------------------------
    # 1) Join raw history into a per-player-per-fixture table
    # -------------------------------------------------------------
    print("[RUN_HISTORY] Building joined history table (simple join)...")
    joined_history = build_history_dataset_simple(history_tables)

    # -------------------------------------------------------------
    # 2) Player baselines
    # -------------------------------------------------------------
    print("[RUN_HISTORY] Building player baselines from history...")
    player_base = weighted_player_baselines.build_player_baselines_by_role_weighted(
    joined_history,
    half_life_days=60.0,   # you can tweak this later
    min_total_minutes=90.0,
    )

    # -------------------------------------------------------------
    # 3) TEAM EXPECTATIONS: match stats → season baselines
    # -------------------------------------------------------------
    print("[RUN_HISTORY] Building team match stats from history...")
    team_match_stats_df = team_expectations.build_team_match_stats(
        fixtures_df=history_tables["fixtures"],
        players_df=joined_history
    )

    print("[RUN_HISTORY] Building team baselines (team expectations)...")
    team_baselines_df = team_expectations.build_team_baselines(team_match_stats_df)

    # -------------------------------------------------------------
    # 4) Substitution profiles
    print("[RUN_HISTORY] Building substitution profiles from history...")
    subs_df = history_tables["subs"]
    sub_profiles = substitution_patterns.build_substitution_profiles(
        subs_df=subs_df,
        players_df=joined_history,
    )

    # -------------------------------------------------------------
    # 5) Team style
    print("[RUN_HISTORY] Building team style table from history...")
    team_style_df = team_style.build_team_style(team_match_stats_df)

    # -------------------------------------------------------------
    # 6) Role-vs-role interaction stats
    # -------------------------------------------------------------
    print("[RUN_HISTORY] Building role-vs-role stats from history...")
    rvr_stats = role_vs_role_stats.build_role_vs_role_stats(joined_history)

    return {
        "joined_history": joined_history,
        "player_baselines": player_base,
        "team_match_stats": team_match_stats_df,
        "team_expectations": team_baselines_df,   # <-- used later in projection
        "sub_profiles": sub_profiles,
        "team_style": team_style_df,
        "role_vs_role_stats": rvr_stats,
    }


# ----------------------------------------------------------------------
# Project all fixtures using existing match_projection logic
# ----------------------------------------------------------------------


def project_all_history_fixtures(
    history_tables: Dict[str, pd.DataFrame],
    globals_: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Use match_projection.project_all_fixtures to project ALL history fixtures
    in one go, using the globals we just built.

    We skip detailed explanations here – the output is mainly for calibration.
    """
    fixtures = history_tables["fixtures"]

    player_base = globals_["player_baselines"]
    team_exp = globals_["team_expectations"]          # this is team_baselines
    sub_profiles = globals_["sub_profiles"]
    team_style_df = globals_["team_style"]
    players_enriched = globals_["joined_history"]

    print(f"[RUN_HISTORY] Projecting {len(fixtures)} fixtures from history using project_all_fixtures()...")

    projections_df = match_projection.project_all_fixtures(
        fixtures_df=fixtures,
        team_baselines=team_exp,
        player_baselines=player_base,
        players_enriched=players_enriched,
        sub_profiles=sub_profiles,
        team_style=team_style_df,
    )

    print("[RUN_HISTORY] Combined history projections shape:", projections_df.shape)

    # --- Attach role / side from joined_history (players_enriched) ---
    if {"role", "side"}.issubset(players_enriched.columns):
        role_small = (
            players_enriched[
                ["fixture_id", "team_id", "player_id", "role", "side"]
            ]
            .drop_duplicates()
            .copy()
        )

        projections_df = projections_df.merge(
            role_small,
            on=["fixture_id", "team_id", "player_id"],
            how="left",
        )

        # Fill missing with sensible defaults
        projections_df["role"] = projections_df["role"].fillna("UNK")
        projections_df["side"] = projections_df["side"].fillna("C")
    else:
        print(
            "[RUN_HISTORY] WARNING: 'role'/'side' columns missing from joined_history; "
            "history projections will not include roles."
        )


    # NOTE: We are NOT adding explanations here. The file name keeps
    # "_with_explanations" so the rest of the code doesn’t need changing,
    # but explanation columns will simply be missing / empty.
    return projections_df


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------


def main():
    # 1) Load raw history tables from CSV
    history_tables = load_history_tables()

    # 2) Build global baselines / team style / role-vs-role from history
    globals_ = build_global_inputs(history_tables)

    # 3) Project all history fixtures
    full_proj = project_all_history_fixtures(history_tables, globals_)

    if full_proj.empty:
        print("[RUN_HISTORY] No projections to save. Exiting.")
        return

    # 4) Save to CSV for calibration + analysis
    hist_dir = get_history_dir()
    out_path = hist_dir / "match_projections_epl_2025_26_history_with_explanations.csv"
    full_proj.to_csv(out_path, index=False)
    print(f"[RUN_HISTORY] Saved history projections to {out_path}")


if __name__ == "__main__":
    main()