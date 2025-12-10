"""
weighted_player_baselines.py – Football Model v1

Build player baselines (per-90 stats) using recency weights
(exponential decay over time) from the joined history table.

This is a *drop-in* replacement for the old "per-player per90"
logic, but it:
  - uses ALL games in history
  - down-weights older games
  - returns the exact schema expected by match_projection.py:

    player_id
    team_id
    player_name
    role
    minutes_total
    games
    avg_minutes
    shots_total_per90
    shots_on_per90
    fouls_committed_per90
    fouls_drawn_per90
    yellow_cards_per90
"""

from __future__ import annotations

from typing import List

import pandas as pd


REQUIRED_COLUMNS: List[str] = [
    "fixture_id",
    "team_id",
    "player_id",
    "player_name",
    "minutes",
    "date",
    "role",
    "shots_total",
    "shots_on",
    "fouls_committed",
    "fouls_drawn",
    "yellow_cards",
]


def _check_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            "[WEIGHTED_BASELINES] Missing required columns in joined history: "
            + ", ".join(missing)
        )


def _compute_recency_weights(
    df: pd.DataFrame,
    half_life_days: float = 60.0,
) -> pd.Series:
    """
    Build an exponential decay weight for each row based on match date.

    weight = 0.5 ** (days_ago / half_life_days)

    So:
      - games ~half_life_days days ago get weight ~0.5
      - games 2*half_life_days days ago get weight ~0.25
      - very recent games have weight close to 1.0
    """
    dates = pd.to_datetime(df["date"], errors="coerce")
    max_date = dates.max()

    # If dates are broken, just use weight=1.0 everywhere
    if pd.isna(max_date):
        return pd.Series(1.0, index=df.index)

    # Days since game
    days_ago = (max_date - dates).dt.days.fillna(0).clip(lower=0)

    # Exponential decay
    weights = 0.5 ** (days_ago / float(half_life_days))
    # Avoid exact zeros
    weights = weights.clip(lower=0.05)
    return weights


def build_player_baselines_by_role_weighted(
    joined_history: pd.DataFrame,
    half_life_days: float = 60.0,
    min_total_minutes: float = 90.0,
) -> pd.DataFrame:
    """
    Build per-player per-90 baselines using exponential recency weighting.

    Parameters
    ----------
    joined_history : pd.DataFrame
        Output of the history join step; must contain:
            fixture_id, team_id, player_id, player_name, role,
            minutes, date,
            shots_total, shots_on, fouls_committed, fouls_drawn, yellow_cards
    half_life_days : float
        Recency half-life in days. Smaller = more weight on recent games.
    min_total_minutes : float
        Minimum minutes required to keep a player in the output.

    Returns
    -------
    pd.DataFrame
        One row per (player_id, team_id, role) with the schema expected
        by match_projection.py.
    """
    df = joined_history.copy()
    _check_required_columns(df)

    # Clean minutes
    df["minutes"] = df["minutes"].fillna(0.0).astype(float)

    # Only keep rows where the player actually appeared
    df = df[df["minutes"] > 0].copy()
    if df.empty:
        raise RuntimeError("[WEIGHTED_BASELINES] No rows with minutes > 0 in history.")

    # Recency weight per row
    df["weight"] = _compute_recency_weights(df, half_life_days=half_life_days)

    # Weighted minutes and weighted stat columns
    df["weighted_minutes"] = df["minutes"] * df["weight"]

    metric_cols = [
        "shots_total",
        "shots_on",
        "fouls_committed",
        "fouls_drawn",
        "yellow_cards",
    ]
    for col in metric_cols:
        df[f"{col}_w"] = df[col].fillna(0.0) * df["weight"]

    # Group by player / team / role
    group_cols = ["player_id", "team_id", "player_name", "role"]

    def games_count(s: pd.Series) -> int:
        # Each row = one appearance
        return int((s > 0).sum())

    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            minutes_total=("minutes", "sum"),
            games=("minutes", games_count),
            weighted_minutes=("weighted_minutes", "sum"),
            shots_total_w=("shots_total_w", "sum"),
            shots_on_w=("shots_on_w", "sum"),
            fouls_committed_w=("fouls_committed_w", "sum"),
            fouls_drawn_w=("fouls_drawn_w", "sum"),
            yellow_cards_w=("yellow_cards_w", "sum"),
        )
        .reset_index()
    )

    # Filter out players with very little time on pitch
    agg = agg[agg["minutes_total"] >= float(min_total_minutes)].copy()
    if agg.empty:
        raise RuntimeError(
            "[WEIGHTED_BASELINES] After filtering on min_total_minutes, no players remain."
        )

    def per90(row, num_col: str) -> float:
        wm = row["weighted_minutes"]
        if wm <= 0:
            return 0.0
        return float(row[num_col]) * 90.0 / float(wm)

    agg["avg_minutes"] = agg["minutes_total"] / agg["games"].clip(lower=1)

    agg["shots_total_per90"] = agg.apply(
        lambda r: per90(r, "shots_total_w"), axis=1
    )
    agg["shots_on_per90"] = agg.apply(
        lambda r: per90(r, "shots_on_w"), axis=1
    )
    agg["fouls_committed_per90"] = agg.apply(
        lambda r: per90(r, "fouls_committed_w"), axis=1
    )
    agg["fouls_drawn_per90"] = agg.apply(
        lambda r: per90(r, "fouls_drawn_w"), axis=1
    )
    agg["yellow_cards_per90"] = agg.apply(
        lambda r: per90(r, "yellow_cards_w"), axis=1
    )

    # Final column order to match what match_projection.py expects
    cols_out = [
        "player_id",
        "team_id",
        "player_name",
        "role",
        "minutes_total",
        "games",
        "avg_minutes",
        "shots_total_per90",
        "shots_on_per90",
        "fouls_committed_per90",
        "fouls_drawn_per90",
        "yellow_cards_per90",
    ]

    # Ensure they all exist
    for c in cols_out:
        if c not in agg.columns:
            agg[c] = 0.0

    result = agg[cols_out].copy()
    result = result.sort_values(["team_id", "player_name"]).reset_index(drop=True)

    print(
        "[WEIGHTED_BASELINES] Built weighted player baselines "
        f"for {len(result)} players (half_life_days={half_life_days})"
    )

    return result