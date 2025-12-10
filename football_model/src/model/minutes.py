"""
minutes.py – very simple minutes model for Football Model v1.

Goal:
    For a given fixture, estimate expected minutes for each player in the lineup.

Approach (v1, deliberately simple):
    - Use API-Football lineups to get starting XI + subs.
    - For each player, fetch season statistics and compute:
          expected_minutes = total_minutes / max(1, appearances)
      capped at 90.
    - If we can't fetch stats (or they are missing), use fallbacks:
          90 minutes for starters, 20 minutes for subs.

Later we can:
    - Incorporate last-N games.
    - Differentiate “regular starters” vs rotation players.
    - Use positions and substitution patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None  # type: ignore

from ..api_sports_client import APISportsClient


@dataclass
class FixtureContext:
    fixture_id: int
    league_id: int
    season: int


def _extract_fixture_context(fixture: Dict[str, Any]) -> FixtureContext:
    """Build a small context object (league, season, fixture id) from a fixture dict."""
    fixture_info = fixture.get("fixture", {}) or {}
    league_info = fixture.get("league", {}) or {}

    fx_id = fixture_info.get("id")
    league_id = league_info.get("id")
    season = league_info.get("season")

    if fx_id is None or league_id is None or season is None:
        raise ValueError(f"Missing fixture/league/season info in fixture={fixture!r}")

    return FixtureContext(
        fixture_id=fx_id,
        league_id=league_id,
        season=season,
    )


def _get_fixture_by_id(
    client: APISportsClient,
    fixture_id: int,
) -> Dict[str, Any]:
    """Fetch a single fixture object by id (helper)."""
    fixtures = client.get_fixtures(fixture_id=fixture_id)
    if not fixtures:
        raise ValueError(f"No fixture found for id={fixture_id}")
    return fixtures[0]


def _safe_get_player_season_minutes(
    client: APISportsClient,
    player_id: int,
    league_id: int,
    season: int,
    team_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fetch season statistics for a player, and extract minutes & appearances.

    Returns:
        dict with keys:
            - total_minutes
            - appearances
        or None if we cannot determine them.
    """
    try:
        stats_list = client.get_player_statistics(
            player_id=player_id,
            league_id=league_id,
            season=season,
            team_id=team_id,
        )
    except Exception:
        return None

    if not stats_list:
        return None

    # stats_list is usually a list where each entry has 'statistics': [ ... ]
    # We just take the first 'statistics' entry for v1.
    entry = stats_list[0]
    statistics = entry.get("statistics") or []
    if not statistics:
        return None

    st0 = statistics[0]
    games = st0.get("games") or {}
    total_minutes = games.get("minutes")
    appearances = games.get("appearences") or games.get("appearances")  # typo-safe

    if total_minutes is None:
        return None

    if appearances is None:
        # fallback: treat appearances as matches played if missing
        appearances = games.get("lineups") or 1

    return {
        "total_minutes": total_minutes,
        "appearances": appearances,
    }


def estimate_minutes_for_fixture(
    client: APISportsClient,
    fixture_id: int,
) -> "pd.DataFrame":
    """
    Estimate expected minutes for all players in the lineups of a fixture.

    Returns
    -------
    pandas.DataFrame with columns:
        - fixture_id
        - league_id
        - season
        - team_id
        - team_name
        - player_id
        - player_name
        - is_starting
        - expected_minutes
        - source  (e.g., "season_avg", "fallback_90", "fallback_20")
    """
    if pd is None:
        raise ImportError(
            "pandas is required for estimate_minutes_for_fixture. "
            "Install with `pip install pandas`."
        )

    # 1) Get fixture context (league & season)
    fixture = _get_fixture_by_id(client, fixture_id)
    ctx = _extract_fixture_context(fixture)

    # 2) Get lineups for the fixture
    lineups_df = client.lineups_df(fixture_id=fixture_id)
    if lineups_df.empty:
        raise ValueError(f"No lineups available for fixture {fixture_id}")

    rows: List[Dict[str, Any]] = []

    for _, row in lineups_df.iterrows():
        team_id = int(row["team_id"])
        team_name = row["team_name"]
        player_id = row["player_id"]
        player_name = row["player_name"]
        is_starting = bool(row["is_starting"])

        if player_id is None:
            # If there's no player id, we can't look up stats; use fallback minutes.
            if is_starting:
                expected_minutes = 90.0
                source = "fallback_90_no_player_id"
            else:
                expected_minutes = 20.0
                source = "fallback_20_no_player_id"
        else:
            # Try to fetch season stats for this player
            stats = _safe_get_player_season_minutes(
                client=client,
                player_id=int(player_id),
                league_id=ctx.league_id,
                season=ctx.season,
                team_id=team_id,
            )
            if stats is None:
                # fallback if we can't get stats
                if is_starting:
                    expected_minutes = 90.0
                    source = "fallback_90_no_stats"
                else:
                    expected_minutes = 20.0
                    source = "fallback_20_no_stats"
            else:
                total_minutes = stats["total_minutes"]
                appearances = max(1, stats["appearances"])
                expected_minutes = float(total_minutes) / float(appearances)

                # cap between 10 and 95 to avoid weird outliers
                expected_minutes = max(10.0, min(expected_minutes, 95.0))
                source = "season_avg"

                # If the player is in the starting XI, slightly bias upwards
                if is_starting and expected_minutes < 70.0:
                    expected_minutes = (expected_minutes + 70.0) / 2.0
                    source = "season_avg_boosted_for_starter"

                # If the player is a sub, bias downwards
                if not is_starting and expected_minutes > 45.0:
                    expected_minutes = (expected_minutes + 30.0) / 2.0
                    source = "season_avg_damped_for_sub"

        rows.append(
            {
                "fixture_id": ctx.fixture_id,
                "league_id": ctx.league_id,
                "season": ctx.season,
                "team_id": team_id,
                "team_name": team_name,
                "player_id": player_id,
                "player_name": player_name,
                "is_starting": is_starting,
                "expected_minutes": expected_minutes,
                "source": source,
            }
        )

    return pd.DataFrame(rows)
