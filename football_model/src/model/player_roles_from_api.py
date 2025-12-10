"""
player_roles_from_api.py – Football Model v1

Build canonical player roles for EPL 2025–26 directly from API-Football /players.

Output CSV:
    data/history/player_roles_epl_2025_26.csv

Columns:
    player_id
    team_id
    player_name
    primary_role   # our model role: GK / CB / CM / ST / etc.
    api_position   # raw API position string, e.g. 'Defender', 'Midfielder', ...

These roles are "typical" positions for the season. We'll later compare them
to per-fixture roles to detect players used out of position.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..api_sports_client import APISportsClient, APISportsError

EPL_LEAGUE_ID = 39
SEASON_2025_26 = 2025  # project rule: 2025 → 2025–26 season


# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_history_dir() -> Path:
    root = get_project_root()
    hist = root / "data" / "history"
    hist.mkdir(parents=True, exist_ok=True)
    return hist


# ---------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------


def _map_api_position_to_primary_role(api_pos: Optional[str]) -> str:
    """
    Map API-Football 'position' / 'games.position' strings to our model roles.

    API positions are generally one of:
        "Goalkeeper", "Defender", "Midfielder", "Attacker"

    We'll keep the mapping simple for now – this is canonical/typical role.
    More detailed per-match roles still come from lineups if we want.
    """
    if not api_pos:
        return "UNK"

    pos = api_pos.strip().lower()

    if pos.startswith("goalkeeper"):
        return "GK"
    if pos.startswith("keeper"):
        return "GK"

    if pos.startswith("defender"):
        # For now treat all defenders as CB in the canonical sense.
        # (We can later split into FB/CB using lineups history.)
        return "CB"

    if pos.startswith("midfielder"):
        # Generic midfielder – central by default.
        return "CM"

    if pos.startswith("attacker") or pos.startswith("forward"):
        # Generic attacker – central striker by default.
        # Later we'll use lineups to split into ST vs W.
        return "ST"

    return "UNK"


def fetch_player_roles_from_api(
    league_id: int = EPL_LEAGUE_ID,
    season: int = SEASON_2025_26,
) -> pd.DataFrame:
    """
    Call /players for the given league+season and build a player_roles table.

    We use APISportsClient._paged_get to handle pagination.
    """
    client = APISportsClient()

    print(
        f"[PLAYER ROLES] Fetching /players for league_id={league_id}, "
        f"season={season} ..."
    )

    try:
        raw_players: List[Dict[str, Any]] = client._paged_get(
            "/players", {"league": league_id, "season": season}
        )
    except APISportsError as e:
        raise SystemExit(f"[PLAYER ROLES] API error: {e}") from e

    print(f"[PLAYER ROLES] Retrieved {len(raw_players)} player records (raw rows).")

    rows: List[Dict[str, Any]] = []

    for item in raw_players:
        player = item.get("player") or {}
        stats_list = item.get("statistics") or []

        if not stats_list:
            continue

        # Statistics can contain multiple competitions – keep only this league+season
        # just to be safe.
        relevant_stats: List[Dict[str, Any]] = []
        for st in stats_list:
            league = st.get("league") or {}
            if (
                int(league.get("id", 0)) == league_id
                and int(league.get("season", 0)) == season
            ):
                relevant_stats.append(st)

        if not relevant_stats:
            continue

        # Take the first relevant stats block (should all be same team for league season)
        st0 = relevant_stats[0]
        team = st0.get("team") or {}
        games = st0.get("games") or {}

        player_id = player.get("id")
        team_id = team.get("id")
        player_name = player.get("name")

        # API position: prefer games.position, fall back to player.position
        api_position = games.get("position") or player.get("position")
        primary_role = _map_api_position_to_primary_role(api_position)

        if player_id is None or team_id is None:
            # Skip any incomplete rows – shouldn't be many.
            continue

        rows.append(
            {
                "player_id": int(player_id),
                "team_id": int(team_id),
                "player_name": str(player_name) if player_name is not None else "",
                "primary_role": primary_role,
                "api_position": api_position,
            }
        )

    df = pd.DataFrame(rows)

    # Drop duplicates (some players may appear more than once if they changed teams;
    # for now we keep the first occurrence).
    df = (
        df.sort_values(["player_id", "team_id"])
        .drop_duplicates(subset=["player_id", "team_id"], keep="first")
        .reset_index(drop=True)
    )

    print(f"[PLAYER ROLES] Built roles table with {len(df)} unique player/team rows.")
    return df


def write_player_roles_csv() -> Path:
    """
    Fetch roles from API and write to data/history/player_roles_epl_2025_26.csv
    (overwriting any existing file).
    """
    history_dir = get_history_dir()
    out_path = history_dir / "player_roles_epl_2025_26.csv"

    df = fetch_player_roles_from_api()
    df.to_csv(out_path, index=False)

    print(f"[PLAYER ROLES] Wrote roles CSV to: {out_path}")
    return out_path


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main() -> None:
    api_key = os.getenv("APISPORTS_API_KEY")
    if not api_key:
        raise SystemExit(
            "APISPORTS_API_KEY environment variable is not set.\n"
            "Set it in your environment or .env file before running."
        )

    write_player_roles_csv()


if __name__ == "__main__":
    main()