"""
upcoming_fixtures.py – Football Model v1

Fetch upcoming EPL fixtures for the 2025–26 season from API-Football
and store them in a local CSV for the UI / projections to use.

Inputs:
    - API-Football via src.api_sports_client.APISportsClient

Outputs:
    data/history/fixtures_epl_2025_26_upcoming.csv

Notes:
    - This script only hits the /fixtures endpoint (cheap on rate limits).
    - It does NOT fetch lineups or player stats yet.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from src.api_sports_client import APISportsClient, APISportsError


# League + season constants for this project
EPL_LEAGUE_ID = 39          # Premier League
SEASON_2025_26 = 2025       # By convention: 2025 => 2025–26 season


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


# ----------------------------------------------------------------------
# Core fixture fetching / normalisation
# ----------------------------------------------------------------------


def fetch_upcoming_fixtures(
    client: APISportsClient,
    league_id: int = EPL_LEAGUE_ID,
    season: int = SEASON_2025_26,
    days_ahead: int = 3,
) -> List[Dict[str, Any]]:
    """
    Fetch upcoming fixtures for the next `days_ahead` days.

    We ask for fixtures whose status is 'NS' (not started).
    """
    today = date.today()
    date_from = today.isoformat()
    date_to = (today + timedelta(days=days_ahead)).isoformat()

    params = {
        "league": league_id,
        "season": season,
        "from": date_from,
        "to": date_to,
        "status": "NS",  # Not started
    }

    # We use the low-level _paged_get just like history_loader does.
    fixtures_raw = client._paged_get("/fixtures", params)
    return fixtures_raw


def normalise_fixtures(fixtures_raw: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert raw API-Football fixture objects into our fixtures schema:

        fixture_id
        league_id
        season
        date
        home_team_id
        away_team_id
        status
        home_goals
        away_goals
    """
    rows = []

    for item in fixtures_raw:
        fixture = item.get("fixture", {}) or {}
        league = item.get("league", {}) or {}
        teams = item.get("teams", {}) or {}
        home = teams.get("home", {}) or {}
        away = teams.get("away", {}) or {}
        goals = item.get("goals", {}) or {}

        rows.append(
            {
                "fixture_id": fixture.get("id"),
                "league_id": league.get("id"),
                "season": league.get("season"),
                "date": fixture.get("date"),
                "home_team_id": home.get("id"),
                "away_team_id": away.get("id"),
                "status": (fixture.get("status") or {}).get("short"),
                "home_goals": goals.get("home"),
                "away_goals": goals.get("away"),
            }
        )

    df = pd.DataFrame(rows)
    return df


# ----------------------------------------------------------------------
# Script entrypoint
# ----------------------------------------------------------------------


if __name__ == "__main__":
    client = APISportsClient()  # API key is read from your env as before

    try:
        fixtures_raw = fetch_upcoming_fixtures(client)
    except APISportsError as e:
        print("Error while fetching upcoming fixtures from API-Football:")
        print("  ", e)
        raise SystemExit(1)

    print(f"Fetched {len(fixtures_raw)} raw fixtures from the API.")

    fixtures_df = normalise_fixtures(fixtures_raw)
    print("Normalised upcoming fixtures shape:", fixtures_df.shape)

    if fixtures_df.empty:
        print("No upcoming fixtures found for the requested window.")
    else:
        print("\nSample rows:")
        print(fixtures_df.head().to_string(index=False))

    history_dir = get_history_dir()
    out_path = history_dir / "fixtures_epl_2025_26_upcoming.csv"
    fixtures_df.to_csv(out_path, index=False)

    print("\nSaved upcoming fixtures to:")
    print("  ", out_path)