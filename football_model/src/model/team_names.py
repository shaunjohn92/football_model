# src/model/team_names.py

from __future__ import annotations

import os
import time
from typing import Optional

import pandas as pd
import requests


API_BASE_URL = "https://v3.football.api-sports.io"
EPL_LEAGUE_ID = 39      # Premier League
EPL_SEASON = 2025       # 2025–26 season
OUTPUT_PATH = "data/history/teams_epl_2025_26.csv"


class APIFootballError(RuntimeError):
    pass


def _get_api_key() -> str:
    """
    Read API key from environment variable.

    You can also adapt this to read from your existing config system
    if you already have one for history_loader.
    """
    key = os.getenv("API_FOOTBALL_KEY")
    if not key:
        raise APIFootballError(
            "API_FOOTBALL_KEY environment variable not set. "
            "Set it to your API-Football key before running."
        )
    return key


def fetch_epl_teams(season: int = EPL_SEASON,
                    league_id: int = EPL_LEAGUE_ID,
                    max_retries: int = 3,
                    backoff_seconds: float = 2.0) -> pd.DataFrame:
    """
    Fetch /teams from API-Football for a given league + season and
    return a DataFrame with columns:
        team_id, team_name, short_name, logo_url
    """
    api_key = _get_api_key()

    url = f"{API_BASE_URL}/teams"
    params = {"league": league_id, "season": season}
    headers = {"x-apisports-key": api_key}

    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            if "response" not in data:
                raise APIFootballError(f"Unexpected response format: {data.keys()}")

            rows = []
            for item in data["response"]:
                team = item.get("team", {})
                team_id = team.get("id")
                name = team.get("name")
                code = team.get("code")    # usually 3-letter code like "ARS", "MUN"
                logo = team.get("logo")

                if team_id is None or name is None:
                    continue

                short_name = code or name  # fall back to full name if code missing

                rows.append(
                    {
                        "team_id": int(team_id),
                        "team_name": name,
                        "short_name": short_name,
                        "logo_url": logo,
                    }
                )

            if not rows:
                raise APIFootballError("No teams returned from API-Football /teams")

            df = pd.DataFrame(rows).drop_duplicates("team_id").sort_values("team_id")
            df.reset_index(drop=True, inplace=True)
            return df

        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt == max_retries:
                break
            time.sleep(backoff_seconds * attempt)

    raise APIFootballError(f"Failed to fetch teams after {max_retries} attempts: {last_err}")


def save_epl_teams_csv(
    output_path: str = OUTPUT_PATH,
    season: int = EPL_SEASON,
    league_id: int = EPL_LEAGUE_ID,
) -> pd.DataFrame:
    """
    Fetch EPL teams for the given season and save them as a CSV file
    at data/history/teams_epl_2025_26.csv (by default).
    """
    df = fetch_epl_teams(season=season, league_id=league_id)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def load_epl_teams(output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    """
    Convenience loader for the teams CSV, to be used by the app and other modules.
    """
    return pd.read_csv(output_path, dtype={"team_id": "int64"})