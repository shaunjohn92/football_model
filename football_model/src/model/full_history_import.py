"""
full_history_import.py

Pull full-season EPL 2025–26 fixtures + per-player match stats
from API-FOOTBALL and save to CSVs:

    data/history/fixtures_epl_2025_26_history.csv
    data/history/player_match_stats_epl_2025_26_history.csv

Requires:
    - Environment variable API_FOOTBALL_KEY set to your API key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import os
import time
import requests
import pandas as pd


API_BASE = "https://v3.football.api-sports.io"
LEAGUE_ID = 39      # EPL
SEASON = 2024       # use 2024 here; we will treat it as 2025-26 in the model


def get_project_root() -> Path:
    # this file is in src/model, so parents[2] is the project root
    return Path(__file__).resolve().parents[2]


def get_history_dir() -> Path:
    root = get_project_root()
    hist = root / "data" / "history"
    hist.mkdir(parents=True, exist_ok=True)
    return hist


def _get_session(api_key: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"x-apisports-key": api_key})
    return s


# ---------------------------------------------------------------------
# Step 1 – Fetch all finished fixtures in the league
# ---------------------------------------------------------------------


def fetch_all_fixtures(api_key: str) -> pd.DataFrame:
    """
    Fetch all finished fixtures for EPL 2025–26 via /fixtures.
    status=FT gives completed matches (no upcoming).
    Handles pagination.
    """
    session = _get_session(api_key)
    page = 1
    rows: List[Dict] = []

    while True:
        print(f"[FULL_HISTORY] Fetching fixtures page {page}...")
        resp = session.get(
            f"{API_BASE}/fixtures",
            params={
               "league": LEAGUE_ID,
                "season": SEASON,
                # Completed matches only (as per API-Football tutorial)
                "status": "FT-AET-PEN",
                "page": page,
            },
            timeout=20,
        )


        resp.raise_for_status()
        data = resp.json()

        fixtures = data.get("response", [])
        if not fixtures:
            break

        for fx in fixtures:
            fixture = fx.get("fixture", {})
            league = fx.get("league", {})
            teams = fx.get("teams", {})
            goals = fx.get("goals", {})

            row = {
                "fixture_id": fixture.get("id"),
                "date": fixture.get("date"),
                "status_short": fixture.get("status", {}).get("short"),
                "league_id": league.get("id"),
                "league_name": league.get("name"),
                "season": league.get("season"),
                "round": league.get("round"),
                "home_team_id": teams.get("home", {}).get("id"),
                "home_team_name": teams.get("home", {}).get("name"),
                "away_team_id": teams.get("away", {}).get("id"),
                "away_team_name": teams.get("away", {}).get("name"),
                "home_goals": goals.get("home"),
                "away_goals": goals.get("away"),
            }
            rows.append(row)

        paging = data.get("paging", {})
        total_pages = paging.get("total", page)
        if page >= total_pages:
            break
        page += 1
        time.sleep(0.25)  # polite delay

    df = pd.DataFrame(rows)
    print(f"[FULL_HISTORY] Retrieved {len(df)} finished fixtures.")
    return df


# ---------------------------------------------------------------------
# Step 2 – Fetch per-player stats for each fixture
# ---------------------------------------------------------------------


def fetch_player_stats_for_fixture(
    session: requests.Session,
    fixture_id: int,
    home_away_map: Dict[int, Tuple[int, int]],
) -> List[Dict]:
    """
    Fetch all player stats for a given fixture via /fixtures/players.

    home_away_map: {fixture_id: (home_team_id, away_team_id)}
    """
    resp = session.get(
        f"{API_BASE}/fixtures/players",
        params={"fixture": fixture_id},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    teams_data = data.get("response", [])

    rows: List[Dict] = []

    home_id, away_id = home_away_map.get(fixture_id, (None, None))

    for team_entry in teams_data:
        team = team_entry.get("team", {})
        team_id = team.get("id")
        if team_id is None:
            continue

        # Determine opponent
        if home_id is not None and away_id is not None:
            if team_id == home_id:
                opponent_id = away_id
            elif team_id == away_id:
                opponent_id = home_id
            else:
                opponent_id = None
        else:
            opponent_id = None

        players = team_entry.get("players", [])
        for p in players:
            player_info = p.get("player", {})
            stats_list = p.get("statistics", []) or []
            if not stats_list:
                continue
            stats = stats_list[0]  # first entry = this competition/match

            games = stats.get("games", {}) or {}
            shots = stats.get("shots", {}) or {}
            fouls = stats.get("fouls", {}) or {}
            cards = stats.get("cards", {}) or {}

            row = {
                "fixture_id": fixture_id,
                "team_id": team_id,
                "opponent_team_id": opponent_id,
                "player_id": player_info.get("id"),
                "player_name": player_info.get("name"),
                "minutes": games.get("minutes"),
                "is_substitute": games.get("substitute"),
                "position": games.get("position"),  # G/D/M/F
                "rating": games.get("rating"),
                "shots_total": shots.get("total"),
                "shots_on": shots.get("on"),
                "fouls_drawn": fouls.get("drawn"),
                "fouls_committed": fouls.get("committed"),
                "yellow_cards": cards.get("yellow"),
                "red_cards": cards.get("red"),
            }
            rows.append(row)

    return rows


def fetch_all_player_stats(fixtures_df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    session = _get_session(api_key)
    rows: List[Dict] = []

    # Map fixture -> (home, away)
    home_away_map: Dict[int, Tuple[int, int]] = {}
    for _, r in fixtures_df.iterrows():
        fid = int(r["fixture_id"])
        home_away_map[fid] = (
            int(r["home_team_id"]) if not pd.isna(r["home_team_id"]) else None,
            int(r["away_team_id"]) if not pd.isna(r["away_team_id"]) else None,
        )

    fixture_ids = fixtures_df["fixture_id"].dropna().astype(int).tolist()

    for idx, fid in enumerate(fixture_ids, start=1):
        print(f"[FULL_HISTORY] Fetching player stats for fixture {fid} ({idx}/{len(fixture_ids)})")
        try:
            rows.extend(fetch_player_stats_for_fixture(session, fid, home_away_map))
        except Exception as e:
            print(f"[FULL_HISTORY]   Failed for fixture {fid}: {e}")
        time.sleep(0.25)  # avoid hammering the API

    df = pd.DataFrame(rows)
    print(f"[FULL_HISTORY] Retrieved {len(df)} player stat rows.")
    return df


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------


def main():
    api_key = os.environ.get("API_FOOTBALL_KEY")
    if not api_key:
        raise RuntimeError(
            "Please set the API_FOOTBALL_KEY environment variable to your API-Football key."
        )

    hist_dir = get_history_dir()

    # 1) Fixtures
    fixtures_df = fetch_all_fixtures(api_key)
    fixtures_path = hist_dir / "fixtures_epl_2025_26_history.csv"
    fixtures_df.to_csv(fixtures_path, index=False)
    print(f"[FULL_HISTORY] Saved fixtures to {fixtures_path}")

    # If we got no fixtures, stop here (nothing else to do)
    if fixtures_df.empty or "fixture_id" not in fixtures_df.columns:
        print("[FULL_HISTORY] No fixtures returned by API – skipping player stats step.")
        return

    # 2) Player stats
    stats_df = fetch_all_player_stats(fixtures_df, api_key)
    stats_path = hist_dir / "player_match_stats_epl_2025_26_history.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"[FULL_HISTORY] Saved player stats to {stats_path}")


if __name__ == "__main__":
    main()