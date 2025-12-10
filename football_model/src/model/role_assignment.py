"""
role_assignment.py

Builds tactical player roles (CB, FB, W, ST, etc.) from API-Football lineups.

Steps:
- For each fixture, fetch lineups (formation + grid + position).
- Infer a role for each player in that fixture.
- Collapse to a primary role per (player_id, team_id).
- Save to data/history/role_assignments_epl_2025_26_history.csv
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import os
import requests
import pandas as pd


API_BASE = "https://v3.football.api-sports.io"


def get_history_dir() -> Path:
    root = Path(__file__).resolve().parents[2]  # project root (where app.py is)
    hist = root / "data" / "history"
    hist.mkdir(parents=True, exist_ok=True)
    return hist


# ---------------------------------------------------------------------
# Low-level API helpers
# ---------------------------------------------------------------------


def _get_session(api_key: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"x-apisports-key": api_key})
    return session


def fetch_lineups_for_fixture(api_key: str, fixture_id: int) -> dict:
    """
    Call /fixtures/lineups for a single fixture.
    Returns the JSON 'response' list (one item per team).
    """
    session = _get_session(api_key)
    url = f"{API_BASE}/fixtures/lineups"
    resp = session.get(url, params={"fixture": fixture_id}, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", [])


# ---------------------------------------------------------------------
# Role inference logic
# ---------------------------------------------------------------------


def _parse_grid(grid: str) -> Tuple[int, int] | Tuple[None, None]:
    """
    API-Football 'grid' is like '3:4' (column:row).
    We convert to integers (x, y).
    """
    if not isinstance(grid, str) or ":" not in grid:
        return None, None
    try:
        x_str, y_str = grid.split(":")
        return int(x_str), int(y_str)
    except ValueError:
        return None, None


def infer_role(formation: str, pos: str, grid: str) -> str:
    """
    Convert (formation, pos, grid) into a tactical role.

    This is a heuristic first pass. It won't be perfect, but it will
    distinguish:
        GK, CB, FB, DM, CM, AM, W, ST
    """
    pos = (pos or "").upper()
    x, y = _parse_grid(grid)

    # Goalkeeper
    if pos == "G":
        return "GK"

    # Fall back if we can't read grid
    if x is None or y is None:
        if pos == "D":
            return "CB"
        if pos == "M":
            return "CM"
        if pos == "F":
            return "ST"
        return "UNK"

    # Rough vertical bands (1 back, 10 very advanced)
    # We'll treat y <= 3 as deep, 4–7 midfield, >= 8 attacking.
    deep = y <= 3
    mid = 4 <= y <= 7
    high = y >= 8

    # Horizontal: 1 = left, high number = right, middle = central.
    if x <= 3:
        side = "L"
    elif x >= 8:
        side = "R"
    else:
        side = "C"

    # Defenders
    if pos == "D":
        if high:
            # Very advanced defender (often wingback)
            return f"FB_{side}"
        if side == "C":
            return "CB"
        else:
            return f"FB_{side}"

    # Midfielders
    if pos == "M":
        if deep:
            return "DM"
        if high and side != "C":
            return f"W_{side}"
        return "CM"

    # Forwards
    if pos == "F":
        if side == "C":
            return "ST"
        else:
            return f"W_{side}"

    # Unknown / other
    return "UNK"


# ---------------------------------------------------------------------
# Build player roles from fixtures
# ---------------------------------------------------------------------


def build_player_roles_from_fixtures(api_key: str) -> pd.DataFrame:
    """
    For all fixtures in fixtures_epl_2025_26_sample.csv, fetch lineups and
    infer roles for each player in each match. Collapse to primary role per
    (player_id, team_id).
    """
    history_dir = get_history_dir()
    fixtures_path = history_dir / "fixtures_epl_2025_26_history.csv"

    if not fixtures_path.exists():
        raise FileNotFoundError(f"Fixtures file not found: {fixtures_path}")

    fixtures_df = pd.read_csv(fixtures_path)
    fixture_ids = fixtures_df["fixture_id"].dropna().unique().tolist()

    print(f"[ROLES] Building from {len(fixture_ids)} fixtures")

    # For each (player_id, team_id) collect all inferred roles
    role_bag: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
    player_names: Dict[Tuple[int, int], str] = {}

    for idx, fid in enumerate(fixture_ids, start=1):
        print(f"[ROLES] Fetching lineups for fixture {fid} ({idx}/{len(fixture_ids)})")
        try:
            lineups = fetch_lineups_for_fixture(api_key, int(fid))
        except Exception as e:
            print(f"[ROLES]   Failed to fetch lineups for fixture {fid}: {e}")
            continue

        for team_entry in lineups:
            team = team_entry.get("team", {})
            team_id = team.get("id")
            if team_id is None:
                continue

            formation = team_entry.get("formation", "")
            players = team_entry.get("startXI", []) + team_entry.get("substitutes", [])

            for p in players:
                player_info = p.get("player", {})
                player_id = player_info.get("id")
                player_name = player_info.get("name")
                pos = player_info.get("pos")  # 'G', 'D', 'M', 'F'
                grid = player_info.get("grid")  # '3:4'

                if player_id is None:
                    continue

                role = infer_role(formation, pos, grid)
                key = (int(player_id), int(team_id))
                role_bag[key][role] += 1
                if player_name:
                    player_names[key] = player_name

    rows = []
    for (player_id, team_id), counts in role_bag.items():
        if not counts:
            primary_role = "UNK"
        else:
            primary_role = counts.most_common(1)[0][0]

        rows.append(
            {
                "player_id": player_id,
                "team_id": team_id,
                "player_name": player_names.get((player_id, team_id), None),
                "primary_role": primary_role,
            }
        )

    roles_df = pd.DataFrame(rows)
    roles_df = roles_df.sort_values(["team_id", "player_name"]).reset_index(drop=True)

    # Save
    out_path = history_dir / "role_assignments_epl_2025_26_history.csv"
    roles_df.to_csv(out_path, index=False)
    print(f"[ROLES] Saved {len(roles_df)} player roles to {out_path}")

    return roles_df


def build_player_roles_cli():
    """
    CLI entry point.

    Usage:
        Set API_FOOTBALL_KEY env var, then run:
            python -m src.model.role_assignment
    """
    api_key = os.environ.get("API_FOOTBALL_KEY")
    if not api_key:
        raise RuntimeError(
            "Please set API_FOOTBALL_KEY environment variable to your API-Football key."
        )
    build_player_roles_from_fixtures(api_key=api_key)


if __name__ == "__main__":
    build_player_roles_cli()