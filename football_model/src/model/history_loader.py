"""
history_loader.py – Football Model v1

History data loader for one league (EPL 2025–26) using API-Football v3.

It builds four tables:

- fixtures
- player_match_stats
- subs
- role_assignments

Season lock:
    We treat API-Football season=2025 as the 2025–26 football season.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd  # type: ignore
except ImportError:  # safety guard
    pd = None  # type: ignore

import time

try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    tqdm = None  # type: ignore

from ..api_sports_client import APISportsClient, APISportsError


# League + season constants
EPL_LEAGUE_ID = 39
SEASON_2025_26 = 2025  # by project rule: this means the 2025–26 season


class HistoryDataLoader:
    """
    Build historical tables for one league & season using API-Football.

    Public entrypoint:
        load_league_history(league_id, season, ...)

    Returns a dict with four DataFrames:
        {
            "fixtures": fixtures_df,
            "player_match_stats": player_match_stats_df,
            "subs": subs_df,
            "role_assignments": role_assignments_df,
        }
    """

    def __init__(self, client: APISportsClient):
        self.client = client

    def _paged_get_with_retry(
        self,
        endpoint: str,
        params: Dict[str, Any],
        max_retries: int = 5,
        base_wait: int = 30,
    ):
        """
        Call APISportsClient._paged_get with simple retry/backoff for rate limits.

        If the API returns a "Too many requests" error, we wait and retry
        up to `max_retries` times, with exponential-ish backoff.
        """
        attempt = 0
        while attempt < max_retries:
            try:
                return self.client._paged_get(endpoint, params)
            except APISportsError as e:
                msg = str(e)
                if "Too many requests" in msg:
                    attempt += 1
                    wait_seconds = base_wait * attempt
                    print(
                        f"[RATE LIMIT] {msg} – waiting {wait_seconds} seconds "
                        f"before retry {attempt}/{max_retries}..."
                    )
                    time.sleep(wait_seconds)
                    continue
                # Non-rate-limit error: re-raise immediately
                raise

        # If we exhaust retries, raise a final error
        raise APISportsError(
            f"Exceeded max retries ({max_retries}) for endpoint {endpoint} "
            f"with params {params}"
        )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def load_league_history(
        self,
        league_id: int = EPL_LEAGUE_ID,
        season: int = SEASON_2025_26,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        finished_only: bool = True,
        max_fixtures: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load fixtures + per-fixture detail tables for a league & season.

        Parameters
        ----------
        league_id : int
            League ID (EPL=39).
        season : int
            Season numeric tag used by API-Football.
            NOTE: by project convention, season=2025 => 2025–26.
        from_date : str, optional
            ISO date 'YYYY-MM-DD' to filter fixtures from.
        to_date : str, optional
            ISO date 'YYYY-MM-DD' to filter fixtures to.
        finished_only : bool
            If True, only include completed fixtures.
        max_fixtures : int, optional
            If set, only process the first N fixtures (good for testing/API limits).

        Returns
        -------
        dict[str, pd.DataFrame]
        """
        if pd is None:
            raise RuntimeError(
                "pandas is not available. Install it in your virtual environment."
            )

        print(
            f"[HISTORY] Loading league_id={league_id}, season={season}, "
            f"from={from_date}, to={to_date}, finished_only={finished_only}, "
            f"max_fixtures={max_fixtures}"
        )

        # 1) Get the list of fixtures for this league + season
        print(f"[FETCH FIXTURES] Loading fixtures {from_date} → {to_date} ...")
        fixtures = self._fetch_fixtures(
            league_id=league_id,
            season=season,
            from_date=from_date,
            to_date=to_date,
            finished_only=finished_only,
        )
        print(f"[FETCH FIXTURES] Retrieved {len(fixtures)} fixtures.")

        if fixtures.empty:
            # Return empty shells with correct columns if no fixtures
            return {
                "fixtures": fixtures,
                "player_match_stats": self._empty_player_match_stats_df(),
                "subs": self._empty_subs_df(),
                "role_assignments": self._empty_role_assignments_df(),
            }

        # 2) For safety, limit how many fixtures we process in one run
        if max_fixtures is not None:
            fixtures = fixtures.head(max_fixtures).copy()
            print(
                f"[HISTORY] Limiting to first {len(fixtures)} fixtures "
                f"(max_fixtures={max_fixtures})."
            )

        # These lists will hold row dictionaries for each table
        player_rows: List[Dict[str, Any]] = []
        subs_rows: List[Dict[str, Any]] = []
        role_rows: List[Dict[str, Any]] = []

        # 3) Loop over each (limited) fixture and fetch details
        total_fixtures = len(fixtures)
        iter_rows = fixtures.iterrows()
        if tqdm is not None:
            # Wrap with tqdm progress bar if available
            iter_rows = tqdm(
                iter_rows,
                total=total_fixtures,
                desc="Fixtures",
                unit="fixture",
            )

        for idx, (fix_idx, fix) in enumerate(iter_rows, start=1):
            fixture_id = int(fix["fixture_id"])
            home_team_id = int(fix["home_team_id"])
            away_team_id = int(fix["away_team_id"])

            if tqdm is None:
                # If no tqdm, print our own progress line
                print(
                    f"[HISTORY] Processing fixture {fixture_id} "
                    f"({idx}/{total_fixtures}) – {away_team_id} @ {home_team_id}"
                )

            # 3a) Player stats per fixture
            players_for_fixture = self._fetch_fixture_players(
                fixture_id=fixture_id,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
            )
            player_rows.extend(players_for_fixture)

            # 3b) Substitutions from events
            subs_for_fixture = self._fetch_fixture_subs(
                fixture_id=fixture_id,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
            )
            subs_rows.extend(subs_for_fixture)

            # 3c) Role assignments from lineups grid
            roles_for_fixture = self._fetch_fixture_role_assignments(
                fixture_id=fixture_id,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
            )
            role_rows.extend(roles_for_fixture)

            time.sleep(1)

        # 4) Turn lists of dicts into DataFrames
        player_match_stats = pd.DataFrame(player_rows)
        subs = pd.DataFrame(subs_rows)
        role_assignments = pd.DataFrame(role_rows)

        # Guarantee the schema is stable even if there were no rows
        if player_match_stats.empty:
            player_match_stats = self._empty_player_match_stats_df()
        if subs.empty:
            subs = self._empty_subs_df()
        if role_assignments.empty:
            role_assignments = self._empty_role_assignments_df()

        return {
            "fixtures": fixtures,
            "player_match_stats": player_match_stats,
            "subs": subs,
            "role_assignments": role_assignments,
        }

    # ------------------------------------------------------------------
    # FIXTURES
    # ------------------------------------------------------------------
    def _fetch_fixtures(
        self,
        league_id: int,
        season: int,
        from_date: Optional[str],
        to_date: Optional[str],
        finished_only: bool,
    ) -> pd.DataFrame:
        """
        Fetch all fixtures for the league+season, with optional date/status filters.

        Returns columns:
            fixture_id, league_id, season, date,
            home_team_id, away_team_id, status,
            home_goals, away_goals
        """
        fixtures_raw = self.client.get_fixtures(
            league_id=league_id,
            season=season,
            from_date=from_date,
            to_date=to_date,
        )

        rows: List[Dict[str, Any]] = []
        allowed_statuses = {"FT", "AET", "PEN"} if finished_only else None

        for item in fixtures_raw:
            fixture = item.get("fixture", {})
            league = item.get("league", {})
            teams = item.get("teams", {})
            goals = item.get("goals", {})

            status_short = (fixture.get("status") or {}).get("short")

            # Keep only finished fixtures if requested
            if finished_only and allowed_statuses is not None:
                if status_short not in allowed_statuses:
                    continue

            rows.append(
                {
                    "fixture_id": fixture.get("id"),
                    "league_id": league.get("id"),
                    # Season lock: treat this as 2025–26 even if API just says 2025
                    "season": season,
                    "date": fixture.get("date"),
                    "home_team_id": (teams.get("home") or {}).get("id"),
                    "away_team_id": (teams.get("away") or {}).get("id"),
                    "status": status_short,
                    "home_goals": goals.get("home"),
                    "away_goals": goals.get("away"),
                }
            )

        fixtures_df = pd.DataFrame(rows)

        if fixtures_df.empty:
            return pd.DataFrame(
                columns=[
                    "fixture_id",
                    "league_id",
                    "season",
                    "date",
                    "home_team_id",
                    "away_team_id",
                    "status",
                    "home_goals",
                    "away_goals",
                ]
            )

        # Drop any rows missing essential IDs
        fixtures_df = fixtures_df.dropna(
            subset=["fixture_id", "home_team_id", "away_team_id"]
        )

        fixtures_df["fixture_id"] = fixtures_df["fixture_id"].astype(int)
        fixtures_df["league_id"] = fixtures_df["league_id"].astype(int)
        fixtures_df["home_team_id"] = fixtures_df["home_team_id"].astype(int)
        fixtures_df["away_team_id"] = fixtures_df["away_team_id"].astype(int)

        return fixtures_df

    # ------------------------------------------------------------------
    # PLAYER MATCH STATS – from /fixtures/players
    # ------------------------------------------------------------------
    def _fetch_fixture_players(
        self,
        fixture_id: int,
        home_team_id: int,
        away_team_id: int,
    ) -> List[Dict[str, Any]]:
        """
        Fetch player statistics for a fixture and flatten to player_match_stats rows.

        Schema:
            fixture_id
            team_id
            opponent_team_id
            player_id
            player_name
            minutes
            position
            role
            side
            band
            shots_total
            shots_on
            fouls_committed
            fouls_drawn
            yellow_cards
            red_cards
            duels_total
            duels_won
            tackles
            dribbles
            subbed_on
            subbed_off
            sub_on_min
            sub_off_min
        """
        # Fetch player stats with retry/backoff around the low-level helper.
        print(f"   [PLAYERS] Fetching players for fixture {fixture_id} ...")
        players_raw = self._paged_get_with_retry(
            "/fixtures/players", {"fixture": fixture_id}
        )

        rows: List[Dict[str, Any]] = []

        for team_block in players_raw:
            team = team_block.get("team") or {}
            team_id = team.get("id")
            team_is_home = team_id == home_team_id
            opponent_team_id = away_team_id if team_is_home else home_team_id

            for player_block in team_block.get("players") or []:
                player = player_block.get("player") or {}
                stats_list = player_block.get("statistics") or []
                if not stats_list:
                    continue
                stats = stats_list[0]  # one entry per match

                minutes = stats.get("games", {}).get("minutes")
                pos = stats.get("games", {}).get("position")

                # Shots / fouls / cards
                shots = stats.get("shots", {}) or {}
                fouls = stats.get("fouls", {}) or {}
                cards = stats.get("cards", {}) or {}
                duels = stats.get("duels", {}) or {}
                tackles = stats.get("tackles", {}) or {}
                dribbles = stats.get("dribbles", {}) or {}

                is_substitute = stats.get("games", {}).get("substitute", False)

                # We'll fill role/side/band from lineups later; defaults here
                rows.append(
                    {
                        "fixture_id": fixture_id,
                        "team_id": team_id,
                        "opponent_team_id": opponent_team_id,
                        "player_id": player.get("id"),
                        "player_name": player.get("name"),
                        "minutes": minutes,
                        "position": pos,
                        "role": None,
                        "side": None,
                        "band": None,
                        "shots_total": shots.get("total"),
                        "shots_on": shots.get("on"),
                        "fouls_committed": fouls.get("committed"),
                        "fouls_drawn": fouls.get("drawn"),
                        "yellow_cards": cards.get("yellow"),
                        "red_cards": cards.get("red"),
                        "duels_total": duels.get("total"),
                        "duels_won": duels.get("won"),
                        "tackles": tackles.get("total"),
                        "dribbles": dribbles.get("attempts"),
                        "subbed_on": bool(is_substitute),
                        "subbed_off": False,  # updated when joining with subs
                        "sub_on_min": None,
                        "sub_off_min": None,
                    }
                )

        return rows

    # ------------------------------------------------------------------
    # SUBSTITUTIONS – from /fixtures/events
    # ------------------------------------------------------------------
    def _fetch_fixture_subs(
        self,
        fixture_id: int,
        home_team_id: int,
        away_team_id: int,
    ) -> List[Dict[str, Any]]:
        """
        Fetch substitution events for a fixture and flatten to subs rows.

        We read /fixtures/events and extract substitutions.

        Schema:
            fixture_id
            team_id
            player_out_id
            player_in_id
            minute
            role_out
            role_in
        """
        print(f"   [EVENTS] Fetching events/subs for fixture {fixture_id} ...")
        events_raw = self._paged_get_with_retry(
            "/fixtures/events", {"fixture": fixture_id}
        )

        rows: List[Dict[str, Any]] = []

        for ev in events_raw:
            if ev.get("type") != "subst":
                continue

            team = ev.get("team") or {}
            team_id = team.get("id")
            team_is_home = team_id == home_team_id
            opponent_team_id = away_team_id if team_is_home else home_team_id

            minute = ev.get("time", {}).get("elapsed")

            player_out = ev.get("player") or {}
            player_in = ev.get("assist") or {}

            # We'll enrich role_out/role_in later when we join with lineups
            rows.append(
                {
                    "fixture_id": fixture_id,
                    "team_id": team_id,
                    "opponent_team_id": opponent_team_id,
                    "player_out_id": player_out.get("id"),
                    "player_in_id": player_in.get("id"),
                    "minute": minute,
                    "role_out": None,
                    "role_in": None,
                }
            )

        return rows

    # ------------------------------------------------------------------
    # ROLE ASSIGNMENTS – from lineups grid
    # ------------------------------------------------------------------
    def _fetch_fixture_role_assignments(
        self,
        fixture_id: int,
        home_team_id: int,
        away_team_id: int,
    ) -> List[Dict[str, Any]]:
        """
        Fetch lineups for a fixture and derive role/side/band from grid.

        Schema:
            fixture_id
            team_id
            player_id
            position_string
            grid_x
            grid_y
            role
            role_profile
            side
            band
        """
        print(f"   [LINEUPS] Fetching lineups for fixture {fixture_id} ...")
        lineups = self._paged_get_with_retry(
        "/fixtures/lineups", {"fixture": fixture_id}
        )

        rows: List[Dict[str, Any]] = []

        for team_block in lineups:
            team = team_block.get("team") or {}
            team_id = team.get("id")

            for player_block in (team_block.get("startXI") or []) + (
                team_block.get("substitutes") or []
            ):
                player = player_block.get("player") or {}
                player_id = player.get("id")
                pos = player.get("pos")
                grid = player.get("grid")

                grid_x, grid_y = self._parse_grid(grid)
                role, side, band, role_profile = self._infer_role_from_grid(
                    pos, grid_x, grid_y
                )

                rows.append(
                    {
                        "fixture_id": fixture_id,
                        "team_id": team_id,
                        "player_id": player_id,
                        "position_string": pos,
                        "grid_x": grid_x,
                        "grid_y": grid_y,
                        "role": role,
                        "role_profile": role_profile,
                        "side": side,
                        "band": band,
                    }
                )

        return rows

    # ------------------------------------------------------------------
    # Helpers for role inference from grid
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_grid(grid: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
        if not grid or not isinstance(grid, str):
            return None, None
        try:
            x_str, y_str = grid.split(":")
            return int(x_str), int(y_str)
        except Exception:
            return None, None

    @staticmethod
    def _infer_role_from_grid(
        position: Optional[str],
        grid_x: Optional[int],
        grid_y: Optional[int],
    ) -> Tuple[str, str, str, str]:
        """
        Grid + coarse position ("G", "D", "M", "F") -> tactical role + side + band.

        Returns:
            role: one of GK/CB/FB/DM/CM/W/ST/UNK
            side: L/C/R/UNK
            band: deep/mid/high/UNK
            role_profile: human-readable text
        """
        # Defaults
        role = "UNK"
        side = "UNK"
        band = "UNK"
        role_profile = "unknown"

        pos = (position or "").upper()

        # --- Side from grid_x -------------------------------------------------
        if grid_x is not None:
            if grid_x <= 2:
                side = "L"
            elif grid_x >= 4:
                side = "R"
            else:
                side = "C"

        # --- Band from grid_y -------------------------------------------------
        if grid_y is not None:
            # API grid: 1 = deepest, larger = more advanced
            if grid_y <= 2:
                band = "deep"
            elif grid_y == 3:
                band = "mid"
            else:
                band = "high"

        # --- Role from coarse position + side/band ----------------------------
        if pos == "G":
            role = "GK"

        elif pos == "D":
            # Defenders
            if band == "high":
                # very advanced defender -> wing-back / attacking full-back
                role = "FB"
            elif side == "C":
                role = "CB"
            else:
                role = "FB"

        elif pos == "M":
            # Midfielders
            if band == "deep":
                role = "DM"
            elif band == "high" and side in {"L", "R"}:
                # wide high midfielder -> winger
                role = "W"
            else:
                # typical central / #8 / #10 area
                role = "CM"

        elif pos == "F":
            # Forwards
            if side == "C":
                role = "ST"
            else:
                role = "W"

        # --- Role profile text ------------------------------------------------
        if role == "GK":
            role_profile = "goalkeeper"
        elif role == "CB":
            role_profile = "central defender"
        elif role == "FB":
            role_profile = "full-back / wing-back"
        elif role == "DM":
            role_profile = "defensive midfielder"
        elif role == "CM":
            role_profile = "central / attacking midfielder"
        elif role == "W":
            role_profile = "wide forward / winger"
        elif role == "ST":
            role_profile = "striker / centre-forward"

        return role, side, band, role_profile

    # ------------------------------------------------------------------
    # Empty DF templates
    # ------------------------------------------------------------------
    @staticmethod
    def _empty_player_match_stats_df() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "fixture_id",
                "team_id",
                "opponent_team_id",
                "player_id",
                "player_name",
                "minutes",
                "position",
                "role",
                "side",
                "band",
                "shots_total",
                "shots_on",
                "fouls_committed",
                "fouls_drawn",
                "yellow_cards",
                "red_cards",
                "duels_total",
                "duels_won",
                "tackles",
                "dribbles",
                "subbed_on",
                "subbed_off",
                "sub_on_min",
                "sub_off_min",
            ]
        )

    @staticmethod
    def _empty_subs_df() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "fixture_id",
                "team_id",
                "opponent_team_id",
                "player_out_id",
                "player_in_id",
                "minute",
                "role_out",
                "role_in",
            ]
        )

    @staticmethod
    def _empty_role_assignments_df() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "fixture_id",
                "team_id",
                "player_id",
                "position_string",
                "grid_x",
                "grid_y",
                "role",
                "role_profile",
                "side",
                "band",
            ]
        )


# ----------------------------------------------------------------------
# Convenience CLI runner for quick testing
# ----------------------------------------------------------------------


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _get_history_dir() -> Path:
    root = _get_project_root()
    history_dir = root / "data" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir


if __name__ == "__main__":
    """
    When run as a script:

        python -m src.model.history_loader

    we fetch a sample of fixtures for EPL 2025–26 and save CSVs under data/history.
    """
    if pd is None:
        raise SystemExit(
            "pandas is required to run history_loader. "
            "Install it in your virtual environment."
        )

    api_key = os.getenv("APISPORTS_API_KEY")
    if not api_key:
        raise SystemExit(
            "APISPORTS_API_KEY environment variable is not set. "
            "Add it to your .env or environment."
        )

    client = APISportsClient()
    loader = HistoryDataLoader(client)

    # You can adjust from_date/to_date and max_fixtures below if needed
    data = loader.load_league_history(
        league_id=EPL_LEAGUE_ID,
        season=SEASON_2025_26,
        from_date="2025-08-01",   # start of season
        to_date="2026-06-30",     # well past end of season
        finished_only=True,
        max_fixtures=None,        # no artificial limit
        # max_fixtures=10,  # keep this commented for future quick tests
    )

    fixtures_df = data["fixtures"]
    player_stats_df = data["player_match_stats"]
    subs_df = data["subs"]
    roles_df = data["role_assignments"]

    print("Fixtures shape:", fixtures_df.shape)
    print("Player match stats shape:", player_stats_df.shape)
    print("Subs shape:", subs_df.shape)
    print("Role assignments shape:", roles_df.shape)

    history_dir = _get_history_dir()
    fixtures_path = history_dir / "fixtures_epl_2025_26_history.csv"
    player_path = history_dir / "player_match_stats_epl_2025_26_history.csv"
    subs_path = history_dir / "subs_epl_2025_26_history.csv"
    roles_path = history_dir / "role_assignments_epl_2025_26_history.csv"

    fixtures_df.to_csv(fixtures_path, index=False)
    player_stats_df.to_csv(player_path, index=False)
    subs_df.to_csv(subs_path, index=False)
    roles_df.to_csv(roles_path, index=False)

    print("\nSaved sample CSVs to:")
    print(f"   {fixtures_path}")
    print(f"   {player_path}")
    print(f"   {subs_path}")
    print(f"   {roles_path}")