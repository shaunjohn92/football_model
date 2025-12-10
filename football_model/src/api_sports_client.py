"""
api_sports_client.py

Thin but robust wrapper around the API-FOOTBALL / API-Sports v3 football API.

Docs:
    https://www.api-football.com/documentation-v3

Core ideas:
    - Single client class: APISportsClient
    - Handles headers, base URL, timeouts, pagination, and basic error checking
    - Exposes a small public surface tailored to a player-prop projection engine:
        * Fixtures (by date / team / league / id)
        * Fixture statistics (team-level)
        * Fixture lineups
        * Team statistics (season-level)
        * Player statistics (season + per-player)
        * Convenience helpers to compute "last N games" for teams / players
    - Provides optional DataFrame helpers for downstream modelling.

Usage:

    from api_sports_client import APISportsClient

    client = APISportsClient(api_key="YOUR_KEY_HERE")

    fixtures = client.get_fixtures(date="2025-12-04", league_id=39, season=2025)
    stats = client.get_fixture_statistics(fixture_id=14025150)
    lineups = client.get_fixture_lineups(fixture_id=14025150)

    player_stats = client.get_player_statistics(
        player_id=1234,
        league_id=39,
        season=2025
    )

    # As DataFrames
    df_team_stats = client.team_statistics_df(team_id=33, league_id=39, season=2025)
"""

from __future__ import annotations

import logging
import time
import os
from typing import Any, Dict, Iterable, List, Optional, Union

import requests

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - optional
    pd = None  # type: ignore


logger = logging.getLogger(__name__)


class APISportsError(Exception):
    """Custom exception for API-Sports / API-FOOTBALL errors."""


class APISportsClient:
    """
    Small, opinionated client for the API-FOOTBALL v3 API (api-football.com).

    Parameters
    ----------
    api_key : str
        Your API key from API-Sports / API-FOOTBALL.
    base_url : str, optional
        Base URL for the API. Default is the official v3 endpoint.
    timeout : int or float, optional
        Timeout (seconds) for HTTP requests. Default is 20.
    rate_limit_per_minute : int, optional
        Soft rate limit; if set, the client will sleep a little between requests
        to avoid hitting hard API limits. Default None = no client-side throttling.
    session : requests.Session, optional
        Optional custom session; if not provided, a new Session is created.

    Notes
    -----
    - Authentication is done via the `x-apisports-key` header.
    - All public methods return the raw JSON `response` list from the API
      (not the top-level object), unless otherwise documented.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://v3.football.api-sports.io",
        timeout: Union[int, float] = 20,
        rate_limit_per_minute: Optional[int] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        """
        Create an API client.

        api_key can be provided directly OR read from the environment variable
        API_FOOTBALL_KEY. This makes it easy to use:

            client = APISportsClient()

        after doing:

            set API_FOOTBALL_KEY=YOUR_KEY_HERE   (on Windows)
        """
        # If no api_key was passed in, try to read it from the environment
        if api_key is None:
            api_key = os.getenv("API_FOOTBALL_KEY")

        if not api_key:
            raise ValueError(
                "API key not provided. Either pass api_key=... to "
                "APISportsClient() or set the API_FOOTBALL_KEY environment "
                "variable."
            )

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.rate_limit_per_minute = rate_limit_per_minute

        self._last_request_ts: float = 0.0
        self._min_interval = (
            60.0 / rate_limit_per_minute if rate_limit_per_minute else 0.0
        )

        self.session.headers.update(
            {
                "x-apisports-key": self.api_key,
                "Accept": "application/json",
            }
        )


    # ------------------------------------------------------------------
    # Low-level request helper
    # ------------------------------------------------------------------
    def _throttle(self) -> None:
        """Respect a soft client-side rate limit, if configured."""
        if not self._min_interval:
            return
        now = time.time()
        elapsed = now - self._last_request_ts
        if elapsed < self._min_interval:
            sleep_for = self._min_interval - elapsed
            logger.debug("Throttling API call for %.3f seconds", sleep_for)
            time.sleep(max(sleep_for, 0))
        self._last_request_ts = time.time()

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform an HTTP request against the API and return the parsed JSON.

        Parameters
        ----------
        method : {"GET", "POST"}
            HTTP method.
        endpoint : str
            Path relative to base_url, e.g. "/fixtures".
        params : dict, optional
            Query parameters.

        Returns
        -------
        dict
            Parsed JSON from the API.

        Raises
        ------
        APISportsError
            On network problems or API error responses.
        """
        self._throttle()

        url = f"{self.base_url}{endpoint}"
        logger.debug("API request: %s %s params=%s", method, url, params)

        try:
            resp = self.session.request(
                method=method.upper(),
                url=url,
                params=params,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:  # pragma: no cover - network
            raise APISportsError(f"Network error: {exc}") from exc

        if not resp.ok:
            raise APISportsError(
                f"API HTTP {resp.status_code} for {url} – body: {resp.text}"
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise APISportsError(f"Invalid JSON response from {url}") from exc

        # API-FOOTBALL standard structure:
        # {"get": "...", "parameters": {...}, "errors": [], "results": ..., "paging": {...}, "response": [...]}
        errors = data.get("errors") or []
        if isinstance(errors, dict):
            # sometimes errors is a dict
            errors = list(errors.values())
        if errors:
            raise APISportsError(f"API returned errors: {errors}")

        return data

    def _paged_get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Handle API-FOOTBALL's `paging` object and gather all pages into a list.

        Strategy:
            - First call WITHOUT any 'page' parameter (some endpoints don't accept it).
            - Inspect the `paging` object.
            - If there are multiple pages, request page=2..N.
        """
        if params is None:
            params = {}

        # 1) First request WITHOUT 'page'
        data = self._request("GET", endpoint, params)
        response_items = data.get("response") or []

        if not isinstance(response_items, list):
            raise APISportsError(
                f"Unexpected response format for {endpoint}: {response_items}"
            )

        all_items: List[Dict[str, Any]] = list(response_items)

        paging = data.get("paging") or {}
        current = paging.get("current", 1)
        total = paging.get("total", current)

        # If there's only one page, we're done
        if not isinstance(total, int) or total <= current:
            return all_items

        # 2) More pages – request page 2..total
        for page in range(current + 1, total + 1):
            page_params = dict(params)
            page_params["page"] = page

            data = self._request("GET", endpoint, page_params)
            response_items = data.get("response") or []

            if not isinstance(response_items, list):
                raise APISportsError(
                    f"Unexpected response format for {endpoint}, page {page}: {response_items}"
                )

            all_items.extend(response_items)

        return all_items


    # ------------------------------------------------------------------
    # Public API wrappers
    # ------------------------------------------------------------------

    # --- Leagues / teams ------------------------------------------------
    def get_leagues(
        self,
        country: Optional[str] = None,
        name: Optional[str] = None,
        current: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch leagues (competitions).

        Parameters
        ----------
        country : str, optional
            Filter by country name (e.g. "England").
        name : str, optional
            Filter by league name (e.g. "Premier League").
        current : bool, optional
            If True/False, filter by current season flag.

        Returns
        -------
        list of dict
            List of league objects.
        """
        params: Dict[str, Any] = {}
        if country:
            params["country"] = country
        if name:
            params["name"] = name
        if current is not None:
            params["current"] = "true" if current else "false"

        return self._paged_get("/leagues", params)

    def get_teams(
        self,
        league_id: Optional[int] = None,
        season: Optional[int] = None,
        search: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch teams, optionally filtered by league, season, or search term.

        Parameters
        ----------
        league_id : int, optional
            League ID.
        season : int, optional
            Season year (e.g., 2025).
        search : str, optional
            Text search for team name.

        Returns
        -------
        list of dict
        """
        params: Dict[str, Any] = {}
        if league_id is not None:
            params["league"] = league_id
        if season is not None:
            params["season"] = season
        if search:
            params["search"] = search

        return self._paged_get("/teams", params)

    def get_team_statistics(
        self,
        team_id: int,
        league_id: int,
        season: int,
    ) -> Dict[str, Any]:
        """
        Fetch season-level team statistics (shots, goals, cards, etc.).

        Parameters
        ----------
        team_id : int
            Team ID.
        league_id : int
            League ID.
        season : int
            Season year.

        Returns
        -------
        dict
            A single statistics object (API returns a list; we take first).
        """
        params = {"team": team_id, "league": league_id, "season": season}
        data = self._request("GET", "/teams/statistics", params)
        resp = data.get("response")
        if isinstance(resp, dict):
            return resp
        if isinstance(resp, list) and resp:
            return resp[0]
        return {}

    # --- Fixtures / matches ---------------------------------------------
    def get_fixtures(
        self,
        fixture_id: Optional[int] = None,
        date: Optional[str] = None,
        league_id: Optional[int] = None,
        season: Optional[int] = None,
        team_id: Optional[int] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch fixtures (matches).

        Parameters
        ----------
        fixture_id : int, optional
            Specific fixture ID.
        date : str, optional
            Specific date in 'YYYY-MM-DD' format.
        league_id : int, optional
            League ID.
        season : int, optional
            Season year.
        team_id : int, optional
            Filter by team.
        from_date : str, optional
            Start date for range ('YYYY-MM-DD').
        to_date : str, optional
            End date for range ('YYYY-MM-DD').
        status : str, optional
            Status filter (e.g. 'NS', 'FT', 'LIVE').

        Returns
        -------
        list of dict
        """
        params: Dict[str, Any] = {}
        if fixture_id is not None:
            params["id"] = fixture_id
        if date:
            params["date"] = date
        if league_id is not None:
            params["league"] = league_id
        if season is not None:
            params["season"] = season
        if team_id is not None:
            params["team"] = team_id
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        if status:
            params["status"] = status

        return self._paged_get("/fixtures", params)

    def get_team_last_fixtures(
        self,
        team_id: int,
        league_id: Optional[int] = None,
        season: Optional[int] = None,
        last_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Convenience wrapper: fetch last N fixtures for a team.

        Parameters
        ----------
        team_id : int
        league_id : int, optional
        season : int, optional
        last_n : int, default 5

        Returns
        -------
        list of dict
        """
        params: Dict[str, Any] = {"team": team_id, "last": last_n}
        if league_id is not None:
            params["league"] = league_id
        if season is not None:
            params["season"] = season
        return self._paged_get("/fixtures", params)

    def get_fixture_statistics(
        self,
        fixture_id: int,
    ) -> List[Dict[str, Any]]:
        """
        Fetch team-level statistics for a given fixture.

        Parameters
        ----------
        fixture_id : int

        Returns
        -------
        list of dict
            Typically length 2: one entry per team, each with stats list.
        """
        params = {"fixture": fixture_id}
        return self._paged_get("/fixtures/statistics", params)

    def get_fixture_lineups(
        self,
        fixture_id: int,
    ) -> List[Dict[str, Any]]:
        """
        Fetch lineups for a given fixture (starting XI + bench + formation).

        Parameters
        ----------
        fixture_id : int

        Returns
        -------
        list of dict
            Usually 2 entries (home/away) containing team and players.
        """
        params = {"fixture": fixture_id}
        return self._paged_get("/fixtures/lineups", params)

    # --- Players / player stats -----------------------------------------
    def get_players(
        self,
        team_id: Optional[int] = None,
        league_id: Optional[int] = None,
        season: Optional[int] = None,
        search: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch players, optionally filtered by team, league, season, or search.

        Returns
        -------
        list of dict
        """
        params: Dict[str, Any] = {}
        if team_id is not None:
            params["team"] = team_id
        if league_id is not None:
            params["league"] = league_id
        if season is not None:
            params["season"] = season
        if search:
            params["search"] = search

        return self._paged_get("/players", params)

    def get_player_statistics(
        self,
        player_id: int,
        league_id: int,
        season: int,
        team_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch statistics for a given player in a given league & season.

        Parameters
        ----------
        player_id : int
        league_id : int
        season : int
        team_id : int, optional
            Optional filter by team; useful when a player has moved clubs.

        Returns
        -------
        list of dict
            Each entry corresponds to a competition/team combination and
            includes aggregated stats and sometimes per-match logs.
        """
        params: Dict[str, Any] = {
            "player": player_id,
            "league": league_id,
            "season": season,
        }
        if team_id is not None:
            params["team"] = team_id

        return self._paged_get("/players/statistics", params)

    def get_player_last_fixtures(
        self,
        player_id: int,
        team_id: int,
        league_id: Optional[int] = None,
        season: Optional[int] = None,
        last_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Convenience wrapper: approximate "last N games" for a player.

        Strategy:
            - Fetch last N fixtures for the team.
            - For each fixture, fetch lineups and/or events (not implemented here)
              and filter for that player.

        NOTE:
            API-FOOTBALL does not have a single dedicated endpoint for player
            last-N stats, so this method is just a structural placeholder for
            your future expansion. For now, it returns the team's last fixtures.
        """
        # This is intentionally simple for v1; extend later if needed.
        return self.get_team_last_fixtures(
            team_id=team_id,
            league_id=league_id,
            season=season,
            last_n=last_n,
        )

    # ------------------------------------------------------------------
    # DataFrame helpers (optional but convenient)
    # ------------------------------------------------------------------
    def _ensure_pandas(self) -> None:
        if pd is None:
            raise ImportError(
                "pandas is required for DataFrame helpers. "
                "Install with `pip install pandas`."
            )

    def fixtures_df(
        self,
        fixtures: Iterable[Dict[str, Any]],
    ) -> "pd.DataFrame":
        """
        Convert a list of fixtures (from get_fixtures / get_team_last_fixtures)
        into a tidy DataFrame.

        Columns include:
            - fixture_id
            - date
            - status_short
            - league_id, league_name
            - home_team_id, home_team_name
            - away_team_id, away_team_name
            - goals_home, goals_away
        """
        self._ensure_pandas()
        rows: List[Dict[str, Any]] = []

        for f in fixtures:
            fixture_info = f.get("fixture", {})
            league_info = f.get("league", {})
            teams_info = f.get("teams", {})
            goals_info = f.get("goals", {})

            row = {
                "fixture_id": fixture_info.get("id"),
                "date": fixture_info.get("date"),
                "status_short": (fixture_info.get("status") or {}).get("short"),
                "league_id": league_info.get("id"),
                "league_name": league_info.get("name"),
                "home_team_id": (teams_info.get("home") or {}).get("id"),
                "home_team_name": (teams_info.get("home") or {}).get("name"),
                "away_team_id": (teams_info.get("away") or {}).get("id"),
                "away_team_name": (teams_info.get("away") or {}).get("name"),
                "goals_home": goals_info.get("home"),
                "goals_away": goals_info.get("away"),
            }
            rows.append(row)

        return pd.DataFrame(rows)  # type: ignore

    def team_statistics_df(
        self,
        team_id: int,
        league_id: int,
        season: int,
    ) -> "pd.DataFrame":
        """
        Fetch team statistics and flatten some common subfields into a DataFrame.

        The raw object from the API has nested structures like:
            stats["shots"]["total"]["home"], ["away"], ["for"], ["against"], etc.

        For v1, we keep it minimal and store the raw JSON in one row;
        you can expand/normalize this to your liking later.
        """
        self._ensure_pandas()
        stats = self.get_team_statistics(team_id, league_id, season)
        return pd.DataFrame([stats])  # type: ignore

    def fixture_statistics_df(
        self,
        fixture_id: int,
    ) -> "pd.DataFrame":
        """
        Fetch fixture statistics and return one row per team,
        with a column 'stats' containing the raw stats list.

        You can then expand the stats list (type/value pairs) as needed.
        """
        self._ensure_pandas()
        stats_list = self.get_fixture_statistics(fixture_id)
        rows: List[Dict[str, Any]] = []

        for entry in stats_list:
            team_info = entry.get("team") or {}
            stats = entry.get("statistics") or []
            rows.append(
                {
                    "fixture_id": fixture_id,
                    "team_id": team_info.get("id"),
                    "team_name": team_info.get("name"),
                    "stats": stats,
                }
            )

        return pd.DataFrame(rows)  # type: ignore

    def lineups_df(
        self,
        fixture_id: int,
    ) -> "pd.DataFrame":
        """
        Fetch fixture lineups and return a DataFrame of players with positions,
        including whether they started or came from the bench.

        Columns:
            - fixture_id
            - team_id, team_name
            - player_id, player_name
            - number
            - position
            - grid (pos on pitch, e.g. "4:2")
            - is_starting
        """
        self._ensure_pandas()
        lineups = self.get_fixture_lineups(fixture_id)
        rows: List[Dict[str, Any]] = []

        for lineup in lineups:
            team_info = lineup.get("team") or {}
            team_id = team_info.get("id")
            team_name = team_info.get("name")

            # starting XI
            for p in lineup.get("startXI") or []:
                player = p.get("player") or {}
                rows.append(
                    {
                        "fixture_id": fixture_id,
                        "team_id": team_id,
                        "team_name": team_name,
                        "player_id": player.get("id"),
                        "player_name": player.get("name"),
                        "number": player.get("number"),
                        "position": player.get("pos"),
                        "grid": player.get("grid"),
                        "is_starting": True,
                    }
                )

            # substitutes
            for p in lineup.get("substitutes") or []:
                player = p.get("player") or {}
                rows.append(
                    {
                        "fixture_id": fixture_id,
                        "team_id": team_id,
                        "team_name": team_name,
                        "player_id": player.get("id"),
                        "player_name": player.get("name"),
                        "number": player.get("number"),
                        "position": player.get("pos"),
                        "grid": player.get("grid"),
                        "is_starting": False,
                    }
                )

        return pd.DataFrame(rows)  # type: ignore


# ----------------------------------------------------------------------
# If you like quick manual testing, you can run this module directly:
# ----------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover - manual test helper
    import os

    logging.basicConfig(level=logging.INFO)

    api_key = os.environ.get("API_FOOTBALL_KEY", "")
    if not api_key:
        raise SystemExit(
            "Set API_FOOTBALL_KEY environment variable to your API key "
            "or modify __main__ in api_sports_client.py"
        )

    client = APISportsClient(api_key=api_key, rate_limit_per_minute=50)

    # Example: get today's Premier League fixtures (league 39)
    fixtures = client.get_fixtures(league_id=39, season=2025, date="2025-12-04")
    print(f"Found {len(fixtures)} fixtures")

    if fixtures:
        fx_id = fixtures[0]["fixture"]["id"]
        print("Testing fixture id:", fx_id)

        stats = client.get_fixture_statistics(fixture_id=fx_id)
        print("Fixture statistics entries:", len(stats))

        lineups = client.get_fixture_lineups(fixture_id=fx_id)
        print("Lineups entries:", len(lineups))

        if pd is not None:  # pragma: no cover
            df_fx = client.fixtures_df(fixtures)
            print(df_fx.head())
