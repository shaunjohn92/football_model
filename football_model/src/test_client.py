"""
Quick sanity check that API-Sports works and our client is wired correctly.
Run with:

    python -m src.test_client

from the project root.
"""

from datetime import date

from .config import load_api_sports_config
from .api_sports_client import APISportsClient


def main() -> None:
    cfg = load_api_sports_config()
    client = APISportsClient(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        rate_limit_per_minute=cfg.rate_limit_per_minute,
    )

    # Use today's date so it works on the free plan window
    today = date.today().isoformat()
    print(f"Requesting fixtures for today: {today}")

    # Don’t filter by league/season yet – just grab whatever is available today
    fixtures = client.get_fixtures(date=today)
    print(f"Found {len(fixtures)} fixtures")

    if not fixtures:
        print("No fixtures found for today (API side). Try again on a match day.")
        return

    first = fixtures[0]
    fx_id = first["fixture"]["id"]
    home = first["teams"]["home"]["name"]
    away = first["teams"]["away"]["name"]
    print("Example fixture:", fx_id, f"{home} vs {away}")

    stats = client.get_fixture_statistics(fixture_id=fx_id)
    print("Fixture statistics entries:", len(stats))

    lineups = client.get_fixture_lineups(fixture_id=fx_id)
    print("Lineups entries:", len(lineups))


if __name__ == "__main__":
    main()
