"""
run_fixture_minutes_demo.py

Run a simple minutes model for the first fixture *today* that has usable lineups.

Usage (from project root):

    python -m src.run_fixture_minutes_demo
"""

from datetime import date

from .config import load_api_sports_config
from .api_sports_client import APISportsClient
from .model.minutes import estimate_minutes_for_fixture


def main() -> None:
    cfg = load_api_sports_config()
    client = APISportsClient(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        rate_limit_per_minute=cfg.rate_limit_per_minute,
    )

    today = date.today().isoformat()
    print(f"Fetching fixtures for today: {today}")
    fixtures = client.get_fixtures(date=today)

    if not fixtures:
        print("No fixtures found for today – nothing to do.")
        return

    chosen_fixture = None

    for f in fixtures:
        fx_id = f["fixture"]["id"]
        home = f["teams"]["home"]["name"]
        away = f["teams"]["away"]["name"]
        league_name = f["league"]["name"]

        # Build the DataFrame the same way the minutes model does
        lineups_df = client.lineups_df(fixture_id=fx_id)

        if lineups_df.empty:
            print(
                f"Skipping fixture {fx_id}: {home} vs {away} "
                f"({league_name}) – no usable lineups (no players in startXI/substitutes)"
            )
            continue

        print(
            f"Using fixture {fx_id}: {home} vs {away} "
            f"({league_name}) – {len(lineups_df)} player rows found"
        )
        chosen_fixture = f
        break

    if chosen_fixture is None:
        print(
            "No fixtures with usable lineups available for today – "
            "cannot run minutes model right now."
        )
        return

    fx_id = chosen_fixture["fixture"]["id"]

    try:
        df_minutes = estimate_minutes_for_fixture(client, fixture_id=fx_id)
    except Exception as exc:
        print(f"Failed to estimate minutes for fixture {fx_id}: {exc}")
        return

    # Show a quick summary in the console
    print("\nEstimated minutes (top 30 rows):\n")
    df_sorted = df_minutes.sort_values(
        by=["team_id", "expected_minutes"], ascending=[True, False]
    )
    print(
        df_sorted[
            ["team_name", "player_name", "is_starting", "expected_minutes", "source"]
        ].head(30)
    )


if __name__ == "__main__":
    main()
