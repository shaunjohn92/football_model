"""
config.py – central configuration for the football model.

For now this just handles the API-Sports key and a few basic settings.
Later we can add league mappings, default seasons, etc.
"""

import os
from dataclasses import dataclass


@dataclass
class APISportsConfig:
    api_key: str
    base_url: str = "https://v3.football.api-sports.io"
    rate_limit_per_minute: int = 50  # adjust based on your plan


def load_api_sports_config() -> APISportsConfig:
    """
    Load API-Sports config from environment variables.

    Set API key in your system or in a .env file (we'll use python-dotenv later):
        API_FOOTBALL_KEY=your_key_here
    """
    api_key = os.getenv("API_FOOTBALL_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "API_FOOTBALL_KEY environment variable not set. "
            "Set it to your API-Sports key."
        )
    return APISportsConfig(api_key=api_key)
