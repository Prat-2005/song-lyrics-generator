"""Fetch a song's lyrics from LRCLIB and append it to the matching artist CSV."""

import logging
import sys
from pathlib import Path

import pandas as pd
import requests

from src.config import DATA_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LRCLIB_GET = "https://lrclib.net/api/get"
LRCLIB_SEARCH = "https://lrclib.net/api/search"


def _fetch(endpoint: str, params: dict) -> str | None:
    """Hit an LRCLIB endpoint and return plain lyrics, or None."""
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        # Handle both single-result and list-result responses
        payload = data[0] if isinstance(data, list) else data
        return payload.get("plainLyrics")
    except Exception as exc:
        logger.debug(f"LRCLIB request failed: {exc}")
        return None


def fetch_lyrics(artist: str, track: str) -> str | None:
    """Fetch plain lyrics for a track. Returns None if not found."""
    params = {"artist_name": artist, "track_name": track}

    # 1. Exact match
    lyrics = _fetch(LRCLIB_GET, params)
    if lyrics:
        logger.info(f"Fetched '{track}' by {artist} (exact match)")
        return lyrics

    # 2. Search fallback
    lyrics = _fetch(LRCLIB_SEARCH, params)
    if lyrics:
        logger.info(f"Fetched '{track}' by {artist} (search match)")
        return lyrics

    logger.warning(f"No lyrics found for '{track}' by {artist}")
    return None


def append_to_csv(artist: str, title: str, lyrics: str) -> None:
    """Append a song to that artist's CSV, creating it if needed."""
    csv_path = Path(DATA_PATH) / f"{artist.replace(' ', '')}.csv"
    row = pd.DataFrame([{"Artist": artist, "Title": title, "Lyric": lyrics}])

    existing = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    combined = pd.concat([existing, row], ignore_index=True)
    combined.to_csv(csv_path, index=False)

    logger.info(f"Saved '{title}' to {csv_path}")


def main(artist: str, track: str) -> None:
    lyrics = fetch_lyrics(artist, track)
    if lyrics:
        append_to_csv(artist, track, lyrics)
    else:
        logger.warning("Nothing saved — no lyrics found.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m scripts.fetch_lyrics <artist> <track>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])