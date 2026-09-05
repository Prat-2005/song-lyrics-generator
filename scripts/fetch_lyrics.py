"""
Fetch a song's lyrics from LRCLIB and append it to the matching artist CSV
in data/.

There's no cache/ directory anymore. The CSVs in data/ are the dataset —
this script writes new songs straight into them instead of stashing a
separate JSON file per song that the retriever then had to know to read.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import requests

from src.config import DATA_PATH

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def fetch_lrclib(artist: str, track: str):
    """Fetch plain lyrics for a track from LRCLIB. Returns None if not found."""
    params = {"artist_name": artist, "track_name": track}

    try:
        response = requests.get("https://lrclib.net/api/get", params=params, timeout=10)
        if response.status_code == 200:
            lyrics = response.json().get("plainLyrics")
            if lyrics:
                logger.info(f"Fetched '{track}' by {artist} from LRCLIB (exact match)")
                return lyrics
    except Exception as e:
        logger.debug(f"LRCLIB exact-match lookup failed for {artist} - {track}: {e}")

    try:
        response = requests.get("https://lrclib.net/api/search", params=params, timeout=10)
        if response.status_code == 200:
            results = response.json()
            if results:
                lyrics = results[0].get("plainLyrics")
                if lyrics:
                    logger.info(f"Fetched '{track}' by {artist} from LRCLIB (search match)")
                    return lyrics
    except Exception as e:
        logger.debug(f"LRCLIB search lookup failed for {artist} - {track}: {e}")

    logger.warning(f"No lyrics found for '{track}' by {artist}")
    return None


def append_to_csv(artist: str, title: str, lyrics: str):
    """Append a fetched song to that artist's CSV in DATA_PATH, creating it if needed."""
    csv_path = Path(DATA_PATH) / f"{artist.replace(' ', '')}.csv"
    row = pd.DataFrame([{"Artist": artist, "Title": title, "Lyric": lyrics}])

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row

    combined.to_csv(csv_path, index=False)
    logger.info(f"Saved '{title}' to {csv_path}")


def main(artist: str, track: str):
    lyrics = fetch_lrclib(artist, track)
    if lyrics:
        append_to_csv(artist, track, lyrics)
    else:
        logger.warning("Nothing saved — no lyrics found.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m scripts.fetch_lyrics <artist> <track>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
