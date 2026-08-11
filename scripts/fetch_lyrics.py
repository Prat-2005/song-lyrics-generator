import time
import json
import csv
import requests
import logging
from pathlib import Path
from src.config import CACHE_DIR
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# Configuration
DATA_DIR = Path("data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def fetch_lrclib(artist: str, track: str) -> Optional[str]:
    """Fetch lyrics from LRCLIB API."""
    # Try /api/get first for exact match
    url = "https://lrclib.net/api/get"
    params = {
        "artist_name": artist,
        "track_name": track
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            lyrics = data.get("plainLyrics")
            if lyrics:
                return lyrics
    except Exception as e:
        logger.debug(f"LRCLIB get error for {artist} - {track}: {e}")

    # Fallback to /api/search if get fails
    search_url = "https://lrclib.net/api/search"
    try:
        response = requests.get(search_url, params=params, timeout=10)
        if response.status_code == 200:
            results = response.json()
            if results and len(results) > 0:
                # Return the first match
                lyrics = results[0].get("plainLyrics")
                if lyrics:
                    return lyrics
    except Exception as e:
        logger.debug(f"LRCLIB search error for {artist} - {track}: {e}")

    return None

def get_cache_path(artist: str, track: str) -> Path:
    """Generate a safe filename for the cache, ensuring cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_artist = "".join(c for c in artist if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')
    safe_track = "".join(c for c in track if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')
    return CACHE_DIR / f"{safe_artist}_{safe_track}.json"

def main(subset_artist: Optional[str] = None, limit_per_artist: Optional[int] = None):
    """Fetch lyrics for songs in the data directory."""
    csv_files = sorted(DATA_DIR.glob("*.csv"))

    total_stats = {"lrclib": 0, "none": 0}
    processed_songs = 0

    for csv_file in csv_files:
        artist_name = csv_file.stem
        if subset_artist and artist_name != subset_artist:
            continue

        logger.info(f"Processing artist: {artist_name}")

        with open(csv_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if limit_per_artist is not None and i >= limit_per_artist:
                    break

                track = row.get('Song') or row.get('Track') or row.get('Title')
                if not track:
                    continue

                processed_songs += 1
                cache_path = get_cache_path(artist_name, track)

                if cache_path.exists():
                    continue

                # Use LRCLIB as the only source
                lyrics = fetch_lrclib(artist_name, track)

                if lyrics:
                    # Save to cache
                    with open(cache_path, 'w', encoding='utf-8') as cache_f:
                        json.dump({
                            "artist": artist_name,
                            "track": track,
                            "lyrics": lyrics,
                            "source": "lrclib"
                        }, cache_f)
                    total_stats["lrclib"] += 1
                    logger.info(f"Fetched {track} from lrclib")
                else:
                    total_stats["none"] += 1
                    logger.warning(f"No lyrics found for {track}")

                # Short delay to be polite to LRCLIB API
                time.sleep(0.5)

    print_stats(total_stats, processed_songs)

def print_stats(stats, total):
    print("\n" + "="*30)
    print("LYRICS FETCHING STATS")
    print("="*30)
    print(f"Total songs processed: {total}")
    print(f"LRCLIB:     {stats['lrclib']}")
    print(f"None:       {stats['none']}")
    print("="*30)

if __name__ == "__main__":
    import sys
    subset = sys.argv[1] if len(sys.argv) > 1 else None
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

    main(subset_artist=subset, limit_per_artist=limit)
