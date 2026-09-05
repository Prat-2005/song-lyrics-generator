import json
import requests
import logging
from pathlib import Path
from src.config import CACHE_DIR
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def get_cache_path(artist: str, track: str) -> Path:
    """Generate a safe filename for the cache, ensuring cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_artist = "".join(c for c in artist if c.isalnum()).strip().replace(' ', '_')
    safe_track = "".join(c for c in track if c.isalnum()).strip().replace(' ', '_')
    return CACHE_DIR / f"{safe_artist}_{safe_track}.json"

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

        cache_path = get_cache_path(artist, track)
            
        if lyrics:
            with open(cache_path, 'w', encoding='utf-8') as cache_f:
                json.dump({
                    "artist": artist,
                    "track": track,
                    "lyrics": lyrics,
                    "source": "lrclib"
              }, cache_f)
                                                    
            logger.info(f"Fetched {track} from lrclib")
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

                cache_path = get_cache_path(artist, track)

                if lyrics:
                    with open(cache_path, 'w', encoding='utf-8') as cache_f:
                        json.dump({
                            "artist": artist,
                            "track": track,
                            "lyrics": lyrics,
                            "source": "lrclib"
                       }, cache_f)
                                        
                    logger.info(f"Fetched {track} from lrclib")
                    return lyrics
                
                else:
                    logger.warning(f"No lyrics found in search results for {track}")
                
    except Exception as e:
        logger.debug(f"LRCLIB search error for {artist} - {track}: {e}")

    return None