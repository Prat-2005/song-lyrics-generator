from unittest.mock import MagicMock, patch

import pandas as pd

from scripts.fetch_lyrics import fetch_lrclib, append_to_csv, main


def test_fetch_lrclib_success_on_exact_match():
    with patch('scripts.fetch_lyrics.requests.get') as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"plainLyrics": "Sample lyrics"}
        )
        lyrics = fetch_lrclib("Artist", "Track")

    assert lyrics == "Sample lyrics"


def test_fetch_lrclib_falls_back_to_search():
    exact_miss = MagicMock(status_code=404)
    search_hit = MagicMock(status_code=200, json=lambda: [{"plainLyrics": "Found via search"}])

    with patch('scripts.fetch_lyrics.requests.get', side_effect=[exact_miss, search_hit]):
        lyrics = fetch_lrclib("Artist", "Track")

    assert lyrics == "Found via search"


def test_fetch_lrclib_returns_none_when_not_found():
    with patch('scripts.fetch_lyrics.requests.get') as mock_get:
        mock_get.return_value = MagicMock(status_code=404)
        lyrics = fetch_lrclib("Artist", "Track")

    assert lyrics is None


def test_append_to_csv_creates_new_file(tmp_path):
    with patch('scripts.fetch_lyrics.DATA_PATH', str(tmp_path)):
        append_to_csv("New Artist", "New Song", "some lyrics")

        csv_path = tmp_path / "NewArtist.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert df.iloc[0]['Lyric'] == "some lyrics"


def test_append_to_csv_appends_to_existing_file(tmp_path):
    csv_path = tmp_path / "Existing.csv"
    pd.DataFrame([{"Artist": "Existing", "Title": "Old Song", "Lyric": "old"}]).to_csv(
        csv_path, index=False
    )

    with patch('scripts.fetch_lyrics.DATA_PATH', str(tmp_path)):
        append_to_csv("Existing", "New Song", "new lyrics")

        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert "new lyrics" in df['Lyric'].values


def test_main_saves_when_lyrics_found():
    with patch('scripts.fetch_lyrics.fetch_lrclib', return_value="lyrics"), \
         patch('scripts.fetch_lyrics.append_to_csv') as mock_append:
        main("Artist", "Track")
        mock_append.assert_called_once_with("Artist", "Track", "lyrics")


def test_main_skips_when_nothing_found():
    with patch('scripts.fetch_lyrics.fetch_lrclib', return_value=None), \
         patch('scripts.fetch_lyrics.append_to_csv') as mock_append:
        main("Artist", "Track")
        mock_append.assert_not_called()
