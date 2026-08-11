import os
from unittest.mock import patch, MagicMock, mock_open
from scripts.fetch_lyrics import fetch_lrclib, main

def test_fetch_lrclib_success():
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"plainLyrics": "Sample LRCLIB lyrics"}
        mock_get.return_value = mock_response

        lyrics = fetch_lrclib("Artist", "Track")
        assert lyrics == "Sample LRCLIB lyrics"

def test_fetch_lrclib_fail():
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        lyrics = fetch_lrclib("Artist", "Track")
        assert lyrics is None

@patch('src.fetch_lyrics.fetch_lrclib')
@patch('src.fetch_lyrics.Path.glob')
@patch('builtins.open', new_callable=mock_open, read_data="Artist,Title\nArtist,Song1")
def test_main_logic(mock_file, mock_glob, mock_lrclib):
    # Mock CSV glob to return one file
    mock_file_obj = MagicMock(spec=os.PathLike)
    mock_file_obj.stem = "Artist"
    mock_glob.return_value = [mock_file_obj]

    # LRCLIB succeeds
    mock_lrclib.return_value = "LRCLIB Lyrics"

    # We need to mock Path.exists to avoid actually checking filesystem for cache
    with patch('src.fetch_lyrics.Path.exists', return_value=False):
        # Run main for a subset
        main(subset_artist="Artist", limit_per_artist=1)

        # Verify LRCLIB was called
        mock_lrclib.assert_called()
