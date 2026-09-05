from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.retrieval import LyricsRetriever


@pytest.fixture
def retriever():
    with patch('src.retrieval.SentenceTransformer'), patch('src.retrieval.login'):
        return LyricsRetriever()


def test_load_lyrics_dataframe_reads_all_csvs(retriever, tmp_path):
    (tmp_path / "Artist1.csv").write_text(
        "Artist,Title,Lyric\nArtist1,Song A,Line one\n"
    )
    # Simulates the real data/ files that carry a leftover unnamed index column.
    (tmp_path / "Artist2.csv").write_text(
        ",Artist,Title,Lyric\n0,Artist2,Song B,Line two\n"
    )

    with patch('src.retrieval.DATA_PATH', str(tmp_path)):
        df = retriever._load_lyrics_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert set(df['Artist']) == {"Artist1", "Artist2"}
    assert list(df.columns) == ['Artist', 'Lyric']


def test_load_lyrics_dataframe_raises_when_no_csvs(retriever, tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with patch('src.retrieval.DATA_PATH', str(empty_dir)):
        with pytest.raises(FileNotFoundError):
            retriever._load_lyrics_dataframe()


def test_build_index(retriever):
    df = pd.DataFrame({'Artist': ['A1'], 'Lyric': ['Line 1 <NEWLINE> Line 2']})

    with patch.object(retriever, '_load_lyrics_dataframe', return_value=df), \
         patch('src.retrieval.faiss.IndexFlatIP') as MockIndex, \
         patch('src.retrieval.faiss.write_index'), \
         patch('builtins.open', MagicMock()), \
         patch('os.makedirs'):

        retriever.model.encode.return_value = np.array([[0.1, 0.2]])

        retriever.build_index()

        assert retriever.index is not None
        retriever.index.add.assert_called()


def test_retrieve_similar_lines(retriever):
    retriever.index = MagicMock()
    retriever.metadata = [
        {"artist": "Artist1", "text": "Line 1"},
        {"artist": "Artist2", "text": "Line 2"},
    ]

    retriever.model.encode.return_value = np.array([[0.1, 0.2]])
    retriever.index.search.return_value = (np.array([[0.9, 0.8]]), np.array([[0, 1]]))

    results = retriever.retrieve_similar_lines("query", artist="Artist1", k=1)
    assert results == ["Line 1"]

    results = retriever.retrieve_similar_lines("query", artist="Artist2", k=1)
    assert results == ["Line 2"]

    results = retriever.retrieve_similar_lines("query", artist=None, k=2)
    assert len(results) == 2
