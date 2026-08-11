import pytest
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
import pandas as pd
from src.retrieval import LyricsRetriever

@pytest.fixture
def retriever():
    with patch('src.retrieval.SentenceTransformer'), patch('src.retrieval.login'):
        return LyricsRetriever()

def test_load_lyrics_dataframe_cache(retriever):
    class MockPath:
        def __init__(self, name):
            self.name = name
        def __lt__(self, other):
            return self.name < other.name
        def __str__(self):
            return self.name

    # Mock CACHE_DIR
    mock_cache = MagicMock()
    mock_cache.exists.return_value = True
    mock_cache.glob.return_value = [MockPath("file2.json"), MockPath("file1.json")]

    with patch('src.retrieval.CACHE_DIR', mock_cache), \
         patch('builtins.open', mock_open(read_data='{"artist": "Artist1", "lyrics": "Lyric 1"}')):

        df = retriever._load_lyrics_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert df.iloc[0]['Artist'] == "Artist1"

def test_build_index(retriever):
    # Mock _load_lyrics_dataframe to return a small df
    df = pd.DataFrame({'Artist': ['A1'], 'Lyric': ['Line 1 <NEWLINE> Line 2']})

    with patch.object(retriever, '_load_lyrics_dataframe', return_value=df), \
         patch('src.retrieval.faiss.IndexFlatIP') as MockIndex, \
         patch('src.retrieval.faiss.write_index'), \
         patch('builtins.open', mock_open()), \
         patch('os.makedirs'):

        # Mock model.encode
        retriever.model.encode.return_value = np.array([[0.1, 0.2]])

        retriever.build_index()

        # Verify index was created and embeddings added
        assert retriever.index is not None
        retriever.index.add.assert_called()

def test_retrieve_similar_lines(retriever):
    # Setup mock index and metadata
    retriever.index = MagicMock()
    retriever.metadata = [
        {"artist": "Artist1", "text": "Line 1"},
        {"artist": "Artist2", "text": "Line 2"},
    ]

    # Mock model.encode
    retriever.model.encode.return_value = np.array([[0.1, 0.2]])

    # Mock FAISS search result: indices = [0, 1]
    retriever.index.search.return_value = (np.array([[0.9, 0.8]]), np.array([[0, 1]]))

    # Retrieve for Artist1
    results = retriever.retrieve_similar_lines("query", artist="Artist1", k=1)
    assert results == ["Line 1"]

    # Retrieve for Artist2
    results = retriever.retrieve_similar_lines("query", artist="Artist2", k=1)
    assert results == ["Line 2"]

    # Retrieve all
    results = retriever.retrieve_similar_lines("query", artist=None, k=2)
    assert len(results) == 2
