import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from src.evaluate import rhyme_density, syllable_consistency, originality_score

def test_rhyme_density():
    # Perfect rhyme
    text = "I love the light\nIt is so bright"
    # light rhymes with bright
    # last_word_i = light, last_word_j = bright
    # get_rhymes("light") should return "bright"
    # Let's mock get_rhymes to be sure
    with patch('src.evaluate.get_rhymes') as mock_rhymes:
        mock_rhymes.return_value = ["bright"]
        assert rhyme_density(text) == 1.0

    # No rhyme
    text_no_rhyme = "I love the light\nThe cat is blue"
    with patch('src.evaluate.get_rhymes') as mock_rhymes:
        mock_rhymes.return_value = ["bright"]
        assert rhyme_density(text_no_rhyme) == 0.0

def test_syllable_consistency():
    # Perfectly consistent
    text = "Hello world\nHow are you" # 3, 3
    assert syllable_consistency(text) == 1.0

    # Inconsistent
    text_inconsistent = "Hello world\nThis is a very long line of text" # 3, 10
    # std_dev will be > 0, so score < 1.0
    assert syllable_consistency(text_inconsistent) < 1.0

    # Single line
    assert syllable_consistency("Hello world") == 1.0

def test_originality_score():
    # Mock LyricsRetriever
    with patch('src.evaluate.LyricsRetriever') as MockRetriever:
        instance = MockRetriever.return_value
        instance.index = MagicMock()
        instance.model = MagicMock()

        # Mock encoding to return a dummy embedding
        instance.model.encode.return_value = np.array([[0.1, 0.2]])

        # Mock FAISS search to return similarity
        # dist[0][0] is the similarity
        instance.index.search.return_value = (np.array([[0.8]]), np.array([[0]]))

        text = "A unique line"
        score = originality_score(text)

        # score = 1.0 - max_sim = 1.0 - 0.8 = 0.2
        assert score == pytest.approx(0.2)
