import unittest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure src is in the python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

# We patch LyricsRetriever before importing tools, because it's instantiated at the module level.
with patch('retrieval.LyricsRetriever', autospec=True) as MockRetriever:
    from tools import check_sentiment, count_syllables, get_rhymes, get_similar_lines, retriever

class TestTools(unittest.TestCase):
    def test_check_sentiment_positive(self):
        """Test positive sentiment"""
        self.assertEqual(check_sentiment("I love this beautiful day and I feel happy"), "positive")

    def test_check_sentiment_negative(self):
        """Test negative sentiment"""
        self.assertEqual(check_sentiment("This is bad, I feel alone in the dark"), "negative")

    def test_check_sentiment_neutral(self):
        """Test neutral sentiment"""
        self.assertEqual(check_sentiment("I am walking down the street"), "neutral")
        
    def test_count_syllables(self):
        """Test syllable counting on words and lines"""
        # Single word (1 syllable)
        self.assertEqual(count_syllables("cat"), 1)
        # Multiple words
        self.assertEqual(count_syllables("hello world"), 3)
        # Empty string
        self.assertEqual(count_syllables(""), 0)
        # Multiple lines
        self.assertEqual(count_syllables("hello\nworld"), [2, 1])

if __name__ == '__main__':
    unittest.main()
