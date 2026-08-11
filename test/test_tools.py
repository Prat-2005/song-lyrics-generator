from src.tools import get_rhymes, count_syllables, check_sentiment

def test_get_rhymes():
    # Basic rhyme check
    rhymes = get_rhymes("blue")
    assert isinstance(rhymes, list)
    assert len(rhymes) >= 0

def test_count_syllables_single_line():
    # Simple words
    assert count_syllables("Hello world") == 3
    assert count_syllables("Syllable") == 3
    assert count_syllables("A") == 1
    # Complex word
    assert count_syllables("Incomprehensible") == 6

def test_count_syllables_multiline():
    text = "Hello world\nHow are you"
    # Hello world (3), How are you (3)
    result = count_syllables(text)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result == [3, 3]

def test_count_syllables_edge_cases():
    assert count_syllables(None) == 0
    assert count_syllables("") == 0
    assert count_syllables("   ") == 0

def test_check_sentiment():
    assert check_sentiment("I love this beautiful day") == "positive"
    assert check_sentiment("I hate this dark pain") == "negative"
    assert check_sentiment("The cat sat on the mat") == "neutral"
    assert check_sentiment("") == "neutral"
