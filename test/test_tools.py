from src.tools import get_rhymes, count_syllables


def test_get_rhymes():
    rhymes = get_rhymes("blue")
    assert isinstance(rhymes, list)
    assert len(rhymes) >= 0


def test_count_syllables_single_line():
    assert count_syllables("Hello world") == 3
    assert count_syllables("Syllable") == 3
    assert count_syllables("A") == 1
    assert count_syllables("Incomprehensible") == 6


def test_count_syllables_multiline():
    text = "Hello world\nHow are you"
    result = count_syllables(text)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result == [3, 3]


def test_count_syllables_edge_cases():
    assert count_syllables(None) == 0
    assert count_syllables("") == 0
    assert count_syllables("   ") == 0
