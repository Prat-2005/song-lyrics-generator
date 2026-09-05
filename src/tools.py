"""
Plain utility functions used directly by agent.py and evaluate.py.

These used to also be registered as OpenAI-style "tools" for the model to
call via function-calling. That's gone — we call these functions ourselves
in Python wherever we need them, which is simpler and doesn't depend on the
local model reliably supporting structured tool-calling.
"""

import re

import pronouncing

from src.retrieval import LyricsRetriever

_retriever = None


def get_shared_retriever():
    """Lazily create a single LyricsRetriever, shared across the app.
    Previously both this module and evaluate.py each created their own
    LyricsRetriever, which meant the embedding model and FAISS index got
    loaded into memory twice. There's only one instance now."""
    global _retriever
    if _retriever is None:
        _retriever = LyricsRetriever()
    return _retriever


def get_rhymes(word: str, max_results: int = 10):
    """Find words that rhyme with the given word."""
    return pronouncing.rhymes(word)[:max_results]


def count_syllables(text: str = None, line: str = None):
    """Count syllables in a line or block of text.
    Returns a single integer if input is a single line,
    or a list of integers if input contains newlines.
    """
    if text is None:
        text = line
    if text is None:
        return 0

    if '\n' in text:
        return [count_syllables(l) for l in text.split('\n') if l.strip()]

    words = re.findall(r'\w+', text.lower())
    total_syllables = 0
    for word in words:
        phones = pronouncing.phones_for_word(word)
        if phones:
            total_syllables += pronouncing.syllable_count(phones[0])
        else:
            total_syllables += _estimate_syllables(word)
    return total_syllables


def _estimate_syllables(word: str) -> int:
    """Fallback vowel-group heuristic for words not in the CMU dictionary."""
    if not word:
        return 0

    vowels = "aeiouy"
    count = 0
    i = 0
    while i < len(word):
        if word[i] in vowels:
            count += 1
            while i + 1 < len(word) and word[i + 1] in vowels:
                i += 1
        i += 1

    if word.endswith("e") and count > 1 and not (len(word) <= 3 and word[0] in vowels):
        count -= 1

    if word.endswith("ed") and count > 1 and len(word) > 2 and word[-3] not in "td":
        count -= 1

    return max(count, 1)


def get_similar_lines(query: str, artist: str = None, k: int = 5):
    """Retrieve reference lines similar to the query, optionally filtered by artist."""
    return get_shared_retriever().retrieve_similar_lines(query, artist, k)
