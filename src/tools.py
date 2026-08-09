import re
import pronouncing
from retrieval import LyricsRetriever

retriever = LyricsRetriever()

def get_rhymes(word: str, max_results: int = 10):
    """Find words that rhyme with the given word."""
    rhymes = pronouncing.rhymes(word)
    return rhymes[:max_results]

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
        return [count_syllables(line) for line in text.split('\n') if line.strip()]

    words = re.findall(r'\w+', text.lower())
    total_syllables = 0
    for word in words:
        phones = pronouncing.phones_for_word(word)
        if phones:
            total_syllables += pronouncing.syllable_count(phones[0])
        else:
            # Improved Fallback: linguistic heuristic
            count = 0
            if not word:
                continue

            # Vowels and diphthongs
            vowels = "aeiouy"
            # Count vowel groups (diphthongs count as one)
            i = 0
            while i < len(word):
                if word[i] in vowels:
                    count += 1
                    while i + 1 < len(word) and word[i+1] in vowels:
                        i += 1
                i += 1

            # Subtract silent 'e' at the end
            if word.endswith("e") and count > 1:
                # Check if it's likely not silent (e.g., 'me', 'be')
                if not (len(word) <= 3 and word[0] in vowels):
                    count -= 1

            # Handle common suffixes that don't add syllables (e.g., -ed)
            if word.endswith("ed") and count > 1:
                # Only subtract if 'ed' doesn't follow 't' or 'd' (e.g., 'wanted' vs 'baked')
                if len(word) > 2 and word[-3] not in "td":
                    count -= 1

            if count <= 0:
                count = 1
            total_syllables += count
    return total_syllables

def check_sentiment(text: str):
    """Simple keyword-based sentiment tagging."""
    positive_words = {"love", "happy", "bright", "shine", "good", "great", "beautiful", "heaven", "joy"}
    negative_words = {"hate", "sad", "dark", "pain", "bad", "worst", "hell", "cry", "alone", "broken"}

    words = re.findall(r'\w+', text.lower())
    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)

    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"

def get_similar_lines(query: str, artist: str = None, k: int = 5):
    """Wrap the retrieval system for tool use."""
    return retriever.retrieve_similar_lines(query, artist, k)

# Tool definitions in OpenAI-style JSON schema
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_rhymes",
            "description": "Find words that rhyme with a given word.",
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {"type": "string", "description": "The word to find rhymes for."},
                    "max_results": {"type": "integer", "description": "Maximum number of rhymes to return.", "default": 10}
                },
                "required": ["word"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_syllables",
            "description": "Count the total syllables in a line of text to check meter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text or lyric line to analyze."},
                    "line": {"type": "string", "description": "Backward-compatible alias for text."}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_sentiment",
            "description": "Analyze the sentiment of a piece of text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to analyze for sentiment."}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_similar_lines",
            "description": "Retrieve existing lines from the dataset that are similar to a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "artist": {"type": "string", "description": "Optional artist filter."},
                    "k": {"type": "integer", "description": "Number of lines to return.", "default": 5}
                },
                "required": ["query"]
            }
        }
    }
]

TOOL_FUNCTIONS = {
    "get_rhymes": get_rhymes,
    "count_syllables": count_syllables,
    "check_sentiment": check_sentiment,
    "get_similar_lines": get_similar_lines
}
