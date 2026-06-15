"""Utility functions for the lyrics generator."""

import torch


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU.

    Returns:
        torch.device: The appropriate device for computation.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_lyrics(text: str) -> str:
    """Light post-processing for display.

    Capitalizes the first word and inserts a newline every 8 words.

    Args:
        text: The input text to format.

    Returns:
        str: The formatted text.
    """
    if not text:
        return text

    # Capitalize first word
    words = text.split()
    if words:
        words[0] = words[0].capitalize()

    # Insert newline every 8 words
    formatted_words = []
    for i, word in enumerate(words):
        if i > 0 and i % 8 == 0:
            formatted_words.append("\n")
        formatted_words.append(word)

    return " ".join(formatted_words)