"""
text_processing.py
Provides normalization and tokenization utilities for user input.
"""

import re
import string

def normalize_text(text: str) -> str:
    """Normalize text: lowercase, remove punctuation and extra spaces."""
    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text: str):
    """Return a list of tokens after normalization."""
    normalized = normalize_text(text)
    return normalized.split()


if __name__ == "__main__":
    s = "Hello, Add this to CART!"
    print("Normalized:", normalize_text(s))
    print("Tokens:", tokenize_text(s))
