"""
Scoring module for intent classification matches.
Calculates confidence for matched intents.
"""
from typing import List, Dict


def calculate_confidence(matches: List[Dict]) -> float:
    """
    Given a list of match dicts from keyword_matcher.match_keywords,
    return the confidence score of the best match.
    """
    if not matches:
        return 0.0  # No match
    best = matches[0]  # match_keywords already sorts by score
    return float(best.get("score", 0.0))


def aggregate_confidence(matches: List[Dict]) -> float:
    """
    Aggregate multiple matches into a single confidence score.
    Weighted average of all matches.
    """
    if not matches:
        return 0.0
    total_score = sum(m.get("score", 0.0) for m in matches)
    return total_score / len(matches)

