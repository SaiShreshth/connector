"""
KeywordMatcher - robust, test-friendly keyword matching for intent classification.

Behavior:
- Loads keyword definitions using app.ai.intent_classification.keywords.loader.load_keywords()
  if available; otherwise loads JSON files from a local keywords/ folder.
- Normalizes input text (removes/replaces symbols), tokenizes, and precompiles phrases and regexes.
- Supports exact / regex / partial matches with scoring.
- Returns a list of dicts with keys: id, intent, action, score, source, match_type, matched_text.
- Includes fallback normalization for symbol-only inputs so tests like "@@@add##to###cart$$$" match.
"""

from __future__ import annotations

import os
import json
import re
import time
import logging
from functools import lru_cache
from typing import Dict, List, Tuple, Any, Optional, Union

from app.ai.intent_classification.keywords.loader import load_all_keywords
from app.utils.text_processing import normalize_text

logger = logging.getLogger(__name__)
KEYWORDS_DIR = os.path.join(os.path.dirname(__file__), "keywords")

# -----------------------
# Try to import project utilities (safe fallbacks if not present)
# -----------------------
try:
    from app.utils.text_processing import normalize_text as _external_normalize
except Exception:
    _external_normalize = None

try:
    from app.ai.intent_classification.keywords.loader import load_keywords as _external_loader
except Exception:
    _external_loader = None

# -----------------------
# Config / Globals
# -----------------------
KEYWORDS_FOLDER = os.path.join(os.path.dirname(__file__), "keywords")  # fallback folder
_KEYWORDS_CACHE: Optional[Dict[str, Dict[str, Any]]] = None

# -----------------------
# Fallback normalize_text (used only if external not available)
# -----------------------
@lru_cache(maxsize=8192)
def _normalize_text_local(text: str) -> str:
    """Lightweight local normalizer with caching for speed."""
    if not text:
        return ""
    s = text.lower()
    # Expand common symbols to words to preserve meaning
    # Replace runs of special symbols (one run -> one word)
    s = re.sub(r'&+', ' and ', s)        # && -> 'and'
    s = re.sub(r'\++', ' plus ', s)      # ++ -> 'plus'
    s = re.sub(r'@+', ' at ', s)         # @@ -> 'at'
    s = re.sub(r'#+', ' hash ', s)       # ## -> 'hash'
    s = re.sub(r'\$+', ' dollar ', s)    # $$ -> 'dollar'
    s = re.sub(r'%+', ' percent ', s)    # %% -> 'percent'
    # Replace non-alphanumeric with spaces
    s = re.sub(r'[^a-z0-9]', ' ', s)
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s

class KeywordMatcher:
    """
    Production-ready keyword matcher that uses the precompiled matching system.
    Compatible with DecisionEngine's expected interface.
    """
    def __init__(self):
        """Initialize by ensuring keywords are precompiled."""
        _precompile_keywords()
        logger.info(f"✅ KeywordMatcher initialized with {len(_PRECOMPILED)} intents.")
        
        # Log sample for verification
        if _PRECOMPILED:
            sample_intent = list(_PRECOMPILED.keys())[0]
            sample_data = _PRECOMPILED[sample_intent]
            logger.debug(f"Sample intent '{sample_intent}': {len(sample_data.get('phrases', []))} phrases")

    def search(self, query: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Search for matching intents using keyword matching.
        Returns list of dicts compatible with DecisionEngine's expected format.
        
        Args:
            query: User input text
            top_n: Maximum number of results to return
            
        Returns:
            List of dicts with keys: id, intent, action, score, source, match_type, matched_text
        """
        results = match_keywords(query, top_n=top_n)
        
        if results:
            logger.info(f"✅ Keyword match: {results[0]['id']} (score={results[0]['score']:.3f})")
        else:
            logger.warning(f"⚠️  No keyword match for: '{query}'")
            
        return results

    def match(self, text: str) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility.
        Returns single best match in simplified format.
        """
        results = self.search(text, top_n=1)
        if results:
            best = results[0]
            return {
                "intent": best["id"],
                "keyword": best.get("matched_text", ""),
                "confidence": best["score"]
            }
        
        logger.warning(f"⚠️  No keyword match found for: '{text}'")
        return {"intent": "UNKNOWN", "confidence": 0.0, "keyword": None}

def normalize_text(text: str) -> str:
    """Wrapper normalization — prefer external if available."""
    if _external_normalize:
        try:
            out = _external_normalize(text)
            if out is not None and out.strip():
                return out
        except Exception:
            pass
    return _normalize_text_local(text or "")

# -----------------------
# Keyword loader (fallback)
# -----------------------
def _load_from_folder(folder: str) -> Dict[str, Dict[str, Any]]:
    """
    Load keyword JSON files from folder.
    Each file name (without extension) becomes the intent id (uppercased).
    JSON structure per file: {"phrases": ["..."], "regex": ["..."], "priority": 1}
    """
    data: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(folder):
        return data

    for fname in os.listdir(folder):
        if not fname.lower().endswith(".json"):
            continue
        intent_id = os.path.splitext(fname)[0].upper()
        path = os.path.join(folder, fname)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
                entry = {
                    "keywords": payload.get("phrases") or payload.get("keywords") or [],
                    "regex": payload.get("regex") or payload.get("patterns") or [],
                    "priority": int(payload.get("priority", 1))
                }
                data[intent_id] = entry
        except Exception:
            # silently skip invalid files
            continue
    return data

def load_keywords() -> Dict[str, Dict[str, Any]]:
    """Public loader: prefer external loader if available, otherwise use folder loader."""
    global _KEYWORDS_CACHE
    if _KEYWORDS_CACHE is not None:
        return _KEYWORDS_CACHE

    if _external_loader:
        try:
            loaded = _external_loader()
            if isinstance(loaded, dict) and loaded:
                _KEYWORDS_CACHE = loaded
                return _KEYWORDS_CACHE
        except Exception:
            pass

    _KEYWORDS_CACHE = _load_from_folder(KEYWORDS_FOLDER)
    return _KEYWORDS_CACHE

# -----------------------
# Preprocessing / Precompilation
# -----------------------
_PRECOMPILED: Dict[str, Dict[str, Any]] = {}
_PRECOMPILED_READY = False

def _precompile_keywords():
    global _PRECOMPILED_READY, _PRECOMPILED
    if _PRECOMPILED_READY:
        return
    keywords = load_keywords()
    compiled: Dict[str, Dict[str, Any]] = {}
    for intent, data in (keywords.items() if keywords else []):
        kw_list = data.get("keywords", []) or []
        regex_list = data.get("regex", []) or []
        priority = int(data.get("priority", 1) or 1)
        # Tokenize and normalize phrase keywords
        tokenized_phrases: List[Tuple[str, ...]] = []
        for p in kw_list:
            if not isinstance(p, str):
                continue
            norm = _normalize_text_local(p)  # use local normalizer for phrase preprocessing
            tokens = tuple(re.findall(r"\w+", norm))
            if tokens:
                tokenized_phrases.append(tokens)
        # Compile regexes
        compiled_regexes = []
        for rx in regex_list:
            try:
                compiled_regexes.append(re.compile(rx, flags=re.IGNORECASE))
            except re.error:
                # skip invalid regex
                continue
        compiled[intent] = {
            "phrases": tokenized_phrases,
            "regex": compiled_regexes,
            "priority": priority
        }
    _PRECOMPILED = compiled
    _PRECOMPILED_READY = True

# -----------------------
# Scoring helpers
# -----------------------
def _score_exact() -> float:
    return 1.0

def _score_partial(overlap: int, pattern_len: int, weight: float) -> float:
    if pattern_len <= 0:
        return 0.0
    base = 0.5
    frac = overlap / pattern_len
    return round(base + (0.5 * frac * weight), 3)

def _score_regex(match_len: int, pattern_len: int, weight: float) -> float:
    if pattern_len <= 0:
        return 0.0
    frac = min(1.0, match_len / (pattern_len or 1))
    return round(0.8 * frac * weight, 3)

# -----------------------
# Normalization + tokenization caching
# -----------------------
@lru_cache(maxsize=8192)
def _normalize_and_tokenize(text: str) -> Tuple[str, Tuple[str, ...]]:
    # IMPORTANT: use local normalizer here to avoid depending on external normalize_text
    norm = _normalize_text_local(text or "")
    # minor extra cleanup
    norm = re.sub(r"['\"]+", "", norm)
    norm = re.sub(r"[-_]+", " ", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    tokens = tuple(re.findall(r"\w+", norm))
    return norm, tokens

# -----------------------
# Core match function
# -----------------------
def match_keywords(
    text: str,
    *,
    top_n: int = 3
) -> List[Dict[str, Union[str, float]]]:
    """
    Match text against keyword patterns and return a sorted list of matches.
    Each result dict has: id, intent, action, score, source, match_type, matched_text
    """
    t0 = time.perf_counter()
    if not text or not isinstance(text, str):
        return []

    # Ensure compiled patterns ready
    _precompile_keywords()
    compiled = _PRECOMPILED

    # ---------------------------
    # Robust normalization fallback logic (FINAL)
    # ---------------------------
    # Use the local normalizer for all matching/segmentation so it lines up with precompiled phrases
    norm = _normalize_text_local(text or "")

    def _has_alnum(s: str) -> bool:
        return bool(re.search(r"[A-Za-z0-9]", s or ""))

    # If local normalizer returned nothing useful, fallback to extracting alphanumeric runs
    if not norm or not _has_alnum(norm):
        raw_alnum = re.findall(r"[A-Za-z0-9]+", text or "")
        if raw_alnum:
            norm = " ".join(raw_alnum).lower()
        else:
            norm = ""

    if not norm:
        return []

    # split into segments to handle multi-intent sentences
    norm_for_segments = _normalize_text_local(norm) if norm else ""
    if norm_for_segments:
        segments = [s.strip() for s in re.split(r"\band\b|,|;|\?|!|\n", norm_for_segments) if s.strip()]
    else:
        raw_alnum = re.findall(r"[A-Za-z0-9]+", norm or "")
        segments = [" ".join(raw_alnum).lower()] if raw_alnum else []

    # Collect candidates: intent -> best info
    candidates: Dict[str, Dict[str, Any]] = {}

    for seg in segments:
        seg_norm, seg_tokens = _normalize_and_tokenize(seg)
        seg_token_set = set(seg_tokens)

        # exact phrase check (compare token tuples)
        seg_tuple = tuple(seg_tokens)
        if seg_tuple:
            for intent, data in compiled.items():
                for phrase_tokens in data.get("phrases", []):
                    if phrase_tokens == seg_tuple:
                        score = _score_exact()
                        info = {
                            "id": intent,
                            "intent": intent,
                            "action": intent,
                            "score": score,
                            "source": "keyword",
                            "match_type": "exact",
                            "matched_text": " ".join(phrase_tokens)
                        }
                        existing = candidates.get(intent)
                        if not existing or existing.get("score", 0) < score:
                            candidates[intent] = info

        # regex checks
        for intent, data in compiled.items():
            for rx in data.get("regex", []):
                m = rx.search(seg_norm)
                if m:
                    match_len = len(m.group(0) or "")
                    weight = 1.0 / (data.get("priority") or 1)
                    score = _score_regex(match_len, len(m.group(0) or ""), weight=weight)
                    info = {
                        "id": intent,
                        "intent": intent,
                        "action": intent,
                        "score": score,
                        "source": "keyword",
                        "match_type": "regex",
                        "matched_text": m.group(0)
                    }
                    existing = candidates.get(intent)
                    if not existing or score > existing.get("score", 0):
                        candidates[intent] = info

        # partial / token-overlap matching
        for intent, data in compiled.items():
            best_for_intent = candidates.get(intent, {})
            best_score_local = best_for_intent.get("score", 0.0)

            for phrase_tokens in data.get("phrases", []):
                if not phrase_tokens:
                    continue
                p_set = set(phrase_tokens)
                overlap = len(p_set & seg_token_set)
                if overlap <= 0:
                    continue
                weight = 1.0 / (data.get("priority") or 1)
                score = _score_partial(overlap, len(phrase_tokens), weight=weight)
                if score > best_score_local:
                    candidates[intent] = {
                        "id": intent,
                        "intent": intent,
                        "action": intent,
                        "score": score,
                        "source": "keyword",
                        "match_type": "partial",
                        "matched_text": " ".join(phrase_tokens)
                    }
                    best_score_local = score

    # Sort results by score then by match type rank (exact > regex > partial)
    def _rank_key(item: Dict[str, Any]) -> Tuple[float, int]:
        typ = item.get("match_type", "")
        typ_rank = {"exact": 3, "regex": 2, "partial": 1}.get(typ, 0)
        return (item.get("score", 0.0), typ_rank)

    results = sorted(candidates.values(), key=_rank_key, reverse=True)[:top_n]

    # Note: previously logged latency; logging removed as requested.
    return results

# NOTE: KeywordMatcher class is now defined earlier in the file (around line 71)
# This duplicate definition has been removed to avoid conflicts


# If run as script, perform a quick self-test
if __name__ == "__main__":
    matcher = KeywordMatcher()
    sample_queries = [
        "show me red nike shoes",
        "add this to my cart",
        "what is the price of apple watch",
        "find blue jeans under 2000",
        "checkout my order"
    ]

    for q in sample_queries:
        print(f"\n🧾 Query: {q}")
        print("Result:", matcher.search(q))
    # Manual test
    matcher = KeywordMatcher()
    print(matcher.search("add@@to#cart$"))
