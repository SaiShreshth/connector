import logging
"""
embedding_matcher.py
Enhanced version — robust against typos, spacing errors, and partial words.
Ensures minimum confidence and better semantic coverage.
"""

import os
import re
import time
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any
from loguru import logger

# --- FIXED IMPORTS ---
from app.ai.intent_classification.keyword_matcher import match_keywords
try:
    from app.ai.intent_classification.intents_modular.definitions import ALL_INTENT_DEFINITIONS  # type: ignore
except Exception:
    ALL_INTENT_DEFINITIONS = None  # type: ignore

# --- Optional fuzzy string helper (no external lib needed) ---
def _soft_normalize(text: str) -> str:
    """Basic normalization for fuzzy matching & embedding robustness."""
    text = text.lower().strip()
    text = re.sub(r"ing\s+", "ing", text)   # Fix "show ing" → "showing"
    text = re.sub(r" +", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


class EmbeddingMatcher:
    """Embedding-based semantic matcher for user intents."""

    def __init__(self, model_name='all-MiniLM-L6-v2', client=None):
        """
        EmbeddingMatcher handles semantic similarity searches.
        :param model_name: Name of the sentence transformer model to use
        :param client: Optional vector database client (e.g., QdrantClient)
        """
        logger.info(f"Loading embedding model '{model_name}'...")
        start = time.time()
        
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.client = client
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self.intent_examples = self._load_reference_phrases()
        self.intent_embeddings = self._precompute_embeddings()
        self._query_cache: Dict[str, Any] = {}
        self._query_cache_max = 512
        logger.info(f"Model ready in {time.time() - start:.2f}s ({len(self.intent_examples)} intents)")

    # ------------------------------------------------------------------
    def _load_reference_phrases(self) -> Dict[str, List[str]]:
        """Loads reference phrases from modular intent definitions."""
        if ALL_INTENT_DEFINITIONS:
            refs: Dict[str, List[str]] = {}
            try:
                for _name, definition in ALL_INTENT_DEFINITIONS.items():  # type: ignore
                    action = getattr(definition, "action_code", None)
                    phrases = getattr(definition, "example_phrases", None)
                    if not action or not phrases:
                        continue
                    if hasattr(action, "name"):
                        action_str = getattr(action, "name")
                    elif isinstance(action, str):
                        action_str = action
                    else:
                        action_str = str(action)
                    clean_phrases = [
                        _soft_normalize(p) for p in phrases if isinstance(p, str) and p.strip()
                    ]
                    if clean_phrases:
                        refs[action_str] = list(set(clean_phrases))
                if refs:
                    logger.info(f"Loaded {len(refs)} embedding references from definitions")
                    return refs
            except Exception as e:
                logger.warning(f"Definition load failed: {e}")
        logger.warning("Using fallback INTENT_EXAMPLES for embeddings")
        return INTENT_EXAMPLES

    # ------------------------------------------------------------------
    def _precompute_embeddings(self) -> Dict[str, Any]:
        """Encodes and stores embeddings for each reference phrase."""
        embeddings: Dict[str, Any] = {}
        for action_code, phrases in self.intent_examples.items():
            try:
                embeddings[action_code] = self.model.encode(phrases, convert_to_tensor=True)
            except Exception as e:
                logger.warning(f"Embedding failure for {action_code}: {e}")
        return embeddings

    # ------------------------------------------------------------------
    def _encode_query_cached(self, query: str):
        """Encodes query with caching to avoid recomputation."""
        key = query.strip().lower()
        if key in self._query_cache:
            return self._query_cache[key]
        emb = self.model.encode(key, convert_to_tensor=True)
        if len(self._query_cache) >= self._query_cache_max:
            self._query_cache.pop(next(iter(self._query_cache)))
        self._query_cache[key] = emb
        return emb

    # ------------------------------------------------------------------
    def search(self, query: str, threshold: float = 0.60) -> List[Dict]:
        """
        Performs semantic similarity matching against all intents.
        Returns a list of matches with their similarity scores.
        """
        if not self.client:
            self.logger.warning("⚠️ No vector DB client provided; returning mock search results.")
            # Mock search result for development
            return [
                {
                    "payload": {"intent": "SEARCH_PRODUCT"},
                    "score": 0.95
                }
            ]
        try:
            if not query or not query.strip():
                return []

            clean_query = _soft_normalize(query)
            if len(clean_query) < 2:
                return []

            query_emb = self._encode_query_cached(clean_query)
            results = []

            for action_code, ref_emb in self.intent_embeddings.items():
                if ref_emb is not None:
                    sim = util.cos_sim(query_emb, ref_emb).max().item()
                    # Apply small boost for keyword overlap
                    kw_overlap = 0.05 if match_keywords(clean_query) else 0.0
                    score = round(min(1.0, sim + kw_overlap), 4)
                    if score >= 0.1:
                        results.append({
                            "id": action_code,
                            "intent": action_code,
                            "score": score,
                            "source": "embedding",
                            "query": clean_query
                        })

            # Sort results by similarity score
            results.sort(key=lambda x: x["score"], reverse=True)

            # ✅ Guarantee minimum confidence > 0 if we have results
            if results and results[0]["score"] < 0.3:
                results[0]["score"] = 0.35  # Slightly bump weak match

            # ✅ Save to Qdrant (safe call)
            if results and "store_vector" in globals() and callable(store_vector):
                try:
                    best = results[0]
                    status = "CONFIDENT" if best["score"] >= threshold else "LOW_CONFIDENCE"
                    store_vector(
                        intent=best["intent"],
                        vector=query_emb.tolist(),
                        confidence=best["score"],
                        status=status,
                        variant="A",
                        query=clean_query,
                    )
                except Exception as e:
                    logger.warning(f"Qdrant store failed: {e}")

            return results
        except Exception as e:
            self.logger.error(f"❌ Error during embedding matching: {e}")
            return [{
                "id": "SEARCH_PRODUCT",
                "intent": "SEARCH_PRODUCT",
                "score": 0.1,
                "source": "embedding_fallback",
                "error": str(e),
            }]

    # ------------------------------------------------------------------
    def match_intent(self, query_vector, collection_name="intents", threshold=0.7):
        """
        Finds the best matching intent for a given query vector.
        :param query_vector: Embedding vector of the query text
        :param collection_name: Name of the vector collection
        :param threshold: Minimum similarity score for a valid match
        :return: Dict with 'intent' and 'score' or None
        """
        results = self.search(query_vector, threshold=threshold)
        if not results:
            self.logger.warning("No matching results found.")
            return None

        # Handle both mock and Qdrant response formats
        best_match = max(results, key=lambda r: r["score"] if isinstance(r, dict) else r.score)
        score = best_match["score"] if isinstance(best_match, dict) else best_match.score
        intent = (
            best_match["payload"]["intent"]
            if isinstance(best_match, dict) and "payload" in best_match
            else best_match.get("intent", "unknown")
        )

        if score >= threshold:
            return {
                "intent": intent,
                "score": round(score, 3)
            }

        self.logger.info("🟡 No intent exceeded threshold confidence.")
        return None


# ----------------------------------------------------------------------
# Default intent examples
# ----------------------------------------------------------------------
INTENT_EXAMPLES = {
    "SEARCH_PRODUCT": [
        "find this item", "show me products", "search for shoes", "look up phone covers", "display clothes"
    ],
    "ADD_TO_CART": [
        "add this to my basket", "put in my cart", "add item to cart", "include this product"
    ],
    "VIEW_CART": [
        "show my cart", "open basket", "view items in cart", "check what’s in my cart"
    ],
    "CHECKOUT": [
        "go to checkout", "buy now", "proceed to payment", "place my order"
    ],
    "PRODUCT_INFO": [
        "tell me about this product", "details of this item", "show specs", "show me information"
    ],
    "COMPARE": [
        "compare this with another", "which is better", "show comparison", "compare two products"
    ],
    "FAQ": [
        "how to return item", "shipping policy", "refund details", "help me with an issue"
    ],
}

# ----------------------------------------------------------------------
# Local Test Mode
# ----------------------------------------------------------------------
if __name__ == "__main__":
    matcher = EmbeddingMatcher()
    while True:
        q = input("🗣️  Query: ")
        if q.lower() == "exit":
            break
        res = matcher.search(q)
        if res:
            print(f"✅ Top match: {res[0]}")
        else:
            print("⚠️ No match found.")
