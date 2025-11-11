import json
import os
import glob

KEYWORDS_DIR = os.path.join("app", "ai", "intent_classification", "keywords")
MIN_KEYWORDS_PER_INTENT = 20

def load_keyword_files():
    return glob.glob(os.path.join(KEYWORDS_DIR, "*_keywords.json"))

def has_multiword(phrases):
    return any(" " in p.strip() for p in phrases if isinstance(p, str))

def test_min_counts_and_multiword_presence():
    files = load_keyword_files()
    assert files, "No keyword JSON files found"
    for path in files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict), f"Invalid structure in {path}"
        for intent, spec in data.items():
            kws = spec.get("keywords", [])
            assert len(kws) >= MIN_KEYWORDS_PER_INTENT, (
                f"{path}:{intent} has {len(kws)} < {MIN_KEYWORDS_PER_INTENT} keywords"
            )
            assert has_multiword(kws), f"{path}:{intent} missing multi-word phrases"

def test_no_duplicate_keywords_case_insensitive():
    files = load_keyword_files()
    for path in files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for intent, spec in data.items():
            kws = [k for k in spec.get("keywords", []) if isinstance(k, str)]
            lowered = [k.lower().strip() for k in kws]
            seen = set()
            dups = set()
            for k in lowered:
                if k in seen:
                    dups.add(k)
                else:
                    seen.add(k)
            assert not dups, f"{path}:{intent} has duplicates"
