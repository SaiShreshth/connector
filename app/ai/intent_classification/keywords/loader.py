import os
import json
from loguru import logger

def load_all_keywords(keywords_dir=None):
    """
    Load all keyword JSON files from the given directory.
    Returns a dictionary: 
    {
        "intent_name": {
            "priority": int,
            "keywords": [list of keyword strings]
        }
    }
    """
    if keywords_dir is None:
        keywords_dir = os.path.dirname(__file__)   # default to current folder

    all_keywords = {}

    for file_name in os.listdir(keywords_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(keywords_dir, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        all_keywords.update(data)
                    else:
                        logger.warning(f"Skipping invalid JSON file: {file_name} (Expected dict, got {type(data)})")
            except Exception as e:
                logger.warning(f"Skipping invalid JSON file: {file_name} ({e})")

    return all_keywords


# ✅ Change made here — added a wrapper for compatibility
def load_keywords(keywords_dir=None):
    """
    Compatibility alias for old code that still calls load_keywords().
    Internally calls load_all_keywords().
    """
    return load_all_keywords(keywords_dir)


# Optional standalone test
if __name__ == "__main__":
    keywords = load_all_keywords()
    print("Loaded keyword intents:")
    for intent, details in keywords.items():
        print(f"{intent} (priority {details.get('priority')}): {len(details.get('keywords', []))} keywords")
