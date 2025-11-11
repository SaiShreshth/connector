#!/usr/bin/env python3
import os
import json
from collections import Counter

from app.ai.intent_classification.ambiguity_resolver import LOG_FILE

def main():
    if not os.path.exists(LOG_FILE):
        print("No ambiguity log found")
        return
    counts = Counter()
    total = 0
    with open(LOG_FILE, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            intents = entry.get("intent_scores", {}).get("intents") or entry.get("intent_scores", {})
            if isinstance(intents, dict):
                for k in intents.keys():
                    counts[k] += 1
            total += 1
    print(f"Total ambiguous/unclear cases: {total}")
    for intent, c in counts.most_common(20):
        print(f"{intent}: {c}")

if __name__ == "__main__":
    main()
