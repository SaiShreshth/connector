import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class EntityExtractor:
    def __init__(self):
        logger.info("✅ EntityExtractor initialized.")

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract simple entities such as product, brand, color, size, and price.
        """

        entities = {
            "product_name": None,
            "brand": None,
            "color": None,
            "size": None,
            "price_range": None,
            "category": None
        }

        # ----- 🔹 Product names -----
        products = ["shoes", "shirt", "laptop", "mobile", "watch", "jeans", "bag"]
        for p in products:
            if p in text.lower():
                entities["product_name"] = p
                break

        # ----- 🔹 Brands -----
        brands = ["nike", "apple", "samsung", "lenovo", "puma", "asus"]
        for b in brands:
            if b in text.lower():
                entities["brand"] = b
                break

        # ----- 🔹 Colors -----
        colors = ["red", "blue", "black", "white", "green", "yellow"]
        for c in colors:
            if c in text.lower():
                entities["color"] = c
                break

        # ----- 🔹 Sizes -----
        size_match = re.search(r"\b(xs|s|m|l|xl|xxl|\d{2})\b", text.lower())
        if size_match:
            entities["size"] = size_match.group()

        # ----- 🔹 Price range -----
        price_match = re.search(r"(\d+)\s*-\s*(\d+)", text)
        if price_match:
            entities["price_range"] = f"{price_match.group(1)}-{price_match.group(2)}"
        elif "under" in text.lower():
            under_match = re.search(r"under\s*(\d+)", text.lower())
            if under_match:
                entities["price_range"] = f"0-{under_match.group(1)}"

        # ----- 🔹 Category -----
        categories = ["electronics", "fashion", "footwear", "accessories"]
        for cat in categories:
            if cat in text.lower():
                entities["category"] = cat
                break

        logger.info(f"🎯 Extracted Entities: {entities}")
        return entities


if __name__ == "__main__":
    extractor = EntityExtractor()
    sample_queries = [
        "Show me red Nike shoes under 3000",
        "Find blue jeans size 32",
        "Apple mobile phones between 20000-40000",
        "White shirt for men",
        "Puma footwear under 5000"
    ]

    for q in sample_queries:
        print(f"\n🧾 Query: {q}")
        print("Entities:", extractor.extract_entities(q))
