import re
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class EntityExtractor:
    def __init__(self):
        logger.info("✅ EntityExtractor initialized.")

    def extract_entities(self, text: str):
        """Main method to extract entities from user input."""
        logger.info(f"🔍 Extracting entities from: {text}")

        text_lower = text.lower()

        # --- Brand extraction (whole word matching to avoid substring issues) ---
        brand = None
        known_brands = ["nike", "adidas", "puma", "reebok", "apple", "samsung"]
        for b in known_brands:
            # Use word boundary regex to match whole words only
            if re.search(r'\b' + re.escape(b) + r'\b', text_lower):
                brand = b
                break

        # --- Color extraction (whole word matching to avoid substring issues like "red" in "ordered") ---
        color = None
        known_colors = ["red", "blue", "black", "white", "green", "yellow", "orange", "purple", "pink", "brown", "gray", "grey"]
        for c in known_colors:
            # Use word boundary regex to match whole words only
            if re.search(r'\b' + re.escape(c) + r'\b', text_lower):
                # Normalize "grey" to "gray" for consistency
                color = "gray" if c == "grey" else c
                break

        # --- Product name / category (whole word matching) ---
        product_name = None
        known_products = ["shoes", "phone", "t-shirt", "watch", "laptop", "shirt", "jeans", "bag"]
        for p in known_products:
            # Use word boundary regex to match whole words only
            # For hyphenated words like "t-shirt", use a more flexible pattern
            if "-" in p:
                # For hyphenated words, match with word boundaries or hyphens
                pattern = r'(?<![a-z])' + re.escape(p) + r'(?![a-z])'
            else:
                pattern = r'\b' + re.escape(p) + r'\b'
            if re.search(pattern, text_lower):
                product_name = p
                break

        # --- Category extraction (whole word matching) ---
        category = None
        known_categories = ["running", "casual", "sports", "formal", "electronics", "fashion", "footwear", "accessories"]
        for cat in known_categories:
            # Use word boundary regex to match whole words only
            if re.search(r'\b' + re.escape(cat) + r'\b', text_lower):
                category = cat
                break

        # --- Size extraction ---
        size = None
        match = re.search(r"size\s*(\d+)", text_lower)
        if match:
            size = match.group(1)

        # --- Price range (under / below / between / from-to) ---
        price_range = {"min": None, "max": None, "currency": "INR"}

        # Case: under or below a price
        if "under" in text_lower or "below" in text_lower:
            match = re.search(r"(?:under|below)\s*\$?(\d+)", text_lower)
            if match:
                price_range["max"] = int(match.group(1))
                price_range["currency"] = "USD" if "$" in text_lower else "INR"

        # Case: from-to or between prices
        elif ("from" in text_lower and "to" in text_lower) or "between" in text_lower:
            match = re.search(r"(?:from|between)\s*\$?(\d+)\s*(?:to|and)\s*\$?(\d+)", text_lower)
            if match:
                price_range["min"] = int(match.group(1))
                price_range["max"] = int(match.group(2))
                price_range["currency"] = "USD" if "$" in text_lower else "INR"

        entities = {
            "product_name": product_name,
            "brand": brand,
            "color": color,
            "size": size,
            "price_range": price_range,
            "category": category
        }

        logger.info(f"🎯 Extracted Entities: {entities} (from: {text})")
        return entities
