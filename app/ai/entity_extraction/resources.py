PRODUCT_KEYWORDS = ["shoes", "phones", "laptop", "shirt", "watch"]
CATEGORIES = ["running", "gaming", "formal", "casual"]
BRANDS = ["nike", "adidas", "apple", "samsung", "puma"]
COLORS = ["black", "white", "red", "blue", "green"]

def find_keyword_from_map(text: str, keywords: list[str]) -> str | None:
    text = text.lower()
    for word in keywords:
        if word in text:
            return word
    return None
