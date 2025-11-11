import re
from typing import Optional

CURRENCY_SYMBOLS = {
    "$": "USD",
    "₹": "INR",
    "€": "EUR",
    "£": "GBP"
}

def normalize_currency(symbol_or_code: str) -> Optional[str]:
    if not symbol_or_code:
        return None
    symbol_or_code = symbol_or_code.strip()
    if symbol_or_code in CURRENCY_SYMBOLS:
        return CURRENCY_SYMBOLS[symbol_or_code]
    if len(symbol_or_code) == 3:
        return symbol_or_code.upper()
    return None

def extract_price_range(text: str):
    text = text.lower()
    # pattern like $200 to $500 or ₹100-₹200
    pattern = r'([$₹€£]?)(\d+)\s*(?:to|-|–|—)\s*([$₹€£]?)(\d+)'
    m = re.search(pattern, text)
    if m:
        cur1, min_p, cur2, max_p = m.groups()
        currency = normalize_currency(cur1 or cur2 or "$")
        return {"min": float(min_p), "max": float(max_p), "currency": currency}

    # pattern like under $100 or below ₹500
    pattern2 = r'(?:under|below)\s*([$₹€£]?)(\d+)'
    m = re.search(pattern2, text)
    if m:
        cur, max_p = m.groups()
        currency = normalize_currency(cur or "$")
        return {"min": None, "max": float(max_p), "currency": currency}

    return {"min": None, "max": None, "currency": None}
