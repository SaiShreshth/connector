import re
from html import unescape

TAG_RE = re.compile(r"<[^>]+>")

MD_PATTERNS = [
    (re.compile(r"(\*\*|__)(.*?)\1", re.S), r"\2"),   # **bold** or __bold__
    (re.compile(r"(\*|_)(.*?)\1", re.S), r"\2"),      # *em* or _em_
    (re.compile(r"^#{1,6}\s*", re.M), ""),            # # headings
    (re.compile(r"^>\s?", re.M), ""),                 # > blockquotes
    (re.compile(r"`{1,3}([^`]+)`{1,3}"), r"\1"),      # `code`
    (re.compile(r"!\[[^\]]*\]\([^)]+\)"), ""),        # images
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),    # [text](url)
]

def clean_text(text: str) -> str:
    if not text:
        return ""
    s = unescape(text)
    s = TAG_RE.sub(" ", s)
    for rx, repl in MD_PATTERNS:
        s = rx.sub(repl, s)
    return re.sub(r"\s+", " ", s).strip()

def parse_price_value(price_like) -> float | None:
    if price_like is None:
        return None
    s = str(price_like).replace(",", "")
    m = re.search(r"(\d+(\.\d+)?)", s)
    try:
        return float(m.group(1)) if m else None
    except Exception:
        return None
