from __future__ import annotations
from typing import Dict, Any

SENSITIVE_HEADER_KEYS = {
    "authorization",
    "x-shopify-access-token",
    "x-api-key",
}


def redact_headers(headers: Dict[str, Any]) -> Dict[str, Any]:
    redacted = {}
    for k, v in (headers or {}).items():
        key_lower = str(k).lower()
        if key_lower in SENSITIVE_HEADER_KEYS:
            redacted[k] = "***REDACTED***"
        else:
            redacted[k] = v
    return redacted


def truncate_text(text: str | None, max_len: int = 4000) -> str | None:
    if text is None:
        return None
    if len(text) <= max_len:
        return text
    return text[: max_len - 20] + "... [truncated]"


