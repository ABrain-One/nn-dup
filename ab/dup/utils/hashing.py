import hashlib
import re
from typing import Optional, Tuple

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8", errors="ignore"))

def canonical_quality_key(name: str, code: str, accuracy: Optional[float]) -> Tuple:
    # Sort by: higher acc first; longer code first; lexicographic name
    acc_key = -(accuracy if (accuracy is not None) else float("-inf"))
    return (acc_key, -len(code), name or "")

def normalize_key(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', (s or '').lower())
