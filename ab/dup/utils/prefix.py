from typing import List, Optional
from .hashing import normalize_key

def name_starts_with_any(name: str, prefixes: List[str]) -> bool:
    n = normalize_key(name)
    return any(n.startswith(normalize_key(p)) for p in prefixes)

def matched_prefix(name: str, prefixes: List[str]) -> Optional[str]:
    n = normalize_key(name)
    for p in prefixes:
        if n.startswith(normalize_key(p)):
            return p
    return None

def prefix_rank(name: str, prefer_order: List[str]) -> int:
    mp = matched_prefix(name, prefer_order)
    if mp is None:
        return len(prefer_order) + 100
    for i, p in enumerate(prefer_order):
        if normalize_key(p) == normalize_key(mp):
            return i
    return len(prefer_order) + 100
