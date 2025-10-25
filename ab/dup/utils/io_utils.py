import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

def dump_accepted_code(kept: Dict[str, dict], out_dir: Path, subdir: str):
    code_dir = out_dir / subdir
    code_dir.mkdir(parents=True, exist_ok=True)

    def sanitize_filename(s: str) -> str:
        s = (s or "model").strip().replace(" ", "_")
        s = re.sub(r'[^a-zA-Z0-9._-]+', "", s)
        return s or "model"

    seen = set()
    for rid, meta in kept.items():
        base = sanitize_filename(meta.get("name") or "model")
        fn = f"{base}__{rid[:12]}.py"
        if fn in seen:
            fn = f"{base}__{rid[:12]}_{len(seen)}.py"
        seen.add(fn)
        (code_dir / fn).write_text(meta["code"], encoding="utf-8")
    return code_dir

def write_sampling_weights(kept: Dict[str, dict], out_dir: Path, upweights: List[Tuple[str, float]]):
    """
    Write sampling_weights.csv with columns: id,name,weight
    Default weight=1.0; multiply by factors for matching prefixes (first match applies).
    """
    def name_starts_with_any(name: str, prefixes: List[str]) -> bool:
        low = (name or "").lower()
        return any(low.startswith(p.lower()) for p in prefixes)

    rows = []
    for rid, meta in kept.items():
        name = meta.get("name", "")
        weight = 1.0
        for pref, mult in upweights:
            if name_starts_with_any(name, [pref]):
                weight *= mult
                break
        rows.append({"id": rid, "name": name, "weight": weight})
    pd.DataFrame(rows).to_csv(out_dir / "sampling_weights.csv", index=False)
