# ab/chatprep/utils/manifest.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Set

def _read_sources(p: Path) -> Set[str]:
    out = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                import json as _json
                ex = _json.loads(line)
                meta = ex.get("meta") or {}
                src = ex.get("source_path") or meta.get("source_path") or meta.get("source")
                if src: out.add(src)
            except Exception:
                pass
    return out

def write_manifest(parent_dir: str) -> str:
    parent = Path(parent_dir)
    tr = parent / "train.jsonl"
    dv = parent / "dev.jsonl"
    te = parent / "test.jsonl"

    counts = { "train": sum(1 for _ in tr.open("r", encoding="utf-8") if _.strip()),
               "dev":   sum(1 for _ in dv.open("r", encoding="utf-8") if _.strip()),
               "test":  sum(1 for _ in te.open("r", encoding="utf-8") if _.strip()) }
    total = counts["train"] + counts["dev"] + counts["test"]

    tr_s, dv_s, te_s = _read_sources(tr), _read_sources(dv), _read_sources(te)
    summary = {
        "counts": counts,
        "total_examples": total,
        "unique_sources": len(tr_s | dv_s | te_s),
        "overlap": {
            "train_dev": len(tr_s & dv_s),
            "train_test": len(tr_s & te_s),
            "dev_test": len(dv_s & te_s),
        }
    }
    out = { "summary": summary }
    (parent / "manifest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return str(parent / "manifest.json")
