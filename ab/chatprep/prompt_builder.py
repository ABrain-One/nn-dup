# ab/chatprep/prompt_builder.py
from __future__ import annotations

import re
import ast
import random
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

from .consts import (
    SYSTEM_POLICY,
    DEFAULT_DATASETS,
    ALLOWED_TRICKS_POOL,
    PARAM_BUCKETS,
    DATASETS,
)
from .utils.code_io import load_py_files
from .example_builder import build_examples_from_code
from .schema import write_jsonl
from .renderer import render_with_template
from .utils.split_utils import stratified_split_by_family

# ----------------------------
# Canonical group key (prefer meta.group_key; fallback: resolved source_path)
# ----------------------------
def _group_key(item: dict) -> str:
    meta = item.get("meta", {}) or {}
    gk = meta.get("group_key")
    if gk:
        return str(gk)
    src = meta.get("source_path")
    if src and isinstance(src, str) and src.strip():
        try:
            return Path(src).resolve().as_posix()
        except Exception:
            return Path(src).as_posix()
    return f"NA::{item.get('id')}"

# ----------------------------
# Message construction helpers
# ----------------------------
def _pick_dataset(family: str) -> Tuple[str, str]:
    cls_only: List[Tuple[str, str]] = [
        (name, shape)
        for (name, shape) in DEFAULT_DATASETS
        if name in DATASETS
        and DATASETS[name].get("modality") == "image"
        and "classification" in DATASETS[name].get("tasks", [])
    ]
    if family in {"mobile", "vgg"}:
        small_pool = [(n, s) for (n, s) in cls_only if s in {"1x28x28", "3x32x32"}]
        pool = small_pool or cls_only
    else:
        pool = cls_only or DEFAULT_DATASETS
    return random.choice(pool)

def _bucket_cap(n_params: int) -> int:
    for cap in PARAM_BUCKETS:
        if n_params <= cap:
            return int(cap)
    return int(PARAM_BUCKETS[-1])

def build_messages(example_id: str, code_text: str, det: Dict[str, Any]) -> Dict[str, Any]:
    ds_name, ds_shape = _pick_dataset(det.get("family", "generic"))
    est = max(int(det.get("param_estimate", 50_000)), 50_000)
    cap = _bucket_cap(int(est * 1.3))
    tricks = ", ".join(sorted(random.sample(ALLOWED_TRICKS_POOL, k=min(3, len(ALLOWED_TRICKS_POOL)))))

    user_text = (
        "Task: Design a PyTorch CV model for image classification.\n"
        f"Dataset: {ds_name} ({ds_shape}, channels-first CxHxW).\n"
        f"Resource limits: params ≤ {cap:.0f}; latency budget: tight (edge-friendly).\n"
        "Constraints: use standard layers only; no pretrained weights.\n"
        f"Allowed training tricks (handled by trainer): {tricks}.\n"
        "Output contract: one Python code block defining a complete nn.Module "
        "(e.g., class Net(nn.Module))."
    )
    return {"system": SYSTEM_POLICY, "user": user_text, "assistant_code": f"```python\n{code_text.strip()}\n```"}

# ----------------------------
# Validation & normalization
# ----------------------------
_CODE_FENCE_RE = re.compile(r"```python\s*(.*?)```", flags=re.S)
_ANY_FENCE_RE  = re.compile(r"```\s*(.*?)```", flags=re.S)

def _sanitize_assistant_to_single_python_block(text: str) -> str:
    import re as _re, ast as _ast
    py_blocks = _re.findall(r"```python\s*(.*?)```", text, flags=_re.S)
    if not py_blocks:
        any_blocks = _re.findall(r"```\s*(.*?)```", text, flags=_re.S)
        py_blocks = any_blocks
    if not py_blocks:
        return f"```python\n{text.strip()}\n```"
    def ok(c: str) -> bool:
        try: _ast.parse(c); return True
        except Exception: return False
    blocks_sorted = sorted(py_blocks, key=len, reverse=True)
    for blk in blocks_sorted:
        if ok(blk): return f"```python\n{blk.strip()}\n```"
    joined = "\n\n".join(b.strip() for b in blocks_sorted)
    if ok(joined): return f"```python\n{joined}\n```"
    return f"```python\n{blocks_sorted[0].strip()}\n```"

def _extract_code(text: str) -> Optional[str]:
    m = _CODE_FENCE_RE.search(text)
    return m.group(1).strip() if m else None

def _is_parseable_python(code: str) -> bool:
    try:
        ast.parse(code); return True
    except Exception:
        return False

def _has_nn_module_subclass(code: str) -> bool:
    try: tree = ast.parse(code)
    except Exception: return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for b in node.bases:
                if getattr(b, "id", "") == "Module": return True
                if getattr(b, "attr", "") == "Module": return True
    return False

# ----------------------------
# Grouped, family-stratified split
# ----------------------------
def _source_grouped_split_with_family_stratification(
    items: List[dict], seed: int = 42, ratios: Tuple[float, float, float] = (0.80, 0.10, 0.10)
) -> Dict[str, List[dict]]:
    from collections import Counter, defaultdict as _dd
    rnd = random.Random(seed)

    source_groups = _dd(list)
    for it in items:
        source_groups[_group_key(it)].append(it)

    g2fam = {}
    for g, rows in source_groups.items():
        fams = [r.get("meta", {}).get("family", "unknown") for r in rows]
        g2fam[g] = Counter(fams).most_common(1)[0][0]

    fam2groups = _dd(list)
    for g, fam in g2fam.items():
        fam2groups[fam].append(g)

    train_g, dev_g, test_g = set(), set(), set()

    def _split_counts(n: int, r: Tuple[float, float, float]) -> Tuple[int, int, int]:
        t = int(n * r[0]); d = int(n * r[1]); te = n - t - d
        if n >= 3 and te == 0:
            if d > 1: d -= 1; te += 1
            elif t > 1: t -= 1; te += 1
        return max(t,0), max(d,0), max(te,0)

    for fam, groups in fam2groups.items():
        rnd.shuffle(groups)
        n = len(groups)
        if n == 1:
            train_g.add(groups[0]); continue
        if n == 2:
            train_g.add(groups[0]); test_g.add(groups[1]); continue
        nt, nd, nte = _split_counts(n, ratios)
        train_g.update(groups[:nt])
        dev_g.update(groups[nt:nt+nd])
        test_g.update(groups[nt+nd:])

    train_items, dev_items, test_items = [], [], []
    for g, rows in source_groups.items():
        if g in train_g: train_items.extend(rows)
        elif g in dev_g: dev_items.extend(rows)
        elif g in test_g: test_items.extend(rows)
        else: train_items.extend(rows)

    return {"train": train_items, "dev": dev_items, "test": test_items}

def _verify_no_overlap(train: List[dict], dev: List[dict], test: List[dict]) -> Dict[str, int]:
    train_sources = {_group_key(item) for item in train}
    dev_sources   = {_group_key(item) for item in dev}
    test_sources  = {_group_key(item) for item in test}
    return {
        "train_dev_overlap": len(train_sources & dev_sources),
        "train_test_overlap": len(train_sources & test_sources),
        "dev_test_overlap": len(dev_sources & test_sources),
    }

def _enforce_disjoint_sources(
    train: List[dict], dev: List[dict], test: List[dict]
) -> Tuple[List[dict], List[dict], List[dict], Dict[str, List[str]]]:
    """Keep each group in a single split (priority: train > dev > test)."""
    from collections import defaultdict as _dd
    g2loc = _dd(set)
    for it in train: g2loc[_group_key(it)].add("train")
    for it in dev:   g2loc[_group_key(it)].add("dev")
    for it in test:  g2loc[_group_key(it)].add("test")

    overlaps = {g: sorted(list(l)) for g, l in g2loc.items() if len(l) > 1}
    if not overlaps: return train, dev, test, {}

    priority = ("train", "dev", "test")
    keep_for = {g: next(p for p in priority if p in locs) for g, locs in overlaps.items()}

    def _filter(rows: List[dict], name: str) -> List[dict]:
        out = []
        for it in rows:
            g = _group_key(it)
            if g in keep_for and keep_for[g] != name:
                continue
            out.append(it)
        return out

    return _filter(train, "train"), _filter(dev, "dev"), _filter(test, "test"), overlaps

# ----------------------------
# Main API
# ----------------------------
@dataclass
class ChatPrepConfig:
    accepted_dir: str = "curation_output/accepted_code"
    out_dir: str = "curation_output/chat_data"

    no_infill: bool = False
    seed: int = 42

    fix_fences: bool = True
    drop_unparseable: bool = True
    require_module_subclass: bool = True
    write_drop_report: bool = True

    group_by_source: bool = True
    split_ratios: Tuple[float, float, float] = (0.80, 0.10, 0.10)

    model_name: Optional[str] = None

    train_name: str = "train.jsonl"
    dev_name: str = "dev.jsonl"
    test_name: str = "test.jsonl"

    def __post_init__(self):
        if self.accepted_dir is None: self.accepted_dir = "curation_output/accepted_code"
        if self.out_dir is None: self.out_dir = "curation_output/chat_data"
        if self.train_name is None: self.train_name = "train.jsonl"
        if self.dev_name is None: self.dev_name = "dev.jsonl"
        if self.test_name is None: self.test_name = "test.jsonl"
        if self.split_ratios is None: self.split_ratios = (0.80, 0.10, 0.10)

    def run(self) -> Dict[str, Any]:
        random.seed(self.seed)
        in_dir = Path(self.accepted_dir)
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) curated code → examples
        pairs = load_py_files(str(in_dir))
        examples = self._build_examples(pairs, add_infill=(not self.no_infill))

        # 2) normalize & validate
        examples = self._normalize_examples_to_dict(examples)
        examples, dropped = self._sanitize_and_filter(examples)

        # 3) split
        if self.group_by_source:
            splits = _source_grouped_split_with_family_stratification(
                examples, seed=self.seed, ratios=self.split_ratios
            )
            train, dev, test = splits["train"], splits["dev"], splits["test"]
        else:
            fallback = stratified_split_by_family(examples, seed=self.seed, ratios=self.split_ratios)
            if isinstance(fallback, dict):
                train, dev, test = fallback["train"], fallback["dev"], fallback["test"]
            else:
                # tuple: (train, dev, test)
                train, dev, test = fallback

        # 3.1) ALWAYS enforce one-group-one-split
        train, dev, test, fixed = _enforce_disjoint_sources(train, dev, test)
        if fixed:
            print(f"[split] Fixed {len(fixed)} overlapping source groups via enforcement.")

        # 3.2) verify after enforcement
        overlap_check = _verify_no_overlap(train, dev, test)
        if any(overlap_check.values()):
            from collections import defaultdict as _dd, json as _json
            offenders = _dd(set)
            for name, rows in (("train", train), ("dev", dev), ("test", test)):
                for it in rows: offenders[_group_key(it)].add(name)
            bad = {g: sorted(list(v)) for g, v in offenders.items() if len(v) > 1}
            with open(out_dir / "_overlap_groups.json", "w", encoding="utf-8") as f:
                _json.dump(bad, f, indent=2)
            raise RuntimeError(f"LEAKAGE after enforcement: {overlap_check}")

        # 4) write jsonl
        train_path = out_dir / self.train_name
        dev_path   = out_dir / self.dev_name
        test_path  = out_dir / self.test_name
        write_jsonl(str(train_path), train)
        write_jsonl(str(dev_path), dev)
        write_jsonl(str(test_path), test)

        # 4.5) manifest (canonical)
        manifest = self._generate_manifest(train, dev, test)
        manifest_path = out_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            import json as _json; _json.dump(manifest, f, indent=2)

        # Double-check post-write
        overlap_check = _verify_no_overlap(train, dev, test)
        if any(overlap_check.values()):
            raise RuntimeError(
                f"LEAKAGE DETECTED! Overlaps found: {overlap_check}. "
                f"Manifest saved to {manifest_path}"
            )

        # 5) optional render
        rendered_paths = {}
        if self.model_name:
            rend_dir = out_dir / "rendered"
            rend_dir.mkdir(exist_ok=True)
            for split_name, split_data in [("train", train), ("dev", dev), ("test", test)]:
                rendered = render_with_template(split_data, self.model_name)
                rp = rend_dir / f"{split_name}.jsonl"
                with open(rp, "w", encoding="utf-8") as f:
                    for item in rendered:
                        f.write(self._json_dumps(item) + "\n")
                rendered_paths[split_name] = str(rp)

        # 6) drop report
        if self.write_drop_report and dropped:
            with open(out_dir / "_drop_report.jsonl", "w", encoding="utf-8") as f:
                for r in dropped:
                    f.write(self._json_dumps(r) + "\n")

        # 7) stats
        def _uniq_sources(rows: List[dict]) -> int:
            return len({_group_key(r) for r in rows})

        return {
            "counts": {"train": len(train), "dev": len(dev), "test": len(test),
                       "total": len(train) + len(dev) + len(test)},
            "paths": {"train": str(train_path), "dev": str(dev_path), "test": str(test_path),
                      "rendered": rendered_paths or None, "manifest": str(manifest_path)},
            "dropped": len(dropped),
            "unique_sources": {"train": _uniq_sources(train), "dev": _uniq_sources(dev), "test": _uniq_sources(test)},
            "config": {
                "fix_fences": self.fix_fences,
                "drop_unparseable": self.drop_unparseable,
                "require_module_subclass": self.require_module_subclass,
                "group_by_source": self.group_by_source,
            },
            "overlap_verification": _verify_no_overlap(train, dev, test),
        }

    # internals
    def _build_examples(self, file_pairs: List[Tuple[str, str]], add_infill: bool) -> List[Any]:
        exs: List[Any] = []
        for path, code in file_pairs:
            exs.extend(build_examples_from_code(path, code, add_infill=add_infill))
        return exs

    @staticmethod
    def _ex_to_dict(ex: Any) -> Dict[str, Any]:
        if isinstance(ex, dict): return ex
        if hasattr(ex, "to_dict") and callable(getattr(ex, "to_dict")): return ex.to_dict()
        if is_dataclass(ex): return asdict(ex)
        return {
            "id": getattr(ex, "id", None),
            "messages": getattr(ex, "messages", None),
            "meta": getattr(ex, "meta", None),
        }

    def _normalize_examples_to_dict(self, examples: List[Any]) -> List[Dict[str, Any]]:
        return [self._ex_to_dict(ex) for ex in examples]

    def _sanitize_and_filter(self, examples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        kept: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []
        for ex in examples:
            msgs = ex.get("messages", [])
            if not (isinstance(msgs, list) and len(msgs) == 3):
                ex["_drop_reason"] = "bad_schema_len"; dropped.append(ex); continue
            roles = [m.get("role") for m in msgs]
            if roles != ["system", "user", "assistant"]:
                ex["_drop_reason"] = "bad_roles"; dropped.append(ex); continue
            if self.fix_fences:
                msgs[-1]["content"] = _sanitize_assistant_to_single_python_block(msgs[-1]["content"])
            code = _extract_code(msgs[-1].get("content", "") or "")
            if code is None:
                ex["_drop_reason"] = "no_code_fence"; dropped.append(ex); continue
            if self.drop_unparseable and not _is_parseable_python(code):
                ex["_drop_reason"] = "parse_fail"; dropped.append(ex); continue
            if self.require_module_subclass and not _has_nn_module_subclass(code):
                ex["_drop_reason"] = "no_nn_module"; dropped.append(ex); continue
            ex["messages"] = msgs
            kept.append(ex)
        return kept, dropped

    def _generate_manifest(self, train: List[dict], dev: List[dict], test: List[dict]) -> Dict[str, Any]:
        from collections import Counter
        def _extract(split: List[dict], name: str) -> Dict[str, Any]:
            srcs = {_group_key(it) for it in split}
            fams = [it.get("meta", {}).get("family", "unknown") for it in split]
            return {
                "split": name,
                "count": len(split),
                "unique_sources": len(srcs),
                "source_keys": sorted(list(srcs)),
                "family_distribution": dict(Counter(fams).most_common()),
            }
        train_m = _extract(train, "train")
        dev_m   = _extract(dev, "dev")
        test_m  = _extract(test, "test")

        train_sources = set(train_m["source_keys"])
        dev_sources   = set(dev_m["source_keys"])
        test_sources  = set(test_m["source_keys"])
        overlaps = {
            "train_dev": sorted(list(train_sources & dev_sources)),
            "train_test": sorted(list(train_sources & test_sources)),
            "dev_test": sorted(list(dev_sources & test_sources)),
        }
        total_sources = len(train_sources | dev_sources | test_sources)
        total_examples = len(train) + len(dev) + len(test)
        return {
            "summary": {
                "counts": {"train": len(train), "dev": len(dev), "test": len(test)},
                "total_examples": total_examples,
                "unique_sources": total_sources,
                "overlap": {
                    "train_dev": len(overlaps["train_dev"]),
                    "train_test": len(overlaps["train_test"]),
                    "dev_test": len(overlaps["dev_test"]),
                },
            },
            "splits": {"train": train_m, "dev": dev_m, "test": test_m},
            "overlaps": overlaps,
            "config": {"group_by_source": self.group_by_source, "split_ratios": self.split_ratios, "seed": self.seed},
        }

    @staticmethod
    def _json_dumps(o: Any) -> str:
        import json
        return json.dumps(o, ensure_ascii=False)
