# ab/chatprep/utils/split_utils.py
from __future__ import annotations
import random
from collections import defaultdict
from typing import Dict, List, Tuple

def stratified_split_by_family(
    items: List[dict], seed: int = 42, ratios: Tuple[float,float,float]=(0.80,0.10,0.10)
) -> Tuple[List[dict], List[dict], List[dict]]:
    # (kept for backward compat) – item-level stratification
    fam2 = defaultdict(list)
    for x in items:
        fam = x.get("meta", {}).get("family", "unknown")
        fam2[fam].append(x)

    rnd = random.Random(seed)
    train, dev, test = [], [], []
    for fam, rows in fam2.items():
        rnd.shuffle(rows)
        n = len(rows)
        n_tr = int(round(n*ratios[0]))
        n_de = int(round(n*ratios[1]))
        train += rows[:n_tr]
        dev   += rows[n_tr:n_tr+n_de]
        test  += rows[n_tr+n_de:]
    return train, dev, test


def stratified_split_by_family_and_source(
    items: List[dict], seed: int = 42, ratios: Tuple[float,float,float]=(0.80,0.10,0.10)
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Source-level split to prevent cross-split contamination.
    All samples sharing the same meta.source_path are kept in the same split.
    """
    # 1) bucket sources per family
    fam2srcs = defaultdict(set)
    src2fam  = {}
    src2rows = defaultdict(list)
    for r in items:
        sp = r.get("meta", {}).get("source_path") or f"NA::{r.get('id')}"
        fam = r.get("meta", {}).get("family", "unknown")
        fam2srcs[fam].add(sp)
        src2fam[sp] = fam
        src2rows[sp].append(r)

    # 2) per-family source shuffle & assign
    rnd = random.Random(seed)
    assign = {}
    for fam, srcs in fam2srcs.items():
        srcs = list(srcs)
        rnd.shuffle(srcs)
        n = len(srcs)
        n_tr = int(round(n*ratios[0]))
        n_de = int(round(n*ratios[1]))
        tr = set(srcs[:n_tr]); de = set(srcs[n_tr:n_tr+n_de]); te = set(srcs[n_tr+n_de:])
        for s in tr: assign[s] = "train"
        for s in de: assign[s] = "dev"
        for s in te: assign[s] = "test"

    # 3) materialize
    out = {"train": [], "dev": [], "test": []}
    for s, rows in src2rows.items():
        split = assign.get(s, "train")
        out[split].extend(rows)

    return out["train"], out["dev"], out["test"]


def _get_source(ex: dict) -> str:
    meta = ex.get("meta") or {}
    return ex.get("source_path") or meta.get("source_path") or meta.get("source") or "unknown"

def _get_family(ex: dict) -> str:
    meta = ex.get("meta") or {}
    return meta.get("family") or ex.get("family") or "generic"

def stratified_family_split_by_source(
    examples: List[dict],
    ratios: Tuple[float, float, float] = (0.80, 0.10, 0.10),
    seed: int = 42,
) -> Dict[str, List[dict]]:
    """
    Group all examples by source file (to prevent leakage), assign each group
    to a split, while keeping family proportions roughly intact.
    """
    rng = random.Random(seed)

    # group by source
    by_src: Dict[str, List[dict]] = defaultdict(list)
    for ex in examples:
        by_src[_get_source(ex)].append(ex)

    # characterize each source by its majority family (lightweight strata)
    src_items = []
    for src, exs in by_src.items():
        fam_count = defaultdict(int)
        for e in exs:
            fam_count[_get_family(e)] += 1
        majority_family = max(fam_count.items(), key=lambda kv: kv[1])[0]
        src_items.append((src, majority_family, exs))

    # stratify by majority family: shuffle inside each family, then fill splits
    fam_buckets: Dict[str, List[Tuple[str, str, List[dict]]]] = defaultdict(list)
    for tup in src_items:
        fam_buckets[tup[1]].append(tup)
    for fam in fam_buckets:
        rng.shuffle(fam_buckets[fam])

    train, dev, test = [], [], []
    for fam, items in fam_buckets.items():
        n = len(items)
        n_train = round(n * ratios[0])
        n_dev   = round(n * ratios[1])
        # ensure sum matches n
        while n_train + n_dev > n:
            n_dev = max(0, n_dev-1)
        n_test = n - n_train - n_dev

        train.extend(items[:n_train])
        dev.extend(items[n_train:n_train+n_dev])
        test.extend(items[n_train+n_dev:])

    # flatten groups back to examples
    out = {
        "train": [e for _,_,grp in train for e in grp],
        "dev":   [e for _,_,grp in dev   for e in grp],
        "test":  [e for _,_,grp in test  for e in grp],
    }

    # safety: no overlap by source
    tr_sources = {_get_source(e) for e in out["train"]}
    dv_sources = {_get_source(e) for e in out["dev"]}
    te_sources = {_get_source(e) for e in out["test"]}
    assert not (tr_sources & dv_sources), "train/dev source leakage detected"
    assert not (tr_sources & te_sources), "train/test source leakage detected"
    assert not (dv_sources & te_sources), "dev/test source leakage detected"

    return out
