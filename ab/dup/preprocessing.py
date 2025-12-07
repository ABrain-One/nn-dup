from pathlib import Path
from typing import Dict, List, Optional, Tuple
import collections
import logging
import random
import json
import pandas as pd
import sys
from tqdm import tqdm

# External API
import ab.nn.api as lemur  # programmatic dataset access

# Consts
from .consts import (
    SHINGLE_K, NUM_PERM, LSH_THRESH,
    JACCARD_THRESH_LEX, JACCARD_THRESH_STRUCT,
    RANDOM_SEED, SPLIT_RATIOS
)

# Utils
from .utils.hashing import sha256_text, canonical_quality_key
from .utils.prefix import name_starts_with_any, matched_prefix, prefix_rank
from .utils.textproc import (
    extract_model_region, strip_comments_and_docstrings,
    collect_declared_names, python_token_stream, shingles, jaccard
)
from .utils.lsh import minhash_from_shingles, cluster_near_duplicates_lsh
from .utils.astfp import StructuralFingerprint, structural_fingerprint_from_source, structural_jaccard
from .utils.family import family_tag_from_source, get_model_family
from .utils.io_utils import dump_accepted_code, write_sampling_weights
from .utils.report import build_report

# ------------------------------------------------------------------------------
# Fetch & filtering
# ------------------------------------------------------------------------------

def fetch_lemur_df() -> pd.DataFrame:
    df = lemur.data(only_best_accuracy=True)  # type: ignore[arg-type]
    if not isinstance(df, pd.DataFrame):
        raise RuntimeError("lemur.data() did not return a pandas.DataFrame")
    return df

def pick_name_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["nn", "nn_name", "model", "model_name", "name", "file", "filename"]:
        if c in df.columns:
            return c
    return None

def _normalize_key(s: str) -> str:
    import re
    return re.sub(r'[^a-z0-9]', '', (s or '').lower())

def apply_prefix_filter(df: pd.DataFrame, includes: List[str]) -> pd.DataFrame:
    if not includes:
        return df
    name_col = pick_name_column(df)
    if not name_col:
        return df.iloc[0:0].copy()
    norm_names = df[name_col].astype(str).map(_normalize_key)
    norm_prefixes = [_normalize_key(p) for p in includes]
    mask = pd.Series(False, index=df.index)
    for p in norm_prefixes:
        mask = mask | norm_names.str.startswith(p)
    return df[mask].copy()

# ------------------------------------------------------------------------------
# Splits
# ------------------------------------------------------------------------------

def build_family_splits(
    families: Dict[str, List[str]],
    ratios= SPLIT_RATIOS,
    seed=RANDOM_SEED
) -> Dict[str, str]:
    keys = list(families.keys())
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_train = int(round(n * ratios[0]))
    n_dev = int(round(n * ratios[1]))
    train_fams = set(keys[:n_train])
    dev_fams = set(keys[n_train:n_train+n_dev])
    test_fams = set(keys[n_train+n_dev:])
    assign: Dict[str, str] = {}
    for fam in keys:
        split = "train" if fam in train_fams else "dev" if fam in dev_fams else "test"
        for rec_id in families[fam]:
            assign[rec_id] = split
    return assign

# ------------------------------------------------------------------------------
# Rescue / Top-up helpers
# ------------------------------------------------------------------------------

def is_near_dup_of_kept(candidate_code: str,
                        kept_items: Dict[str, dict],
                        compare_with_prefix: Optional[str],
                        verify_thresh_generic: float,
                        verify_thresh_fractal: float) -> bool:
    model_region = extract_model_region(candidate_code)
    declared_names = collect_declared_names(model_region)
    cand_norm = strip_comments_and_docstrings(model_region)
    cand_tokens = python_token_stream(cand_norm, declared_names)
    cand_sh = shingles(cand_tokens, SHINGLE_K)
    if not cand_sh:
        return False

    for rid, meta in kept_items.items():
        if compare_with_prefix is not None:
            mp = matched_prefix(meta.get("name", ""), [compare_with_prefix])
            if mp is None:
                continue
        other_region = extract_model_region(meta["code"])
        dec2 = collect_declared_names(other_region)
        oth_norm = strip_comments_and_docstrings(other_region)
        oth_tokens = python_token_stream(oth_norm, dec2)
        oth_sh = shingles(oth_tokens, SHINGLE_K)
        if not oth_sh:
            continue
        fam = (meta.get("family_tag") or "generic").lower()
        thresh = verify_thresh_fractal if fam.startswith("fractal") else verify_thresh_generic
        if jaccard(cand_sh, oth_sh) >= thresh:
            return True
    return False

def ensure_prefix_representation(kept: Dict[str, dict],
                                 includes: List[str],
                                 record_index: Dict[str, dict],
                                 tombstones: Dict[str, dict],
                                 min_per_prefix: int,
                                 verify_thresh_generic: float,
                                 verify_thresh_fractal: float,
                                 logger: logging.Logger) -> Dict[str, dict]:
    if not includes:
        return kept

    kept_by_prefix = collections.defaultdict(list)
    for rid, meta in kept.items():
        mp = matched_prefix(meta.get("name", ""), includes)
        if mp:
            kept_by_prefix[mp].append(rid)

    removed_ids = set(tombstones.keys())
    removed_ids |= {rid for rid, v in tombstones.items() if v.get("reason") == "exact_duplicate"}

    kept_hashes = {sha256_text(meta["code"]) for meta in kept.values()}

    for p in includes:
        have = len(kept_by_prefix.get(p, []))
        need = max(0, min_per_prefix - have)
        if need == 0:
            continue

        pool = []
        for rid in removed_ids:
            rec = record_index.get(rid)
            if not rec:
                continue
            if matched_prefix(rec.get("name", ""), [p]) is None:
                continue
            if rec["code_hash"] in kept_hashes:
                continue
            pool.append(rec)

        if not pool:
            logger.warning(f"No promotable candidates found for prefix '{p}'.")
            continue

        pool.sort(key=lambda r: canonical_quality_key(r["name"], r["code"], r["accuracy"]))
        promoted = 0
        for rec in pool:
            if is_near_dup_of_kept(rec["code"], kept, p, verify_thresh_generic, verify_thresh_fractal):
                continue
            kept[rec["id"]] = {
                "name": rec["name"],
                "accuracy": rec["accuracy"],
                "code": rec["code"],
                "family_tag": rec["family_tag"],
                "reason": "rescued_by_prefix",
            }
            kept_hashes.add(rec["code_hash"])
            promoted += 1
            logger.info(f"Promoted '{rec['name']}' ({rec['id'][:12]}) to satisfy prefix '{p}'")
            if promoted >= need:
                break

        if promoted < need:
            logger.warning(f"Could only promote {promoted}/{need} needed for prefix '{p}'.")

    return kept

def diversity_topup(kept: Dict[str, dict],
                    record_index: Dict[str, dict],
                    topup_prefixes: List[str],
                    tombstones: Dict[str, dict],
                    topup_per_prefix: int,
                    topup_lex_max: float,
                    topup_struct_max: float,
                    logger: logging.Logger) -> Dict[str, dict]:
    """
    *Small* number of candidates per prefix, ONLY if they are lexically and
    structurally diverse vs kept items of the same prefix.
    """
    if not topup_prefixes or topup_per_prefix <= 0:
        return kept

    kept_fp: Dict[str, StructuralFingerprint] = {}
    logger.info("  Computing structural fingerprints for top-up...")
    for rid, meta in tqdm(kept.items(), desc="Top-up fingerprints", leave=False):
        kept_fp[rid] = structural_fingerprint_from_source(meta["code"])

    def lex_jaccard_to_kept_same_prefix(code: str, prefix: str) -> float:
        region = extract_model_region(code)
        names = collect_declared_names(region)
        norm = strip_comments_and_docstrings(region)
        toks = python_token_stream(norm, names)
        sh = shingles(toks, SHINGLE_K)
        if not sh:
            return 0.0
        best = 0.0
        for rid, meta in kept.items():
            mp = matched_prefix(meta.get("name", ""), [prefix])
            if mp is None:
                continue
            other_region = extract_model_region(meta["code"])
            dn = collect_declared_names(other_region)
            on = strip_comments_and_docstrings(other_region)
            ot = python_token_stream(on, dn)
            osh = shingles(ot, SHINGLE_K)
            if not osh:
                continue
            from math import fsum
            best = max(best, (len(sh & osh) / len(sh | osh)) if (sh | osh) else 0.0)
        return best

    def struct_jaccard_to_kept_same_prefix(code: str, prefix: str) -> float:
        fp = structural_fingerprint_from_source(code)
        best = 0.0
        for rid, meta in kept.items():
            mp = matched_prefix(meta.get("name", ""), [prefix])
            if mp is None:
                continue
            best = max(best, structural_jaccard(fp, kept_fp[rid]))
        return best

    removed_ids = list(tombstones.keys())
    added_total = 0
    logger.info("  Processing diversity top-up candidates...")
    for pref in tqdm(topup_prefixes, desc="Diversity top-up", leave=False):
        pool = []
        for rid in removed_ids:
            rec = record_index.get(rid)
            if not rec:
                continue
            if matched_prefix(rec.get("name", ""), [pref]) is None:
                continue
            if rec["code_hash"] in {sha256_text(m["code"]) for m in kept.values()}:
                continue
            pool.append(rec)

        if not pool:
            logger.info(f"[TopUp] No candidates found for prefix '{pref}'.")
            continue

        pool.sort(key=lambda r: canonical_quality_key(r["name"], r["code"], r["accuracy"]))
        added = 0
        for rec in tqdm(pool, desc=f"Top-up {pref}", leave=False):
            best_lex = lex_jaccard_to_kept_same_prefix(rec["code"], pref)
            best_struct = struct_jaccard_to_kept_same_prefix(rec["code"], pref)
            if best_lex > topup_lex_max:
                continue
            if best_struct > topup_struct_max:
                continue
            kept[rec["id"]] = {
                "name": rec["name"],
                "accuracy": rec["accuracy"],
                "code": rec["code"],
                "family_tag": rec["family_tag"],
                "reason": "diversity_topup",
            }
            kept_fp[rec["id"]] = structural_fingerprint_from_source(rec["code"])
            added += 1
            added_total += 1
            logger.info(f"[TopUp] Added '{rec['name']}' (lex_max={best_lex:.3f}, struct_max={best_struct:.3f}) for '{pref}'")
            if added >= topup_per_prefix:
                break

        if added == 0:
            logger.info(f"[TopUp] No diverse candidates passed for prefix '{pref}'.")
    if added_total == 0:
        logger.info("[TopUp] No records added by diversity top-up.")
    return kept

# ------------------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------------------

def curate_from_lemur(out_dir: Path = Path("./curation_output"),
                      includes: List[str] = None,
                      prefer_order: List[str] = None,
                      min_per_prefix: int = 1,
                      keep_per_family: int = 5,
                      lex_thresh_fractal: float = 0.97,
                      topup_prefixes: List[str] = None,
                      topup_per_prefix: int = 10,
                      topup_lex_max: float = 0.85,
                      topup_struct_max: float = 0.60,
                      dump_code_subdir: str = "accepted_code",
                      upweights: List[Tuple[str, float]] = None,
                      logger: logging.Logger = None):

    # Set defaults for None values
    if includes is None:
        includes = []
    if prefer_order is None:
        prefer_order = includes[:] if includes else []
    if topup_prefixes is None:
        topup_prefixes = []
    if upweights is None:
        upweights = []
    if logger is None:
        from .utils.logutils import setup_logging
        logger = setup_logging(verbose=False)

    # Configure tqdm to work better with logging
    # tqdm automatically handles TTY detection and logging interference
    out_dir.mkdir(parents=True, exist_ok=True)

    # 0) Fetch & filter
    logger.info("[0] Fetching LEMUR data…")
    try:
        df = fetch_lemur_df()
        logger.info(f"  Received {len(df)} rows.")
    except Exception as e:
        logger.error(f"Failed to fetch LEMUR data: {e}")
        raise
    df = apply_prefix_filter(df, includes)
    logger.info(f"  After --include filter: {len(df)} rows.")

    if df.empty:
        (out_dir / "dedup_report.md").write_text("# Curation Report\n\nNo records after filtering.\n", encoding="utf-8")
        logger.warning("No records after filtering. Done.")
        return

    name_col = pick_name_column(df)
    code_col = "nn_code"
    acc_col = "accuracy" if "accuracy" in df.columns else None

    # Diagnostics
    if name_col and code_col in df.columns:
        from collections import Counter, defaultdict
        norm_names = df[name_col].astype(str).map(_normalize_key)
        cnt = Counter()
        for p in includes:
            cnt[p] = int((norm_names.str.startswith(_normalize_key(p))).sum())
        logger.info(f"  Name-start counts by prefix (strict): {dict(cnt)}")

        code_hashes = df[code_col].astype(str).map(sha256_text)
        uniq_by_pref = defaultdict(set)
        for idx in df.index:
            nm = df.at[idx, name_col]
            mp = matched_prefix(nm, includes)
            if mp:
                uniq_by_pref[mp].add(code_hashes.at[idx])
        dbg = {p: len(s) for p, s in uniq_by_pref.items()}
        logger.info(f"  Unique code-hash counts by prefix (upper bound): {dbg}")

    # Build records
    records = []
    empty_code_count = 0
    seen_hashes = set()
    duplicate_code_count = 0

    logger.info("  Building records from dataframe...")
    # Use index iteration for better performance with tqdm
    pbar = tqdm(total=len(df), desc="Building records", leave=False, mininterval=0.5)
    for idx in df.index:
        try:
            row = df.loc[idx]
            name = str(row[name_col]) if name_col else ""
            code = str(row[code_col]) if code_col in df.columns else ""
            if not code.strip():
                empty_code_count += 1
                pbar.update(1)
                continue

            acc = float(row[acc_col]) if acc_col and pd.notnull(row[acc_col]) else None
            code_hash = sha256_text(code)
            rid = f"{code_hash}_{idx}"

            if code_hash in seen_hashes:
                duplicate_code_count += 1
            seen_hashes.add(code_hash)

            fam_tag = get_model_family(name, code)
            records.append({
                "id": rid,
                "name": name,
                "code": code,
                "accuracy": acc,
                "code_hash": code_hash,
                "row_idx": idx,
                "family_tag": fam_tag
            })
        except Exception as e:
            logger.warning(f"Error processing row {idx}: {e}")
        finally:
            pbar.update(1)
    pbar.close()

    logger.info(f"  Records with non-empty code: {len(records)}")
    logger.info(f"  Unique code hashes: {len(seen_hashes)}")
    logger.info(f"  Duplicate code instances: {duplicate_code_count}")
    if empty_code_count > 0:
        logger.info(f"  Records with empty code (skipped): {empty_code_count}")

    record_index = {r["id"]: r for r in records}

    # 1) Exact dedup (prefix-aware canonicalization)
    logger.info("[1] Exact dedup (prefix-aware)…")
    by_hash = collections.defaultdict(list)
    for r in tqdm(records, desc="Grouping by hash", leave=False):
        by_hash[r["code_hash"]].append(r)

    def choose_canonical_exact(group: List[dict]) -> dict:
        def sort_key(g: dict):
            pr = prefix_rank(g["name"], prefer_order)
            base = canonical_quality_key(g["name"], g["code"], g["accuracy"])
            return (pr,) + base
        return sorted(group, key=sort_key)[0]

    exact_removed = {}
    kept: Dict[str, dict] = {}
    for h, group in tqdm(by_hash.items(), desc="Exact dedup", leave=False):
        if len(group) == 1:
            r = group[0]
            kept[r["id"]] = {"name": r["name"], "accuracy": r["accuracy"], "code": r["code"],
                             "family_tag": r["family_tag"], "reason": "unique"}
        else:
            canon = choose_canonical_exact(group)
            kept[canon["id"]] = {"name": canon["name"], "accuracy": canon["accuracy"], "code": canon["code"],
                                 "family_tag": canon["family_tag"], "reason": "canonical_of_exact_group"}
            for g in group:
                if g["id"] != canon["id"]:
                    exact_removed[g["id"]] = {"reason": "exact_duplicate", "canonical": canon["id"],
                                              "name": g["name"], "canonical_name": canon["name"]}

    logger.info(f"  After exact dedup: {len(kept)} records kept, {len(exact_removed)} exact duplicates removed")

    # 2) Lexical near-dedup (family aware + per-prefix coverage)
    logger.info(f"[2] Lexical near-dedup (MinHash+LSH)… Processing {len(kept)} records")
    norm, toks, shing, mhash = {}, {}, {}, {}
    family_label_map = {}

    logger.info("  Computing MinHash signatures...")
    pbar = tqdm(total=len(kept), desc="Computing MinHash", leave=False, mininterval=0.5)
    for i, (k, meta) in enumerate(kept.items()):
        family_label_map[k] = get_model_family(meta["name"], meta["code"])
        model_region = extract_model_region(meta["code"])
        declared_names = collect_declared_names(model_region)
        s2 = strip_comments_and_docstrings(model_region)
        norm[k] = s2
        t = python_token_stream(s2, declared_names)
        toks[k] = t
        sh = shingles(t, SHINGLE_K)
        shing[k] = sh
        mhash[k] = minhash_from_shingles(sh, NUM_PERM)
        pbar.update(1)
    pbar.close()

    doc_keys = list(kept.keys())
    lsh_clusters = cluster_near_duplicates_lsh(doc_keys, mhash, LSH_THRESH)
    logger.info(f"  LSH candidate groups: {len(lsh_clusters)}")

    near_removed = {}
    logger.info("  Processing LSH clusters...")
    for cluster in tqdm(lsh_clusters, desc="Lexical dedup", leave=False):
        g = list(cluster)
        pairs = []
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g[i], g[j]
                famA, famB = family_label_map.get(a, ""), family_label_map.get(b, "")
                if famA != famB:
                    continue
                sim = jaccard(shing[a], shing[b])
                thresh = lex_thresh_fractal if famA.lower().startswith("fractal") else JACCARD_THRESH_LEX
                if sim >= thresh:
                    pairs.append((a, b, sim))
        if not pairs:
            continue

        parent = {x: x for x in g}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        for a, b, _ in pairs:
            union(a, b)

        comps = collections.defaultdict(list)
        for x in g:
            comps[find(x)].append(x)

        for members in comps.values():
            if len(members) <= 1:
                continue
            by_tag = collections.defaultdict(list)
            for rid in members:
                by_tag[kept[rid].get("family_tag", "generic")].append(rid)

            keep_global: set = set()
            for tag, rids in by_tag.items():
                ranked = sorted(
                    rids,
                    key=lambda rid: canonical_quality_key(
                        kept[rid]["name"], kept[rid]["code"], kept[rid]["accuracy"]
                    )
                )
                keep_tag = ranked[:keep_per_family]
                keep_global.update(keep_tag)

            # ensure each include-prefix present in this component appears at least once in keep_global
            present_prefixes = set()
            candidates_by_prefix = collections.defaultdict(list)
            for rid in members:
                mp = matched_prefix(kept[rid]["name"], includes)
                if mp:
                    present_prefixes.add(mp)
                    candidates_by_prefix[mp].append(rid)
            covered_prefixes = {
                matched_prefix(kept[r]["name"], includes)
                for r in keep_global
                if matched_prefix(kept[r]["name"], includes)
            }
            missing = [p for p in present_prefixes if p not in covered_prefixes]
            for p in missing:
                ranked_p = sorted(
                    candidates_by_prefix[p],
                    key=lambda rid: canonical_quality_key(
                        kept[rid]["name"], kept[rid]["code"], kept[rid]["accuracy"]
                    )
                )
                pick = next((r for r in ranked_p if r not in keep_global), None)
                if pick:
                    keep_global.add(pick)

            for rid in members:
                if rid in keep_global:
                    continue
                ref = next(iter(keep_global))
                sim = jaccard(shing[ref], shing[rid])
                near_removed[rid] = {
                    "reason": "lexical_near_duplicate",
                    "canonical": ref,
                    "jaccard": round(sim, 3),
                    "family_tag": kept[rid].get("family_tag", "generic"),
                    "name": kept[rid]["name"],
                    "canonical_name": kept[ref]["name"],
                }
                kept.pop(rid, None)

    logger.info(f"  After lexical near-dedup: {len(kept)} records kept, {len(near_removed)} lexical near-duplicates removed")

    # 3) Structural (AST) dedup
    logger.info("[3] Structural (AST) dedup…")
    fp: Dict[str, StructuralFingerprint] = {}
    logger.info("  Computing structural fingerprints...")
    for i, (rid, meta) in enumerate(tqdm(kept.items(), desc="AST fingerprints", leave=False)):
        fp[rid] = structural_fingerprint_from_source(meta["code"])

    by_struct_hash = collections.defaultdict(list)
    for rid, f in fp.items():
        by_struct_hash[f.as_hash()].append(rid)

    structural_removed = {}
    for sig, group in by_struct_hash.items():
        if len(group) <= 1:
            continue
        canon = sorted(group,
                       key=lambda rid: canonical_quality_key(
                           kept[rid]["name"], kept[rid]["code"], kept[rid]["accuracy"]))[0]
        for rid in group:
            if rid != canon:
                structural_removed[rid] = {
                    "reason": "structural_duplicate_exact",
                    "canonical": canon,
                    "name": kept[rid]["name"],
                    "canonical_name": kept[canon]["name"]
                }
                kept.pop(rid, None)

    keys_remaining = list(kept.keys())
    logger.info("  Comparing structural fingerprints...")
    total_comparisons = len(keys_remaining) * (len(keys_remaining) - 1) // 2
    if total_comparisons > 0:
        pbar = tqdm(total=total_comparisons, desc="Structural comparison", leave=False, mininterval=1.0)
        for i in range(len(keys_remaining)):
            for j in range(i + 1, len(keys_remaining)):
                a, b = keys_remaining[i], keys_remaining[j]
                if a not in kept or b not in kept:
                    pbar.update(1)
                    continue
                sim = structural_jaccard(fp[a], fp[b])
                if sim >= JACCARD_THRESH_STRUCT:
                    canon = a if canonical_quality_key(
                        kept[a]["name"], kept[a]["code"], kept[a]["accuracy"]
                    ) <= canonical_quality_key(
                        kept[b]["name"], kept[b]["code"], kept[b]["accuracy"]
                    ) else b
                    other = b if canon == a else a
                    structural_removed[other] = {
                        "reason": "structural_duplicate_near",
                        "canonical": canon,
                        "struct_jaccard": round(sim, 3),
                        "name": kept[other]["name"],
                        "canonical_name": kept[canon]["name"]
                    }
                    kept.pop(other, None)
                pbar.update(1)
        pbar.close()

    logger.info(f"  After structural dedup: {len(kept)} records kept, {len(structural_removed)} structural duplicates removed")

    # 4) Prefix-minimum rescue
    logger.info("[4] Ensuring prefix representation…")
    tombstones = {}
    tombstones.update(exact_removed)
    tombstones.update(near_removed)
    tombstones.update(structural_removed)

    kept = ensure_prefix_representation(
        kept, includes, record_index, tombstones, min_per_prefix,
        JACCARD_THRESH_LEX, lex_thresh_fractal, logger
    )

    # 4.5) Diversity top-up (small & safe)
    if not topup_prefixes:
        inferred = [p for p in includes if _normalize_key(p).startswith("fractalnet")]
        topup_prefixes = inferred
    if topup_prefixes:
        logger.info(f"[TopUp] Attempting diversity top-up for prefixes: {topup_prefixes}")
        kept = diversity_topup(kept, record_index, topup_prefixes, tombstones,
                               topup_per_prefix, topup_lex_max, topup_struct_max, logger)

    # 5) Families & splits
    logger.info("[5] Building family-holdout splits…")
    logger.info("  Computing missing fingerprints...")
    for rid, meta in tqdm(list(kept.items()), desc="Family fingerprints", leave=False):
        if rid not in fp:
            fp[rid] = structural_fingerprint_from_source(meta["code"])

    families: Dict[str, List[str]] = collections.defaultdict(list)
    for rid in kept.keys():
        fam_id = fp[rid].as_hash()
        families[fam_id].append(rid)
    split_assign = build_family_splits(families, SPLIT_RATIOS, RANDOM_SEED)

    # 6) Save artifacts
    kept_list = [{
        "id": rid,
        "name": kept[rid]["name"],
        "accuracy": kept[rid]["accuracy"],
        "family_tag": kept[rid]["family_tag"]
    } for rid in sorted(kept.keys())]

    (out_dir / "kept_records.json").write_text(json.dumps(kept_list, indent=2), encoding="utf-8")
    (out_dir / "tombstones.json").write_text(json.dumps(tombstones, indent=2), encoding="utf-8")
    (out_dir / "splits.json").write_text(json.dumps(split_assign, indent=2), encoding="utf-8")

    # Dump code files
    code_dir = dump_accepted_code(kept, out_dir, dump_code_subdir)
    logger.info(f"Accepted code files written to: {code_dir}")

    # Optional sampling weights
    if upweights:
        write_sampling_weights(kept, out_dir, upweights)
        logger.info(f"Sampling weights written to: {out_dir / 'sampling_weights.csv'}")

    # Report
    total_rows = len(df)
    kept_n = len(kept_list)
    report_md = build_report(
        total_rows, kept_n,
        len(exact_removed), len(near_removed), len(structural_removed),
        includes, prefer_order, lex_thresh_fractal, keep_per_family, min_per_prefix,
        topup_prefixes, topup_per_prefix, topup_lex_max, topup_struct_max
    )
    (out_dir / "dedup_report.md").write_text(report_md, encoding="utf-8")

    # Visibility
    from collections import Counter
    by_prefix = Counter()
    for rid, meta in kept.items():
        mp = matched_prefix(meta.get("name"), includes)
        if mp:
            by_prefix[mp] += 1
    by_family = Counter(kept[r]["family_tag"] for r in kept)
    logger.info(f"Kept by family: {dict(by_family)}")
    logger.info(f"Kept by prefix (strict name-only): {dict(by_prefix)}")
    logger.info("Done.")
    logger.info(f"Artifacts written to: {out_dir}")

    total_removed = len(exact_removed) + len(near_removed) + len(structural_removed)
    logger.info(f"Summary: {total_rows} → {len(records)} (non-empty code) → {kept_n} records")
    logger.info(f"  - Empty code records skipped: {total_rows - len(records)}")
    logger.info(f"  - Total duplicates removed: {total_removed}")
    logger.info(f"    - Exact duplicates: {len(exact_removed)}")
    logger.info(f"    - Lexical near-duplicates: {len(near_removed)}")
    logger.info(f"    - Structural duplicates: {len(structural_removed)}")
    logger.info(f"  - Final kept records: {kept_n}")
