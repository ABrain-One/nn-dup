import time
from ..consts import SHINGLE_K, NUM_PERM, LSH_THRESH, JACCARD_THRESH_LEX, JACCARD_THRESH_STRUCT, SPLIT_RATIOS

def build_report(total_rows: int,
                 kept_n: int,
                 exact_removed_n: int,
                 near_removed_n: int,
                 struct_removed_n: int,
                 includes,
                 prefer_order,
                 lex_thresh_fractal,
                 keep_per_family,
                 min_per_prefix,
                 topup_prefixes,
                 topup_per_prefix,
                 topup_lex_max,
                 topup_struct_max):
    return f"""# Curation Report (LEMUR API)

Generated at: {time.ctime()}

## Summary
- Total rows fetched from LEMUR: **{total_rows}**
- Exact duplicates removed: **{exact_removed_n}**
- Lexical near-duplicates removed (verify ≥ {JACCARD_THRESH_LEX} / Fractal ≥ {lex_thresh_fractal}): **{near_removed_n}**
- Structural duplicates removed (exact/near, ≥ {JACCARD_THRESH_STRUCT}): **{struct_removed_n}**
- **Kept for training/eval:** **{kept_n}** records

## Filters
- Include prefixes (strict name-only): {includes if includes else "ALL (no prefix filter)"}
- Prefer prefix order (exact-canonical): {prefer_order if prefer_order else "(none)"}

## Parameters
- Shingle length (k): `{SHINGLE_K}`, MinHash permutations: `{NUM_PERM}`, LSH retrieval: `{LSH_THRESH}`
- Lexical Jaccard verify (generic): `{JACCARD_THRESH_LEX}`, (Fractal): `{lex_thresh_fractal}`
- Keep per family (K): `{keep_per_family}`, Min per prefix: `{min_per_prefix}`
- Diversity Top-Up: prefixes={topup_prefixes or []}, per_prefix={topup_per_prefix}, lex≤{topup_lex_max}, struct≤{topup_struct_max}
- Train/dev/test ratios: `{SPLIT_RATIOS}`

## Notes
- Source: `ab.nn.api.data(only_best_accuracy=True)`.
- `--include` matches normalized *name/file* prefix only.
- We did **not** globally relax dedup rules; top-up is small and **diversity-screened**.
"""
