import argparse
from pathlib import Path

from .preprocessing import curate_from_lemur
from .utils.logutils import setup_logging
from .consts import RANDOM_SEED

def parse_upweights(items):
    import re
    rules = []
    for it in items:
        m = re.match(r'^\s*([^:]+)\s*:\s*([0-9]*\.?[0-9]+)\s*$', it)
        if not m:
            continue
        pref = m.group(1).strip()
        mult = float(m.group(2))
        if mult <= 0:
            continue
        rules.append((pref, mult))
    return rules

def main():
    ap = argparse.ArgumentParser(description="Data curation & near-dedup from LEMUR (ab.nn.api) with diversity top-up.")
    ap.add_argument("--out", type=str, default="./curation_output", help="Output directory")
    ap.add_argument("--include", action="append", default=[],
                    help="Prefix of model/file names to include (repeatable). Example: --include rag --include FractalNet")
    ap.add_argument("--prefer-prefix-order", action="append", default=[],
                    help="Prefix priority order for exact-dedup canonicalization (repeatable). "
                         "Defaults to the order of --include if not provided.")
    ap.add_argument("--min-per-prefix", type=int, default=1,
                    help="Guarantee at least this many kept records per --include prefix (after dedup).")
    ap.add_argument("--keep-per-family", type=int, default=5,
                    help="Keep up to K exemplars per family inside a lexical near-dup cluster.")
    ap.add_argument("--lex-thresh-fractal", type=float, default=0.97,
                    help="Verification Jaccard threshold just for 'Fractal' family.")
    # Top-up controls (small, safe)
    ap.add_argument("--topup-prefix", action="append", default=[],
                    help="Run diversity top-up for this prefix (repeatable). If omitted and 'FractalNet' was included, it will be inferred.")
    ap.add_argument("--topup-per-prefix", type=int, default=10,
                    help="Max number of diversity-screened rescues per prefix.")
    ap.add_argument("--topup-lex-max", type=float, default=0.85,
                    help="Max lexical Jaccard allowed vs kept for top-up candidate (lower = more diverse).")
    ap.add_argument("--topup-struct-max", type=float, default=0.60,
                    help="Max structural Jaccard allowed vs kept for top-up candidate.")
    # Outputs
    ap.add_argument("--dump-accepted-code-dir", type=str, default="accepted_code",
                    help="Subdirectory under --out to write accepted code .py files.")
    ap.add_argument("--upweight", action="append", default=[],
                    help="Optional sampling upweight rule PREFIX:FACTOR (repeatable). Example: --upweight FractalNet:3")
    ap.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = ap.parse_args()

    logger = setup_logging(verbose=args.verbose)

    out_dir = Path(args.out).resolve()
    includes = [s.strip() for s in args.include if s and s.strip()]
    prefer_order = [s.strip() for s in args.prefer_prefix_order if s and s.strip()] or includes[:]
    topup_prefixes = [s.strip() for s in args.topup_prefix if s and s.strip()]
    upweights = parse_upweights(args.upweight)

    logger.info(f"Starting LEMUR preprocessing with {len(includes)} include filters: {includes}")
    if prefer_order:
        logger.info(f"Exact-dedup canonicalization prefers prefixes in this order: {prefer_order}")

    curate_from_lemur(out_dir,
                      includes, prefer_order,
                      args.min_per_prefix,
                      args.keep_per_family,
                      args.lex_thresh_fractal,
                      topup_prefixes,
                      args.topup_per_prefix,
                      args.topup_lex_max,
                      args.topup_struct_max,
                      args.dump_accepted_code_dir,
                      upweights,
                      logger)

if __name__ == "__main__":
    main()
