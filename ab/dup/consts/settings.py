# Global tunables & safe defaults

SHINGLE_K = 10                  # token shingle length
NUM_PERM = 256                  # MinHash permutations

# LSH retrieval threshold (looser) and verification thresholds (stricter)
LSH_THRESH = 0.85
JACCARD_THRESH_LEX = 0.90       # lexical near-dup verify (generic)
JACCARD_THRESH_STRUCT = 0.90    # structural near-dup verify

RANDOM_SEED = 42
SPLIT_RATIOS = (0.80, 0.10, 0.10)

# Token placeholders
STRING_PLACEHOLDER = "STR"
NUMBER_PLACEHOLDER = "NUM"
