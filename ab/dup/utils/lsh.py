import collections
from typing import Dict, List, Set
from datasketch import MinHash, MinHashLSH

from ..consts import NUM_PERM, LSH_THRESH

def minhash_from_shingles(sh: set, num_perm: int = NUM_PERM) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for s in sh:
        m.update(s.encode("utf-8"))
    return m

def cluster_near_duplicates_lsh(
    doc_keys: List[str],
    doc_minhashes: Dict[str, MinHash],
    threshold: float = LSH_THRESH,
) -> List[Set[str]]:
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    for k in doc_keys:
        lsh.insert(k, doc_minhashes[k])

    parent = {k: k for k in doc_keys}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for k in doc_keys:
        for n in lsh.query(doc_minhashes[k]):
            if n != k:
                union(k, n)

    groups = collections.defaultdict(set)
    for k in doc_keys:
        groups[find(k)].add(k)
    return [g for g in groups.values() if len(g) >= 2]
