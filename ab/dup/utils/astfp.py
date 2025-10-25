import ast
import dataclasses
import json
import math
from typing import List, Tuple, Optional
from .hashing import sha256_text

HYPERPARAM_KEYS = {
    "kernel_size","stride","padding","groups",
    "num_heads","heads","expansion","hidden_size",
    "in_channels","out_channels","dim","width","depth",
}

def _attr_tail(n: ast.AST) -> Optional[str]:
    if isinstance(n, ast.Attribute):
        return n.attr
    if isinstance(n, ast.Name):
        return n.id
    return None

def _kw_value_bucket(v: ast.AST) -> Optional[str]:
    try:
        if isinstance(v, ast.Constant) and isinstance(v.value, (int, float)):
            x = float(v.value)
            if x <= 0:
                return "0"
            b = int(2 ** round(math.log2(x)))  # bucket to power-of-two
            return f"{b}"
        if isinstance(v, ast.Tuple):
            vals = []
            for elt in v.elts:
                s = _kw_value_bucket(elt)
                vals.append(s or "X")
            return "(" + ",".join(vals) + ")"
    except Exception:
        pass
    return None

@dataclasses.dataclass(frozen=True)
class StructuralFingerprint:
    layer_seq: Tuple[str, ...]
    layer_bag: Tuple[Tuple[str, int], ...]
    hyperparams_bag: Tuple[Tuple[str, str], ...]

    def as_hash(self) -> str:
        data = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return sha256_text(data)

def structural_fingerprint_from_source(source: str) -> StructuralFingerprint:
    try:
        tree = ast.parse(source)
    except Exception:
        return StructuralFingerprint((), (), ())

    layer_seq: List[str] = []
    layer_counter = {}
    hp_tokens: List[Tuple[str, str]] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            tail = _attr_tail(node.func)
            if tail:
                name = tail
                layer_seq.append(name)
                layer_counter[name] = layer_counter.get(name, 0) + 1
                for kw in node.keywords or []:
                    if isinstance(kw, ast.keyword) and isinstance(kw.arg, str):
                        if kw.arg in HYPERPARAM_KEYS:
                            b = _kw_value_bucket(kw.value)
                            if b is not None:
                                hp_tokens.append((kw.arg, b))
            self.generic_visit(node)
    Visitor().visit(tree)
    layer_bag = tuple(sorted(layer_counter.items()))
    return StructuralFingerprint(
        layer_seq=tuple(layer_seq[:256]),
        layer_bag=layer_bag,
        hyperparams_bag=tuple(sorted(hp_tokens)),
    )

def structural_jaccard(a: StructuralFingerprint, b: StructuralFingerprint) -> float:
    set_a = set(a.layer_bag) | set(a.hyperparams_bag)
    set_b = set(b.layer_bag) | set(b.hyperparams_bag)
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0
