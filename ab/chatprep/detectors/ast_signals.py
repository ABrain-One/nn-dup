# ab/chatprep/detectors/ast_signals.py
import ast
from typing import Dict, Any, Optional

_FRACTAL_TOKENS = {"FractalBlock", "FractalUnit", "fractal_fn"}
_ATTENTION_TOKENS = {"MultiheadAttention", "TransformerEncoderLayer", "TransformerDecoderLayer"}
_VGG_HINTS = {"MaxPool2d"}  # stacked conv+pool is common; VGG often lacks skip-add
_DENSE_HINTS = {"Dense", "DenseNet"}
_RES_HINTS = {"residual", "BasicBlock", "Bottleneck", "add", "Identity"}
_MOBILE_HINTS = {"depthwise", "MobileNet"}

class ASTSummary(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls = []
        self.class_names = set()
        self.has_module_subclass = False
        self.conv_layers = []   # (in_ch, out_ch, k)
        self.linear_layers = [] # (in_f, out_f)

    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_names.add(node.name)
        # detect subclass of nn.Module
        for b in node.bases:
            if isinstance(b, ast.Attribute) and b.attr == "Module":
                self.has_module_subclass = True
            if isinstance(b, ast.Name) and b.id == "Module":
                self.has_module_subclass = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        fn = node.func
        name = None
        if isinstance(fn, ast.Attribute):
            name = fn.attr
        elif isinstance(fn, ast.Name):
            name = fn.id
        if name:
            self.calls.append(name)

        # collect Conv2d/Linear param hints (static)
        try:
            if name == "Conv2d":
                args = node.args
                in_ch = _as_int(args, node, "in_channels", idx=0)
                out_ch = _as_int(args, node, "out_channels", idx=1)
                k = _kernel_size(node)
                self.conv_layers.append((in_ch, out_ch, k))
            elif name == "Linear":
                args = node.args
                in_f = _as_int(args, node, "in_features", idx=0)
                out_f = _as_int(args, node, "out_features", idx=1)
                self.linear_layers.append((in_f, out_f))
        except Exception:
            pass

        self.generic_visit(node)

def _as_int(args, node, kw: str, idx: int) -> int:
    # try kw first
    for k in node.keywords or []:
        if k.arg == kw and isinstance(k.value, ast.Constant) and isinstance(k.value.value, int):
            return int(k.value.value)
    # positional
    if len(args) > idx and isinstance(args[idx], ast.Constant) and isinstance(args[idx].value, int):
        return int(args[idx].value)
    return 0

def _kernel_size(node: ast.Call) -> int:
    # handle kw/pos; for tuples pick first elem
    for k in node.keywords or []:
        if k.arg == "kernel_size":
            v = k.value
            if isinstance(v, ast.Constant) and isinstance(v.value, int): return int(v.value)
            if isinstance(v, ast.Tuple) and v.elts and isinstance(v.elts[0], ast.Constant): return int(v.elts[0].value)
    if node.args and len(node.args) >= 3:
        v = node.args[2]
        if isinstance(v, ast.Constant) and isinstance(v.value, int): return int(v.value)
        if isinstance(v, ast.Tuple) and v.elts and isinstance(v.elts[0], ast.Constant): return int(v.elts[0].value)
    return 3

def estimate_params(summary: ASTSummary) -> int:
    # very rough upper bound (ignores bias; treats kxk convs)
    params = 0
    for (ic, oc, k) in summary.conv_layers:
        if ic and oc:
            params += ic * oc * (k * k)
    for (i, o) in summary.linear_layers:
        if i and o:
            params += i * o
    return int(params)

def detect_family(summary: ASTSummary, source_lower: str) -> str:
    # priority: explicit tokens > attention > mobile > resnet > densenet > vgg > generic
    if any(t in source_lower for t in map(str.lower, _FRACTAL_TOKENS)): return "fractal"
    if any(n in summary.calls for n in _ATTENTION_TOKENS): return "transformer"
    if any(w in source_lower for w in map(str.lower, _MOBILE_HINTS)): return "mobile"
    if any(w in source_lower for w in map(str.lower, _RES_HINTS)): return "resnet"
    if any(w in source_lower for w in map(str.lower, _DENSE_HINTS)): return "densenet"
    if any(n in summary.calls for n in _VGG_HINTS): return "vgg"
    return "generic"

def summarize_source(code_text: str) -> Dict[str, Any]:
    tree = ast.parse(code_text)
    s = ASTSummary()
    s.visit(tree)
    fam = detect_family(s, code_text.lower())
    pcount = estimate_params(s)
    return {
        "has_module": s.has_module_subclass,
        "family": fam,
        "param_estimate": pcount,
        "class_names": sorted(s.class_names),
        "calls": s.calls,
    }
