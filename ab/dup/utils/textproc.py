import ast
import io
import re
import tokenize
from keyword import iskeyword
from typing import List, Set, Tuple

from ..consts import SHINGLE_K, STRING_PLACEHOLDER, NUMBER_PLACEHOLDER

# Common layer/type hints – kept as tokens (helps preserve architectural signal)
LAYER_HINTS = {
    "Conv1d","Conv2d","Conv3d",
    "BatchNorm1d","BatchNorm2d","BatchNorm3d",
    "LayerNorm","GroupNorm","InstanceNorm2d",
    "ReLU","GELU","SiLU","LeakyReLU","ELU",
    "MaxPool2d","AvgPool2d","AdaptiveAvgPool2d",
    "Dropout","Dropout2d",
    "Linear","Flatten",
    "LSTM","GRU",
    "MultiheadAttention","TransformerEncoderLayer","TransformerDecoderLayer",
    "Upsample","ConvTranspose2d",
    "SEBlock","SqueezeExcite",
    "Sequential","ModuleList",
}

def extract_model_region(source: str) -> str:
    """
    Return only text of classes that subclass nn.Module and helper functions
    they call (e.g., fractal_fn). Fallback to full source on failure.
    """
    try:
        tree = ast.parse(source)
        lines = source.splitlines(True)
        spans: List[Tuple[int,int]] = []
        module_class_names = set()

        class FirstPass(ast.NodeVisitor):
            def visit_ClassDef(self, node):
                is_module = any(
                    (isinstance(b, ast.Attribute) and b.attr == "Module") or
                    (isinstance(b, ast.Name) and b.id == "Module")
                    for b in node.bases
                )
                if is_module:
                    module_class_names.add(node.name)
                    spans.append((node.lineno, node.end_lineno))
                self.generic_visit(node)
        FirstPass().visit(tree)

        helpers = []
        class RefGrab(ast.NodeVisitor):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    helpers.append(node.func.id)
                self.generic_visit(node)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in module_class_names:
                RefGrab().visit(node)

        helper_names = set(helpers)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in helper_names:
                spans.append((node.lineno, node.end_lineno))

        if not spans:
            return source
        spans = sorted(spans)
        chunks = []
        cur_s, cur_e = spans[0]
        for s, e in spans[1:]:
            if s <= cur_e + 1:
                cur_e = max(cur_e, e)
            else:
                chunks.append("".join(lines[cur_s - 1:cur_e]))
                cur_s, cur_e = s, e
        chunks.append("".join(lines[cur_s - 1:cur_e]))
        return "\n".join(chunks)
    except Exception:
        return source

def strip_comments_and_docstrings(text: str) -> str:
    io_obj = io.StringIO(text)
    out = []
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    try:
        for tok in tokenize.generate_tokens(io_obj.readline):
            tok_type, tok_string, (srow, scol), (erow, ecol), _ = tok
            if tok_type == tokenize.COMMENT:
                continue
            if tok_type == tokenize.STRING:
                if prev_toktype in (tokenize.INDENT, tokenize.NEWLINE):
                    prev_toktype = tok_type
                    continue
            if srow > last_lineno:
                last_col = 0
            if scol > last_col:
                out.append(" " * (scol - last_col))
            out.append(tok_string)
            prev_toktype = tok_type
            last_col = ecol
            last_lineno = erow
    except Exception:
        return re.sub(r"(?m)#.*$", "", text)
    return "".join(out)

def collect_declared_names(source: str) -> Set[str]:
    declared_names: Set[str] = set()
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                declared_names.add(node.name)
            elif isinstance(node, ast.FunctionDef):
                declared_names.add(node.name)
    except Exception:
        pass
    return declared_names

def python_token_stream(normalized_source: str, declared_names: Set[str] = None) -> List[str]:
    if declared_names is None:
        declared_names = set()
    tokens: List[str] = []
    try:
        g = tokenize.generate_tokens(io.StringIO(normalized_source).readline)
        for tok_type, tok_string, *_ in g:
            if tok_type == tokenize.NAME:
                if (iskeyword(tok_string) or
                    tok_string in LAYER_HINTS or
                    tok_string in declared_names or
                    tok_string.startswith((
                        'Conv','Batch','Layer','Group','Instance','Max','Avg','Adaptive',
                        'Dropout','Linear','Flatten','LSTM','GRU','Multihead','Transformer',
                        'Upsample','SE','Squeeze','Sequential','Module','Fractal','ResNet','Dense','VGG'
                    )) or
                    any(char.isupper() for char in tok_string[1:])):
                    tokens.append(tok_string)
                else:
                    tokens.append("ID")
            elif tok_type == tokenize.OP:
                tokens.append(tok_string)
            elif tok_type == tokenize.NUMBER:
                tokens.append(NUMBER_PLACEHOLDER)
            elif tok_type == tokenize.STRING:
                tokens.append(STRING_PLACEHOLDER)
            else:
                continue
    except (IndentationError, SyntaxError, tokenize.TokenError):
        return []
    return tokens

def shingles(tokens: List[str], k: int = SHINGLE_K) -> set:
    if len(tokens) < k:
        return set([" ".join(tokens)]) if tokens else set()
    return {" ".join(tokens[i:i+k]) for i in range(0, len(tokens) - k + 1)}

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0
