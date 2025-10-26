# ab/chatprep/validate.py
from __future__ import annotations
import re, ast
from typing import Dict, Any, Optional

_CODE_FENCE_RE = re.compile(r"```python\s*(.*?)```", flags=re.S)
_ANY_FENCE_RE  = re.compile(r"```\s*(.*?)```", flags=re.S)

def sanitize_to_single_python_block(text: str) -> str:
    """Return content with exactly one ```python ...``` block (strip extras/non-code prose)."""
    m = _CODE_FENCE_RE.search(text)
    if m:
        code = m.group(1).strip()
    else:
        m_any = _ANY_FENCE_RE.search(text)
        code = (m_any.group(1) if m_any else text).strip()
    return f"```python\n{code}\n```"

def extract_code(text: str) -> Optional[str]:
    m = _CODE_FENCE_RE.search(text)
    if not m:
        return None
    return m.group(1).strip()

def is_parseable_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False

def has_nn_module_subclass(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for b in node.bases:
                # class Net(nn.Module) or class X(torch.nn.Module)
                if getattr(b, "id", "") == "Module":
                    return True
                if getattr(b, "attr", "") == "Module":
                    return True
    return False

def validate_and_fix_assistant(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure exactly one fenced python block; returns updated message dict.
    """
    content = message.get("content", "")
    fixed = sanitize_to_single_python_block(content)
    message["content"] = fixed
    return message
