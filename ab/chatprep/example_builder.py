# ab/chatprep/example_builder.py
import ast
import re
import uuid
from typing import Dict, Any, List, Optional
from .schema import Message, ChatExample
from .detectors.ast_signals import summarize_source

def _mask_forward_body(src: str) -> Optional[Dict[str, str]]:
    """
    Create an infilling variant by replacing the body of `forward()` with a marker.
    Returns dict with 'user_code' and 'assistant_code' (full filled code) if success.
    """
    try:
        tree = ast.parse(src)
    except Exception:
        return None

    class Rewriter(ast.NodeTransformer):
        def __init__(self):
            self.forward_src = None
            super().__init__()

        def visit_FunctionDef(self, node: ast.FunctionDef):
            if node.name == "forward":
                # capture original forward code span
                self.forward_src = (node.lineno, node.end_lineno)
            return node

    r = Rewriter()
    r.visit(tree)
    if not r.forward_src:
        return None

    lines = src.splitlines()
    s, e = r.forward_src
    masked = lines[:s] + ["        # >>> FILL HERE: implement forward() <<<"] + lines[e:]
    user_code = "\n".join(masked)

    return {
        "user_code": user_code,
        "assistant_code": f"```python\n{src.strip()}\n```"
    }

def build_examples_from_code(path: str, code_text: str, add_infill: bool = True, is_test: bool = False) -> List[ChatExample]:
    det = summarize_source(code_text)
    if not det["has_module"]:
        return []

    eid = str(uuid.uuid4())
    msgs = _chat_messages(eid, code_text, det, is_test=is_test)
    ex = ChatExample(id=eid, messages=msgs, meta={"source_path": path, **det, "type": "full"})

    out = [ex]
    if add_infill:
        inf = _mask_forward_body(code_text)
        if inf:
            ieid = str(uuid.uuid4())
            imsgs = _chat_messages(ieid, inf["user_code"], det, assistant_code=inf["assistant_code"], is_test=is_test)
            out.append(ChatExample(id=ieid, messages=imsgs, meta={"source_path": path, **det, "type": "infill"}))
    return out

def _chat_messages(eid: str, code_or_user: str, det: Dict[str, Any], assistant_code: Optional[str] = None, is_test: bool = False) -> List[Message]:
    from .prompt_builder import build_messages, build_test_messages
    
    # Use test-specific prompts for test split
    builder = build_test_messages if is_test else build_messages
    m = builder(eid, code_or_user if assistant_code else code_or_user, det)
    
    sys = Message(role="system", content=m["system"])
    usr = Message(role="user", content=(code_or_user if assistant_code else m["user"]))
    asst = Message(role="assistant", content=(assistant_code if assistant_code else m["assistant_code"]))
    return [sys, usr, asst]
