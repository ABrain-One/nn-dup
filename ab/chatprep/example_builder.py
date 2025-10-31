# ab/chatprep/example_builder.py
import ast
import re
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from .schema import Message, ChatExample
from .detectors.ast_signals import summarize_source

def _mask_forward_body(src: str) -> Optional[Dict[str, str]]:
    """Create an infilling variant by replacing the body of `forward()`."""
    try:
        tree = ast.parse(src)
    except Exception:
        return None

    class ForwardRewriter(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == "forward":
                node.body = [
                    ast.Raise(
                        exc=ast.Call(func=ast.Name(id="NotImplementedError", ctx=ast.Load()), args=[], keywords=[]),
                        cause=None,
                    )
                ]
            return node

    try:
        rewriter = ForwardRewriter()
        new_tree = rewriter.visit(tree)
        ast.fix_missing_locations(new_tree)
        user_code = ast.unparse(new_tree) if hasattr(ast, "unparse") else src
        return {"user_code": user_code, "assistant_code": src}
    except Exception:
        return None

def _canon_group_key(path: str) -> str:
    try:
        return Path(path).resolve().as_posix()
    except Exception:
        return Path(path).as_posix()

def build_examples_from_code(path: str, code_text: str, add_infill: bool = True) -> List[ChatExample]:
    det = summarize_source(code_text)
    if not det.get("has_module", False):
        return []

    group_key = _canon_group_key(path)

    eid = str(uuid.uuid4())
    msgs = _chat_messages(eid, code_text, det)
    ex = ChatExample(
        id=eid,
        messages=msgs,
        meta={"source_path": path, "group_key": group_key, **det, "type": "full"},
    )

    out = [ex]
    if add_infill:
        inf = _mask_forward_body(code_text)
        if inf:
            ieid = str(uuid.uuid4())
            imsgs = _chat_messages(ieid, inf["user_code"], det, assistant_code=inf["assistant_code"])
            out.append(
                ChatExample(
                    id=ieid,
                    messages=imsgs,
                    meta={"source_path": path, "group_key": group_key, **det, "type": "infill"},
                )
            )
    return out

def _chat_messages(eid: str, code_or_user: str, det: Dict[str, Any], assistant_code: Optional[str] = None) -> List[Message]:
    from .prompt_builder import build_messages
    m = build_messages(eid, code_or_user, det)
    sys = Message(role="system", content=m["system"])
    usr = Message(role="user", content=(code_or_user if assistant_code else m["user"]))
    asst = Message(role="assistant", content=(assistant_code if assistant_code else m["assistant_code"]))
    return [sys, usr, asst]
