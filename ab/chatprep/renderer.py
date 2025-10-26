# ab/chatprep/renderer.py
from typing import List, Dict, Any, Optional
from .schema import ChatExample
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None  # optional

def _get_messages(row):
    """Extract messages from row, handling both ChatExample and dict formats."""
    if hasattr(row, "messages"):
        return row.messages
    if isinstance(row, dict):
        return row.get("messages", [])
    raise TypeError(f"Unsupported row type: {type(row)!r}")

def _get_id(row):
    """Extract id from row, handling both ChatExample and dict formats."""
    if hasattr(row, "id"):
        return row.id
    if isinstance(row, dict):
        return row.get("id")
    raise TypeError(f"Unsupported row type: {type(row)!r}")

def _get_meta(row):
    """Extract meta from row, handling both ChatExample and dict formats."""
    if hasattr(row, "meta"):
        return row.meta
    if isinstance(row, dict):
        return row.get("meta")
    raise TypeError(f"Unsupported row type: {type(row)!r}")

def render_with_template(examples: List[Any], model_name: str) -> List[Dict[str, Any]]:
    """Render examples with chat template, supporting both ChatExample and dict formats."""
    if AutoTokenizer is None:
        raise RuntimeError("transformers is not installed. Please `pip install transformers`.")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    rendered = []
    for ex in examples:
        messages = _get_messages(ex)
        # Convert messages to the format expected by transformers
        if hasattr(messages[0], 'role'):  # ChatMessage objects
            msgs = [{"role": m.role, "content": m.content} for m in messages]
        else:  # Already dict format
            msgs = messages
        
        # Prepare an assistant-generation prompt; label is assistant content.
        prompt_ids = tok.apply_chat_template(
            msgs[:-1],  # system+user only as prompt
            tokenize=False,
            add_generation_prompt=True,
        )
        rendered.append({
            "id": _get_id(ex),
            "prompt": prompt_ids,
            "assistant": msgs[-1]["content"],
            "meta": _get_meta(ex),
        })
    return rendered
