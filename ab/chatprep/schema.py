# ab/chatprep/schema.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import json
from dataclasses import is_dataclass, asdict
from typing import Iterable, Any, Dict

@dataclass
class Message:
    role: str
    content: Any

@dataclass
class ChatExample:
    id: str
    messages: List[Message]
    meta: Dict[str, Any]

    def to_json(self) -> str:
        payload = {
            "id": self.id,
            "messages": [asdict(m) for m in self.messages],
            "meta": self.meta,
        }
        return json.dumps(payload, ensure_ascii=False)

def write_jsonl(path: str, examples: List["ChatExample"]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.to_json() + "\n")


def _row_to_json_line(ex: Any) -> str:
    """
    Accepts:
      - objects with .to_json()  -> use it directly (legacy ChatExample)
      - objects with .to_dict()  -> json.dumps(to_dict())
      - dataclasses              -> json.dumps(asdict(...))
      - plain dict               -> json.dumps(dict)
    """
    if hasattr(ex, "to_json") and callable(getattr(ex, "to_json")):
        # Expect .to_json() to return a JSON string
        s = ex.to_json()
        # Best-effort sanity: ensure it's valid JSON
        try:
            json.loads(s)
            return s
        except Exception:
            # Fall through to dict serialization
            pass

    if hasattr(ex, "to_dict") and callable(getattr(ex, "to_dict")):
        return json.dumps(ex.to_dict(), ensure_ascii=False)

    if is_dataclass(ex):
        return json.dumps(asdict(ex), ensure_ascii=False)

    if isinstance(ex, dict):
        return json.dumps(ex, ensure_ascii=False)

    # Last resort: try to project common attributes
    try:
        proj: Dict[str, Any] = {
            "id": getattr(ex, "id", None),
            "messages": getattr(ex, "messages", None),
            "meta": getattr(ex, "meta", None),
        }
        return json.dumps(proj, ensure_ascii=False)
    except Exception as e:
        raise TypeError(f"Unsupported row type in write_jsonl: {type(ex)!r}") from e


def write_jsonl(path: str, rows: Iterable[Any]) -> None:
    """
    Writes an iterable of rows that may be ChatExample, dict, or dataclass.
    Each row is serialized to one compact JSON line.
    """
    with open(path, "w", encoding="utf-8") as f:
        for ex in rows:
            f.write(_row_to_json_line(ex) + "\n")