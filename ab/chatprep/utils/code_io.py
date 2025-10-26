# ab/chatprep/utils/code_io.py
import os
from typing import List, Tuple

def load_py_files(root: str) -> List[Tuple[str, str]]:
    paths = []
    for dp, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(".py"):
                p = os.path.join(dp, fn)
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        paths.append((p, f.read()))
                except Exception:
                    continue
    return paths
