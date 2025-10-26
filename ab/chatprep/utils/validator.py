import re, ast

def sanitize_assistant_to_single_python_block(text: str) -> str:
    m = re.search(r"```python\s*(.*?)```", text, flags=re.S)
    if m:
        code = m.group(1).strip()
    else:
        m_any = re.search(r"```\s*(.*?)```", text, flags=re.S)
        code = (m_any.group(1) if m_any else text).strip()
    return f"```python\n{code}\n```"

def is_parseable_python_block(text: str) -> bool:
    m = re.search(r"```python\s*(.*?)```", text, flags=re.S)
    if not m:
        return False
    code = m.group(1).strip()
    try:
        ast.parse(code)
        return True
    except Exception:
        return False
