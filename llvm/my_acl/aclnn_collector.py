import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

# Скомпилированное регулярное выражение для поиска начала объявления
ACLNN_DECL_RE = re.compile(r'ACLNN_API\s+aclnnStatus\s+(\w+)\s*\(')

def find_closing_paren(text: str, start: int) -> int:
    """Возвращает позицию парной закрывающей скобки для '(' в позиции start."""
    depth = 1
    i = start + 1
    while i < len(text) and depth:
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
        i += 1
    return i - 1 if depth == 0 else -1

def extract_signature(text: str, pos: int) -> Optional[str]:
    """Извлекает сигнатуру функции начиная с 'ACLNN_API aclnnStatus ...' до ';'."""
    m = ACLNN_DECL_RE.search(text[pos:])
    if not m:
        return None
    paren_open = pos + m.end() - 1  # позиция '('
    paren_close = find_closing_paren(text, paren_open)
    if paren_close == -1:
        return None
    semicolon = text.find(';', paren_close + 1)
    if semicolon == -1:
        return None
    return text[pos:semicolon + 1]

def parse_header(file_path: Path) -> Dict[str, List[Optional[str]]]:
    """Обрабатывает один .h файл, возвращает словарь {имя_операции: [main_sig, ws_sig]}."""
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception:
        return {}
    result: Dict[str, List[Optional[str]]] = {}
    for m in ACLNN_DECL_RE.finditer(content):
        func_name = m.group(1)
        sig = extract_signature(content, m.start())
        if sig is None:
            continue
        if func_name.endswith('GetWorkspaceSize'):
            base = func_name[:-len('GetWorkspaceSize')]
            result.setdefault(base, [None, None])[1] = sig
        else:
            result.setdefault(func_name, [None, None])[0] = sig
    return result

def scan_aclnn(root_dir: str) -> Dict[str, List[Optional[str]]]:
    """Рекурсивно обходит root_dir, собирает объявления aclnn функций."""
    aggregated: Dict[str, List[Optional[str]]] = defaultdict(lambda: [None, None])
    for h_file in Path(root_dir).rglob('*.h'):
        for name, sigs in parse_header(h_file).items():
            existing = aggregated[name]
            if existing[0] is None and sigs[0] is not None:
                existing[0] = sigs[0]
            if existing[1] is None and sigs[1] is not None:
                existing[1] = sigs[1]
    return dict(sorted(aggregated.items()))

def generate_unimplemented_macro(sig: str) -> str:
    assert sig.startswith("ACLNN_API aclnnStatus") and sig.endswith(';')
    sig = sig[len("ACLNN_API aclnnStatus"):-1].strip()
    assert sig.count('(') == 1 and sig.count(')') == 1
    sig = sig.replace('(', ",\n")
    sig = f"DEFINE_UNIMPLEMENTED_ACLNN({sig}"
    pos = sig.index("(")
    pad = " " * (pos + 1)

    it = iter(sig.split("\n"))
    result = "\n".join((
        next(it),
        *(pad + line.strip() for line in it if line.strip())
    ))
    return result


if __name__ == '__main__':
    root = Path.home() / "tmp" / "Ascend-cann-950" / "run_package"
    decls = scan_aclnn(root)

    prev_print = False
    for op, (main_sig, ws_sig) in decls.items():
       #print(f"[OP] {op}:")
       #print(f"  Main: {main_sig}")
       #print(f"  WS:   {ws_sig}")
        assert main_sig is not None

        if ws_sig is not None:
            if not prev_print:
                print()
            print(generate_unimplemented_macro(ws_sig))

        print(generate_unimplemented_macro(main_sig))
        prev_print = ws_sig is not None
        if prev_print:
            print()
