import re
import typing as py_typing
from pathlib import Path
from typing import Optional


def _fix_exits(text: str) -> str:
    return re.sub(
        r"def __exit__\([^)]+\)[^:]+:",
        "def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:",
        text,
        flags=re.DOTALL,
    )


def _fix_imports(text: str) -> str:
    """Ensure 'from . import typing as duckdb_typing' exists (replace plain 'from . import typing'),
    and ensure 'import typing' (stdlib) is present somewhere.
    """
    duckdb_typing_import = "from . import typing as duckdb_typing"
    # Replace exact `from . import typing` occurrences (line anchored)
    text = re.sub(r"(?m)^\s*from \. import typing\s*$", duckdb_typing_import, text)

    # Ensure stdlib typing present somewhere
    if not re.search(r"(?m)^\s*(import typing|from typing import\b)", text):
        lines = text.splitlines()
        for i, ln in enumerate(lines):
            if ln.strip() == duckdb_typing_import:
                lines.insert(i, "import typing")
                text = "\n".join(lines)
                break
        else:
            # If we didn't find the duckdb_typing import, put stdlib import at top
            text = "import typing\n" + text
    return text


# -------------------------
# Overload consolidation
# -------------------------
def _collect_block_boundaries(lines: list[str], start: int) -> int:
    """Given a starting index pointing at a decorator line (starting with '@'),
    return the index just after the entire function block (decorators + def + body).
    If def not found after decorators, return start+1 (conservative).
    """
    n = len(lines)
    j = start
    # skip contiguous decorator lines
    while j < n and lines[j].lstrip().startswith("@"):
        j += 1
    if j >= n:
        return start + 1
    # must find a def
    if not lines[j].lstrip().startswith("def "):
        return start + 1
    def_line = lines[j]
    def_indent = len(def_line) - len(def_line.lstrip(" "))
    j += 1
    # collect body: lines with indentation > def_indent (allow blank lines that are followed by indented lines)
    while j < n:
        ln = lines[j]
        if ln.strip() == "":
            # include blank line only if next non-empty line is more-indented than def line
            k = j + 1
            while k < n and lines[k].strip() == "":
                k += 1
            if k < n and (len(lines[k]) - len(lines[k].lstrip(" "))) > def_indent:
                j += 1
                continue
            break
        indent = len(ln) - len(ln.lstrip(" "))
        if indent > def_indent:
            j += 1
            continue
        break
    return j


def _extract_def_name(def_line: str) -> Optional[str]:
    m = re.search(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", def_line)
    return m.group(1) if m else None


def _remove_docstring_from_block(block: list[str]) -> tuple[list[str], Optional[list[str]]]:
    """Remove the first docstring in the function body (if present).
    Return (cleaned_block, docstring_lines_or_None).
    Docstring lines are returned verbatim (including the triple quotes) so they can be reinserted.
    Be conservative: if the docstring terminator is missing, we do not remove anything.
    """
    def_idx = None
    for idx, ln in enumerate(block):
        if ln.lstrip().startswith("def "):
            def_idx = idx
            break
    if def_idx is None:
        return block, None

    i = def_idx + 1
    while i < len(block):
        ln = block[i]
        if '"""' in ln or "'''" in ln:
            delim = '"""' if '"""' in ln else "'''"
            if ln.count(delim) >= 2:
                # single-line docstring
                doc_lines = [ln]
                new_block = block[:i] + block[i + 1 :]
                return new_block, doc_lines
            # multi-line: find closing delimiter
            doc_lines = [ln]
            j = i + 1
            while j < len(block) and delim not in block[j]:
                doc_lines.append(block[j])
                j += 1
            if j < len(block):
                doc_lines.append(block[j])  # include closing line
                new_block = block[:i] + block[j + 1 :]
                return new_block, doc_lines
            # unterminated docstring -> conservative: don't remove
            return block, None
        i += 1
    return block, None


def _indent_to_string(s: str) -> str:
    return s[: len(s) - len(s.lstrip(" "))]


def _fix_overloaded_functions(text: str) -> str:
    """Consolidate consecutive @typing.overload blocks for the same function name.
    - For groups with multiple overloads: remove docstrings from earlier overloads and replace them with `...`.
      Insert the first found non-empty docstring into the last overload.
    - Be robust to various indent and spacing styles.
    """
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        if lines[i].lstrip().startswith("@typing.overload"):
            # attempt to build a group of consecutive overload blocks for the same function
            group_blocks: list[list[str]] = []
            group_name: Optional[str] = None
            j = i
            while j < n and lines[j].lstrip().startswith("@typing.overload"):
                block_end = _collect_block_boundaries(lines, j)
                block = lines[j:block_end]
                # find def line in block (if any) to get function name
                def_line = None
                for ln in block:
                    if ln.lstrip().startswith("def "):
                        def_line = ln
                        break
                name = _extract_def_name(def_line) if def_line else None
                if group_name is None:
                    group_name = name
                # stop grouping if name differs (do not consume the differing block)
                if name != group_name:
                    break
                group_blocks.append(block)
                j = block_end

            # if single block only, emit as-is
            if len(group_blocks) <= 1:
                if j > i:
                    out.extend(lines[i:j])
                    i = j
                else:
                    out.append(lines[i])
                    i += 1
                continue

            # multiple overloads for same function name -> clean docstrings
            best_doc: Optional[list[str]] = None
            cleaned_blocks: list[list[str]] = []
            for blk in group_blocks:
                cleaned, doc = _remove_docstring_from_block(blk)
                cleaned_blocks.append(cleaned)
                if best_doc is None and doc:
                    # choose the first non-empty docstring
                    joined = "\n".join(line.strip() for line in doc)
                    if joined.strip():
                        best_doc = doc

            last_idx = len(cleaned_blocks) - 1

            for idx_bl, blk in enumerate(cleaned_blocks):
                is_last = idx_bl == last_idx
                blk_copy = list(blk)

                # drop final @typing.overload on last overload
                if is_last:
                    # remove first decorator if it's exactly '@typing.overload'
                    for d_idx, ln in enumerate(blk_copy):
                        if ln.lstrip().startswith("@"):
                            if ln.lstrip() == "@typing.overload":
                                blk_copy.pop(d_idx)
                            break

                # find def index in blk_copy
                def_idx = None
                for k, ln in enumerate(blk_copy):
                    if ln.lstrip().startswith("def "):
                        def_idx = k
                        break
                if def_idx is None:
                    # malformed block: emit as-is
                    out.extend(blk_copy)
                    continue

                # determine body indentation for inserted lines
                body_indent = None
                for k in range(def_idx + 1, len(blk_copy)):
                    if blk_copy[k].strip() != "":
                        body_indent = _indent_to_string(blk_copy[k])
                        break
                if body_indent is None:
                    def_line = blk_copy[def_idx]
                    def_indent = _indent_to_string(def_line)
                    body_indent = def_indent + "    "

                if not is_last:
                    # non-last overloads: ensure single `...` body
                    new_blk = [*blk_copy[: def_idx + 1], body_indent + "..."]
                    out.extend(new_blk)
                else:
                    # last overload: if we have best_doc, reinsert it (reindented)
                    if best_doc:
                        # compute minimal leading indent in best_doc (so we can re-indent)
                        minimal = None
                        for ln in best_doc:
                            if ln.strip() == "":
                                continue
                            lead = len(ln) - len(ln.lstrip(" "))
                            if minimal is None or lead < minimal:
                                minimal = lead
                        if minimal is None:
                            minimal = 0
                        stripped_doc = [body_indent + (ln[minimal:]) for ln in best_doc]
                        rem = blk_copy[def_idx + 1 :]
                        new_blk = blk_copy[: def_idx + 1] + stripped_doc + rem
                        out.extend(new_blk)
                    else:
                        out.extend(blk_copy)
            # advance index past the group
            i = j
        else:
            out.append(lines[i])
            i += 1

    return "\n".join(out)


# -------------------------
# Typing shadow replacement
# -------------------------
def _fix_typing_shadowing(text: str) -> tuple[str, set, set]:
    """Replace occurrences of `typing.Symbol` with `duckdb_typing.Symbol` when the symbol
    is not present in the stdlib `typing` module. Use a negative lookbehind so we don't
    touch occurrences already prefixed with `duckdb_typing.`.
    Returns (new_text, replaced_set, kept_set).
    """
    typing_pattern = re.compile(r"(?<!duckdb_typing\.)\btyping\.([A-Za-z_][A-Za-z0-9_]*)\b")
    replaced_typing = set()
    kept_typing = set()

    def typing_repl(m) -> str:
        symbol = m.group(1)
        if hasattr(py_typing, symbol):
            kept_typing.add(symbol)
            return m.group(0)
        replaced_typing.add(symbol)
        return f"duckdb_typing.{symbol}"

    new_text = typing_pattern.sub(typing_repl, text)
    return new_text, replaced_typing, kept_typing


# -------------------------
# Optional wrapping (fixed)
# -------------------------
def _fix_optionals(text: str) -> str:
    """Wrap parameter annotations that have a default `= None` in `typing.Optional[...]`
    unless the annotation already indicates optional (contains 'None' or 'Optional' or 'Union[..., None]')
    or is in the allowlist (Any, object, ClassVar, etc.).

    This intentionally identifies the annotation *after* the colon and stops at the character
    right before the `=`; it does not try to parse entire function signatures but is much
    less greedy than prior attempts.
    """
    # Capture colon + annotation (non-greedy) up to the equals; ensure we stop before comma, ), or newline
    pattern = re.compile(r"(?P<colon>:\s*)(?P<ann>[^=,\)\n]+?)(?P<trail>\s*)(?=\=\s*None)")

    def repl(m: re.Match) -> str:
        colon = m.group("colon")
        ann_raw = m.group("ann")
        trail = m.group("trail") or ""
        ann = ann_raw.strip()

        if not ann:
            return colon + ann_raw + trail

        ann_lower = ann.lower()

        # Skip if annotation already explicitly optional / contains None (covers `X | None`, `Union[..., None]`, etc.)
        if "none" in ann_lower:
            return colon + ann_raw + trail
        if ("optional[" in ann_lower) or ("typing.optional[" in ann_lower) or ("duckdb_typing.optional[" in ann_lower):
            return colon + ann_raw + trail
        if ("classvar[" in ann_lower) or ("typing.classvar[" in ann_lower):
            return colon + ann_raw + trail
        if ann in {"Any", "typing.Any", "duckdb_typing.Any", "object", "ClassVar"}:
            return colon + ann_raw + trail

        # Otherwise wrap conservatively preserving original whitespace and annotation text
        return f"{colon}typing.Optional[{ann}]{trail}"

    return pattern.sub(repl, text)


# -------------------------
# Main fixer
# -------------------------
def fix_stub(path: Path):
    """Apply transformations to a .pyi stub file:
    1. Normalize/ensure imports
    2. Replace tabs with four spaces
    3. Consolidate overloaded functions safely
    4. Replace shadowed typing symbols with duckdb_typing when appropriate
    5. Wrap `= None` defaults in typing.Optional[...] where safe.
    """
    print(f"=== Fixing {path}")
    text = path.read_text()

    text = _fix_imports(text)

    text = _fix_exits(text)

    # Normalize tabs early for stable indentation handling
    text = text.replace("\t", "    ")

    text = _fix_overloaded_functions(text)

    text, replaced_typing, kept_typing = _fix_typing_shadowing(text)

    text = _fix_optionals(text)

    path.write_text(text)

    print(f"[stub fixer] Replaced typing symbols: {sorted(replaced_typing)}")
    print(f"[stub fixer] Kept stdlib typing symbols: {sorted(kept_typing)}")


# -------------------------
# Quick demonstration (non-executing example)
# -------------------------
if __name__ == "__main__":
    fix_stub(Path("_duckdb-stubs/__init__.pyi"))
    fix_stub(Path("_duckdb-stubs/typing.pyi"))
    fix_stub(Path("_duckdb-stubs/functional.pyi"))
