import argparse
import re
import typing
from pathlib import Path


def _fix_exits(text: str) -> str:
    return re.sub(
        r"def __exit__\([^)]+\)[^:]+:",
        "def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:",
        text,
        flags=re.DOTALL,
    )


def _fix_imports(text: str) -> str:
    """Replace 'import typing' with 'import typing as pytyping'."""
    import_line = "import typing as pytyping"
    return re.sub(r"(?m)^\s*import typing\s*$", import_line, text)


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


def _extract_def_name(def_line: str) -> str | None:
    m = re.search(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", def_line)
    return m.group(1) if m else None


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
            group_name: str | None = None
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

            # Check if there's a non-overloaded function immediately following
            # that has the same name (this would be the actual implementation)
            has_implementation = False
            if j < n and not lines[j].lstrip().startswith("@typing.overload"):
                # Look for a def line in the next block
                impl_block_end = _collect_block_boundaries(lines, j)
                impl_block = lines[j:impl_block_end]
                for ln in impl_block:
                    if ln.lstrip().startswith("def "):
                        impl_name = _extract_def_name(ln)
                        if impl_name == group_name:
                            has_implementation = True
                        break

            # if single block only, emit as-is
            if len(group_blocks) <= 1:
                if j > i:
                    out.extend(lines[i:j])
                    i = j
                else:
                    out.append(lines[i])
                    i += 1
                continue

            # Process multiple overload blocks
            last_idx = len(group_blocks) - 1

            for idx_bl, blk in enumerate(group_blocks):
                is_last = idx_bl == last_idx
                blk_copy = list(blk)

                # Only drop @typing.overload on the last overload if there's no separate implementation
                if is_last and not has_implementation:
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

                if not is_last or has_implementation:
                    # For non-last blocks or when there's a separate implementation,
                    # just emit the signature
                    out.extend([*blk_copy[: def_idx + 1]])
                else:
                    # This is the last block and there's no separate implementation
                    out.extend(blk_copy)

            # advance index past the group
            i = j
        else:
            out.append(lines[i])
            i += 1

    return "\n".join(out) + "\n"


# -------------------------
# Typing shadow replacement
# -------------------------
def _fix_typing_shadowing(text: str) -> tuple[str, set, set]:
    """Replace occurrences of `typing.Symbol` with `pytyping.Symbol` when the symbol
    is present in the stdlib `typing` module. Use a negative lookbehind so we don't
    touch occurrences already prefixed with `pytyping.`.
    Returns (new_text, replaced_set, kept_set).
    """
    typing_pattern = re.compile(r"(?<!pytyping\.)\btyping\.([A-Za-z_][A-Za-z0-9_]*)\b")
    replaced_typing = set()
    kept_typing = set()

    def typing_repl(m) -> str:
        symbol = m.group(1)
        if hasattr(typing, symbol):
            replaced_typing.add(symbol)
            return f"pytyping.{symbol}"
        kept_typing.add(symbol)
        return m.group(0)

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
        if "optional[" in ann_lower:
            return colon + ann_raw + trail
        if "classvar[" in ann_lower:
            return colon + ann_raw + trail
        if ann in {"Any", "typing.Any", "pytyping.Any", "object", "ClassVar"}:
            return colon + ann_raw + trail

        # Otherwise wrap conservatively preserving original whitespace and annotation text
        # return f"{colon}typing.Optional[{ann}]{trail}"
        return f"{colon}{ann} | None {trail}"

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

    if replaced_typing:
        print(f"[stub fixer] Replaced stdlib typing symbols: {sorted(replaced_typing)}")
    if kept_typing:
        print(f"[stub fixer] Kept duckdb typing symbols: {sorted(kept_typing)}")


def _is_valid_stubfile(path: Path) -> bool:
    assert path.is_file()
    return path.suffix == ".pyi"


if __name__ == "__main__":
    description = "Post-processing script for stubs generated with pybind11-stubgen."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-p", "--path", type=Path, required=True, help="Path to .pyi stub file or dir containing .pyi stubs"
    )
    parser.add_argument("-r", "--recursive", action="store_true", default=False)
    args = parser.parse_args()
    if args.path.is_file():
        if not _is_valid_stubfile(args.path):
            print(f"{args.path} is a file, but not a pyi file.")
        else:
            fix_stub(args.path)
    elif args.path.is_dir():
        dirs: list[Path] = [args.path]
        files: list[Path] = []
        while len(dirs) > 0:
            curdir = dirs.pop()
            for path in curdir.iterdir():
                if path.is_file() and _is_valid_stubfile(path):
                    files.append(path)
                elif path.is_dir() and args.recursive:
                    dirs.append(path)
        if len(files) == 0:
            print(f"No .pyi stub files found in {args.path}")
        else:
            for path in files:
                fix_stub(path)
    else:
        print(f"I can't process {args.path}. Does it exist? Do I have read permission?")
