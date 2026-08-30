"""One spelling for a SQL type, so `TEXT` and `VARCHAR` compare equal, as the engine treats them.

A declared heading is written by hand and an engine answer is the engine's
canonical text, so comparing the two needs both on one spelling.
"""

from __future__ import annotations

import re

#: Alias to canonical name, as `duckdb_types()` lists them. Pinned to the
#: engine by a test, so a bump that adds or moves an alias is noticed.
ALIASES: dict[str, str] = {
    "array": "ARRAY",
    "bigint": "BIGINT",
    "int64": "BIGINT",
    "int8": "BIGINT",
    "long": "BIGINT",
    "oid": "BIGINT",
    "bignum": "BIGNUM",
    "varint": "BIGNUM",
    "bit": "BIT",
    "bitstring": "BIT",
    "binary": "BLOB",
    "blob": "BLOB",
    "bytea": "BLOB",
    "varbinary": "BLOB",
    "bool": "BOOLEAN",
    "boolean": "BOOLEAN",
    "logical": "BOOLEAN",
    "date": "DATE",
    "dec": "DECIMAL",
    "decimal": "DECIMAL",
    "numeric": "DECIMAL",
    "double": "DOUBLE",
    "float8": "DOUBLE",
    "enum": "ENUM",
    "float": "FLOAT",
    "float4": "FLOAT",
    "real": "FLOAT",
    "geometry": "GEOMETRY",
    "hugeint": "HUGEINT",
    "int128": "HUGEINT",
    "int": "INTEGER",
    "int32": "INTEGER",
    "int4": "INTEGER",
    "integer": "INTEGER",
    "integral": "INTEGER",
    "signed": "INTEGER",
    "interval": "INTERVAL",
    "list": "LIST",
    "map": "MAP",
    "null": "NULL",
    "int16": "SMALLINT",
    "int2": "SMALLINT",
    "short": "SMALLINT",
    "smallint": "SMALLINT",
    "row": "STRUCT",
    "struct": "STRUCT",
    "time": "TIME",
    "time with time zone": "TIME WITH TIME ZONE",
    "timetz": "TIME WITH TIME ZONE",
    "datetime": "TIMESTAMP",
    "timestamp": "TIMESTAMP",
    "timestamp_us": "TIMESTAMP",
    "timestamp with time zone": "TIMESTAMP WITH TIME ZONE",
    "timestamptz": "TIMESTAMP WITH TIME ZONE",
    "timestamptz_ns": "TIMESTAMPTZ_NS",
    "timestamp_ms": "TIMESTAMP_MS",
    "timestamp_ns": "TIMESTAMP_NS",
    "timestamp_s": "TIMESTAMP_S",
    "time_ns": "TIME_NS",
    "int1": "TINYINT",
    "tinyint": "TINYINT",
    "tuple": "TUPLE",
    "type": "TYPE",
    "ubigint": "UBIGINT",
    "uint64": "UBIGINT",
    "uhugeint": "UHUGEINT",
    "uint128": "UHUGEINT",
    "uint32": "UINTEGER",
    "uinteger": "UINTEGER",
    "union": "UNION",
    "uint16": "USMALLINT",
    "usmallint": "USMALLINT",
    "uint8": "UTINYINT",
    "utinyint": "UTINYINT",
    "guid": "UUID",
    "uuid": "UUID",
    "json": "VARCHAR",
    "bpchar": "VARCHAR",
    "char": "VARCHAR",
    "nvarchar": "VARCHAR",
    "string": "VARCHAR",
    "text": "VARCHAR",
    "varchar": "VARCHAR",
    "variant": "VARIANT",
}

#: Types whose values compare as numbers.
NUMERIC = {
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "UHUGEINT",
    "FLOAT",
    "DOUBLE",
    "DECIMAL",
    "BIGNUM",
}


def canonical(text: str) -> str:
    """The engine's spelling of a type written by hand.

    `text` becomes `VARCHAR`, `int[]` becomes `INTEGER[]`, a bare `DECIMAL`
    becomes `DECIMAL(18,3)`, and `struct(a text)` becomes `STRUCT(a VARCHAR)`.
    A name the table does not know, an extension's type, passes through
    upper-cased, so it still compares equal to itself.
    """
    text = text.strip()
    if text.endswith("]"):
        # A list `T[]` or an array `T[n]`: the element is canonical too.
        open_at = text.rindex("[")
        return canonical(text[:open_at]) + text[open_at:].replace(" ", "")
    head, _, rest = text.partition("(")
    name = " ".join(head.split()).lower()
    name = ALIASES.get(name, name.upper())
    if not rest:
        return "DECIMAL(18,3)" if name == "DECIMAL" else name
    inner = rest[: rest.rindex(")")]
    if name == "VARCHAR":
        return name  # a length is accepted and ignored by the engine
    if name == "DECIMAL":
        return "DECIMAL(" + ",".join(p.strip() for p in inner.split(",")) + ")"
    if name in {"STRUCT", "UNION"}:
        fields = ", ".join(_field(part) for part in _split_top(inner))
        return f"{name}({fields})"
    if name == "MAP":
        return "MAP(" + ", ".join(canonical(part) for part in _split_top(inner)) + ")"
    return f"{name}({inner.strip()})"


def _field(part: str) -> str:
    """A `name TYPE` member of a STRUCT or UNION, with the type made canonical."""
    part = part.strip()
    if part.startswith('"'):
        # A quote inside the name is written doubled, so the name ends at
        # the first quote not followed by another.
        end = 1
        while True:
            end = part.index('"', end) + 1
            if end >= len(part) or part[end] != '"':
                break
            end += 1
        return f"{part[:end]} {canonical(part[end:])}"
    name, _, type_text = part.partition(" ")
    return f"{name} {canonical(type_text)}"


def _split_top(text: str) -> list[str]:
    """Split on the commas outside parentheses and quotes."""
    parts: list[str] = []
    depth = 0
    quote = ""
    start = 0
    for i, c in enumerate(text):
        if quote:
            if c == quote:
                quote = ""
        elif c in "\"'":
            quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def is_numeric(text: str) -> bool:
    """Whether values of this type compare as numbers."""
    return re.split(r"[(\[]", canonical(text), maxsplit=1)[0] in NUMERIC
