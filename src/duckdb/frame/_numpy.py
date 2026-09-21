"""Results to numpy, imported inside the functions so that importing duckdb never pulls it in."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .. import _duckdb

#: DuckDB's type-id numbers, copied rather than imported; the round-trip tests check the copy against DuckDB.
_BOOLEAN = 10
_TINYINT = 11
_SMALLINT = 12
_INTEGER = 13
_BIGINT = 14
_DATE = 15
_TS_S = 17
_TS_MS = 18
_TS = 19
_TS_NS = 20
_DECIMAL = 21
_FLOAT = 22
_DOUBLE = 23
_INTERVAL = 27
_UTINYINT = 28
_USMALLINT = 29
_UINTEGER = 30
_UBIGINT = 31
_TS_TZ = 32
_UHUGEINT = 49
_HUGEINT = 50
_ENUM = 104

_NUMERIC_DTYPE = {
    _BOOLEAN: "bool",
    _TINYINT: "int8",
    _SMALLINT: "int16",
    _INTEGER: "int32",
    _BIGINT: "int64",
    _UTINYINT: "uint8",
    _USMALLINT: "uint16",
    _UINTEGER: "uint32",
    _UBIGINT: "uint64",
    _FLOAT: "float32",
    _DOUBLE: "float64",
}

_TS_UNIT = {_TS_S: "s", _TS_MS: "ms", _TS: "us", _TS_NS: "ns", _TS_TZ: "us"}

#: DuckDB's infinite date and timestamp markers, and the instants they clamp to, matching what fetching rows gives.
_DATE_INF = 2147483647
_DATE_CLAMP = (2932896, -719162)
_TS_INF = 9223372036854775807
_TS_CLAMP = {
    "s": (253402300799, -62135596800),
    "ms": (253402300799999, -62135596800000),
    "us": (253402300799999999, -62135596800000000),
}


def _mask(np: Any, validity: Any, count: int) -> Any:
    """The invalid-rows bool mask from the 64-bit validity words, or None."""
    if validity is None:
        return None
    bits = np.unpackbits(np.frombuffer(validity, dtype="uint8"), bitorder="little")[:count]
    mask = bits == 0
    return mask if mask.any() else None


def _int128_to_float(np: Any, data: Any, mask: Any, *, signed: bool) -> Any:
    """128-bit ints to float64 by sign and magnitude: combining two's-complement halves loses small negatives."""
    kind = "<i8" if signed else "<u8"
    limbs = np.frombuffer(data, dtype=np.dtype([("lower", "<u8"), ("upper", kind)]))
    lower = limbs["lower"]
    upper = limbs["upper"]
    if mask is not None:
        # NULL slots hold undefined halves; zeroed so masked positions stay a predictable 0.0.
        lower = np.where(mask, np.uint64(0), lower)
        upper = np.where(mask, 0, upper)
    scale = 18446744073709551616.0
    if not signed:
        return upper.astype("float64") * scale + lower.astype("float64")
    negative = upper < 0
    upper_bits = upper.astype("uint64")
    # The +1 of a two's-complement negation carries into the upper half exactly when the lower half is zero.
    magnitude_lower = np.where(negative, ~lower + np.uint64(1), lower)
    magnitude_upper = np.where(negative, ~upper_bits + (lower == 0).astype("uint64"), upper_bits)
    combined = magnitude_upper.astype("float64") * scale + magnitude_lower.astype("float64")
    return np.where(negative, -combined, combined)


def _convert_column(np: Any, view: _duckdb.ChunkView, column: int, count: int) -> tuple[Any, Any, str, Any]:
    """One column of a batch as (values, mask, kind, meta), where the kind names how the pieces go together."""
    type_id = view.type_id(column)
    mask = _mask(np, view.validity(column), count)

    dtype = _NUMERIC_DTYPE.get(type_id)
    if dtype is not None:
        return np.frombuffer(view.data(column), dtype=dtype).copy(), mask, "numeric", None

    if type_id == _DATE:
        days = np.frombuffer(view.data(column), dtype="int32").copy()
        if mask is not None:
            # NULL slots hold undefined day counts; zeroed first, or extreme garbage overflows datetime64.
            days[mask] = 0
        days[days == _DATE_INF] = _DATE_CLAMP[0]
        days[days == -_DATE_INF] = _DATE_CLAMP[1]
        # DATE lands as datetime64[us], as the previous duckdb package had it.
        return days.astype("datetime64[D]").astype("datetime64[us]"), mask, "date", "us"

    unit = _TS_UNIT.get(type_id)
    if unit is not None:
        raw = np.frombuffer(view.data(column), dtype="int64").copy()
        if mask is not None:
            raw[mask] = 0
        if unit != "ns":
            # In nanoseconds the marker is already the last instant there is, and its negative is NaT + 1.
            raw[raw == _TS_INF] = _TS_CLAMP[unit][0]
            raw[raw == -_TS_INF] = _TS_CLAMP[unit][1]
        values = raw.view(f"datetime64[{unit}]")
        return values, mask, ("datetimetz" if type_id == _TS_TZ else "datetime"), unit

    if type_id == _INTERVAL:
        record = np.frombuffer(
            view.data(column), dtype=np.dtype([("months", "<i4"), ("days", "<i4"), ("micros", "<i8")])
        )
        # Months count as 30 days, the same as fetching rows does.
        total = (record["months"].astype("int64") * 30 + record["days"]) * 86_400_000_000 + record["micros"]
        if mask is not None:
            total[mask] = 0
        return total.view("timedelta64[us]"), mask, "timedelta", None

    if type_id == _DECIMAL:
        # DECIMAL becomes float64 here as the previous duckdb package had it; fetching rows gives exact Decimals.
        decimal_data = cast("memoryview", view.data(column))  # every decimal width is fixed-width, so never None
        element = len(decimal_data) // count if count else 8
        if element == 16:
            ints = _int128_to_float(np, decimal_data, mask, signed=True)
        else:
            ints = np.frombuffer(decimal_data, dtype=f"int{8 * element}").astype("float64")
        return ints / (10.0 ** view.decimal_scale(column)), mask, "numeric", None

    if type_id in (_HUGEINT, _UHUGEINT):
        # float64 as before: the column layout is worth losing precision past 2^53, and rows stay exact.
        values = _int128_to_float(np, view.data(column), mask, signed=type_id == _HUGEINT)
        return values, mask, "numeric", None

    if type_id == _ENUM:
        data = view.data(column)
        if data is not None:
            element = len(data) // count if count else 1
            codes = np.frombuffer(data, dtype=f"uint{8 * element}").astype("int64")
            if mask is not None:
                codes = codes.copy()
                codes[mask] = -1
            return codes, mask, "enum", view.enum_values(column)

    values = np.empty(count, dtype=object)
    values[:] = view.values(column)
    if mask is None:
        nones = np.array([v is None for v in values], dtype=bool)
        mask = nones if nones.any() else None
    return values, mask, "object", None


def _empty_column(np: Any, type_id: int, enum_values: Any) -> tuple[Any, Any, str, Any]:
    """A zero-row column with the dtype and kind its rows would have had."""
    dtype = _NUMERIC_DTYPE.get(type_id)
    if dtype is not None:
        return np.empty(0, dtype=dtype), None, "numeric", None
    if type_id == _DATE:
        return np.empty(0, dtype="datetime64[us]"), None, "date", "us"
    unit = _TS_UNIT.get(type_id)
    if unit is not None:
        kind = "datetimetz" if type_id == _TS_TZ else "datetime"
        return np.empty(0, dtype=f"datetime64[{unit}]"), None, kind, unit
    if type_id == _INTERVAL:
        return np.empty(0, dtype="timedelta64[us]"), None, "timedelta", None
    if type_id in (_DECIMAL, _HUGEINT, _UHUGEINT):
        return np.empty(0, dtype="float64"), None, "numeric", None
    if type_id == _ENUM:
        return np.empty(0, dtype="int64"), None, "enum", list(enum_values or [])
    return np.empty(0, dtype=object), None, "object", None


def _columns_from_views(
    np: Any, names: list[str], views: Any, types_meta: list[tuple[int, int, Any]]
) -> list[tuple[str, Any, Any, str, Any]]:
    """Batches of rows turned into per-column (name, values, mask, kind, meta)."""
    pieces: list[list[tuple[Any, Any, str, Any]]] = [[] for _ in names]
    for view in views:
        count = view.row_count
        skip = view.row_offset
        for i in range(len(names)):
            values, mask, kind, meta = _convert_column(np, view, i, count)
            if skip:
                # A prior row fetch consumed the head of this batch, and delivering it again would repeat rows.
                values = values[skip:]
                if mask is not None:
                    mask = mask[skip:]
                    if not mask.any():
                        mask = None
            pieces[i].append((values, mask, kind, meta))

    out: list[tuple[str, Any, Any, str, Any]] = []
    for i, name in enumerate(names):
        if not pieces[i]:
            type_id, _, enum_values = types_meta[i]
            values, mask, kind, meta = _empty_column(np, type_id, enum_values)
            out.append((name, values, mask, kind, meta))
            continue
        kind = pieces[i][-1][2]
        meta = pieces[i][-1][3]
        values = np.concatenate([p[0] for p in pieces[i]])
        if any(p[1] is not None for p in pieces[i]):
            mask = np.concatenate([p[1] if p[1] is not None else np.zeros(len(p[0]), dtype=bool) for p in pieces[i]])
        else:
            mask = None
        out.append((name, values, mask, kind, meta))
    return out


def _all_views(result: _duckdb.Result) -> Any:
    while (view := result.fetch_chunk_view()) is not None:
        yield view


def _result_to_columns(result: _duckdb.Result) -> list[tuple[str, Any, Any, str, Any]]:
    """Consume a whole result into per-column (name, values, mask, kind, meta)."""
    import numpy as np

    names = [name for name, _ in result.schema]
    columns = _columns_from_views(np, names, _all_views(result), result.schema_types)
    result.close()
    return columns


def fetch_numpy(result: _duckdb.Result) -> dict[str, Any]:
    """The whole result as {column: ndarray}; NULL-bearing columns come back masked."""
    import numpy as np

    out: dict[str, Any] = {}
    for name, values, mask, kind, meta in _result_to_columns(result):
        if kind == "enum":
            categories = np.array(meta, dtype=object)
            strings = np.empty(len(values), dtype=object)
            valid = values >= 0
            strings[valid] = categories[values[valid]]
            strings[~valid] = None
            values = strings
        out[name] = np.ma.masked_array(values, mask) if mask is not None else values
    return out
