"""Results to pandas, assembled from the numpy egress's columns and never through Arrow.

The dtypes are the previous duckdb package's, except where its frames disagreed with its own rows; pandas is
imported inside the functions, so importing duckdb never pulls it in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._sources import _unique_names
from ..exceptions import ConversionError
from ._numpy import _TS_CLAMP, _result_to_columns

if TYPE_CHECKING:
    from .. import _duckdb

#: DuckDB's type-id number for VARCHAR, copied rather than imported; the tests check the copy against DuckDB.
_VARCHAR = 25

_NULLABLE_KINDS = frozenset("iu")

#: The days either side of the epoch that a datetime64[us] holds.
_US_DAYS = 106_751_991


def _pandas() -> Any:
    try:
        import pandas
    except ImportError as error:
        message = "to_pandas needs pandas, which is not installed"
        raise ImportError(message) from error
    return pandas


def _numeric(pd: Any, np: Any, values: Any, mask: Any) -> Any:
    if mask is None:
        return values
    if values.dtype.kind == "b":
        return pd.arrays.BooleanArray(values, mask)
    if values.dtype.kind in _NULLABLE_KINDS:
        return pd.arrays.IntegerArray(values, mask)
    # Floats keep numpy's own missing value, as the previous package had it, rather than a nullable Float64.
    values[mask] = np.nan
    return values


def _date(np: Any, values: Any, mask: Any) -> Any:
    days = values.view("int64")
    outside = (days < -_US_DAYS) | (days > _US_DAYS)
    if mask is not None:
        outside &= ~mask
    if outside.any():
        day = np.datetime_as_string(values[np.flatnonzero(outside)[0]])
        message = f"Conversion Error: DATE {day} is outside the range pandas' datetime64[us] can represent"
        raise ConversionError(message)
    out = values.astype("datetime64[us]")
    if mask is not None:
        out[mask] = np.datetime64("NaT", "us")
    return out


def _with_nat(np: Any, values: Any, mask: Any) -> Any:
    if mask is not None:
        values[mask] = values.dtype.type("NaT", np.datetime_data(values.dtype)[0])
    return values


def _zone(time_zone: str) -> Any:
    """The session's zone for pandas, as the standard library's own rules."""
    import zoneinfo

    try:
        # A name alone would reach pytz on pandas 2, which stops applying daylight saving after 2038.
        return zoneinfo.ZoneInfo(time_zone)
    except (zoneinfo.ZoneInfoNotFoundError, ValueError) as error:
        message = f"Conversion Error: the session TimeZone {time_zone!r} is not a zone the standard library knows"
        raise ConversionError(message) from error


def _inward(pd: Any, np: Any, end: Any, zone: Any, *, upper: bool) -> Any:
    """`end` moved inward by the zone's offset, when that offset would show it past the end in local time.

    The offset is pandas' own, measured two days inside the range, where no zone changes it.
    """
    safe = np.array([end + np.timedelta64(-2 if upper else 2, "D")])
    local = pd.Series(safe).dt.tz_localize("UTC").dt.tz_convert(zone).dt.tz_localize(None).to_numpy()
    offset = (local - safe)[0]
    zero = np.timedelta64(0, "s")
    return end - offset if (offset > zero if upper else offset < zero) else end


def _zoned(pd: Any, np: Any, values: Any, mask: Any, time_zone: str) -> Any:
    zone = _zone(time_zone)
    unit = np.datetime_data(values.dtype)[0]
    top, bottom = _TS_CLAMP[unit]
    raw = values.view("int64")
    # An infinity is the end of the range in local time: the same end in UTC can land past the year 9999 or past
    # what nanoseconds hold once shown in the zone, and pandas then fails to display or even read the value.
    for end, upper in ((top, True), (bottom, False)):
        where = raw == end
        if where.any():
            raw[where] = _inward(pd, np, np.datetime64(end, unit), zone, upper=upper).astype("int64")
    return pd.Series(_with_nat(np, values, mask)).dt.tz_localize("UTC").dt.tz_convert(zone)


def _column(pd: Any, np: Any, type_id: int, column: tuple[Any, Any, str, Any], time_zone: str, *, dates: bool) -> Any:
    values, mask, kind, meta = column
    if kind == "numeric":
        return _numeric(pd, np, values, mask)
    if kind == "date":
        out = pd.Series(_date(np, values, mask))
        return out.dt.date if dates else out
    if kind in ("datetime", "timedelta"):
        return _with_nat(np, values, mask)
    if kind == "datetimetz":
        return _zoned(pd, np, values, mask, time_zone)
    if kind == "enum":
        return pd.Categorical.from_codes(values, dtype=pd.CategoricalDtype(meta, ordered=True))
    if type_id == _VARCHAR:
        # The string dtype pandas chooses for itself, also for an empty or all-NULL column, which inference leaves
        # as object.
        return pd.Series(values, dtype="str")
    if mask is not None:
        values[mask] = pd.NA
    return pd.Series(values, dtype=object)


def fetch_pandas(result: _duckdb.Result, *, time_zone: str, date_as_object: bool) -> Any:
    """The whole result as a pandas DataFrame, its TIMESTAMPTZ columns in `time_zone`."""
    import numpy as np

    pd = _pandas()
    type_ids = [type_id for type_id, _, _ in result.schema_types]
    columns = _result_to_columns(result)
    names = _unique_names([name for name, *_ in columns])
    data = {
        name: _column(pd, np, type_id, column[1:], time_zone, dates=date_as_object)
        for name, type_id, column in zip(names, type_ids, columns, strict=True)
    }
    # Every array above is freshly built here, so the frame may own it rather than copy it.
    return pd.DataFrame(data, copy=False)
