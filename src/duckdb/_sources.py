"""What a Python object registered as a table is, decided without importing the library it comes from."""

from __future__ import annotations

import datetime

#: The capsule type, which the C API exposes no other way; the datetime module's own capsule has it.
_CAPSULE = type(datetime.datetime_CAPI)

#: Read once: a stream, or a reader whose exported streams all drain the same source.
STREAM = "stream"
#: Exports a fresh stream every time it is read.
OBJECT = "object"


def kind_of(obj: object) -> str:
    """Whether `obj` is read once or as often as asked.

    An Arrow stream capsule and a pyarrow RecordBatchReader are streams. Anything else exporting `__arrow_c_stream__`,
    a pyarrow Table or RecordBatch for instance, serves a fresh stream per read.
    """
    if isinstance(obj, _CAPSULE):
        return STREAM
    if not hasattr(obj, "__arrow_c_stream__"):
        message = (
            f"a registered object must export an Arrow stream through __arrow_c_stream__, or be an Arrow stream "
            f"capsule; {type(obj).__name__} does neither"
        )
        raise TypeError(message)
    if any(cls.__name__ == "RecordBatchReader" for cls in type(obj).__mro__):
        return STREAM
    return OBJECT
