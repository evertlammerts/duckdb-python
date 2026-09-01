"""The old client's connection surface, for migration.

The shipping client's `duckdb.connect()` returns a connection that executes
and fetches directly: `conn.execute(...).fetchall()`, with `description` and
`rowcount` on the connection, `cursor()` handing back a duplicate whose
lifetime is tied to its parent, and `sql()`/`table()`/`query()` returning
lazy relations. This face reproduces that contract over the new seam so
migrating code keeps running while it moves. The native surface and the
strict PEP 249 face in `duckdb.dbapi` stay what they are; new code has no
reason to be here.

Old habits are reproduced deliberately, the questionable ones included:
`cursor()` is an independent transaction unlike dbapi's shared one, closing
a connection closes its cursors, `rowcount` is always -1, relations compose
SQL text, `connect()` hands out connections into one shared database
instance per path (with the old config-mismatch refusal), and a module-level
default connection backs `execute()` and friends. What the face still lacks
is measured by the adopted suite under compat/, and every remaining
difference is a behavior-change-log entry, not an accident.
"""

from __future__ import annotations

import sys
import threading
import weakref
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from . import _duckdb
from .connection import Connection, LiveResult, _Catalog
from .dbapi import BINARY, DATETIME, NUMBER, ROWID, STRING, apilevel, paramstyle, threadsafety
from .exceptions import CatalogError, InterfaceError, InterruptError, InvalidInputError
from .expr import qualified, quote

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "BINARY",
    "DATETIME",
    "NUMBER",
    "ROWID",
    "STRING",
    "CatalogException",
    "CompatConnection",
    "CompatRelation",
    "ConnectionException",
    "InterruptException",
    "InvalidInputException",
    "apilevel",
    "connect",
    "default_connection",
    "description",
    "execute",
    "from_query",
    "paramstyle",
    "query",
    "sql",
    "threadsafety",
]

#: The old exception names. ConnectionException covered "closed or unusable",
#: which this package reports as InterfaceError.
CatalogException = CatalogError
ConnectionException = InterfaceError
InterruptException = InterruptError
InvalidInputException = InvalidInputError

#: The old client advertised itself to the engine as python/<major.minor>.
__formatted_python_version__ = f"{sys.version_info.major}.{sys.version_info.minor}"


def _name(key: object) -> str:
    """A parameter name the old client's way: str as is, bytes decoded, anything else through str()."""
    if isinstance(key, str):
        return key
    if isinstance(key, bytes):
        return key.decode()
    return str(key)


class CompatRelation:
    """A lazy relation the old client's way: SQL text composed over one connection.

    The old verbs take SQL fragments as text, so composition is textual and
    never touches the native expression layer. Fetching follows the old
    model: `fetchone`/`fetchmany` run the relation once and drain the held
    result; a bare `fetchall` with nothing held runs fresh every call;
    `execute()` resets the held result.
    """

    def __init__(self, connection: CompatConnection, sql: str, alias: str | None = None) -> None:
        #: Held strongly, so a relation keeps its cursor alive (old GH-315).
        self._connection = connection
        self._sql = sql
        self._relation_alias = alias
        self._result: LiveResult | None = None
        self._closed = False

    def _source(self) -> str:
        aliased = f" AS {quote(self._relation_alias)}" if self._relation_alias else ""
        return f"({self._sql}){aliased}"

    def set_alias(self, alias: str) -> CompatRelation:
        """The same relation under a name later fragments can reference."""
        return CompatRelation(self._connection, self._sql, alias)

    def filter(self, condition: str) -> CompatRelation:
        """The rows where the SQL fragment holds."""
        return CompatRelation(self._connection, f"SELECT * FROM {self._source()} WHERE {condition}")

    def project(self, columns: str) -> CompatRelation:
        """The columns of the SQL select list."""
        return CompatRelation(self._connection, f"SELECT {columns} FROM {self._source()}")

    def aggregate(self, aggregates: str, groups: str = "") -> CompatRelation:
        """Aggregate over the whole relation, or per group."""
        grouped = f" GROUP BY {groups}" if groups else ""
        return CompatRelation(self._connection, f"SELECT {aggregates} FROM {self._source()}{grouped}")

    def query(self, virtual_table_name: str, sql: str) -> CompatRelation:
        """Run SQL that refers to this relation by the given name."""
        return CompatRelation(self._connection, f"WITH {quote(virtual_table_name)} AS ({self._sql}) {sql}")

    def execute(self) -> CompatRelation:
        """Run the relation and hold its rows, resetting whatever was held."""
        self._close_result()
        self._closed = False
        self._result = self._connection._execute(self._sql)
        return self

    def close(self) -> None:
        """Release the held result; fetches afterwards say `result closed`."""
        self._close_result()
        self._closed = True

    def fetchone(self) -> tuple[Any, ...] | None:
        """The next row, running the relation first if nothing is held."""
        rows = self._held().fetch_rows(1)
        return rows[0] if rows else None

    def fetchmany(self, size: int = 1) -> list[tuple[Any, ...]]:
        """Up to `size` rows, running the relation first if nothing is held."""
        if size <= 0:
            return []
        return self._held().fetch_rows(size)

    def fetchall(self) -> list[tuple[Any, ...]]:
        """Every row: the rest of a held result, or a fresh run each call."""
        if self._closed:
            message = "result closed"
            raise InvalidInputException(message)
        if self._result is not None:
            rows = self._result.fetch_all()
            self._close_result()
            return rows
        with self._connection._execute(self._sql) as result:
            return result.fetch_all()

    @property
    def description(self) -> list[tuple[Any, ...]]:
        """Column metadata, from the held result or from binding the text."""
        if self._result is not None:
            schema = self._result.result.schema
        else:
            schema, _ = self._connection._engine().bind(self._sql)
        return [(column, type_text, None, None, None, None, None) for column, type_text in schema]

    def _close_result(self) -> None:
        if self._result is not None:
            self._result.close()
            self._result = None

    def _held(self) -> LiveResult:
        if self._closed:
            message = "result closed"
            raise InvalidInputException(message)
        if self._result is None:
            self.execute()
        return cast("LiveResult", self._result)


class CompatConnection(Connection):
    """A connection speaking the old client's execute-and-fetch vocabulary."""

    def __init__(self, database: _duckdb.Database, catalog: _Catalog | None = None) -> None:
        super().__init__(database, catalog)
        self._held: LiveResult | None = None
        self._compat_description: list[tuple[Any, ...]] | None = None
        #: Cursors made from this connection; the old client closed them with it.
        self._cursors: weakref.WeakSet[CompatConnection] = weakref.WeakSet()
        #: The shared-instance holder, kept so the cache entry lives while we do.
        self._instance: _Instance | None = None

    def execute(self, sql: str, parameters: Sequence[Any] | Mapping[Any, Any] | None = None) -> CompatConnection:
        """Run a statement, holding its rows for the fetch family. Returns self.

        A statement that produces no rows is applied on the spot: the old
        client's INSERT took effect at execute, and this face keeps that.
        """
        self._release_held()
        if isinstance(parameters, Mapping):
            # The old client took any parameter-dict key, so {1: v} fills $1;
            # the native seam refuses non-strings, and the leniency lives here.
            parameters = {_name(key): value for key, value in parameters.items()}
        result = self._execute(sql, parameters)
        if result.result.result_type == "rows":
            self._held = result
            self._compat_description = [
                (name, type_text, None, None, None, None, None) for name, type_text in result.result.schema
            ]
        else:
            with result:
                result.drain()
        return self

    def executemany(self, sql: str, seq_of_parameters: Iterable[Sequence[Any] | Mapping[Any, Any]]) -> CompatConnection:
        """Run one statement once per parameter set. Returns self."""
        for parameters in seq_of_parameters:
            self.execute(sql, parameters)
        return self

    def fetchone(self) -> tuple[Any, ...] | None:
        """The next row, or None when the held result is exhausted."""
        rows = self._require_held().fetch_rows(1)
        return rows[0] if rows else None

    def fetchmany(self, size: int = 1) -> list[tuple[Any, ...]]:
        """Up to `size` rows; the old client's default is one."""
        if size <= 0:
            return []
        return self._require_held().fetch_rows(size)

    def fetchall(self) -> list[tuple[Any, ...]]:
        """Every remaining row of the held result."""
        return self._require_held().fetch_all()

    @property
    def description(self) -> list[tuple[Any, ...]] | None:
        """Column metadata of the held result, or None when there is none."""
        return self._compat_description

    @property
    def rowcount(self) -> int:
        """Always -1, as the old client reported it. Real counts live in `run()` and dbapi."""
        return -1

    def sql(self, query: str) -> CompatRelation:
        """A lazy relation over this SQL, the old client's way."""
        return CompatRelation(self, query)

    def query(self, query: str) -> CompatRelation:
        """The old alias of `sql`."""
        return self.sql(query)

    def table(self, name: str) -> CompatRelation:
        """A relation reading a table by name."""
        return CompatRelation(self, f"SELECT * FROM {qualified(name)}")

    def begin(self) -> CompatConnection:
        """BEGIN TRANSACTION, as a method because the old connection had one."""
        self.run("BEGIN TRANSACTION")
        return self

    def commit(self) -> CompatConnection:
        """COMMIT the open transaction."""
        self.run("COMMIT")
        return self

    def rollback(self) -> CompatConnection:
        """ROLLBACK the open transaction."""
        self.run("ROLLBACK")
        return self

    def cursor(self) -> CompatConnection:
        """An independent duplicate with its own transaction and temp schema.

        The old client's `cursor()` behaved this way, and this face keeps it:
        the PEP 249 cursor that shares its parent's transaction lives in
        `duckdb.dbapi`, not here. Closing this connection closes the cursor,
        as the old client did.
        """
        child = cast("CompatConnection", self.duplicate())
        self._cursors.add(child)
        return child

    def close(self) -> None:
        """Close this connection and every cursor made from it."""
        for child in list(self._cursors):
            child.close()
        self._release_held()
        self._instance = None
        super().close()

    def _release_held(self) -> None:
        if self._held is not None:
            self._held.close()
            self._held = None
        self._compat_description = None

    def _require_held(self) -> LiveResult:
        # The held check speaks first, so fetching after close() reports the
        # old client's "No open result set" rather than a closed connection.
        if self._held is None:
            message = "No open result set"
            raise InvalidInputException(message)
        return self._held


class _Instance:
    """One shared database and its catalog, held so the cache can reference it weakly.

    The engine's Database object cannot take weak references, so this plain
    holder stands in: every connection to the path keeps the holder alive,
    and when the last one closes, the holder dies and the cache entry with
    it, releasing the file.
    """

    __slots__ = ("__weakref__", "catalog", "database")

    def __init__(self, database: _duckdb.Database, catalog: _Catalog) -> None:
        self.database = database
        self.catalog = catalog


#: One shared database instance per path, as the old client's connect() gave
#: out. Weak, so the instance dies with its last connection and a later
#: connect() opens the file afresh.
_instances: dict[str, tuple[weakref.ref[_Instance], frozenset[tuple[str, str]]]] = {}
_instances_lock = threading.Lock()


def connect(
    database: str = ":memory:", read_only: bool = False, config: Mapping[str, object] | None = None
) -> CompatConnection:
    """Open a connection the old client's way: positional path, read_only flag, config dict.

    Connections to the same path share one database instance, so two
    writers see each other and `:memory:name` is shared; asking for the
    same path with a different configuration is refused in the old
    client's words. Plain in-memory databases are never shared.
    """
    options: dict[str, object] = dict(config or {})
    if read_only:
        options["access_mode"] = "read_only"
    options.setdefault("duckdb_api", f"python/{__formatted_python_version__}")
    rendered = [(key, str(value)) for key, value in options.items()]
    if database in ("", ":memory:"):
        return CompatConnection(_duckdb.Database(database, rendered))
    key = database if database.startswith(":memory:") else str(Path(database).resolve())
    fingerprint = frozenset(rendered)
    with _instances_lock:
        entry = _instances.get(key)
        shared = entry[0]() if entry is not None else None
        if shared is not None and entry is not None:
            if fingerprint != entry[1]:
                message = (
                    "Can't open a connection to same database file with a "
                    "different configuration than existing connections"
                )
                raise ConnectionException(message)
        else:
            if entry is not None:
                del _instances[key]
            shared = _Instance(_duckdb.Database(database, rendered), _Catalog())
            _instances[key] = (weakref.ref(shared), fingerprint)
        connection = CompatConnection(shared.database, shared.catalog)
        connection._instance = shared
        return connection


_default: CompatConnection | None = None
_default_lock = threading.Lock()


def default_connection() -> CompatConnection:
    """The module-level connection behind `execute()` and friends, made lazily."""
    global _default
    with _default_lock:
        if _default is None or _default._database is None:
            _default = connect(":memory:")
        return _default


def execute(sql: str, parameters: Sequence[Any] | Mapping[Any, Any] | None = None) -> CompatConnection:
    """`execute` on the default connection, as the old module surface had."""
    return default_connection().execute(sql, parameters)


def sql(query: str) -> CompatRelation:
    """A relation over the default connection."""
    return default_connection().sql(query)


def query(query: str) -> CompatRelation:
    """The old alias of module-level `sql`."""
    return default_connection().sql(query)


def from_query(query: str) -> CompatRelation:
    """The old alias of module-level `sql`."""
    return default_connection().sql(query)


def description() -> list[tuple[Any, ...]] | None:
    """The default connection's `description`."""
    return default_connection().description
