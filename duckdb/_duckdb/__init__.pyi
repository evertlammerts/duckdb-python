import typing

"""
DuckDB is an embeddable SQL OLAP Database Management System
"""
from __future__ import annotations
import collections.abc

import polars

import duckdb
import fsspec
import pandas
import pyarrow.lib
from . import functional
from . import typing as duckdb_typing

__all__: list[str] = [
    "ALTER",
    "ANALYZE",
    "ATTACH",
    "BinderException",
    "CALL",
    "CARRIAGE_RETURN_LINE_FEED",
    "CHANGED_ROWS",
    "COLUMNS",
    "COPY",
    "COPY_DATABASE",
    "CREATE",
    "CREATE_FUNC",
    "CSVLineTerminator",
    "CaseExpression",
    "CatalogException",
    "CoalesceOperator",
    "ColumnExpression",
    "ConnectionException",
    "ConstantExpression",
    "ConstraintException",
    "ConversionException",
    "DEFAULT",
    "DELETE",
    "DETACH",
    "DROP",
    "DataError",
    "DatabaseError",
    "DefaultExpression",
    "DependencyException",
    "DuckDBPyConnection",
    "DuckDBPyRelation",
    "EXECUTE",
    "EXPLAIN",
    "EXPORT",
    "EXTENSION",
    "Error",
    "ExpectedResultType",
    "ExplainType",
    "Expression",
    "FatalException",
    "FunctionExpression",
    "HTTPException",
    "INSERT",
    "INVALID",
    "IOException",
    "IntegrityError",
    "InternalError",
    "InternalException",
    "InterruptException",
    "InvalidInputException",
    "InvalidTypeException",
    "LINE_FEED",
    "LOAD",
    "LOGICAL_PLAN",
    "LambdaExpression",
    "MERGE_INTO",
    "MULTI",
    "NOTHING",
    "NotImplementedException",
    "NotSupportedError",
    "OperationalError",
    "OutOfMemoryException",
    "OutOfRangeException",
    "PRAGMA",
    "PREPARE",
    "ParserException",
    "PermissionException",
    "ProgrammingError",
    "PythonExceptionHandling",
    "QUERY_RESULT",
    "RELATION",
    "RETURN_NULL",
    "ROWS",
    "RenderMode",
    "SELECT",
    "SET",
    "SQLExpression",
    "STANDARD",
    "SequenceException",
    "SerializationException",
    "StarExpression",
    "Statement",
    "StatementType",
    "SyntaxException",
    "TRANSACTION",
    "TransactionException",
    "TypeMismatchException",
    "UPDATE",
    "VACUUM",
    "VARIABLE_SET",
    "Warning",
    "aggregate",
    "alias",
    "apilevel",
    "append",
    "array_type",
    "arrow",
    "begin",
    "checkpoint",
    "close",
    "comment",
    "commit",
    "connect",
    "create_function",
    "cursor",
    "decimal_type",
    "default_connection",
    "description",
    "df",
    "distinct",
    "dtype",
    "duplicate",
    "enum_type",
    "execute",
    "executemany",
    "extract_statements",
    "fetch_arrow_table",
    "fetch_df",
    "fetch_df_chunk",
    "fetch_record_batch",
    "fetchall",
    "fetchdf",
    "fetchmany",
    "fetchnumpy",
    "fetchone",
    "filesystem_is_registered",
    "filter",
    "from_arrow",
    "from_csv_auto",
    "from_df",
    "from_parquet",
    "from_query",
    "functional",
    "get_table_names",
    "identifier",
    "install_extension",
    "interrupt",
    "keyword",
    "limit",
    "list_filesystems",
    "list_type",
    "load_extension",
    "map_type",
    "numeric_const",
    "operator",
    "order",
    "paramstyle",
    "pl",
    "project",
    "query",
    "query_df",
    "query_progress",
    "read_csv",
    "read_json",
    "read_parquet",
    "register",
    "register_filesystem",
    "remove_function",
    "rollback",
    "row_type",
    "rowcount",
    "set_default_connection",
    "sql",
    "sqltype",
    "string_const",
    "string_type",
    "struct_type",
    "table",
    "table_function",
    "tf",
    "threadsafety",
    "token_type",
    "tokenize",
    "torch",
    "type",
    "typing",
    "union_type",
    "unregister",
    "unregister_filesystem",
    "values",
    "view",
    "write_csv",
]

class BinderException(ProgrammingError):
    pass

class CSVLineTerminator:
    """
    Members:

      LINE_FEED

      CARRIAGE_RETURN_LINE_FEED
    """

    CARRIAGE_RETURN_LINE_FEED: typing.ClassVar[
        CSVLineTerminator
    ]  # value = <CSVLineTerminator.CARRIAGE_RETURN_LINE_FEED: 1>
    LINE_FEED: typing.ClassVar[CSVLineTerminator]  # value = <CSVLineTerminator.LINE_FEED: 0>
    __members__: typing.ClassVar[
        dict[str, CSVLineTerminator]
    ]  # value = {'LINE_FEED': <CSVLineTerminator.LINE_FEED: 0>, 'CARRIAGE_RETURN_LINE_FEED': <CSVLineTerminator.CARRIAGE_RETURN_LINE_FEED: 1>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class CatalogException(ProgrammingError):
    pass

class ConnectionException(OperationalError):
    pass

class ConstraintException(IntegrityError):
    pass

class ConversionException(DataError):
    pass

class DataError(DatabaseError):
    pass

class DatabaseError(Error):
    pass

class DependencyException(DatabaseError):
    pass

class DuckDBPyConnection:
    def __del__(self) -> None: ...
    def __enter__(self) -> DuckDBPyConnection: ...
    def __exit__(self, exc_type: typing.Any, exc: typing.Any, traceback: typing.Any) -> None: ...
    def append(self, table_name: str, df: pandas.DataFrame, *, by_name: bool = False) -> DuckDBPyConnection:
        """
        Append the passed DataFrame to the named table
        """
    def array_type(self, type: duckdb_typing.DuckDBPyType, size: typing.SupportsInt) -> duckdb_typing.DuckDBPyType:
        """
        Create an array type object of 'type'
        """
    def arrow(self, rows_per_batch: typing.SupportsInt = 1000000) -> pyarrow.lib.RecordBatchReader:
        """
        Fetch an Arrow RecordBatchReader following execute()
        """
    def begin(self) -> DuckDBPyConnection:
        """
        Start a new transaction
        """
    def checkpoint(self) -> DuckDBPyConnection:
        """
        Synchronizes data in the write-ahead log (WAL) to the database data file (no-op for in-memory connections)
        """
    def close(self) -> None:
        """
        Close the connection
        """
    def commit(self) -> DuckDBPyConnection:
        """
        Commit changes performed within a transaction
        """
    def create_function(
        self,
        name: str,
        function: collections.abc.Callable,
        parameters: typing.Any = None,
        return_type: typing.Optional[duckdb_typing.DuckDBPyType] = None,
        *,
        type: functional.PythonUDFType = functional.PythonUDFType.NATIVE,
        null_handling: functional.FunctionNullHandling = functional.FunctionNullHandling.DEFAULT,
        exception_handling: PythonExceptionHandling = PythonExceptionHandling.DEFAULT,
        side_effects: bool = False,
    ) -> DuckDBPyConnection:
        """
        Create a DuckDB function out of the passing in Python function so it can be used in queries
        """
    def cursor(self) -> DuckDBPyConnection:
        """
        Create a duplicate of the current connection
        """
    def decimal_type(self, width: typing.SupportsInt, scale: typing.SupportsInt) -> duckdb_typing.DuckDBPyType:
        """
        Create a decimal type with 'width' and 'scale'
        """
    def df(self, *, date_as_object: bool = False) -> pandas.DataFrame:
        """
        Fetch a result as DataFrame following execute()
        """
    def dtype(self, type_str: str) -> duckdb_typing.DuckDBPyType:
        """
        Create a type object by parsing the 'type_str' string
        """
    def duplicate(self) -> DuckDBPyConnection:
        """
        Create a duplicate of the current connection
        """
    def enum_type(self, name: str, type: duckdb_typing.DuckDBPyType, values: list) -> duckdb_typing.DuckDBPyType:
        """
        Create an enum type of underlying 'type', consisting of the list of 'values'
        """
    def execute(self, query: typing.Any, parameters: typing.Any = None) -> DuckDBPyConnection:
        """
        Execute the given SQL query, optionally using prepared statements with parameters set
        """
    def executemany(self, query: typing.Any, parameters: typing.Any = None) -> DuckDBPyConnection:
        """
        Execute the given prepared statement multiple times using the list of parameter sets in parameters
        """
    def extract_statements(self, query: str) -> list:
        """
        Parse the query string and extract the Statement object(s) produced
        """
    def fetch_arrow_table(self, rows_per_batch: typing.SupportsInt = 1000000) -> pyarrow.lib.Table:
        """
        Fetch a result as Arrow table following execute()
        """
    def fetch_df(self, *, date_as_object: bool = False) -> pandas.DataFrame:
        """
        Fetch a result as DataFrame following execute()
        """
    def fetch_df_chunk(
        self, vectors_per_chunk: typing.SupportsInt = 1, *, date_as_object: bool = False
    ) -> pandas.DataFrame:
        """
        Fetch a chunk of the result as DataFrame following execute()
        """
    def fetch_record_batch(self, rows_per_batch: typing.SupportsInt = 1000000) -> pyarrow.lib.RecordBatchReader:
        """
        Fetch an Arrow RecordBatchReader following execute()
        """
    def fetchall(self) -> list:
        """
        Fetch all rows from a result following execute
        """
    def fetchdf(self, *, date_as_object: bool = False) -> pandas.DataFrame:
        """
        Fetch a result as DataFrame following execute()
        """
    def fetchmany(self, size: typing.SupportsInt = 1) -> list:
        """
        Fetch the next set of rows from a result following execute
        """
    def fetchnumpy(self) -> dict:
        """
        Fetch a result as list of NumPy arrays following execute
        """
    def fetchone(self) -> tuple | None:
        """
        Fetch a single row from a result following execute
        """
    def filesystem_is_registered(self, name: str) -> bool:
        """
        Check if a filesystem with the provided name is currently registered
        """
    def from_arrow(self, arrow_object: typing.Any) -> DuckDBPyRelation:
        """
        Create a relation object from an Arrow object
        """
    def from_csv_auto(self, path_or_buffer: typing.Any, **kwargs) -> DuckDBPyRelation:
        """
        Create a relation object from the CSV file in 'name'
        """
    def from_df(self, df: pandas.DataFrame) -> DuckDBPyRelation:
        """
        Create a relation object from the DataFrame in df
        """
    @typing.overload
    def from_parquet(
        self,
        file_glob: str,
        binary_as_string: bool = False,
        *,
        file_row_number: bool = False,
        filename: bool = False,
        hive_partitioning: bool = False,
        union_by_name: bool = False,
        compression: typing.Any = None,
    ) -> DuckDBPyRelation:
        """
        Create a relation object from the Parquet files in file_glob
        """
    @typing.overload
    def from_parquet(
        self,
        file_globs: collections.abc.Sequence[str],
        binary_as_string: bool = False,
        *,
        file_row_number: bool = False,
        filename: bool = False,
        hive_partitioning: bool = False,
        union_by_name: bool = False,
        compression: typing.Any = None,
    ) -> DuckDBPyRelation:
        """
        Create a relation object from the Parquet files in file_globs
        """
    def from_query(self, query: typing.Any, *, alias: str = "", params: typing.Any = None) -> DuckDBPyRelation:
        """
        Run a SQL query. If it is a SELECT statement, create a relation object from the given SQL query, otherwise run the query as-is.
        """
    def get_table_names(self, query: str, *, qualified: bool = False) -> set[str]:
        """
        Extract the required table names from a query
        """
    def install_extension(
        self,
        extension: str,
        *,
        force_install: bool = False,
        repository: typing.Any = None,
        repository_url: typing.Any = None,
        version: typing.Any = None,
    ) -> None:
        """
        Install an extension by name, with an optional version and/or repository to get the extension from
        """
    def interrupt(self) -> None:
        """
        Interrupt pending operations
        """
    def list_filesystems(self) -> list:
        """
        List registered filesystems, including builtin ones
        """
    def list_type(self, type: duckdb_typing.DuckDBPyType) -> duckdb_typing.DuckDBPyType:
        """
        Create a list type object of 'type'
        """
    def load_extension(self, extension: str) -> None:
        """
        Load an installed extension
        """
    def map_type(
        self, key: duckdb_typing.DuckDBPyType, value: duckdb_typing.DuckDBPyType
    ) -> duckdb_typing.DuckDBPyType:
        """
        Create a map type object from 'key_type' and 'value_type'
        """
    def pl(self, rows_per_batch: typing.SupportsInt = 1000000, *, lazy: bool = False) -> polars.DataFrame:
        """
        Fetch a result as Polars DataFrame following execute()
        """
    def query(self, query: typing.Any, *, alias: str = "", params: typing.Any = None) -> DuckDBPyRelation:
        """
        Run a SQL query. If it is a SELECT statement, create a relation object from the given SQL query, otherwise run the query as-is.
        """
    def query_progress(self) -> float:
        """
        Query progress of pending operation
        """
    def read_csv(self, path_or_buffer: typing.Any, **kwargs) -> DuckDBPyRelation:
        """
        Create a relation object from the CSV file in 'name'
        """
    def read_json(
        self,
        path_or_buffer: typing.Any,
        *,
        columns: typing.Optional[typing.Any | None] = None,
        sample_size: typing.Optional[typing.Any | None] = None,
        maximum_depth: typing.Optional[typing.Any | None] = None,
        records: typing.Optional[str | None] = None,
        format: typing.Optional[str | None] = None,
        date_format: typing.Optional[typing.Any | None] = None,
        timestamp_format: typing.Optional[typing.Any | None] = None,
        compression: typing.Optional[typing.Any | None] = None,
        maximum_object_size: typing.Optional[typing.Any | None] = None,
        ignore_errors: typing.Optional[typing.Any | None] = None,
        convert_strings_to_integers: typing.Optional[typing.Any | None] = None,
        field_appearance_threshold: typing.Optional[typing.Any | None] = None,
        map_inference_threshold: typing.Optional[typing.Any | None] = None,
        maximum_sample_files: typing.Optional[typing.Any | None] = None,
        filename: typing.Optional[typing.Any | None] = None,
        hive_partitioning: typing.Optional[typing.Any | None] = None,
        union_by_name: typing.Optional[typing.Any | None] = None,
        hive_types: typing.Optional[typing.Any | None] = None,
        hive_types_autocast: typing.Optional[typing.Any | None] = None,
    ) -> DuckDBPyRelation:
        """
        Create a relation object from the JSON file in 'name'
        """
    @typing.overload
    def read_parquet(
        self,
        file_glob: str,
        binary_as_string: bool = False,
        *,
        file_row_number: bool = False,
        filename: bool = False,
        hive_partitioning: bool = False,
        union_by_name: bool = False,
        compression: typing.Any = None,
    ) -> DuckDBPyRelation:
        """
        Create a relation object from the Parquet files in file_glob
        """
    @typing.overload
    def read_parquet(
        self,
        file_globs: collections.abc.Sequence[str],
        binary_as_string: bool = False,
        *,
        file_row_number: bool = False,
        filename: bool = False,
        hive_partitioning: bool = False,
        union_by_name: bool = False,
        compression: typing.Any = None,
    ) -> DuckDBPyRelation:
        """
        Create a relation object from the Parquet files in file_globs
        """
    def register(self, view_name: str, python_object: typing.Any) -> DuckDBPyConnection:
        """
        Register the passed Python Object value for querying with a view
        """
    def register_filesystem(self, filesystem: fsspec.AbstractFileSystem) -> None:
        """
        Register a fsspec compliant filesystem
        """
    def remove_function(self, name: str) -> DuckDBPyConnection:
        """
        Remove a previously created function
        """
    def rollback(self) -> DuckDBPyConnection:
        """
        Roll back changes performed within a transaction
        """
    def row_type(self, fields: typing.Any) -> duckdb_typing.DuckDBPyType:
        """
        Create a struct type object from 'fields'
        """
    def sql(self, query: typing.Any, *, alias: str = "", params: typing.Any = None) -> DuckDBPyRelation:
        """
        Run a SQL query. If it is a SELECT statement, create a relation object from the given SQL query, otherwise run the query as-is.
        """
    def sqltype(self, type_str: str) -> duckdb_typing.DuckDBPyType:
        """
        Create a type object by parsing the 'type_str' string
        """
    def string_type(self, collation: str = "") -> duckdb_typing.DuckDBPyType:
        """
        Create a string type with an optional collation
        """
    def struct_type(self, fields: typing.Any) -> duckdb_typing.DuckDBPyType:
        """
        Create a struct type object from 'fields'
        """
    def table(self, table_name: str) -> DuckDBPyRelation:
        """
        Create a relation object for the named table
        """
    def table_function(self, name: str, parameters: typing.Any = None) -> DuckDBPyRelation:
        """
        Create a relation object from the named table function with given parameters
        """
    def tf(self) -> dict:
        """
        Fetch a result as dict of TensorFlow Tensors following execute()
        """
    def torch(self) -> dict:
        """
        Fetch a result as dict of PyTorch Tensors following execute()
        """
    def type(self, type_str: str) -> duckdb_typing.DuckDBPyType:
        """
        Create a type object by parsing the 'type_str' string
        """
    def union_type(self, members: typing.Any) -> duckdb_typing.DuckDBPyType:
        """
        Create a union type object from 'members'
        """
    def unregister(self, view_name: str) -> DuckDBPyConnection:
        """
        Unregister the view name
        """
    def unregister_filesystem(self, name: str) -> None:
        """
        Unregister a filesystem
        """
    def values(self, *args) -> DuckDBPyRelation:
        """
        Create a relation object from the passed values
        """
    def view(self, view_name: str) -> DuckDBPyRelation:
        """
        Create a relation object for the named view
        """
    @property
    def description(self) -> list | None:
        """
        Get result set attributes, mainly column names
        """
    @property
    def rowcount(self) -> int:
        """
        Get result set row count
        """

class DuckDBPyRelation:
    def __arrow_c_stream__(self, requested_schema: typing.Any = None) -> typing.Any:
        """
        Execute and return an ArrowArrayStream through the Arrow PyCapsule Interface.

        https://arrow.apache.org/docs/dev/format/CDataInterface/PyCapsuleInterface.html
        """
    def __contains__(self, name: str) -> bool: ...
    def __getattr__(self, name: str) -> DuckDBPyRelation:
        """
        Get a projection relation created from this relation, on the provided column name
        """
    def __getitem__(self, name: str) -> DuckDBPyRelation:
        """
        Get a projection relation created from this relation, on the provided column name
        """
    def __len__(self) -> int:
        """
        Number of rows in relation.
        """
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def aggregate(self, aggr_expr: typing.Any, group_expr: str = "") -> DuckDBPyRelation:
        """
        Compute the aggregate aggr_expr by the optional groups group_expr on the relation
        """
    def any_value(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Returns the first non-null value from a given column
        """
    def apply(
        self,
        function_name: str,
        function_aggr: str,
        group_expr: str = "",
        function_parameter: str = "",
        projected_columns: str = "",
    ) -> DuckDBPyRelation:
        """
        Compute the function of a single column or a list of columns by the optional groups on the relation
        """
    def arg_max(
        self, arg_column: str, value_column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Finds the row with the maximum value for a value column and returns the value of that row for an argument column
        """
    def arg_min(
        self, arg_column: str, value_column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Finds the row with the minimum value for a value column and returns the value of that row for an argument column
        """
    def arrow(self, batch_size: typing.SupportsInt = 1000000) -> pyarrow.lib.RecordBatchReader:
        """
        Execute and return an Arrow Record Batch Reader that yields all rows
        """
    def avg(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Computes the average on a given column
        """
    def bit_and(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Computes the bitwise AND of all bits present in a given column
        """
    def bit_or(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Computes the bitwise OR of all bits present in a given column
        """
    def bit_xor(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Computes the bitwise XOR of all bits present in a given column
        """
    def bitstring_agg(
        self,
        column: str,
        min: typing.Optional[typing.Any | None] = None,
        max: typing.Optional[typing.Any | None] = None,
        groups: str = "",
        window_spec: str = "",
        projected_columns: str = "",
    ) -> DuckDBPyRelation:
        """
        Computes a bitstring with bits set for each distinct value in a given column
        """
    def bool_and(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Computes the logical AND of all values present in a given column
        """
    def bool_or(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Computes the logical OR of all values present in a given column
        """
    def close(self) -> None:
        """
        Closes the result
        """
    def count(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Computes the number of elements present in a given column
        """
    def create(self, table_name: str) -> None:
        """
        Creates a new table named table_name with the contents of the relation object
        """
    def create_view(self, view_name: str, replace: bool = True) -> DuckDBPyRelation:
        """
        Creates a view named view_name that refers to the relation object
        """
    def cross(self, other_rel: DuckDBPyRelation) -> DuckDBPyRelation:
        """
        Create cross/cartesian product of two relational objects
        """
    def cume_dist(self, window_spec: str, projected_columns: str = "") -> DuckDBPyRelation:
        """
        Computes the cumulative distribution within the partition
        """
    def dense_rank(self, window_spec: str, projected_columns: str = "") -> DuckDBPyRelation:
        """
        Computes the dense rank within the partition
        """
    def describe(self) -> DuckDBPyRelation:
        """
        Gives basic statistics (e.g., min, max) and if NULL exists for each column of the relation.
        """
    def df(self, *, date_as_object: bool = False) -> pandas.DataFrame:
        """
        Execute and fetch all rows as a pandas DataFrame
        """
    def distinct(self) -> DuckDBPyRelation:
        """
        Retrieve distinct rows from this relation object
        """
    def except_(self, other_rel: DuckDBPyRelation) -> DuckDBPyRelation:
        """
        Create the set except of this relation object with another relation object in other_rel
        """
    def execute(self) -> DuckDBPyRelation:
        """
        Transform the relation into a result set
        """
    def explain(self, type: ExplainType = "standard") -> str: ...
    def favg(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Computes the average of all values present in a given column using a more accurate floating point summation (Kahan Sum)
        """
    def fetch_arrow_reader(self, batch_size: typing.SupportsInt = 1000000) -> pyarrow.lib.RecordBatchReader:
        """
        Execute and return an Arrow Record Batch Reader that yields all rows
        """
    def fetch_arrow_table(self, batch_size: typing.SupportsInt = 1000000) -> pyarrow.lib.Table:
        """
        Execute and fetch all rows as an Arrow Table
        """
    def fetch_df_chunk(
        self, vectors_per_chunk: typing.SupportsInt = 1, *, date_as_object: bool = False
    ) -> pandas.DataFrame:
        """
        Execute and fetch a chunk of the rows
        """
    def fetch_record_batch(self, rows_per_batch: typing.SupportsInt = 1000000) -> pyarrow.lib.RecordBatchReader:
        """
        Execute and return an Arrow Record Batch Reader that yields all rows
        """
    def fetchall(self) -> list:
        """
        Execute and fetch all rows as a list of tuples
        """
    def fetchdf(self, *, date_as_object: bool = False) -> pandas.DataFrame:
        """
        Execute and fetch all rows as a pandas DataFrame
        """
    def fetchmany(self, size: typing.SupportsInt = 1) -> list:
        """
        Execute and fetch the next set of rows as a list of tuples
        """
    def fetchnumpy(self) -> dict:
        """
        Execute and fetch all rows as a Python dict mapping each column to one numpy arrays
        """
    def fetchone(self) -> tuple | None:
        """
        Execute and fetch a single row as a tuple
        """
    def filter(self, filter_expr: typing.Any) -> DuckDBPyRelation:
        """
        Filter the relation object by the filter in filter_expr
        """
    def first(self, column: str, groups: str = "", projected_columns: str = "") -> DuckDBPyRelation:
        """
        Returns the first value of a given column
        """
    def first_value(self, column: str, window_spec: str = "", projected_columns: str = "") -> DuckDBPyRelation:
        """
        Computes the first value within the group or partition
        """
    def fsum(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Computes the sum of all values present in a given column using a more accurate floating point summation (Kahan Sum)
        """
    def geomean(self, column: str, groups: str = "", projected_columns: str = "") -> DuckDBPyRelation:
        """
        Computes the geometric mean over all values present in a given column
        """
    def histogram(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Computes the histogram over all values present in a given column
        """
    def insert(self, values: typing.Any) -> None:
        """
        Inserts the given values into the relation
        """
    def insert_into(self, table_name: str) -> None:
        """
        Inserts the relation object into an existing table named table_name
        """
    def intersect(self, other_rel: DuckDBPyRelation) -> DuckDBPyRelation:
        """
        Create the set intersection of this relation object with another relation object in other_rel
        """
    def join(self, other_rel: DuckDBPyRelation, condition: typing.Any, how: str = "inner") -> DuckDBPyRelation:
        """
        Join the relation object with another relation object in other_rel using the join condition expression in join_condition. Types supported are 'inner', 'left', 'right', 'outer', 'semi' and 'anti'
        """
    def lag(
        self,
        column: str,
        window_spec: str,
        offset: typing.SupportsInt = 1,
        default_value: str = "NULL",
        ignore_nulls: bool = False,
        projected_columns: str = "",
    ) -> DuckDBPyRelation:
        """
        Computes the lag within the partition
        """
    def last(self, column: str, groups: str = "", projected_columns: str = "") -> DuckDBPyRelation:
        """
        Returns the last value of a given column
        """
    def last_value(self, column: str, window_spec: str = "", projected_columns: str = "") -> DuckDBPyRelation:
        """
        Computes the last value within the group or partition
        """
    def lead(
        self,
        column: str,
        window_spec: str,
        offset: typing.SupportsInt = 1,
        default_value: str = "NULL",
        ignore_nulls: bool = False,
        projected_columns: str = "",
    ) -> DuckDBPyRelation:
        """
        Computes the lead within the partition
        """
    def limit(self, n: typing.SupportsInt, offset: typing.SupportsInt = 0) -> DuckDBPyRelation:
        """
        Only retrieve the first n rows from this relation object, starting at offset
        """
    def list(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Returns a list containing all values present in a given column
        """
    def map(
        self, map_function: collections.abc.Callable, *, schema: typing.Optional[typing.Any | None] = None
    ) -> DuckDBPyRelation:
        """
        Calls the passed function on the relation
        """
    def max(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Returns the maximum value present in a given column
        """
    def mean(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Computes the average on a given column
        """
    def median(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Computes the median over all values present in a given column
        """
    def min(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Returns the minimum value present in a given column
        """
    def mode(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Computes the mode over all values present in a given column
        """
    def n_tile(
        self, window_spec: str, num_buckets: typing.SupportsInt, projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Divides the partition as equally as possible into num_buckets
        """
    def nth_value(
        self,
        column: str,
        window_spec: str,
        offset: typing.SupportsInt,
        ignore_nulls: bool = False,
        projected_columns: str = "",
    ) -> DuckDBPyRelation:
        """
        Computes the nth value within the partition
        """
    def order(self, order_expr: str) -> DuckDBPyRelation:
        """
        Reorder the relation object by order_expr
        """
    def percent_rank(self, window_spec: str, projected_columns: str = "") -> DuckDBPyRelation:
        """
        Computes the relative rank within the partition
        """
    def pl(self, batch_size: typing.SupportsInt = 1000000, *, lazy: bool = False) -> polars.DataFrame:
        """
        Execute and fetch all rows as a Polars DataFrame
        """
    def product(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Returns the product of all values present in a given column
        """
    def project(self, *args, groups: str = "") -> DuckDBPyRelation:
        """
        Project the relation object by the projection in project_expr
        """
    def quantile(
        self, column: str, q: typing.Any = 0.5, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Computes the exact quantile value for a given column
        """
    def quantile_cont(
        self, column: str, q: typing.Any = 0.5, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Computes the interpolated quantile value for a given column
        """
    def quantile_disc(
        self, column: str, q: typing.Any = 0.5, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Computes the exact quantile value for a given column
        """
    def query(self, virtual_table_name: str, sql_query: str) -> DuckDBPyRelation:
        """
        Run the given SQL query in sql_query on the view named virtual_table_name that refers to the relation object
        """
    def rank(self, window_spec: str, projected_columns: str = "") -> DuckDBPyRelation:
        """
        Computes the rank within the partition
        """
    def rank_dense(self, window_spec: str, projected_columns: str = "") -> DuckDBPyRelation:
        """
        Computes the dense rank within the partition
        """
    def record_batch(self, batch_size: typing.SupportsInt = 1000000) -> typing.Any: ...
    def row_number(self, window_spec: str, projected_columns: str = "") -> DuckDBPyRelation:
        """
        Computes the row number within the partition
        """
    def select(self, *args, groups: str = "") -> DuckDBPyRelation:
        """
        Project the relation object by the projection in project_expr
        """
    def select_dtypes(self, types: typing.Any) -> DuckDBPyRelation:
        """
        Select columns from the relation, by filtering based on type(s)
        """
    def select_types(self, types: typing.Any) -> DuckDBPyRelation:
        """
        Select columns from the relation, by filtering based on type(s)
        """
    def set_alias(self, alias: str) -> DuckDBPyRelation:
        """
        Rename the relation object to new alias
        """
    def show(
        self,
        *,
        max_width: typing.Optional[typing.SupportsInt | None] = None,
        max_rows: typing.Optional[typing.SupportsInt | None] = None,
        max_col_width: typing.Optional[typing.SupportsInt | None] = None,
        null_value: typing.Optional[str | None] = None,
        render_mode: typing.Any = None,
    ) -> None:
        """
        Display a summary of the data
        """
    def sort(self, *args) -> DuckDBPyRelation:
        """
        Reorder the relation object by the provided expressions
        """
    def sql_query(self) -> str:
        """
        Get the SQL query that is equivalent to the relation
        """
    def std(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Computes the sample standard deviation for a given column
        """
    def stddev(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Computes the sample standard deviation for a given column
        """
    def stddev_pop(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Computes the population standard deviation for a given column
        """
    def stddev_samp(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Computes the sample standard deviation for a given column
        """
    def string_agg(
        self, column: str, sep: str = ",", groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Concatenates the values present in a given column with a separator
        """
    def sum(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Computes the sum of all values present in a given column
        """
    def tf(self) -> dict:
        """
        Fetch a result as dict of TensorFlow Tensors
        """
    def to_arrow_table(self, batch_size: typing.SupportsInt = 1000000) -> pyarrow.lib.Table:
        """
        Execute and fetch all rows as an Arrow Table
        """
    def to_csv(
        self,
        file_name: str,
        *,
        sep: typing.Any = None,
        na_rep: typing.Any = None,
        header: typing.Any = None,
        quotechar: typing.Any = None,
        escapechar: typing.Any = None,
        date_format: typing.Any = None,
        timestamp_format: typing.Any = None,
        quoting: typing.Any = None,
        encoding: typing.Any = None,
        compression: typing.Any = None,
        overwrite: typing.Any = None,
        per_thread_output: typing.Any = None,
        use_tmp_file: typing.Any = None,
        partition_by: typing.Any = None,
        write_partition_columns: typing.Any = None,
    ) -> None:
        """
        Write the relation object to a CSV file in 'file_name'
        """
    def to_df(self, *, date_as_object: bool = False) -> pandas.DataFrame:
        """
        Execute and fetch all rows as a pandas DataFrame
        """
    def to_parquet(
        self,
        file_name: str,
        *,
        compression: typing.Any = None,
        field_ids: typing.Any = None,
        row_group_size_bytes: typing.Any = None,
        row_group_size: typing.Any = None,
        overwrite: typing.Any = None,
        per_thread_output: typing.Any = None,
        use_tmp_file: typing.Any = None,
        partition_by: typing.Any = None,
        write_partition_columns: typing.Any = None,
        append: typing.Any = None,
    ) -> None:
        """
        Write the relation object to a Parquet file in 'file_name'
        """
    def to_table(self, table_name: str) -> None:
        """
        Creates a new table named table_name with the contents of the relation object
        """
    def to_view(self, view_name: str, replace: bool = True) -> DuckDBPyRelation:
        """
        Creates a view named view_name that refers to the relation object
        """
    def torch(self) -> dict:
        """
        Fetch a result as dict of PyTorch Tensors
        """
    def union(self, union_rel: DuckDBPyRelation) -> DuckDBPyRelation:
        """
        Create the set union of this relation object with another relation object in other_rel
        """
    def unique(self, unique_aggr: str) -> DuckDBPyRelation:
        """
        Returns the distinct values in a column.
        """
    def update(self, set: typing.Any, *, condition: typing.Any = None) -> None:
        """
        Update the given relation with the provided expressions
        """
    def value_counts(self, column: str, groups: str = "") -> DuckDBPyRelation:
        """
        Computes the number of elements present in a given column, also projecting the original column
        """
    def var(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Computes the sample variance for a given column
        """
    def var_pop(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Computes the population variance for a given column
        """
    def var_samp(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Computes the sample variance for a given column
        """
    def variance(
        self, column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation:
        """
        Computes the sample variance for a given column
        """
    def write_csv(
        self,
        file_name: str,
        *,
        sep: typing.Any = None,
        na_rep: typing.Any = None,
        header: typing.Any = None,
        quotechar: typing.Any = None,
        escapechar: typing.Any = None,
        date_format: typing.Any = None,
        timestamp_format: typing.Any = None,
        quoting: typing.Any = None,
        encoding: typing.Any = None,
        compression: typing.Any = None,
        overwrite: typing.Any = None,
        per_thread_output: typing.Any = None,
        use_tmp_file: typing.Any = None,
        partition_by: typing.Any = None,
        write_partition_columns: typing.Any = None,
    ) -> None:
        """
        Write the relation object to a CSV file in 'file_name'
        """
    def write_parquet(
        self,
        file_name: str,
        *,
        compression: typing.Any = None,
        field_ids: typing.Any = None,
        row_group_size_bytes: typing.Any = None,
        row_group_size: typing.Any = None,
        overwrite: typing.Any = None,
        per_thread_output: typing.Any = None,
        use_tmp_file: typing.Any = None,
        partition_by: typing.Any = None,
        write_partition_columns: typing.Any = None,
        append: typing.Any = None,
    ) -> None:
        """
        Write the relation object to a Parquet file in 'file_name'
        """
    @property
    def alias(self) -> str:
        """
        Get the name of the current alias
        """
    @property
    def columns(self) -> list:
        """
        Return a list containing the names of the columns of the relation.
        """
    @property
    def description(self) -> list:
        """
        Return the description of the result
        """
    @property
    def dtypes(self) -> list:
        """
        Return a list containing the types of the columns of the relation.
        """
    @property
    def shape(self) -> tuple:
        """
        Tuple of # of rows, # of columns in relation.
        """
    @property
    def type(self) -> str:
        """
        Get the type of the relation.
        """
    @property
    def types(self) -> list:
        """
        Return a list containing the types of the columns of the relation.
        """

class Error(Exception):
    pass

class ExpectedResultType:
    """
    Members:

      QUERY_RESULT

      CHANGED_ROWS

      NOTHING
    """

    CHANGED_ROWS: typing.ClassVar[ExpectedResultType]  # value = <ExpectedResultType.CHANGED_ROWS: 1>
    NOTHING: typing.ClassVar[ExpectedResultType]  # value = <ExpectedResultType.NOTHING: 2>
    QUERY_RESULT: typing.ClassVar[ExpectedResultType]  # value = <ExpectedResultType.QUERY_RESULT: 0>
    __members__: typing.ClassVar[
        dict[str, ExpectedResultType]
    ]  # value = {'QUERY_RESULT': <ExpectedResultType.QUERY_RESULT: 0>, 'CHANGED_ROWS': <ExpectedResultType.CHANGED_ROWS: 1>, 'NOTHING': <ExpectedResultType.NOTHING: 2>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class ExplainType:
    """
    Members:

      STANDARD

      ANALYZE
    """

    ANALYZE: typing.ClassVar[ExplainType]  # value = <ExplainType.ANALYZE: 1>
    STANDARD: typing.ClassVar[ExplainType]  # value = <ExplainType.STANDARD: 0>
    __members__: typing.ClassVar[
        dict[str, ExplainType]
    ]  # value = {'STANDARD': <ExplainType.STANDARD: 0>, 'ANALYZE': <ExplainType.ANALYZE: 1>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class Expression:
    __hash__: typing.ClassVar[None] = None
    def __add__(self, expr: Expression) -> Expression:
        """
        Add expr to self

        Parameters:
                expr: The expression to add together with

        Returns:
                FunctionExpression: self '+' expr
        """
    def __and__(self, arg0: Expression) -> Expression:
        """
        Binary-and self together with expr

        Parameters:
                expr: The expression to AND together with self

        Returns:
                FunctionExpression: self '&' expr
        """
    def __div__(self, arg0: Expression) -> Expression:
        """
        Divide self by expr

        Parameters:
                expr: The expression to divide by

        Returns:
                FunctionExpression: self '/' expr
        """
    def __eq__(self, arg0: Expression) -> Expression:
        """
        Create an equality expression between two expressions

        Parameters:
                expr: The expression to check equality with

        Returns:
                FunctionExpression: self '=' expr
        """
    def __floordiv__(self, arg0: Expression) -> Expression:
        """
        (Floor) Divide self by expr

        Parameters:
                expr: The expression to (floor) divide by

        Returns:
                FunctionExpression: self '//' expr
        """
    def __ge__(self, arg0: Expression) -> Expression:
        """
        Create a greater than or equal expression between two expressions

        Parameters:
                expr: The expression to check

        Returns:
                FunctionExpression: self '>=' expr
        """
    def __gt__(self, arg0: Expression) -> Expression:
        """
        Create a greater than expression between two expressions

        Parameters:
                expr: The expression to check

        Returns:
                FunctionExpression: self '>' expr
        """
    @typing.overload
    def __init__(self, arg0: str) -> None: ...
    @typing.overload
    def __init__(self, arg0: typing.Any) -> None: ...
    def __invert__(self) -> Expression:
        """
        Create a binary-not expression from self

        Returns:
                FunctionExpression: ~self
        """
    def __le__(self, arg0: Expression) -> Expression:
        """
        Create a less than or equal expression between two expressions

        Parameters:
                expr: The expression to check

        Returns:
                FunctionExpression: self '<=' expr
        """
    def __lt__(self, arg0: Expression) -> Expression:
        """
        Create a less than expression between two expressions

        Parameters:
                expr: The expression to check

        Returns:
                FunctionExpression: self '<' expr
        """
    def __mod__(self, arg0: Expression) -> Expression:
        """
        Modulo self by expr

        Parameters:
                expr: The expression to modulo by

        Returns:
                FunctionExpression: self '%' expr
        """
    def __mul__(self, arg0: Expression) -> Expression:
        """
        Multiply self by expr

        Parameters:
                expr: The expression to multiply by

        Returns:
                FunctionExpression: self '*' expr
        """
    def __ne__(self, arg0: Expression) -> Expression:
        """
        Create an inequality expression between two expressions

        Parameters:
                expr: The expression to check inequality with

        Returns:
                FunctionExpression: self '!=' expr
        """
    def __neg__(self) -> Expression:
        """
        Negate the expression.

        Returns:
                FunctionExpression: -self
        """
    def __or__(self, arg0: Expression) -> Expression:
        """
        Binary-or self together with expr

        Parameters:
                expr: The expression to OR together with self

        Returns:
                FunctionExpression: self '|' expr
        """
    def __pow__(self, arg0: Expression) -> Expression:
        """
        Power self by expr

        Parameters:
                expr: The expression to power by

        Returns:
                FunctionExpression: self '**' expr
        """
    def __radd__(self, arg0: Expression) -> Expression:
        """
        Add expr to self

        Parameters:
                expr: The expression to add together with

        Returns:
                FunctionExpression: self '+' expr
        """
    def __rand__(self, arg0: Expression) -> Expression:
        """
        Binary-and self together with expr

        Parameters:
                expr: The expression to AND together with self

        Returns:
                FunctionExpression: expr '&' self
        """
    def __rdiv__(self, arg0: Expression) -> Expression:
        """
        Divide self by expr

        Parameters:
                expr: The expression to divide by

        Returns:
                FunctionExpression: self '/' expr
        """
    def __repr__(self) -> str:
        """
        Return the stringified version of the expression.

        Returns:
                str: The string representation.
        """
    def __rfloordiv__(self, arg0: Expression) -> Expression:
        """
        (Floor) Divide self by expr

        Parameters:
                expr: The expression to (floor) divide by

        Returns:
                FunctionExpression: self '//' expr
        """
    def __rmod__(self, arg0: Expression) -> Expression:
        """
        Modulo self by expr

        Parameters:
                expr: The expression to modulo by

        Returns:
                FunctionExpression: self '%' expr
        """
    def __rmul__(self, arg0: Expression) -> Expression:
        """
        Multiply self by expr

        Parameters:
                expr: The expression to multiply by

        Returns:
                FunctionExpression: self '*' expr
        """
    def __ror__(self, arg0: Expression) -> Expression:
        """
        Binary-or self together with expr

        Parameters:
                expr: The expression to OR together with self

        Returns:
                FunctionExpression: expr '|' self
        """
    def __rpow__(self, arg0: Expression) -> Expression:
        """
        Power self by expr

        Parameters:
                expr: The expression to power by

        Returns:
                FunctionExpression: self '**' expr
        """
    def __rsub__(self, arg0: Expression) -> Expression:
        """
        Subtract expr from self

        Parameters:
                expr: The expression to subtract from

        Returns:
                FunctionExpression: self '-' expr
        """
    def __rtruediv__(self, arg0: Expression) -> Expression:
        """
        Divide self by expr

        Parameters:
                expr: The expression to divide by

        Returns:
                FunctionExpression: self '/' expr
        """
    def __sub__(self, arg0: Expression) -> Expression:
        """
        Subtract expr from self

        Parameters:
                expr: The expression to subtract from

        Returns:
                FunctionExpression: self '-' expr
        """
    def __truediv__(self, arg0: Expression) -> Expression:
        """
        Divide self by expr

        Parameters:
                expr: The expression to divide by

        Returns:
                FunctionExpression: self '/' expr
        """
    def alias(self, arg0: str) -> Expression:
        """
        Create a copy of this expression with the given alias.

        Parameters:
                name: The alias to use for the expression, this will affect how it can be referenced.

        Returns:
                Expression: self with an alias.
        """
    def asc(self) -> Expression:
        """
        Set the order by modifier to ASCENDING.
        """
    def between(self, lower: Expression, upper: Expression) -> Expression: ...
    def cast(self, type: duckdb_typing.DuckDBPyType) -> Expression:
        """
        Create a CastExpression to type from self

        Parameters:
                type: The type to cast to

        Returns:
                CastExpression: self::type
        """
    def collate(self, collation: str) -> Expression: ...
    def desc(self) -> Expression:
        """
        Set the order by modifier to DESCENDING.
        """
    def get_name(self) -> str:
        """
        Return the stringified version of the expression.

        Returns:
                str: The string representation.
        """
    def isin(self, *args) -> Expression:
        """
        Return an IN expression comparing self to the input arguments.

        Returns:
                DuckDBPyExpression: The compare IN expression
        """
    def isnotin(self, *args) -> Expression:
        """
        Return a NOT IN expression comparing self to the input arguments.

        Returns:
                DuckDBPyExpression: The compare NOT IN expression
        """
    def isnotnull(self) -> Expression:
        """
        Create a binary IS NOT NULL expression from self

        Returns:
                DuckDBPyExpression: self IS NOT NULL
        """
    def isnull(self) -> Expression:
        """
        Create a binary IS NULL expression from self

        Returns:
                DuckDBPyExpression: self IS NULL
        """
    def nulls_first(self) -> Expression:
        """
        Set the NULL order by modifier to NULLS FIRST.
        """
    def nulls_last(self) -> Expression:
        """
        Set the NULL order by modifier to NULLS LAST.
        """
    def otherwise(self, value: Expression) -> Expression:
        """
        Add an ELSE <value> clause to the CaseExpression.

        Parameters:
                value: The value to use if none of the WHEN conditions are met.

        Returns:
                CaseExpression: self with an ELSE clause.
        """
    def show(self) -> None:
        """
        Print the stringified version of the expression.
        """
    def when(self, condition: Expression, value: Expression) -> Expression:
        """
        Add an additional WHEN <condition> THEN <value> clause to the CaseExpression.

        Parameters:
                condition: The condition that must be met.
                value: The value to use if the condition is met.

        Returns:
                CaseExpression: self with an additional WHEN clause.
        """

class FatalException(DatabaseError):
    pass

class HTTPException(IOException):
    """
    Thrown when an error occurs in the httpfs extension, or whilst downloading an extension.
    """

class IOException(OperationalError):
    pass

class IntegrityError(DatabaseError):
    pass

class InternalError(DatabaseError):
    pass

class InternalException(InternalError):
    pass

class InterruptException(DatabaseError):
    pass

class InvalidInputException(ProgrammingError):
    pass

class InvalidTypeException(ProgrammingError):
    pass

class NotImplementedException(NotSupportedError):
    pass

class NotSupportedError(DatabaseError):
    pass

class OperationalError(DatabaseError):
    pass

class OutOfMemoryException(OperationalError):
    pass

class OutOfRangeException(DataError):
    pass

class ParserException(ProgrammingError):
    pass

class PermissionException(DatabaseError):
    pass

class ProgrammingError(DatabaseError):
    pass

class PythonExceptionHandling:
    """
    Members:

      DEFAULT

      RETURN_NULL
    """

    DEFAULT: typing.ClassVar[PythonExceptionHandling]  # value = <PythonExceptionHandling.DEFAULT: 0>
    RETURN_NULL: typing.ClassVar[PythonExceptionHandling]  # value = <PythonExceptionHandling.RETURN_NULL: 1>
    __members__: typing.ClassVar[
        dict[str, PythonExceptionHandling]
    ]  # value = {'DEFAULT': <PythonExceptionHandling.DEFAULT: 0>, 'RETURN_NULL': <PythonExceptionHandling.RETURN_NULL: 1>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class RenderMode:
    """
    Members:

      ROWS

      COLUMNS
    """

    COLUMNS: typing.ClassVar[RenderMode]  # value = <RenderMode.COLUMNS: 1>
    ROWS: typing.ClassVar[RenderMode]  # value = <RenderMode.ROWS: 0>
    __members__: typing.ClassVar[
        dict[str, RenderMode]
    ]  # value = {'ROWS': <RenderMode.ROWS: 0>, 'COLUMNS': <RenderMode.COLUMNS: 1>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class SequenceException(DatabaseError):
    pass

class SerializationException(OperationalError):
    pass

class Statement:
    @property
    def expected_result_type(self) -> list:
        """
        Get the expected type of result produced by this statement, actual type may vary depending on the statement.
        """
    @property
    def named_parameters(self) -> set:
        """
        Get the map of named parameters this statement has.
        """
    @property
    def query(self) -> str:
        """
        Get the query equivalent to this statement.
        """
    @property
    def type(self) -> StatementType:
        """
        Get the type of the statement.
        """

class StatementType:
    """
    Members:

      INVALID

      SELECT

      INSERT

      UPDATE

      CREATE

      DELETE

      PREPARE

      EXECUTE

      ALTER

      TRANSACTION

      COPY

      ANALYZE

      VARIABLE_SET

      CREATE_FUNC

      EXPLAIN

      DROP

      EXPORT

      PRAGMA

      VACUUM

      CALL

      SET

      LOAD

      RELATION

      EXTENSION

      LOGICAL_PLAN

      ATTACH

      DETACH

      MULTI

      COPY_DATABASE

      MERGE_INTO
    """

    ALTER: typing.ClassVar[StatementType]  # value = <StatementType.ALTER: 8>
    ANALYZE: typing.ClassVar[StatementType]  # value = <StatementType.ANALYZE: 11>
    ATTACH: typing.ClassVar[StatementType]  # value = <StatementType.ATTACH: 25>
    CALL: typing.ClassVar[StatementType]  # value = <StatementType.CALL: 19>
    COPY: typing.ClassVar[StatementType]  # value = <StatementType.COPY: 10>
    COPY_DATABASE: typing.ClassVar[StatementType]  # value = <StatementType.COPY_DATABASE: 28>
    CREATE: typing.ClassVar[StatementType]  # value = <StatementType.CREATE: 4>
    CREATE_FUNC: typing.ClassVar[StatementType]  # value = <StatementType.CREATE_FUNC: 13>
    DELETE: typing.ClassVar[StatementType]  # value = <StatementType.DELETE: 5>
    DETACH: typing.ClassVar[StatementType]  # value = <StatementType.DETACH: 26>
    DROP: typing.ClassVar[StatementType]  # value = <StatementType.DROP: 15>
    EXECUTE: typing.ClassVar[StatementType]  # value = <StatementType.EXECUTE: 7>
    EXPLAIN: typing.ClassVar[StatementType]  # value = <StatementType.EXPLAIN: 14>
    EXPORT: typing.ClassVar[StatementType]  # value = <StatementType.EXPORT: 16>
    EXTENSION: typing.ClassVar[StatementType]  # value = <StatementType.EXTENSION: 23>
    INSERT: typing.ClassVar[StatementType]  # value = <StatementType.INSERT: 2>
    INVALID: typing.ClassVar[StatementType]  # value = <StatementType.INVALID: 0>
    LOAD: typing.ClassVar[StatementType]  # value = <StatementType.LOAD: 21>
    LOGICAL_PLAN: typing.ClassVar[StatementType]  # value = <StatementType.LOGICAL_PLAN: 24>
    MERGE_INTO: typing.ClassVar[StatementType]  # value = <StatementType.MERGE_INTO: 30>
    MULTI: typing.ClassVar[StatementType]  # value = <StatementType.MULTI: 27>
    PRAGMA: typing.ClassVar[StatementType]  # value = <StatementType.PRAGMA: 17>
    PREPARE: typing.ClassVar[StatementType]  # value = <StatementType.PREPARE: 6>
    RELATION: typing.ClassVar[StatementType]  # value = <StatementType.RELATION: 22>
    SELECT: typing.ClassVar[StatementType]  # value = <StatementType.SELECT: 1>
    SET: typing.ClassVar[StatementType]  # value = <StatementType.SET: 20>
    TRANSACTION: typing.ClassVar[StatementType]  # value = <StatementType.TRANSACTION: 9>
    UPDATE: typing.ClassVar[StatementType]  # value = <StatementType.UPDATE: 3>
    VACUUM: typing.ClassVar[StatementType]  # value = <StatementType.VACUUM: 18>
    VARIABLE_SET: typing.ClassVar[StatementType]  # value = <StatementType.VARIABLE_SET: 12>
    __members__: typing.ClassVar[
        dict[str, StatementType]
    ]  # value = {'INVALID': <StatementType.INVALID: 0>, 'SELECT': <StatementType.SELECT: 1>, 'INSERT': <StatementType.INSERT: 2>, 'UPDATE': <StatementType.UPDATE: 3>, 'CREATE': <StatementType.CREATE: 4>, 'DELETE': <StatementType.DELETE: 5>, 'PREPARE': <StatementType.PREPARE: 6>, 'EXECUTE': <StatementType.EXECUTE: 7>, 'ALTER': <StatementType.ALTER: 8>, 'TRANSACTION': <StatementType.TRANSACTION: 9>, 'COPY': <StatementType.COPY: 10>, 'ANALYZE': <StatementType.ANALYZE: 11>, 'VARIABLE_SET': <StatementType.VARIABLE_SET: 12>, 'CREATE_FUNC': <StatementType.CREATE_FUNC: 13>, 'EXPLAIN': <StatementType.EXPLAIN: 14>, 'DROP': <StatementType.DROP: 15>, 'EXPORT': <StatementType.EXPORT: 16>, 'PRAGMA': <StatementType.PRAGMA: 17>, 'VACUUM': <StatementType.VACUUM: 18>, 'CALL': <StatementType.CALL: 19>, 'SET': <StatementType.SET: 20>, 'LOAD': <StatementType.LOAD: 21>, 'RELATION': <StatementType.RELATION: 22>, 'EXTENSION': <StatementType.EXTENSION: 23>, 'LOGICAL_PLAN': <StatementType.LOGICAL_PLAN: 24>, 'ATTACH': <StatementType.ATTACH: 25>, 'DETACH': <StatementType.DETACH: 26>, 'MULTI': <StatementType.MULTI: 27>, 'COPY_DATABASE': <StatementType.COPY_DATABASE: 28>, 'MERGE_INTO': <StatementType.MERGE_INTO: 30>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class SyntaxException(ProgrammingError):
    pass

class TransactionException(OperationalError):
    pass

class TypeMismatchException(DataError):
    pass

class Warning(Exception):
    pass

class token_type:
    """
    Members:

      identifier

      numeric_const

      string_const

      operator

      keyword

      comment
    """

    __members__: typing.ClassVar[
        dict[str, token_type]
    ]  # value = {'identifier': <token_type.identifier: 0>, 'numeric_const': <token_type.numeric_const: 1>, 'string_const': <token_type.string_const: 2>, 'operator': <token_type.operator: 3>, 'keyword': <token_type.keyword: 4>, 'comment': <token_type.comment: 5>}
    comment: typing.ClassVar[token_type]  # value = <token_type.comment: 5>
    identifier: typing.ClassVar[token_type]  # value = <token_type.identifier: 0>
    keyword: typing.ClassVar[token_type]  # value = <token_type.keyword: 4>
    numeric_const: typing.ClassVar[token_type]  # value = <token_type.numeric_const: 1>
    operator: typing.ClassVar[token_type]  # value = <token_type.operator: 3>
    string_const: typing.ClassVar[token_type]  # value = <token_type.string_const: 2>
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

def CaseExpression(condition: Expression, value: Expression) -> Expression: ...
def CoalesceOperator(*args) -> Expression: ...
def ColumnExpression(*args) -> Expression:
    """
    Create a column reference from the provided column name
    """

def ConstantExpression(value: typing.Any) -> Expression:
    """
    Create a constant expression from the provided value
    """

def DefaultExpression() -> Expression: ...
def FunctionExpression(function_name: str, *args) -> Expression: ...
def LambdaExpression(lhs: typing.Any, rhs: Expression) -> Expression: ...
def SQLExpression(expression: str) -> Expression: ...
@typing.overload
def StarExpression(*, exclude: typing.Any = None) -> Expression: ...
@typing.overload
def StarExpression() -> Expression: ...
def aggregate(
    df: pandas.DataFrame,
    aggr_expr: typing.Any,
    group_expr: str = "",
    *,
    connection: typing.Optional[duckdb.DuckDBPyConnection] = None,
) -> DuckDBPyRelation:
    """
    Compute the aggregate aggr_expr by the optional groups group_expr on the relation
    """

def alias(
    df: pandas.DataFrame, alias: str, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> DuckDBPyRelation:
    """
    Rename the relation object to new alias
    """

def append(
    table_name: str,
    df: pandas.DataFrame,
    *,
    by_name: bool = False,
    connection: typing.Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyConnection:
    """
    Append the passed DataFrame to the named table
    """

def array_type(
    type: duckdb_typing.DuckDBPyType,
    size: typing.SupportsInt,
    *,
    connection: typing.Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb_typing.DuckDBPyType:
    """
    Create an array type object of 'type'
    """

@typing.overload
def arrow(
    rows_per_batch: typing.SupportsInt = 1000000, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> pyarrow.lib.RecordBatchReader:
    """
    Fetch an Arrow RecordBatchReader following execute()
    """

@typing.overload
def arrow(
    rows_per_batch: typing.SupportsInt = 1000000, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> pyarrow.lib.Table:
    """
    Fetch a result as Arrow table following execute()
    """

@typing.overload
def arrow(
    arrow_object: typing.Any, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> DuckDBPyRelation:
    """
    Create a relation object from an Arrow object
    """

def begin(*, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyConnection:
    """
    Start a new transaction
    """

def checkpoint(*, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyConnection:
    """
    Synchronizes data in the write-ahead log (WAL) to the database data file (no-op for in-memory connections)
    """

def close(*, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> None:
    """
    Close the connection
    """

def commit(*, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyConnection:
    """
    Commit changes performed within a transaction
    """

def connect(
    database: typing.Any = ":memory:", read_only: bool = False, config: typing.Optional[dict] = None
) -> duckdb.DuckDBPyConnection:
    """
    Create a DuckDB database instance. Can take a database file name to read/write persistent data and a read_only flag if no changes are desired
    """

def create_function(
    name: str,
    function: collections.abc.Callable,
    parameters: typing.Any = None,
    return_type: typing.Optional[duckdb_typing.DuckDBPyType] = None,
    *,
    type: functional.PythonUDFType = functional.PythonUDFType.NATIVE,
    null_handling: functional.FunctionNullHandling = functional.FunctionNullHandling.DEFAULT,
    exception_handling: PythonExceptionHandling = PythonExceptionHandling.DEFAULT,
    side_effects: bool = False,
    connection: typing.Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyConnection:
    """
    Create a DuckDB function out of the passing in Python function so it can be used in queries
    """

def cursor(*, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyConnection:
    """
    Create a duplicate of the current connection
    """

def decimal_type(
    width: typing.SupportsInt,
    scale: typing.SupportsInt,
    *,
    connection: typing.Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb_typing.DuckDBPyType:
    """
    Create a decimal type with 'width' and 'scale'
    """

def default_connection() -> duckdb.DuckDBPyConnection:
    """
    Retrieve the connection currently registered as the default to be used by the module
    """

def description(*, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> list | None:
    """
    Get result set attributes, mainly column names
    """

@typing.overload
def df(
    *, date_as_object: bool = False, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> pandas.DataFrame:
    """
    Fetch a result as DataFrame following execute()
    """

@typing.overload
def df(
    *, date_as_object: bool = False, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> pandas.DataFrame:
    """
    Fetch a result as DataFrame following execute()
    """

@typing.overload
def df(df: pandas.DataFrame, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> DuckDBPyRelation:
    """
    Create a relation object from the DataFrame df
    """

def distinct(
    df: pandas.DataFrame, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> DuckDBPyRelation:
    """
    Retrieve distinct rows from this relation object
    """

def dtype(
    type_str: str, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> duckdb_typing.DuckDBPyType:
    """
    Create a type object by parsing the 'type_str' string
    """

def duplicate(*, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyConnection:
    """
    Create a duplicate of the current connection
    """

def enum_type(
    name: str,
    type: duckdb_typing.DuckDBPyType,
    values: list,
    *,
    connection: typing.Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb_typing.DuckDBPyType:
    """
    Create an enum type of underlying 'type', consisting of the list of 'values'
    """

def execute(
    query: typing.Any, parameters: typing.Any = None, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> duckdb.DuckDBPyConnection:
    """
    Execute the given SQL query, optionally using prepared statements with parameters set
    """

def executemany(
    query: typing.Any, parameters: typing.Any = None, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> duckdb.DuckDBPyConnection:
    """
    Execute the given prepared statement multiple times using the list of parameter sets in parameters
    """

def extract_statements(query: str, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> list:
    """
    Parse the query string and extract the Statement object(s) produced
    """

def fetch_arrow_table(
    rows_per_batch: typing.SupportsInt = 1000000, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> pyarrow.lib.Table:
    """
    Fetch a result as Arrow table following execute()
    """

def fetch_df(
    *, date_as_object: bool = False, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> pandas.DataFrame:
    """
    Fetch a result as DataFrame following execute()
    """

def fetch_df_chunk(
    vectors_per_chunk: typing.SupportsInt = 1,
    *,
    date_as_object: bool = False,
    connection: typing.Optional[duckdb.DuckDBPyConnection] = None,
) -> pandas.DataFrame:
    """
    Fetch a chunk of the result as DataFrame following execute()
    """

def fetch_record_batch(
    rows_per_batch: typing.SupportsInt = 1000000, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> pyarrow.lib.RecordBatchReader:
    """
    Fetch an Arrow RecordBatchReader following execute()
    """

def fetchall(*, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> list:
    """
    Fetch all rows from a result following execute
    """

def fetchdf(
    *, date_as_object: bool = False, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> pandas.DataFrame:
    """
    Fetch a result as DataFrame following execute()
    """

def fetchmany(size: typing.SupportsInt = 1, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> list:
    """
    Fetch the next set of rows from a result following execute
    """

def fetchnumpy(*, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> dict:
    """
    Fetch a result as list of NumPy arrays following execute
    """

def fetchone(*, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> tuple | None:
    """
    Fetch a single row from a result following execute
    """

def filesystem_is_registered(name: str, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> bool:
    """
    Check if a filesystem with the provided name is currently registered
    """

def filter(
    df: pandas.DataFrame, filter_expr: typing.Any, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> DuckDBPyRelation:
    """
    Filter the relation object by the filter in filter_expr
    """

def from_arrow(
    arrow_object: typing.Any, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> DuckDBPyRelation:
    """
    Create a relation object from an Arrow object
    """

def from_csv_auto(path_or_buffer: typing.Any, **kwargs) -> DuckDBPyRelation:
    """
    Create a relation object from the CSV file in 'name'
    """

def from_df(df: pandas.DataFrame, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> DuckDBPyRelation:
    """
    Create a relation object from the DataFrame in df
    """

@typing.overload
def from_parquet(
    file_glob: str,
    binary_as_string: bool = False,
    *,
    file_row_number: bool = False,
    filename: bool = False,
    hive_partitioning: bool = False,
    union_by_name: bool = False,
    compression: typing.Any = None,
    connection: typing.Optional[duckdb.DuckDBPyConnection] = None,
) -> DuckDBPyRelation:
    """
    Create a relation object from the Parquet files in file_glob
    """

@typing.overload
def from_parquet(
    file_globs: collections.abc.Sequence[str],
    binary_as_string: bool = False,
    *,
    file_row_number: bool = False,
    filename: bool = False,
    hive_partitioning: bool = False,
    union_by_name: bool = False,
    compression: typing.Any = None,
    connection: typing.Optional[duckdb.DuckDBPyConnection] = None,
) -> DuckDBPyRelation:
    """
    Create a relation object from the Parquet files in file_globs
    """

def from_query(
    query: typing.Any,
    *,
    alias: str = "",
    params: typing.Any = None,
    connection: typing.Optional[duckdb.DuckDBPyConnection] = None,
) -> DuckDBPyRelation:
    """
    Run a SQL query. If it is a SELECT statement, create a relation object from the given SQL query, otherwise run the query as-is.
    """

def get_table_names(
    query: str, *, qualified: bool = False, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> set[str]:
    """
    Extract the required table names from a query
    """

def install_extension(
    extension: str,
    *,
    force_install: bool = False,
    repository: typing.Any = None,
    repository_url: typing.Any = None,
    version: typing.Any = None,
    connection: typing.Optional[duckdb.DuckDBPyConnection] = None,
) -> None:
    """
    Install an extension by name, with an optional version and/or repository to get the extension from
    """

def interrupt(*, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> None:
    """
    Interrupt pending operations
    """

def limit(
    df: pandas.DataFrame,
    n: typing.SupportsInt,
    offset: typing.SupportsInt = 0,
    *,
    connection: typing.Optional[duckdb.DuckDBPyConnection] = None,
) -> DuckDBPyRelation:
    """
    Only retrieve the first n rows from this relation object, starting at offset
    """

def list_filesystems(*, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> list:
    """
    List registered filesystems, including builtin ones
    """

def list_type(
    type: duckdb_typing.DuckDBPyType, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> duckdb_typing.DuckDBPyType:
    """
    Create a list type object of 'type'
    """

def load_extension(extension: str, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> None:
    """
    Load an installed extension
    """

def map_type(
    key: duckdb_typing.DuckDBPyType,
    value: duckdb_typing.DuckDBPyType,
    *,
    connection: typing.Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb_typing.DuckDBPyType:
    """
    Create a map type object from 'key_type' and 'value_type'
    """

def order(
    df: pandas.DataFrame, order_expr: str, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> DuckDBPyRelation:
    """
    Reorder the relation object by order_expr
    """

def pl(
    rows_per_batch: typing.SupportsInt = 1000000,
    *,
    lazy: bool = False,
    connection: typing.Optional[duckdb.DuckDBPyConnection] = None,
) -> polars.DataFrame:
    """
    Fetch a result as Polars DataFrame following execute()
    """

def project(
    df: pandas.DataFrame, *args, groups: str = "", connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> DuckDBPyRelation:
    """
    Project the relation object by the projection in project_expr
    """

def query(
    query: typing.Any,
    *,
    alias: str = "",
    params: typing.Any = None,
    connection: typing.Optional[duckdb.DuckDBPyConnection] = None,
) -> DuckDBPyRelation:
    """
    Run a SQL query. If it is a SELECT statement, create a relation object from the given SQL query, otherwise run the query as-is.
    """

def query_df(
    df: pandas.DataFrame,
    virtual_table_name: str,
    sql_query: str,
    *,
    connection: typing.Optional[duckdb.DuckDBPyConnection] = None,
) -> DuckDBPyRelation:
    """
    Run the given SQL query in sql_query on the view named virtual_table_name that refers to the relation object
    """

def query_progress(*, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> float:
    """
    Query progress of pending operation
    """

def read_csv(path_or_buffer: typing.Any, **kwargs) -> DuckDBPyRelation:
    """
    Create a relation object from the CSV file in 'name'
    """

def read_json(
    path_or_buffer: typing.Any,
    *,
    columns: typing.Optional[typing.Any | None] = None,
    sample_size: typing.Optional[typing.Any | None] = None,
    maximum_depth: typing.Optional[typing.Any | None] = None,
    records: typing.Optional[str | None] = None,
    format: typing.Optional[str | None] = None,
    date_format: typing.Optional[typing.Any | None] = None,
    timestamp_format: typing.Optional[typing.Any | None] = None,
    compression: typing.Optional[typing.Any | None] = None,
    maximum_object_size: typing.Optional[typing.Any | None] = None,
    ignore_errors: typing.Optional[typing.Any | None] = None,
    convert_strings_to_integers: typing.Optional[typing.Any | None] = None,
    field_appearance_threshold: typing.Optional[typing.Any | None] = None,
    map_inference_threshold: typing.Optional[typing.Any | None] = None,
    maximum_sample_files: typing.Optional[typing.Any | None] = None,
    filename: typing.Optional[typing.Any | None] = None,
    hive_partitioning: typing.Optional[typing.Any | None] = None,
    union_by_name: typing.Optional[typing.Any | None] = None,
    hive_types: typing.Optional[typing.Any | None] = None,
    hive_types_autocast: typing.Optional[typing.Any | None] = None,
    connection: typing.Optional[duckdb.DuckDBPyConnection] = None,
) -> DuckDBPyRelation:
    """
    Create a relation object from the JSON file in 'name'
    """

@typing.overload
def read_parquet(
    file_glob: str,
    binary_as_string: bool = False,
    *,
    file_row_number: bool = False,
    filename: bool = False,
    hive_partitioning: bool = False,
    union_by_name: bool = False,
    compression: typing.Any = None,
    connection: typing.Optional[duckdb.DuckDBPyConnection] = None,
) -> DuckDBPyRelation:
    """
    Create a relation object from the Parquet files in file_glob
    """

@typing.overload
def read_parquet(
    file_globs: collections.abc.Sequence[str],
    binary_as_string: bool = False,
    *,
    file_row_number: bool = False,
    filename: bool = False,
    hive_partitioning: bool = False,
    union_by_name: bool = False,
    compression: typing.Any = None,
    connection: typing.Optional[duckdb.DuckDBPyConnection] = None,
) -> DuckDBPyRelation:
    """
    Create a relation object from the Parquet files in file_globs
    """

def register(
    view_name: str, python_object: typing.Any, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> duckdb.DuckDBPyConnection:
    """
    Register the passed Python Object value for querying with a view
    """

def register_filesystem(
    filesystem: fsspec.AbstractFileSystem, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> None:
    """
    Register a fsspec compliant filesystem
    """

def remove_function(
    name: str, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> duckdb.DuckDBPyConnection:
    """
    Remove a previously created function
    """

def rollback(*, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyConnection:
    """
    Roll back changes performed within a transaction
    """

def row_type(
    fields: typing.Any, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> duckdb_typing.DuckDBPyType:
    """
    Create a struct type object from 'fields'
    """

def rowcount(*, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> int:
    """
    Get result set row count
    """

def set_default_connection(connection: duckdb.DuckDBPyConnection) -> None:
    """
    Register the provided connection as the default to be used by the module
    """

def sql(
    query: typing.Any,
    *,
    alias: str = "",
    params: typing.Any = None,
    connection: typing.Optional[duckdb.DuckDBPyConnection] = None,
) -> DuckDBPyRelation:
    """
    Run a SQL query. If it is a SELECT statement, create a relation object from the given SQL query, otherwise run the query as-is.
    """

def sqltype(
    type_str: str, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> duckdb_typing.DuckDBPyType:
    """
    Create a type object by parsing the 'type_str' string
    """

def string_type(
    collation: str = "", *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> duckdb_typing.DuckDBPyType:
    """
    Create a string type with an optional collation
    """

def struct_type(
    fields: typing.Any, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> duckdb_typing.DuckDBPyType:
    """
    Create a struct type object from 'fields'
    """

def table(table_name: str, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> DuckDBPyRelation:
    """
    Create a relation object for the named table
    """

def table_function(
    name: str, parameters: typing.Any = None, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> DuckDBPyRelation:
    """
    Create a relation object from the named table function with given parameters
    """

def tf(*, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> dict:
    """
    Fetch a result as dict of TensorFlow Tensors following execute()
    """

def tokenize(query: str) -> list:
    """
    Tokenizes a SQL string, returning a list of (position, type) tuples that can be used for e.g., syntax highlighting
    """

def torch(*, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> dict:
    """
    Fetch a result as dict of PyTorch Tensors following execute()
    """

def type(type_str: str, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> duckdb_typing.DuckDBPyType:
    """
    Create a type object by parsing the 'type_str' string
    """

def union_type(
    members: typing.Any, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> duckdb_typing.DuckDBPyType:
    """
    Create a union type object from 'members'
    """

def unregister(
    view_name: str, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None
) -> duckdb.DuckDBPyConnection:
    """
    Unregister the view name
    """

def unregister_filesystem(name: str, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> None:
    """
    Unregister a filesystem
    """

def values(*args, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> DuckDBPyRelation:
    """
    Create a relation object from the passed values
    """

def view(view_name: str, *, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> DuckDBPyRelation:
    """
    Create a relation object for the named view
    """

def write_csv(
    df: pandas.DataFrame,
    filename: str,
    *,
    sep: typing.Any = None,
    na_rep: typing.Any = None,
    header: typing.Any = None,
    quotechar: typing.Any = None,
    escapechar: typing.Any = None,
    date_format: typing.Any = None,
    timestamp_format: typing.Any = None,
    quoting: typing.Any = None,
    encoding: typing.Any = None,
    compression: typing.Any = None,
    overwrite: typing.Any = None,
    per_thread_output: typing.Any = None,
    use_tmp_file: typing.Any = None,
    partition_by: typing.Any = None,
    write_partition_columns: typing.Any = None,
    connection: typing.Optional[duckdb.DuckDBPyConnection] = None,
) -> None:
    """
    Write the relation object to a CSV file in 'file_name'
    """

ALTER: StatementType  # value = <StatementType.ALTER: 8>
ANALYZE: StatementType  # value = <StatementType.ANALYZE: 11>
ATTACH: StatementType  # value = <StatementType.ATTACH: 25>
CALL: StatementType  # value = <StatementType.CALL: 19>
CARRIAGE_RETURN_LINE_FEED: CSVLineTerminator  # value = <CSVLineTerminator.CARRIAGE_RETURN_LINE_FEED: 1>
CHANGED_ROWS: ExpectedResultType  # value = <ExpectedResultType.CHANGED_ROWS: 1>
COLUMNS: RenderMode  # value = <RenderMode.COLUMNS: 1>
COPY: StatementType  # value = <StatementType.COPY: 10>
COPY_DATABASE: StatementType  # value = <StatementType.COPY_DATABASE: 28>
CREATE: StatementType  # value = <StatementType.CREATE: 4>
CREATE_FUNC: StatementType  # value = <StatementType.CREATE_FUNC: 13>
DEFAULT: PythonExceptionHandling  # value = <PythonExceptionHandling.DEFAULT: 0>
DELETE: StatementType  # value = <StatementType.DELETE: 5>
DETACH: StatementType  # value = <StatementType.DETACH: 26>
DROP: StatementType  # value = <StatementType.DROP: 15>
EXECUTE: StatementType  # value = <StatementType.EXECUTE: 7>
EXPLAIN: StatementType  # value = <StatementType.EXPLAIN: 14>
EXPORT: StatementType  # value = <StatementType.EXPORT: 16>
EXTENSION: StatementType  # value = <StatementType.EXTENSION: 23>
INSERT: StatementType  # value = <StatementType.INSERT: 2>
INVALID: StatementType  # value = <StatementType.INVALID: 0>
LINE_FEED: CSVLineTerminator  # value = <CSVLineTerminator.LINE_FEED: 0>
LOAD: StatementType  # value = <StatementType.LOAD: 21>
LOGICAL_PLAN: StatementType  # value = <StatementType.LOGICAL_PLAN: 24>
MERGE_INTO: StatementType  # value = <StatementType.MERGE_INTO: 30>
MULTI: StatementType  # value = <StatementType.MULTI: 27>
NOTHING: ExpectedResultType  # value = <ExpectedResultType.NOTHING: 2>
PRAGMA: StatementType  # value = <StatementType.PRAGMA: 17>
PREPARE: StatementType  # value = <StatementType.PREPARE: 6>
QUERY_RESULT: ExpectedResultType  # value = <ExpectedResultType.QUERY_RESULT: 0>
RELATION: StatementType  # value = <StatementType.RELATION: 22>
RETURN_NULL: PythonExceptionHandling  # value = <PythonExceptionHandling.RETURN_NULL: 1>
ROWS: RenderMode  # value = <RenderMode.ROWS: 0>
SELECT: StatementType  # value = <StatementType.SELECT: 1>
SET: StatementType  # value = <StatementType.SET: 20>
STANDARD: ExplainType  # value = <ExplainType.STANDARD: 0>
TRANSACTION: StatementType  # value = <StatementType.TRANSACTION: 9>
UPDATE: StatementType  # value = <StatementType.UPDATE: 3>
VACUUM: StatementType  # value = <StatementType.VACUUM: 18>
VARIABLE_SET: StatementType  # value = <StatementType.VARIABLE_SET: 12>
__formatted_python_version__: str = "3.11"
__git_revision__: str = "b8a06e4a22"
__interactive__: bool = False
__jupyter__: bool = False
__standard_vector_size__: int = 2048
__version__: str = "1.4.0"
_clean_default_connection: typing.Any  # value = <capsule object>
apilevel: str = "2.0"
comment: token_type  # value = <token_type.comment: 5>
identifier: token_type  # value = <token_type.identifier: 0>
keyword: token_type  # value = <token_type.keyword: 4>
numeric_const: token_type  # value = <token_type.numeric_const: 1>
operator: token_type  # value = <token_type.operator: 3>
paramstyle: str = "qmark"
string_const: token_type  # value = <token_type.string_const: 2>
threadsafety: int = 1
