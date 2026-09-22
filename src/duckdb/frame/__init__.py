"""Build a query as a plan, without a connection, and run it by passing one.

from duckdb import frame
from duckdb.frame import col, table

con = frame.connect("shop.db")
table("orders").filter(col("total") > 1000).rows(con)
"""

from .._expressions import (
    Expr,
    Side,
    coalesce,
    col,
    count_all,
    dense_rank,
    first_value,
    fn,
    lag,
    last_value,
    lead,
    lit,
    ntile,
    param,
    rank,
    row_number,
    sql_expr,
    star,
    when,
)
from .connection import Connection, connect
from .plan import (
    Bound,
    Column,
    Frame,
    NeedsConnection,
    Step,
    read_csv,
    read_json,
    read_parquet,
    sql,
    table,
    table_function,
    values,
)

__all__ = [
    "Bound",
    "Column",
    "Connection",
    "Expr",
    "Frame",
    "NeedsConnection",
    "Side",
    "Step",
    "coalesce",
    "col",
    "connect",
    "count_all",
    "dense_rank",
    "first_value",
    "fn",
    "lag",
    "last_value",
    "lead",
    "lit",
    "ntile",
    "param",
    "rank",
    "read_csv",
    "read_json",
    "read_parquet",
    "row_number",
    "sql",
    "sql_expr",
    "star",
    "table",
    "table_function",
    "values",
    "when",
]
