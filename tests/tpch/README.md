# TPC-H queries

The 22 queries, copied from DuckDB's own `extension/tpch/dbgen/queries`.

They are here as a **coverage probe, not a conformance target**. TPC-H is
external and fixed, so it cannot be shaped to flatter the API, and every query
has a verifiable answer. What it demands tells us which verbs the frame layer
needs.

Its blind spots matter as much as its contents. TPC-H contains **no window
functions at all**, no set operations, no DML and no nested types. Passing all
22 would say nothing about any of those, so this suite is a floor rather than a
definition of done.

Derived from TPC-H. Not comparable to published TPC-H results.
