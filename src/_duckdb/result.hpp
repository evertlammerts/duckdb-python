//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/result.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nanobind/nanobind.h>

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "chunkview.hpp"
#include "lifetime.hpp"

namespace duckdb_python {

/// Column names paired with the text form of their type.
std::vector<std::pair<std::string, std::string>> FieldTexts(const cxx::Schema &schema);

/// The columns' logical types, which this facade keeps on the schema, not on
/// the vectors.
std::vector<cxx::LogicalType> FieldTypes(const cxx::Schema &schema);

/// One statement's result, streamed a chunk at a time.
///
/// Single reader by construction: the parked chunk, its offset, the end flag
/// and the type cache are unguarded, written by the fetching thread alone,
/// and stepping releases the GIL, so two threads fetching from one Result
/// race even on a GIL build. A close from another thread is the one
/// cross-thread call that is safe: it releases through the guard and touches
/// no fetch state.
class Result {
public:
	Result(nb::object database, std::shared_ptr<ModuleState> module, cxx::QueryResult result);

	nb::handle Parent() const {
		return result.Parent();
	}

	/// Column names paired with the text form of their type.
	std::vector<std::pair<std::string, std::string>> Schema();

	/// Whether this result carries rows, a changed-row count, or nothing.
	std::string ResultType();

	/// The kind of SQL statement this result came from, as lowercase text.
	std::string StatementTypeName();

	/// Run the statement to completion and report how many rows it changed.
	///
	/// Side effects land when a result is drained, so a statement whose result
	/// is dropped without draining never takes effect at all, except one
	/// carrying RETURNING, which the engine applies at execute. Stepped here
	/// rather than by the engine's blocking drain, so a Ctrl-C can land midway.
	/// Every chunk is consumed inside the quantum rather than surfaced: this
	/// loop does no Python work, and per-chunk GIL churn starves every other
	/// Python thread, the one delivering the Ctrl-C included. The count of a
	/// changed-rows result travels as a chunk, so it is harvested here.
	cxx::idx_t Drain();

	/// Release the result so the connection can run another query.
	///
	/// The engine allows one live result per connection, so a caller that
	/// stops reading early needs a way to say so. Relying on the Python object
	/// being collected would make the moment of release depend on refcount
	/// timing, which is exactly the wrong thing for an exclusive resource.
	///
	/// A closed result pins nothing: holding the database here kept the file
	/// "in use" past a close that had returned, until garbage collection
	/// happened to run.
	void Close();

	/// Up to `count` more rows, or every remaining row when `count` is zero.
	///
	/// Streaming is the only execution model here, so rows are taken a chunk
	/// at a time and a partly-read chunk is carried across calls. Buffering
	/// the whole result to serve fetchone() would defeat the point.
	nb::list FetchRows(size_t count);

	/// Drain the result into a list of row tuples.
	nb::list FetchAll();

	/// The next chunk column-wise for the numpy converter, or None at the end.
	///
	/// A chunk partly consumed by a prior row fetch is handed out whole with
	/// its row offset, so the converter delivers only the remaining rows.
	nb::object FetchChunkView();

	/// Per-column (type id, decimal scale, enum dictionary or None), from the
	/// schema alone, so an empty result still assembles with the dtypes its
	/// rows would have had.
	std::vector<std::tuple<int, int, std::optional<std::vector<std::string>>>> SchemaTypes();

private:
	enum class Pumped { Chunk, Finished, Cancelled, Quantum };

	Pinned<cxx::QueryResult> Live();

	/// One quantum of stepping with the GIL released: until `sink` keeps a
	/// chunk, the result ends, the query is cancelled, or the time is up.
	template <class SINK>
	static Pumped Pump(cxx::QueryResult &live, SINK &sink);

	/// Pump until `sink` keeps a chunk (true) or the result ends (false),
	/// under the Ctrl-C rules. A cancel is InterruptError. A signal is
	/// checked between quanta and after a kept chunk, which the sink parks
	/// first so a Ctrl-C on a chunk boundary loses no rows: a caller that
	/// catches it finds the chunk still pending and can keep fetching. It is
	/// never checked after the end: the statement completed and its side
	/// effects landed, and a signal pending this very quantum must not turn
	/// that into an exception a caller would read as "did not happen". The
	/// interpreter's own next check delivers it instead. The check is a no-op
	/// off the main thread, where Python runs no signal handlers.
	template <class SINK>
	bool Stream(SINK &&sink);

	/// Step until a chunk arrives, parked in `pending`, or the result ends.
	/// False once it has ended.
	bool Advance();

	/// The columns' logical types, cached: the row path dispatches on them
	/// per chunk, and this facade keeps them on the schema, not the vectors.
	const ColumnTypes &Types();

	Owned<cxx::QueryResult> result;
	std::optional<cxx::DataChunk> pending;
	ColumnTypes types;
	cxx::idx_t offset = 0;
	bool finished = false;
};

} // namespace duckdb_python
