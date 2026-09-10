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

/// The columns' types, which DuckDB's C++ API keeps on the schema rather than on the column data.
std::vector<cxx::LogicalType> FieldTypes(const cxx::Schema &schema);

/// One statement's result, delivered a batch of rows at a time.
///
/// Only one thread may fetch: the held batch, its offset and the flags are unguarded, and fetching drops the
/// GIL, so two readers race even there. Closing from another thread is the one safe cross-thread call.
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
	/// A statement takes effect only once its result is run out, so dropping one unread does nothing, unless
	/// it has RETURNING, which DuckDB applies at execute. Short slices let a Ctrl-C land partway, and every
	/// batch is consumed inside the slice, since handing them back would starve the thread delivering it.
	cxx::idx_t Drain();

	/// Release the result: DuckDB allows one live result per connection, and holding it keeps the file in use.
	void Close();

	/// Up to `count` more rows, or every remaining row when `count` is zero; a partly read batch carries over.
	nb::list FetchRows(size_t count);

	/// Every remaining row, as a list of tuples.
	nb::list FetchAll();

	/// The next batch of rows column by column, or None at the end; a partly read one comes with its offset.
	nb::object FetchChunkView();

	/// Per column: type id, decimal scale, and ENUM labels, from the schema, so an empty result still types.
	std::vector<std::tuple<int, int, std::optional<std::vector<std::string>>>> SchemaTypes();

private:
	enum class Pumped { Chunk, Finished, Cancelled, Quantum };

	Pinned<cxx::QueryResult> Live();

	/// Advance with the GIL released until `sink` keeps a batch, the result ends or is cancelled, or time is up.
	template <class SINK>
	static Pumped Pump(cxx::QueryResult &live, SINK &sink);

	/// Run until `sink` keeps a batch (true) or the result ends (false); a cancelled query raises.
	///
	/// A pending Ctrl-C is checked between slices and after a batch has been stored, so catching it loses no
	/// rows. It is never checked once the result ends: the statement already took effect, and raising there
	/// would read as "it did not happen". Python delivers the signal at its own next check instead.
	template <class SINK>
	bool Stream(SINK &&sink);

	/// Fetch the next batch into `pending`, or return false once the result has ended.
	bool Advance();

	/// The columns' types, cached because every batch dispatches on them and only the schema carries them.
	const ColumnTypes &Types();

	Owned<cxx::QueryResult> result;
	std::optional<cxx::DataChunk> pending;
	ColumnTypes types;
	cxx::idx_t offset = 0;
	bool finished = false;
};

} // namespace duckdb_python
