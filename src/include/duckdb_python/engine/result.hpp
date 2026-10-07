//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/engine/result.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/arrow/arrow_format.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/atomic.hpp"
#include "duckdb/common/box_renderer.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/main/query_result_stream.hpp"

#include <functional>

namespace duckdb {
class Relation;

namespace engine {

//! Runs between engine tasks of a blocking call and throws to abandon it
using InterruptCheck = std::function<void()>;

//! The unit a result is produced in, fixed when the statement is submitted
class Format {
public:
	static Format Chunks();
	//! batch_size is a maximum: the engine also cuts an array at row group and producer boundaries
	static Format Arrow(idx_t batch_size);

public:
	bool IsArrow() const {
		return arrow_batch_size != 0;
	}
	idx_t ArrowBatchSize() const {
		return arrow_batch_size;
	}
	//! Null for chunks, the engine's default format
	const shared_ptr<ResultFormat> &EngineFormat() const {
		return engine_format;
	}

private:
	Format(shared_ptr<ResultFormat> engine_format, idx_t arrow_batch_size);

private:
	shared_ptr<ResultFormat> engine_format;
	idx_t arrow_batch_size;
};

//! A submitted statement. Retention is settled by the first consumer: a fetch opens a stream, Retain and the
//! Take calls keep every row. Every call that drives the query takes the client context lock and may wait for
//! running tasks, so callers must not hold a lock those tasks need.
class Result {
public:
	Result(unique_ptr<QueryResult> submitted, Format format, shared_ptr<Relation> relation);
	~Result();

	Result(const Result &) = delete;
	Result &operator=(const Result &) = delete;

public:
	const vector<Identifier> &Names() const {
		return names;
	}
	const vector<LogicalType> &Types() const {
		return types;
	}
	const StatementProperties &Properties() const {
		return properties;
	}
	const ClientProperties &GetClientProperties() const {
		return client_properties;
	}
	const Format &GetFormat() const {
		return format;
	}
	bool IsRetained() const {
		return state == State::RETAINED;
	}
	//! Whether a stream's query has ended underneath it, also by a statement the seam did not run. Takes the
	//! client context lock.
	bool StreamEnded() const;
	//! The planner settles such a statement before its result is returned. Goes away with planner eagerness.
	bool CompletesBeforeReturning() const {
		return properties.result_eagerness == ResultEagerness::FORCED;
	}

	//! Keeps every row: drives the query to its end unless it already ended
	void Retain(const InterruptCheck &check);
	//! Rows of a retained result. Materializes it first.
	idx_t RowCount(const InterruptCheck &check);
	//! The next chunk, not flattened; null at the end. Opens a stream unless the result is retained.
	unique_ptr<DataChunk> FetchChunk(const InterruptCheck &check);
	//! The next Arrow array; null at the end. Opens a stream unless the result is retained.
	unique_ptr<ArrowArrayWrapper> FetchArray(const InterruptCheck &check);
	//! Every remaining row, the result is empty afterwards
	unique_ptr<ColumnDataCollection> TakeChunks(const InterruptCheck &check);
	//! Every remaining array, the result is empty afterwards
	vector<unique_ptr<ArrowArrayWrapper>> TakeArrays(const InterruptCheck &check);
	//! A copy of the result's Arrow schema, owned by the caller. Arrow results only.
	void CopyArrowSchema(ArrowSchema &out) const;
	//! An Arrow schema for the result's types under the given names, built in a transaction on its connection
	void BuildArrowSchema(const vector<string> &names, ArrowSchema &out) const;
	//! The remaining rows, after the leading ones, scanned again by the engine in another format. The result
	//! is empty afterwards. The scan is a new statement on the result's connection.
	shared_ptr<Result> Reformat(const Format &target, unique_ptr<DataChunk> leading, const InterruptCheck &check);
	//! Renders the rows of a retained result
	string ToBox(BoxRendererContext &context, const BoxRendererConfig &config);

	//! Ends the query if it is still running. Idempotent, and safe while another thread is fetching.
	void Close();
	//! Closes the result because another statement is about to run on its connection
	void Supersede();

private:
	enum class State : uint8_t { PENDING, STREAMING, RETAINED, DRAINED, CLOSED };

	void ThrowIfSuperseded() const;
	void OpenStream();
	template <class FORMAT>
	unique_ptr<typename FORMAT::T> FetchStreamUnit(const InterruptCheck &check);
	void EndStream();

private:
	unique_ptr<QueryResult> handle;
	unique_ptr<ResultStreamBase> stream;
	Format format;
	atomic<State> state;
	bool retained_fetched = false;
	atomic<bool> superseded;
	//! Guards handle and stream against Supersede from another thread; never held while a task runs
	mutable mutex handle_lock;
	//! Keeps the relation's external dependencies alive while its query runs
	shared_ptr<Relation> relation;
	vector<Identifier> names;
	vector<LogicalType> types;
	StatementProperties properties;
	ClientProperties client_properties;
	//! Copied at submission, while the producing transaction is live
	ArrowSchemaWrapper arrow_schema;
};

//! Runs a result to its end and keeps every row, as a blocking call
void Complete(QueryResult &result, const InterruptCheck &check);

} // namespace engine
} // namespace duckdb
