#include "duckdb_python/engine/result.hpp"
#include "duckdb_python/engine/connection.hpp"

#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/arrow/nanoarrow/nanoarrow.hpp"
#include "duckdb/common/enums/query_result_state.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/relation.hpp"

namespace duckdb {
namespace engine {

Format::Format(shared_ptr<ResultFormat> engine_format_p) : engine_format(std::move(engine_format_p)) {
}

Format Format::Chunks() {
	return Format(nullptr);
}

Format Format::Arrow(idx_t batch_size) {
	if (batch_size == 0) {
		throw InvalidInputException("The Arrow batch size must be larger than 0");
	}
	return Format(make_shared_ptr<ArrowFormat>(batch_size));
}

void Complete(QueryResult &result, const InterruptCheck &check) {
	if (result.HasError()) {
		result.ThrowError();
	}
	result.Materialize();
	// A result that arrives finished has no context to run tasks on; Poll reports that without touching it.
	auto state = result.Poll();
	while (!IsTerminal(state)) {
		check();
		if (state == QueryResultState::BLOCKED || state == QueryResultState::READY ||
		    state == QueryResultState::NO_TASKS_AVAILABLE) {
			result.WaitForTask();
		}
		state = result.ExecuteTask();
	}
	// FINISHED only means the executor is done: Complete takes the rows and ends the query with its
	// transaction step. Left open, the next statement would roll it back.
	result.Complete();
	if (result.HasError()) {
		result.ThrowError();
	}
}

Result::Result(unique_ptr<QueryResult> submitted, Format format_p, shared_ptr<Relation> relation_p)
    : handle(std::move(submitted)), format(std::move(format_p)), state(State::PENDING), superseded(false),
      relation(std::move(relation_p)) {
	if (!handle) {
		throw InternalException("engine::Result created without a query result");
	}
	if (handle->HasError()) {
		handle->ThrowError();
	}
	names = handle->GetNames();
	types = handle->GetTypes();
	properties = handle->GetStatementProperties();
	client_properties = handle->client_properties;
	if (format.IsArrow()) {
		auto &schema = handle->FormatState<ArrowFormat>().Schema();
		if (duckdb_nanoarrow::ArrowSchemaDeepCopy(&schema, &arrow_schema.arrow_schema) != NANOARROW_OK) {
			throw OutOfMemoryException("Failed to copy the Arrow schema of the query result");
		}
	}
}

Result::~Result() {
	try {
		Close();
	} catch (...) { // NOLINT: a destructor must not throw
	}
}

void Result::ThrowIfSuperseded() const {
	if (superseded) {
		// The engine reports a query ended by a later statement as an interrupt, and so does the client
		throw InterruptException(
		    "The result was cancelled because another statement ran on its connection before it was fully read");
	}
}

void Result::OpenStream() {
	// Such a statement cannot be streamed; its rows are served from the retained handle instead
	if (CompletesBeforeReturning()) {
		return;
	}
	lock_guard<mutex> guard(handle_lock);
	// A concurrent first consumer can have settled the result while this one waited for the lock
	if (state != State::PENDING || !handle) {
		return;
	}
	try {
		if (format.IsArrow()) {
			stream = make_uniq<QueryResultStream<ArrowFormat>>(std::move(handle));
		} else {
			stream = make_uniq<QueryResultStream<ChunkFormat>>(std::move(handle));
		}
	} catch (...) {
		// A stream that fails to open has consumed and ended the query
		state = State::CLOSED;
		throw;
	}
	state = State::STREAMING;
}

void Result::EndStream() {
	// The stream object stays alive: another thread can still be inside a fetch on it, and a drained
	// stream keeps reporting its terminal state
	state = State::DRAINED;
}

template <class FORMAT>
unique_ptr<typename FORMAT::T> Result::FetchStreamUnit(const InterruptCheck &check) {
	auto &typed = static_cast<QueryResultStream<FORMAT> &>(*stream);
	while (true) {
		unique_ptr<typename FORMAT::T> unit;
		auto query_state = typed.TryFetch(unit);
		if (unit) {
			return unit;
		}
		if (query_state == QueryResultState::FINISHED) {
			ThrowIfSuperseded();
			EndStream();
			return nullptr;
		}
		if (query_state == QueryResultState::EXECUTION_ERROR) {
			ThrowIfSuperseded();
			// A thread that polls after another thread drained the stream sees the ended query as an
			// interrupt; it reached the end, not an error
			if (state == State::DRAINED) {
				return nullptr;
			}
			typed.GetErrorObject().Throw();
		}
		check();
		query_state = typed.ExecuteTask();
		if (query_state == QueryResultState::BLOCKED || query_state == QueryResultState::NO_TASKS_AVAILABLE) {
			typed.WaitForTask();
		}
	}
}

void Result::Retain(const InterruptCheck &check) {
	ThrowIfSuperseded();
	if (state != State::PENDING) {
		return;
	}
	Complete(*handle, check);
	state = State::RETAINED;
}

idx_t Result::RowCount(const InterruptCheck &check) {
	Retain(check);
	if (state != State::RETAINED) {
		throw InternalException("engine::Result::RowCount on a result that is not retained");
	}
	return handle->RowCount();
}

unique_ptr<DataChunk> Result::FetchChunk(const InterruptCheck &check) {
	D_ASSERT(!format.IsArrow());
	ThrowIfSuperseded();
	if (state == State::PENDING) {
		OpenStream();
		if (state == State::PENDING) {
			Retain(check);
		}
	}
	switch (state) {
	case State::STREAMING:
		return FetchStreamUnit<ChunkFormat>(check);
	case State::RETAINED: {
		retained_fetched = true;
		auto chunk = handle->FetchRaw();
		if (handle->HasError()) {
			handle->ThrowError();
		}
		return chunk;
	}
	default:
		return nullptr;
	}
}

unique_ptr<ArrowArrayWrapper> Result::FetchArray(const InterruptCheck &check) {
	D_ASSERT(format.IsArrow());
	ThrowIfSuperseded();
	if (state == State::PENDING) {
		OpenStream();
		if (state == State::PENDING) {
			Retain(check);
		}
	}
	switch (state) {
	case State::STREAMING:
		return FetchStreamUnit<ArrowFormat>(check);
	case State::RETAINED:
		retained_fetched = true;
		return handle->Fetch<ArrowFormat>();
	default:
		return nullptr;
	}
}

unique_ptr<ColumnDataCollection> Result::TakeChunks(const InterruptCheck &check) {
	D_ASSERT(!format.IsArrow());
	ThrowIfSuperseded();
	if (state == State::PENDING) {
		Retain(check);
	}
	if (state == State::RETAINED && !retained_fetched) {
		auto collection = handle->TakeCollection<ChunkFormat>();
		state = State::DRAINED;
		return collection;
	}
	auto collection = make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), types);
	if (state == State::RETAINED) {
		// Fetching copies out of the collection, so only the rows after the fetch cursor remain
		while (auto chunk = handle->FetchRaw()) {
			collection->Append(*chunk);
		}
		state = State::DRAINED;
	} else if (state == State::STREAMING) {
		while (auto chunk = FetchStreamUnit<ChunkFormat>(check)) {
			collection->Append(*chunk);
		}
	}
	return collection;
}

vector<unique_ptr<ArrowArrayWrapper>> Result::TakeArrays(const InterruptCheck &check) {
	D_ASSERT(format.IsArrow());
	ThrowIfSuperseded();
	vector<unique_ptr<ArrowArrayWrapper>> arrays;
	if (state == State::PENDING) {
		Retain(check);
	}
	if (state == State::RETAINED && retained_fetched) {
		while (auto array = handle->Fetch<ArrowFormat>()) {
			arrays.push_back(std::move(array));
		}
		state = State::DRAINED;
		return arrays;
	}
	if (state == State::RETAINED) {
		auto collection = handle->TakeCollection<ArrowFormat>();
		state = State::DRAINED;
		arrays.reserve(collection->size());
		for (auto &owner : *collection) {
			arrays.push_back(ArrowFormat::ShareArray(owner));
		}
		return arrays;
	}
	if (state == State::STREAMING) {
		while (auto array = FetchStreamUnit<ArrowFormat>(check)) {
			arrays.push_back(std::move(array));
		}
	}
	return arrays;
}

void Result::CopyArrowSchema(ArrowSchema &out) const {
	if (!format.IsArrow()) {
		throw InternalException("engine::Result::CopyArrowSchema on a result that is not in the Arrow format");
	}
	out.release = nullptr;
	if (duckdb_nanoarrow::ArrowSchemaDeepCopy(&arrow_schema.arrow_schema, &out) != NANOARROW_OK) {
		throw OutOfMemoryException("Failed to copy the Arrow schema of the query result");
	}
}

void Result::BuildArrowSchema(const vector<string> &schema_names, ArrowSchema &out) const {
	auto context = client_properties.client_context;
	if (!context) {
		throw ConnectionException("Cannot build an Arrow schema without a valid connection");
	}
	auto properties = client_properties;
	context->RunFunctionInTransaction([&]() { ArrowConverter::ToArrowSchema(&out, types, schema_names, properties); });
}

shared_ptr<Result> Result::Reformat(const Format &target, const InterruptCheck &check) {
	auto context = client_properties.client_context;
	if (!context) {
		throw ConnectionException("Cannot convert a result whose connection is closed");
	}
	auto session = Session(context->shared_from_this());
	return session.Submit(TakeChunks(check), names, target);
}

string Result::ToBox(BoxRendererContext &context, const BoxRendererConfig &config) {
	if (state != State::RETAINED) {
		throw InternalException("engine::Result::ToBox on a result that is not retained");
	}
	return handle->ToBox(context, config);
}

bool Result::RowsRead() const {
	return retained_fetched || state == State::STREAMING || state == State::DRAINED;
}

int64_t Result::ChangedRows() {
	if (properties.return_type != StatementReturnType::CHANGED_ROWS || state != State::RETAINED) {
		return -1;
	}
	auto &collection = handle->Collection<ChunkFormat>();
	if (collection.Count() == 0) {
		return -1;
	}
	return collection.GetValue(0, 0).GetValue<int64_t>();
}

bool Result::StreamEnded() const {
	if (superseded) {
		return true;
	}
	lock_guard<mutex> guard(handle_lock);
	return stream && !stream->IsOpen();
}

void Result::Close() {
	if (state == State::CLOSED) {
		return;
	}
	{
		lock_guard<mutex> guard(handle_lock);
		if (stream) {
			stream->Close();
		} else if (handle) {
			handle->Close();
		}
	}
	state = State::CLOSED;
}

void Result::Supersede() {
	lock_guard<mutex> guard(handle_lock);
	if (stream && stream->IsOpen()) {
		superseded = true;
		stream->Close();
	} else if (handle && handle->IsOpen()) {
		superseded = true;
		handle->Close();
	}
}

} // namespace engine
} // namespace duckdb
