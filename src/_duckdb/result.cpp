//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/result.cpp
//
//
//===----------------------------------------------------------------------===//

#include "result.hpp"

#include <chrono>

namespace duckdb_python {
namespace {

// Step participates in executing the query: WAITING means "call again", never
// "wait for someone else". Sleeping between calls only serializes the work,
// so the step loop runs hot, releasing the GIL for one quantum of stepping at
// a time and retaking it for the Ctrl-C check in between.
constexpr std::chrono::milliseconds kStepQuantum {20};

/// The kind of SQL statement a result came from, as lowercase text.
const char *StatementTypeText(cxx::QueryResult::StatementType type) {
	using Kind = cxx::QueryResult::StatementType;
	static constexpr std::pair<Kind, const char *> kNames[] = {
	    {Kind::SELECT, "select"},
	    {Kind::INSERT, "insert"},
	    {Kind::UPDATE, "update"},
	    {Kind::CREATE, "create"},
	    {Kind::DELETE, "delete"},
	    {Kind::PREPARE, "prepare"},
	    {Kind::EXECUTE, "execute"},
	    {Kind::ALTER, "alter"},
	    {Kind::TRANSACTION, "transaction"},
	    {Kind::COPY, "copy"},
	    {Kind::ANALYZE, "analyze"},
	    {Kind::VARIABLE_SET, "variable_set"},
	    {Kind::CREATE_FUNC, "create_func"},
	    {Kind::EXPLAIN, "explain"},
	    {Kind::DROP, "drop"},
	    {Kind::EXPORT, "export"},
	    {Kind::PRAGMA, "pragma"},
	    {Kind::VACUUM, "vacuum"},
	    {Kind::CALL, "call"},
	    {Kind::SET, "set"},
	    {Kind::LOAD, "load"},
	    {Kind::RELATION, "relation"},
	    {Kind::EXTENSION, "extension"},
	    {Kind::LOGICAL_PLAN, "logical_plan"},
	    {Kind::ATTACH, "attach"},
	    {Kind::DETACH, "detach"},
	    {Kind::MULTI, "multi"},
	    {Kind::COPY_DATABASE, "copy_database"},
	    {Kind::UPDATE_EXTENSIONS, "update_extensions"},
	    {Kind::MERGE_INTO, "merge_into"},
	};
	for (const auto &[kind, name] : kNames) {
		if (kind == type) {
			return name;
		}
	}
	return "unknown";
}

} // namespace

std::vector<std::pair<std::string, std::string>> FieldTexts(const cxx::Schema &schema) {
	std::vector<std::pair<std::string, std::string>> out;
	const auto count = schema.GetFieldCount();
	out.reserve(count);
	for (cxx::idx_t i = 0; i < count; i++) {
		out.emplace_back(std::string(schema.GetFieldName(i)), schema.GetFieldType(i).ToText());
	}
	return out;
}

std::vector<cxx::LogicalType> FieldTypes(const cxx::Schema &schema) {
	std::vector<cxx::LogicalType> out;
	const auto count = schema.GetFieldCount();
	out.reserve(count);
	for (cxx::idx_t i = 0; i < count; i++) {
		out.push_back(schema.GetFieldType(i));
	}
	return out;
}

Result::Result(nb::object database, std::shared_ptr<ModuleState> module, cxx::QueryResult result)
    : result(std::move(module), std::move(database), std::move(result)) {
}

std::vector<std::pair<std::string, std::string>> Result::Schema() {
	return FieldTexts(Live().engine->GetSchema());
}

std::string Result::ResultType() {
	switch (Live().engine->GetResultType()) {
	case cxx::QueryResult::ResultType::QUERY_RESULT:
		return "rows";
	case cxx::QueryResult::ResultType::CHANGED_ROWS:
		return "changed_rows";
	default:
		return "nothing";
	}
}

std::string Result::StatementTypeName() {
	return StatementTypeText(Live().engine->GetStatementType());
}

cxx::idx_t Result::Drain() {
	pending.reset();
	cxx::idx_t changed = 0;
	Stream([&](cxx::QueryResult &live, cxx::DataChunk chunk) {
		// Asked per chunk: a statement group knows its result type only
		// once stepping has prepared the statement that produces it.
		if (chunk.GetRowCount() > 0 && live.GetResultType() == cxx::QueryResult::ResultType::CHANGED_ROWS) {
			changed += static_cast<cxx::idx_t>(chunk.GetVector(0).GetValue(0).Get<int64_t>());
		}
		return false;
	});
	return changed;
}

void Result::Close() {
	result.Release();
}

nb::list Result::FetchRows(size_t count) {
	auto live = Live();
	auto &ctx = result.Module()->conversion;
	const auto column_types = Types();
	nb::list rows;
	while (count == 0 || nb::len(rows) < count) {
		// Anything left over from the previous call comes first.
		if (pending) {
			const auto available = pending->GetRowCount();
			if (offset < available) {
				auto want = available - offset;
				if (count != 0) {
					const auto remaining = count - nb::len(rows);
					if (remaining < want) {
						want = remaining;
					}
				}
				AppendChunkRows(*pending, *column_types, offset, offset + want, ctx, rows);
				// Advanced only on success, so a failed conversion can
				// be retried from the same row.
				offset += want;
			}
			if (offset < available) {
				return rows; // count reached mid-chunk
			}
			pending.reset();
			offset = 0;
		}
		if (finished) {
			return rows;
		}
		if (!Advance()) {
			return rows;
		}
	}
	return rows;
}

nb::list Result::FetchAll() {
	return FetchRows(0);
}

nb::object Result::FetchChunkView() {
	Live();
	if (!pending && (finished || !Advance())) {
		return nb::none();
	}
	auto chunk = std::move(*pending);
	const auto consumed = offset;
	pending.reset();
	offset = 0;
	return nb::cast(ChunkView(std::move(chunk), Types(), consumed));
}

std::vector<std::tuple<int, int, std::optional<std::vector<std::string>>>> Result::SchemaTypes() {
	Live();
	const auto &column_types = *Types();
	std::vector<std::tuple<int, int, std::optional<std::vector<std::string>>>> out;
	out.reserve(column_types.size());
	for (const auto &type : column_types) {
		const auto id = type.GetTypeId();
		int scale = 0;
		std::optional<std::vector<std::string>> dictionary;
		if (id == cxx::LogicalTypeId::DECIMAL) {
			scale = static_cast<int>(type.GetDecimalScale());
		} else if (id == cxx::LogicalTypeId::ENUM) {
			dictionary = EnumValues(type);
		}
		out.emplace_back(static_cast<int>(id), scale, std::move(dictionary));
	}
	return out;
}

Pinned<cxx::QueryResult> Result::Live() {
	return result.Acquire("result is closed");
}

template <class SINK>
Result::Pumped Result::Pump(cxx::QueryResult &live, SINK &sink) {
	// The engine may run this thread's share of the query inside Step;
	// never hold the GIL across it, or a callback into Python deadlocks.
	nb::gil_scoped_release release;
	const auto deadline = std::chrono::steady_clock::now() + kStepQuantum;
	while (std::chrono::steady_clock::now() < deadline) {
		auto step = live.Step();
		switch (step.status) {
		case cxx::QueryResult::StepStatus::CHUNK:
			if (sink(live, std::move(step.chunk))) {
				return Pumped::Chunk;
			}
			break;
		case cxx::QueryResult::StepStatus::FINISHED:
			return Pumped::Finished;
		case cxx::QueryResult::StepStatus::CANCELLED:
			return Pumped::Cancelled;
		default:
			// WAITING: stepping is the progress; go again.
			break;
		}
	}
	return Pumped::Quantum;
}

template <class SINK>
bool Result::Stream(SINK &&sink) {
	while (true) {
		// Pinned before the GIL is dropped and for the whole quantum, so
		// a concurrent Close cannot free the engine result underneath.
		auto live = Live();
		const auto pumped = Pump(*live.engine, sink);
		if (pumped == Pumped::Cancelled) {
			Raise(result.Module()->InterruptError(), "query was cancelled");
		}
		if (pumped == Pumped::Finished) {
			finished = true;
			return false;
		}
		if (PyErr_CheckSignals() != 0) {
			throw nb::python_error();
		}
		if (pumped == Pumped::Chunk) {
			return true;
		}
	}
}

bool Result::Advance() {
	return Stream([&](cxx::QueryResult &, cxx::DataChunk chunk) {
		pending = std::move(chunk);
		offset = 0;
		return true;
	});
}

const ColumnTypes &Result::Types() {
	if (!types) {
		types = std::make_shared<std::vector<cxx::LogicalType>>(FieldTypes(Live().engine->GetSchema()));
	}
	return types;
}

} // namespace duckdb_python
