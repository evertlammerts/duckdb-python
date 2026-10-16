//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/result.cpp
//
//
//===----------------------------------------------------------------------===//

#include "result.hpp"

#include "conversion/value_to_python.hpp"

#include <chrono>

namespace duckdb_python {
namespace {

// Stepping does the query's own work, so WAITING means "call again" and sleeping would only serialise it.
constexpr std::chrono::milliseconds kStepQuantum {20};

} // namespace

/// The kind of SQL statement, as lowercase text.
const char *StatementTypeText(cxx::StatementType type) {
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

Result::Result(nb::object database, std::shared_ptr<ModuleState> module, cxx::QueryResult result,
               ConnectionUse::Hold hold)
    : result(std::move(module), std::move(database), OpenQuery {std::move(hold), std::move(result)}) {
}

std::vector<std::pair<std::string, std::string>> Result::Schema() {
	return FieldTexts(Live().engine->query.GetSchema());
}

std::string Result::ResultType() {
	switch (Live().engine->query.GetResultType()) {
	case cxx::QueryResult::ResultType::QUERY_RESULT:
		return "rows";
	case cxx::QueryResult::ResultType::CHANGED_ROWS:
		return "changed_rows";
	default:
		return "nothing";
	}
}

std::string Result::StatementTypeName() {
	return StatementTypeText(Live().engine->query.GetStatementType());
}

cxx::idx_t Result::Drain() {
	pending.reset();
	cxx::idx_t changed = 0;
	Stream([&](cxx::QueryResult &live, cxx::DataChunk chunk) {
		// Asked per batch: a multi-statement result knows its type only once stepping reaches that statement.
		if (chunk.GetRowCount() > 0 && live.GetResultType() == cxx::QueryResult::ResultType::CHANGED_ROWS) {
			changed += static_cast<cxx::idx_t>(chunk.GetVector(0).GetValue(0).Get<int64_t>());
		}
		return false;
	});
	return changed;
}

void Result::Finish(const std::function<void()> &cancel, Ending &ending) {
	pending.reset();
	ending = Ending::FAILED;
	bool reported = false;
	auto sink = [&](cxx::QueryResult &, cxx::DataChunk) {
		reported = true;
		return false;
	};
	std::optional<nb::python_error> signalled;
	while (true) {
		auto live = Live();
		Pumped pumped;
		try {
			pumped = Pump(*live.engine, sink);
		} catch (const cxx::Exception &error) {
			if (reported) {
				ending = Ending::TOOK_EFFECT;
			} else {
				ending = error.GetCode() == DUCKDB_V2_ERROR_RUNTIME_INTERRUPT ? Ending::CANCELLED : Ending::FAILED;
			}
			if (signalled) {
				if (ending == Ending::FAILED) {
					// The held exception is raised, so the statement's own failure goes with it as its cause.
					auto cause = result.Module()->ClassForCode(error.GetCode())(error.what());
					PyException_SetCause(signalled->value().ptr(), cause.release().ptr());
				}
				throw *signalled;
			}
			throw;
		}
		if (pumped == Pumped::Finished || pumped == Pumped::Cancelled) {
			// A result that reported its rows took effect, whatever its last step says.
			const bool took_effect = pumped == Pumped::Finished || reported;
			ending = took_effect ? Ending::TOOK_EFFECT : Ending::CANCELLED;
			finished = pumped == Pumped::Finished;
			if (signalled) {
				throw *signalled;
			}
			if (!took_effect) {
				Raise(result.Module()->InterruptError(), "query was cancelled");
			}
			return;
		}
		if (!signalled && PyErr_CheckSignals() != 0) {
			signalled.emplace();
			try {
				cancel();
			} catch (const cxx::Exception &) {
				// Not cancelled, the statement runs to its end, and how it ended still decides the rows.
			}
		}
	}
}

void Result::Close() {
	result.Release();
}

void Result::Settle() {
	if (pending || finished) {
		return;
	}
	Advance();
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
				// Advanced only on success, so a failed conversion can be retried from the same row.
				offset += want;
			}
			if (offset < available) {
				return rows; // count reached mid-batch
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

Pinned<OpenQuery> Result::Live() {
	return result.Acquire("result is closed");
}

template <class SINK>
Result::Pumped Result::Pump(OpenQuery &live, SINK &sink) {
	const ConnectionUse::Reading reading(live.hold);
	// Step may run part of the query here, and a Python callback inside it would deadlock on a held GIL.
	nb::gil_scoped_release release;
	const auto deadline = std::chrono::steady_clock::now() + kStepQuantum;
	while (std::chrono::steady_clock::now() < deadline) {
		// The engine lets go of the connection once the query ends, whether it finished, failed or was cancelled.
		auto step = [&] {
			try {
				return live.query.Step();
			} catch (...) {
				live.hold.Release();
				throw;
			}
		}();
		switch (step.status) {
		case cxx::QueryResult::StepStatus::CHUNK:
			if (sink(live.query, std::move(step.chunk))) {
				return Pumped::Chunk;
			}
			break;
		case cxx::QueryResult::StepStatus::FINISHED:
			live.hold.Release();
			return Pumped::Finished;
		case cxx::QueryResult::StepStatus::CANCELLED:
			live.hold.Release();
			return Pumped::Cancelled;
		default:
			// WAITING: stepping is itself the progress, so go again.
			break;
		}
	}
	return Pumped::Quantum;
}

template <class SINK>
bool Result::Stream(SINK &&sink) {
	while (true) {
		// Held for the whole burst, before the GIL is dropped, so a concurrent close cannot free the result.
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
		types = std::make_shared<std::vector<cxx::LogicalType>>(FieldTypes(Live().engine->query.GetSchema()));
	}
	return types;
}

} // namespace duckdb_python
