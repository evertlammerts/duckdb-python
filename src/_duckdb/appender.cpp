//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/appender.cpp
//
//
//===----------------------------------------------------------------------===//

#include "appender.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <utility>

#include "conversion/conversion.hpp"
#include "conversion/python_to_value.hpp"
#include "conversion/sql_types.hpp"
#include "conversion/untyped.hpp"
#include "result.hpp"

namespace duckdb_python {
namespace {

constexpr const char *kClosed = "connection is closed";
constexpr const char *kInUse =
    "the appender is in use by another call, from another thread or from Python code that call runs; it takes one "
    "caller at a time";
constexpr const char *kBroken =
    "the appender is broken: a write into its buffer failed part way, so the rows it holds are unknown; clear() "
    "drops them and makes it usable again";

// The names the statement reads the buffer by.
constexpr const char *kDataName = "appender_data";
constexpr const char *kBufferName = "appender_buffer";
// The alias every action but the insert gives its target, so its columns stay apart from the buffer's whatever the
// table is called; without it an upsert into a table named excluded fails, since the engine rewrites the upsert into
// a merge that uses that name too.
constexpr const char *kTargetName = "appender_target";

/// The names and types of the columns a table appender buffers.
struct BufferColumns {
	std::vector<std::string> names;
	std::vector<cxx::LogicalType> types;
};

std::vector<cxx::ColumnDescription> AllColumns(const cxx::TableDescription &table) {
	std::vector<cxx::ColumnDescription> all;
	for (cxx::idx_t i = 0; i < table.GetColumnCount(); i++) {
		all.push_back(table.GetColumn(i));
	}
	return all;
}

/// The columns `listed` names, in that order, each matched ignoring ASCII case; `argument` names the list in a refusal.
BufferColumns ResolveListed(const std::vector<cxx::ColumnDescription> &all, const std::string &target,
                            const std::vector<std::string> &listed, const std::string &argument) {
	BufferColumns resolved;
	for (const auto &name : listed) {
		const auto found = std::find_if(all.begin(), all.end(), [&](const cxx::ColumnDescription &column) {
			return EqualIgnoringCase(column.GetName(), name);
		});
		if (found == all.end()) {
			throw nb::value_error((name + " is not a column of " + target).c_str());
		}
		if (found->HasGenerated()) {
			throw nb::value_error(
			    (name + " is a generated column: the engine computes it, so it takes no value").c_str());
		}
		const auto repeated = std::any_of(resolved.names.begin(), resolved.names.end(),
		                                  [&](const std::string &taken) { return EqualIgnoringCase(taken, name); });
		if (repeated) {
			throw nb::value_error((argument + " names " + name + " twice").c_str());
		}
		resolved.names.emplace_back(found->GetName());
		resolved.types.push_back(found->GetType());
	}
	return resolved;
}

BufferColumns ChooseColumns(const std::vector<cxx::ColumnDescription> &all, const std::string &target,
                            const std::optional<std::vector<std::string>> &requested) {
	if (!requested) {
		BufferColumns chosen;
		for (const auto &column : all) {
			if (!column.HasGenerated()) {
				chosen.names.emplace_back(column.GetName());
				chosen.types.push_back(column.GetType());
			}
		}
		return chosen;
	}
	if (requested->empty()) {
		throw nb::value_error("columns names no column; an appender writes at least one");
	}
	return ResolveListed(all, target, *requested, "columns");
}

bool Contains(const std::vector<std::string> &names, const std::string &name) {
	return std::any_of(names.begin(), names.end(),
	                   [&](const std::string &listed) { return EqualIgnoringCase(listed, name); });
}

/// The buffered columns an update sets: every one not in the key, whose columns must all be buffered. `listed` is
/// the key as the caller spelled it, for the refusal.
std::vector<std::string> ColumnsToSet(const std::vector<std::string> &buffered, const std::vector<std::string> &key,
                                      const std::vector<std::string> &listed) {
	for (std::size_t i = 0; i < key.size(); i++) {
		if (!Contains(buffered, key[i])) {
			throw nb::value_error(("key column " + listed[i] +
			                       " is not one of the appender's columns, so no row has a value to match it on")
			                          .c_str());
		}
	}
	std::vector<std::string> assigned;
	for (const auto &name : buffered) {
		if (!Contains(key, name)) {
			assigned.push_back(name);
		}
	}
	if (assigned.empty()) {
		throw nb::value_error("an update sets the appender's columns other than the key, and there are none");
	}
	return assigned;
}

template <class Render>
std::string Join(const std::vector<std::string> &names, const std::string &separator, const Render &render) {
	std::string joined;
	for (std::size_t i = 0; i < names.size(); i++) {
		joined += (i == 0 ? "" : separator) + render(names[i]);
	}
	return joined;
}

/// The statement a flush runs over the buffer, which holds `buffered`: `key` serves the update and the delete, and
/// `assigned` the update.
std::string RenderStatement(TableAction action, const std::string &target, const std::vector<std::string> &buffered,
                            const std::vector<std::string> &key, const std::vector<std::string> &assigned) {
	const auto quoted = [](const std::string &name) {
		return cxx::RenderQuotedIdentifier(name);
	};
	const auto read = [&](const std::string &name) {
		return std::string(kDataName) + "." + quoted(name);
	};
	const auto match = [&](const std::string &name) {
		return std::string(kTargetName) + "." + quoted(name) + " = " + read(name);
	};
	const auto set = [&](const std::string &name) {
		return quoted(name) + " = " + read(name);
	};
	const auto aliased = target + " AS " + kTargetName;
	switch (action) {
	case TableAction::INSERT_ROWS:
		return "INSERT INTO " + target + " (" + Join(buffered, ", ", quoted) + ") SELECT * FROM " + kDataName;
	case TableAction::UPSERT_ROWS:
		return "INSERT OR REPLACE INTO " + aliased + " (" + Join(buffered, ", ", quoted) + ") SELECT * FROM " +
		       kDataName;
	case TableAction::UPDATE_ROWS:
		return "UPDATE " + aliased + " SET " + Join(assigned, ", ", set) + " FROM " + kDataName + " WHERE " +
		       Join(key, " AND ", match);
	case TableAction::DELETE_ROWS:
		break;
	}
	return "DELETE FROM " + aliased + " USING " + kDataName + " WHERE " + Join(key, " AND ", match);
}

/// `value` cast to `type` as a bound INSERT casts it: one row through the vector cast, which raises the failing cast's
/// own error.
cxx::Value CastAsAnInsert(cxx::Factory &factory, const cxx::Value &value, const cxx::LogicalType &type) {
	std::vector<cxx::LogicalType> from;
	from.push_back(value.GetLogicalType());
	cxx::DataChunk source(factory, from);
	auto input = source.GetVector(0);
	input.SetSize(1);
	input.SetValue(0, value);
	std::vector<cxx::LogicalType> to;
	to.push_back(type.Copy());
	cxx::DataChunk target(factory, to);
	auto output = target.GetVector(0);
	input.Cast(factory, output);
	return output.GetValue(0);
}

/// A row's values as a tuple, or nothing for a row that is not a sequence of values: a string, bytes, a mapping, or
/// what does not iterate. Reading a row runs its own Python code.
nb::object RowValues(nb::handle row, nb::handle mapping_cls) {
	if (nb::isinstance<nb::str>(row) || nb::isinstance<nb::bytes>(row) || nb::isinstance(row, mapping_cls)) {
		return nb::object();
	}
	if (nb::isinstance<nb::tuple>(row)) {
		return nb::borrow(row);
	}
	nb::object items = nb::borrow(row);
	if (!nb::isinstance<nb::list>(row)) {
		items = nb::steal(PyObject_GetIter(row.ptr()));
		if (!items.is_valid()) {
			if (!PyErr_ExceptionMatches(PyExc_TypeError)) {
				throw nb::python_error();
			}
			PyErr_Clear();
			return nb::object();
		}
	}
	auto values = nb::steal(PySequence_Tuple(items.ptr()));
	if (!values.is_valid()) {
		throw nb::python_error();
	}
	return values;
}

bool KeepsRows(int code) {
	return code == DUCKDB_V2_ERROR_RESOURCE_IN_USE || code == DUCKDB_V2_ERROR_RUNTIME_INTERRUPT;
}

} // namespace

AppenderBuffer::AppenderBuffer(cxx::Connection &connection, const std::string &sql,
                               const std::vector<std::string> &names, const std::vector<cxx::LogicalType> &types)
    : rows(connection.GetFactory(), types), staging(connection.GetFactory(), types),
      statement(connection.ParseSQL(sql).Next()) {
	capacity = staging.GetCapacity();
	statement.AddCollection(kDataName, rows, names);
	statement.AddCollection(kBufferName, rows, names);
}

void AppenderBuffer::Reset() {
	rows.Reset();
	staging.Clear();
	staged = 0;
}

AppenderBuffers::~AppenderBuffers() {
	ReleaseAll();
}

void AppenderBuffers::Add(const std::shared_ptr<Owned<AppenderBuffer>> &buffer) {
	{
		nb::ft_lock_guard guard(lock);
		if (!closed) {
			buffers.erase(
			    std::remove_if(buffers.begin(), buffers.end(),
			                   [](const std::weak_ptr<Owned<AppenderBuffer>> &held) { return held.expired(); }),
			    buffers.end());
			buffers.push_back(buffer);
			return;
		}
	}
	// Released here rather than by its owner's destructor, which would destroy it inside DuckDB holding the GIL.
	buffer->Release();
	Raise(buffer->Module()->InterfaceError(), kClosed);
}

void AppenderBuffers::ReleaseAll() {
	std::vector<std::weak_ptr<Owned<AppenderBuffer>>> released;
	{
		nb::ft_lock_guard guard(lock);
		closed = true;
		released.swap(buffers);
	}
	// Outside the lock: a release lets go of the GIL.
	for (const auto &held : released) {
		if (auto buffer = held.lock()) {
			buffer->Release();
		}
	}
}

/// Holds the appender for one call, or refuses at once when another call holds it.
class Appender::OneCaller {
public:
	explicit OneCaller(Appender &appender) : appender(appender) {
		bool expected = false;
		if (!appender.in_use.compare_exchange_strong(expected, true)) {
			throw cxx::Exception(DUCKDB_V2_ERROR_RESOURCE_IN_USE, std::string("Resource In Use Error: ") + kInUse,
			                     kInUse);
		}
	}
	~OneCaller() {
		appender.in_use.store(false);
	}
	OneCaller(const OneCaller &) = delete;
	OneCaller &operator=(const OneCaller &) = delete;

private:
	Appender &appender;
};

/// A call's own references to the connection and to the buffer, so a close on another thread or from Python code the
/// call runs frees neither under it. The buffer is let go first, while its database is still open.
struct Appender::Pins {
	Pins(Pinned<cxx::Connection> connection, Pinned<AppenderBuffer> buffer)
	    : connection(std::move(connection)), buffer(std::move(buffer)) {
	}
	~Pins() {
		// Last references once a close ran during the call, and destroying a connection waits for its lock.
		if (buffer.engine.use_count() == 1 || connection.engine.use_count() == 1) {
			nb::gil_scoped_release unlock;
			buffer.engine.reset();
			connection.engine.reset();
		}
	}
	Pins(const Pins &) = delete;
	Pins &operator=(const Pins &) = delete;

	Pinned<cxx::Connection> connection;
	Pinned<AppenderBuffer> buffer;
};

Appender::Appender(AppenderConnection connection, std::shared_ptr<Owned<AppenderBuffer>> buffer,
                   std::vector<cxx::LogicalType> types)
    : connection(std::move(connection)), module(buffer->Module()), buffer(std::move(buffer)), types(std::move(types)) {
}

Appender::~Appender() {
	buffer->Release();
}

std::unique_ptr<Appender> Appender::ForTable(const std::shared_ptr<Owned<cxx::Connection>> &engine,
                                             const std::shared_ptr<ConnectionUse> &use, AppenderBuffers &buffers,
                                             const std::vector<std::string> &parts,
                                             const std::optional<std::vector<std::string>> &columns, TableAction action,
                                             const std::optional<std::vector<std::string>> &key) {
	Claimed claimed(engine->Acquire(kClosed), *use, ConnectionUse::Needs::NOT_RUNNING);
	auto &live = *claimed.held.engine;
	const auto name = cxx::QualifiedName::Create(parts);
	const auto table = WithoutGil([&] { return live.DescribeTable(name); });
	const auto target = table.GetQualifiedName().Render();
	if (table.IsReadOnly()) {
		const auto body = "an appender cannot write to " + target + ": its database is read-only";
		throw cxx::InvalidInputException("Invalid Input Error: " + body, body);
	}
	const auto all = AllColumns(table);
	auto keyed = key ? ResolveListed(all, target, *key, "key") : BufferColumns {};
	const auto key_names = keyed.names;
	auto chosen = action == TableAction::DELETE_ROWS ? std::move(keyed) : ChooseColumns(all, target, columns);
	const auto assigned = action == TableAction::UPDATE_ROWS
	                          ? ColumnsToSet(chosen.names, key_names, key.value_or(key_names))
	                          : std::vector<std::string> {};
	const auto sql = RenderStatement(action, target, chosen.names, key_names, assigned);
	auto made = WithoutGil([&] {
		AppenderBuffer buffer(live, sql, chosen.names, chosen.types);
		live.Bind(buffer.statement);
		return buffer;
	});
	auto buffer = std::make_shared<Owned<AppenderBuffer>>(engine->Module(), claimed.held.database, std::move(made));
	buffers.Add(buffer);
	return std::make_unique<Appender>(AppenderConnection {engine, use}, std::move(buffer), std::move(chosen.types));
}

Appender::Pins Appender::Pin() {
	auto owner = connection.engine.lock();
	if (!owner) {
		Raise(module->InterfaceError(), kClosed);
	}
	return Pins(owner->Acquire(kClosed), buffer->Acquire(kClosed));
}

bool Appender::Closed() {
	auto owner = connection.engine.lock();
	return !owner || owner->Released() || buffer->Released();
}

void Appender::RefuseBroken() const {
	if (broken) {
		Raise(module->InterfaceError(), kBroken);
	}
}

void Appender::AppendRow(nb::tuple values) {
	const OneCaller caller(*this);
	auto pins = Pin();
	RefuseBroken();
	StageRow(pins, values);
}

void Appender::AppendRows(nb::handle rows) {
	const OneCaller caller(*this);
	{
		const auto pins = Pin();
		RefuseBroken();
	}
	if (nb::isinstance<nb::str>(rows) || nb::isinstance<nb::bytes>(rows)) {
		const auto kind = nb::cast<std::string>(nb::handle(Py_TYPE(rows.ptr())).attr("__name__"));
		throw nb::type_error(
		    ("append_rows takes an iterable of rows, not a " + kind + "; one row is append_row(...)").c_str());
	}
	auto iterator = nb::steal(PyObject_GetIter(rows.ptr()));
	if (!iterator.is_valid()) {
		throw nb::python_error();
	}
	while (true) {
		// Rows are read with nothing pinned, so Python code reading them may close the connection and reopen its
		// database; converting needs the connection, so that runs pinned.
		auto row = nb::steal(PyIter_Next(iterator.ptr()));
		if (!row.is_valid()) {
			if (PyErr_Occurred() != nullptr) {
				throw nb::python_error();
			}
			return;
		}
		auto values = RowValues(row, module->conversion.mapping_cls);
		auto pins = Pin();
		if (!values.is_valid()) {
			const auto kind = nb::cast<std::string>(nb::handle(Py_TYPE(row.ptr())).attr("__name__"));
			throw nb::type_error(("a row is a sequence of values, one per column, not a " + kind).c_str());
		}
		StageRow(pins, values);
	}
}

void Appender::StageRow(Pins &pins, nb::handle row) {
	const auto count = static_cast<std::size_t>(PyTuple_Size(row.ptr()));
	if (count != types.size()) {
		const auto message = "a row of this appender holds " + std::to_string(types.size()) +
		                     " values, one per column, not " + std::to_string(count);
		throw nb::value_error(message.c_str());
	}
	auto &factory = pins.connection.engine->GetFactory();
	std::vector<cxx::Value> values;
	values.reserve(count);
	for (std::size_t i = 0; i < count; i++) {
		values.push_back(Convert(factory, PyTuple_GetItem(row.ptr(), static_cast<Py_ssize_t>(i)), types[i]));
	}
	auto &held = *pins.buffer.engine;
	const auto at = held.staged;
	try {
		for (cxx::idx_t i = 0; i < values.size(); i++) {
			auto column = held.staging.GetVector(i);
			column.SetSize(at + 1);
			column.SetValue(at, values[i]);
		}
	} catch (...) {
		broken = true;
		throw;
	}
	held.staged = at + 1;
	buffered.fetch_add(1);
	if (held.staged == held.capacity) {
		MoveStaged(held);
	}
}

cxx::Value Appender::Convert(cxx::Factory &factory, nb::handle object, const cxx::LogicalType &type) {
	// As a query parameter of `executemany` converts: untyped, then given the type the statement expects at its
	// place, which for an INSERT's VALUES is the column's, then cast to the column as the bound INSERT casts it.
	bool untyped = false;
	auto value = [&] {
		try {
			return PythonToValue(factory, object, module->conversion, TimestampPrecision::MICROSECONDS, &untyped);
		} catch (const UnsupportedTypeException &error) {
			ThrowInvalidInput("cannot append a value of type " + error.TypeName());
		}
	}();
	if (untyped) {
		if (value.IsNull()) {
			value = cxx::Value::CreateNull(factory, type);
		} else {
			FillUntypedFromExpected(factory, value, type);
		}
	}
	if (value.GetLogicalType() == type) {
		return value;
	}
	try {
		return value.Cast(factory, type);
	} catch (const cxx::Exception &) {
		// A value's cast reports every soft failure as one InvalidInputError, where the INSERT's cast raises the
		// failing cast's own error; casting as a vector does what the INSERT does.
		return CastAsAnInsert(factory, value, type);
	}
}

void Appender::MoveStaged(AppenderBuffer &held) {
	if (held.staged == 0) {
		return;
	}
	try {
		// Without an append state kept between moves: one keeps the last chunk's blocks pinned, which a flush under a
		// tight memory limit needs for its own reading.
		WithoutGil([&] {
			held.rows.Append(held.staging);
			held.staging.Clear();
		});
	} catch (...) {
		broken = true;
		throw;
	}
	held.staged = 0;
}

void Appender::Empty(AppenderBuffer &held) {
	try {
		WithoutGil([&] { held.Reset(); });
	} catch (...) {
		broken = true;
		throw;
	}
	broken = false;
	buffered.store(0);
}

void Appender::Flush(nb::handle before_run) {
	const OneCaller caller(*this);
	if (Closed()) {
		Raise(module->InterfaceError(), kClosed);
	}
	RefuseBroken();
	if (buffered.load() == 0) {
		return;
	}
	// Run with nothing pinned, as rows are read, so its Python code may close the connection and reopen its database.
	if (!before_run.is_none()) {
		before_run();
	}
	auto pins = Pin();
	const ConnectionUse::Call call(*connection.use, ConnectionUse::Needs::FREE);
	auto &held = *pins.buffer.engine;
	MoveStaged(held);
	auto &live = *pins.connection.engine;
	// Only for a failure of its own: the flush's error is the one raised, and a failed reset leaves it broken.
	const auto empty_after_failure = [&] {
		try {
			Empty(held);
		} catch (...) {
		}
	};
	std::unique_ptr<Result> running;
	try {
		auto query = WithoutGil([&] { return live.Execute(held.statement); });
		running = std::make_unique<Result>(pins.connection.database, module, std::move(query),
		                                   ConnectionUse::Hold(connection.use));
	} catch (const cxx::Exception &error) {
		if (!KeepsRows(error.GetCode())) {
			empty_after_failure();
		}
		throw;
	} catch (...) {
		empty_after_failure();
		throw;
	}
	auto ending = Result::Ending::FAILED;
	try {
		running->Finish([&] { live.Interrupt(); }, ending);
	} catch (...) {
		// Only a statement the engine cancelled left the rows unwritten; one that took effect or failed has them gone.
		running->Close();
		if (ending != Result::Ending::CANCELLED) {
			empty_after_failure();
		}
		throw;
	}
	running->Close();
	Empty(held);
}

void Appender::Clear() {
	const OneCaller caller(*this);
	if (Closed()) {
		// The close already dropped the rows, and this error tells of it.
		buffered.store(0);
		broken = false;
		Raise(module->InterfaceError(), kClosed);
	}
	const auto pins = Pin();
	Empty(*pins.buffer.engine);
}

} // namespace duckdb_python
