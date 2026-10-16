//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/appender.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nanobind/nanobind.h>

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "lifetime.hpp"

namespace duckdb_python {

/// The rows an appender holds and the statement its flush runs, all made from its connection. Rows are staged in the
/// chunk and move into the collection a whole chunk at a time; the statement reads the collection.
struct AppenderBuffer {
	AppenderBuffer(cxx::Connection &connection, const std::string &sql, const std::vector<std::string> &names,
	               const std::vector<cxx::LogicalType> &types);

	/// Drop every row and let go of the memory they took.
	void Reset();

	cxx::ColumnDataCollection rows;
	cxx::DataChunk staging;
	cxx::idx_t staged = 0;
	cxx::idx_t capacity = 0;
	// Declared after the collection it reads, so it is destroyed first.
	cxx::SqlStatement statement;
};

/// The buffers of one connection's appenders, which its close releases before the connection itself: a buffer
/// destroyed after its database closed reads freed memory.
class AppenderBuffers {
public:
	AppenderBuffers() = default;
	~AppenderBuffers();
	AppenderBuffers(const AppenderBuffers &) = delete;
	AppenderBuffers &operator=(const AppenderBuffers &) = delete;

	/// Track `buffer`, or raise InterfaceError once the connection is closed.
	void Add(const std::shared_ptr<Owned<AppenderBuffer>> &buffer);

	/// Release every buffer tracked and refuse any added later; repeatable.
	void ReleaseAll();

private:
	nb::ft_mutex lock;
	bool closed = false;
	std::vector<std::weak_ptr<Owned<AppenderBuffer>>> buffers;
};

/// What an appender needs of its connection: the engine connection, held weakly so an appender never keeps a closed
/// connection's database open, and the record of how the connection is in use, which a flush claims as any call does.
struct AppenderConnection {
	std::weak_ptr<Owned<cxx::Connection>> engine;
	std::shared_ptr<ConnectionUse> use;
};

/// Rows buffered on the client and written to the database by one statement at each flush, and in no other way.
///
/// One caller at a time: a second one, from another thread or from Python code the first call runs, is refused at
/// once and never waits.
class Appender {
public:
	Appender(AppenderConnection connection, std::shared_ptr<Owned<AppenderBuffer>> buffer,
	         std::vector<cxx::LogicalType> types);
	~Appender();
	Appender(const Appender &) = delete;
	Appender &operator=(const Appender &) = delete;

	/// An appender inserting into the table `parts` names, resolved now, over `columns` in that order, or else over
	/// every column the table does not generate, in its order.
	static std::unique_ptr<Appender> ForTable(const std::shared_ptr<Owned<cxx::Connection>> &engine,
	                                          const std::shared_ptr<ConnectionUse> &use, AppenderBuffers &buffers,
	                                          const std::vector<std::string> &parts,
	                                          const std::optional<std::vector<std::string>> &columns);

	/// Buffer one row, one value per column.
	void AppendRow(nb::tuple values);

	/// Buffer each row of an iterable, stopping at the first that fails and keeping the rows before it.
	void AppendRows(nb::handle rows);

	/// Write every buffered row with one run of the statement, then empty the buffer. `before_run`, unless None, is
	/// called once the appender is held and found to have rows, before the statement claims the connection.
	void Flush(nb::handle before_run);

	/// Drop every buffered row.
	void Clear();

	/// How many rows are buffered, also once the connection's close has dropped them.
	cxx::idx_t Buffered() const {
		return buffered.load();
	}

	/// Whether the connection is closed.
	bool Closed();

	/// The Database reference, for the garbage collector's visit.
	nb::handle Parent() const {
		return buffer->Parent();
	}

	/// Let go of the buffer; every later call raises InterfaceError.
	void Close() {
		buffer->Release();
	}

private:
	class OneCaller;
	struct Pins;

	/// The call's own references to the connection and the buffer, or InterfaceError once the connection is closed.
	Pins Pin();
	void RefuseBroken() const;
	void StageRow(Pins &pins, nb::handle row);
	cxx::Value Convert(cxx::Factory &factory, nb::handle object, const cxx::LogicalType &type);
	/// Move the staged rows into the collection; a failure leaves the appender broken.
	void MoveStaged(AppenderBuffer &buffer);
	/// Empty the buffer; a failure leaves the appender broken.
	void Empty(AppenderBuffer &buffer);

	AppenderConnection connection;
	std::shared_ptr<ModuleState> module;
	std::shared_ptr<Owned<AppenderBuffer>> buffer;
	std::vector<cxx::LogicalType> types;
	std::atomic<bool> in_use {false};
	std::atomic<cxx::idx_t> buffered {0};
	// Set when a write into the buffer failed part way, so it holds an unknown part of the rows until `Clear`.
	bool broken = false;
};

} // namespace duckdb_python
