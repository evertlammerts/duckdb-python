//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/engine/connection.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb_python/engine/result.hpp"

#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/client_context_state.hpp"
#include "duckdb/main/prepared_statement.hpp"
#include "duckdb/parser/sql_statement.hpp"
#include "duckdb/parser/tableref.hpp"

#include <thread>

namespace duckdb {
class Relation;

namespace engine {

//! The tables a replacement scan resolved, by the name exactly as the statement used it, which is how the
//! client's scan looks it up
using ResolvedTables = unordered_map<string, unique_ptr<TableRef>>;

//! Client state kept on the engine's connection. It never owns a result: a result holds the context, and
//! the context holds this state.
class ConnectionState : public ClientContextState {
public:
	static shared_ptr<ConnectionState> Get(ClientContext &context);

public:
	void SetOpen(const shared_ptr<Supersedable> &result);
	//! Closes the result that still holds the connection, so the next statement never finds one open
	void SupersedeOpen();

	//! Between the two calls, every table the client's replacement scan resolves is recorded
	void BeginResolving();
	ResolvedTables EndResolving();
	void RecordResolved(const string &name, TableRef &table);
	//! While pinned, the replacement scan answers a recorded name with the recorded table
	void Pin(optional_ptr<const ResolvedTables> tables);
	unique_ptr<TableRef> RecallResolved(const string &name);

private:
	mutex lock;
	weak_ptr<Supersedable> open;
	//! Both belong to the thread binding the statement, so another thread's bind on the context does not see them
	unique_ptr<ResolvedTables> resolving;
	std::thread::id resolving_thread;
	optional_ptr<const ResolvedTables> pinned;
	std::thread::id pinned_thread;
};

//! A prepared statement with the tables its replacement scans resolved. A rebind at submission, after the
//! catalog moved, finds the same tables instead of looking the names up again where the result is read.
struct Prepared {
	unique_ptr<PreparedStatement> statement;
	ResolvedTables tables;
};

//! The statements, prepared statements and results of one engine connection
class Session {
public:
	explicit Session(shared_ptr<ClientContext> context);

public:
	ClientContext &Context() const {
		return *context;
	}
	vector<unique_ptr<SQLStatement>> Parse(const string &sql);
	Prepared Prepare(unique_ptr<SQLStatement> statement);
	shared_ptr<Result> Submit(unique_ptr<SQLStatement> statement, identifier_map_t<BoundParameterData> &values,
	                          const Format &format);
	shared_ptr<Result> Submit(Prepared &prepared, identifier_map_t<BoundParameterData> &values, const Format &format);
	//! A relation that writes is submitted as itself; one that reads as a statement over its query
	shared_ptr<Result> Submit(const shared_ptr<Relation> &relation, const Format &format);
	//! Scans a collection, for example to produce a retained result in another format
	shared_ptr<Result> Submit(unique_ptr<ColumnDataCollection> collection, const vector<Identifier> &names,
	                          const Format &format);
	//! Runs a statement to its end and drops its rows
	void Run(unique_ptr<SQLStatement> statement, const InterruptCheck &check);

private:
	shared_ptr<Result> Track(unique_ptr<QueryResult> handle, const Format &format, shared_ptr<Relation> relation);

private:
	shared_ptr<ClientContext> context;
	shared_ptr<ConnectionState> state;
};

} // namespace engine
} // namespace duckdb
