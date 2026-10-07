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

namespace duckdb {
class Relation;

namespace engine {

//! Client state kept on the engine's connection. It never owns a result: a result holds the context, and
//! the context holds this state.
class ConnectionState : public ClientContextState {
public:
	static shared_ptr<ConnectionState> Get(ClientContext &context);

public:
	void SetOpen(const shared_ptr<Result> &result);
	//! Closes the result whose query is still running, so the next statement never finds one open
	void SupersedeOpen();

private:
	mutex lock;
	weak_ptr<Result> open;
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
	unique_ptr<PreparedStatement> Prepare(unique_ptr<SQLStatement> statement);
	shared_ptr<Result> Submit(unique_ptr<SQLStatement> statement, identifier_map_t<BoundParameterData> &values,
	                          const Format &format);
	shared_ptr<Result> Submit(PreparedStatement &prepared, identifier_map_t<BoundParameterData> &values,
	                          const Format &format);
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
