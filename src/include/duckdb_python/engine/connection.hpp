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
	void QueryBegin(ClientContext &context) override;
	//! Moves whenever a statement starts on the connection, also one that did not go through a Session
	idx_t Generation() const {
		return generation;
	}

	void SetOpen(const shared_ptr<Supersedable> &result);
	void ClearOpen(const Supersedable &result);
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
	atomic<idx_t> generation {0};
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

//! A statement bound without running, with its signature and the tables its replacement scans resolved
struct Bound {
	unique_ptr<SQLStatement> statement;
	StatementSignature signature;
	ResolvedTables tables;
};

class Deferred;

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
	//! Binds without running and without touching an open result
	Bound Bind(unique_ptr<SQLStatement> statement);
	shared_ptr<Result> Submit(unique_ptr<SQLStatement> statement, identifier_map_t<BoundParameterData> &values,
	                          const Format &format);
	shared_ptr<Result> Submit(Prepared &prepared, identifier_map_t<BoundParameterData> &values, const Format &format);
	//! Binding at submission finds the tables the statement was bound with
	shared_ptr<Result> Submit(Bound &bound, identifier_map_t<BoundParameterData> &values, const Format &format);
	//! A relation that writes is submitted as itself; one that reads as a statement over its query
	shared_ptr<Result> Submit(const shared_ptr<Relation> &relation, const Format &format);
	//! Scans a collection, for example to produce a retained result in another format
	shared_ptr<Result> Submit(unique_ptr<ColumnDataCollection> collection, const vector<Identifier> &names,
	                          const Format &format);
	//! Runs a statement to its end and drops its rows
	void Run(unique_ptr<SQLStatement> statement, const InterruptCheck &check);
	//! Holds a statement that can be deferred until its first consumer picks the format. Checks the values
	//! against its parameters and ends the open result, as a submission would.
	shared_ptr<Deferred> Defer(Bound bound, identifier_map_t<BoundParameterData> values);
	//! Whether the open transaction has failed, so that only a rollback can run
	bool TransactionFailed() const;

private:
	shared_ptr<Result> Track(unique_ptr<QueryResult> handle, const Format &format, shared_ptr<Relation> relation);

private:
	friend class Deferred;
	shared_ptr<ClientContext> context;
	shared_ptr<ConnectionState> state;
};

//! A prepared statement that returns rows, not submitted until its first consumer picks the format. Abandoning
//! it follows the engine's rule for an unfinished result: one that reads is dropped, one that may write fails
//! an explicit transaction.
class Deferred : public Supersedable {
public:
	//! Queries and DML, whose effects stay inside the transaction, can wait for their consumer
	static bool MayDefer(StatementType type);
	static bool CanDefer(const Bound &bound);

public:
	Deferred(Session session, Bound bound, identifier_map_t<BoundParameterData> values);
	~Deferred() override;

public:
	const vector<Identifier> &Names() const;
	const vector<LogicalType> &Types() const;
	const ClientProperties &GetClientProperties() const {
		return client_properties;
	}
	//! Submits the statement. Throws when another statement ran on the connection since it was deferred.
	shared_ptr<Result> Start(const Format &format);
	void Close();
	void Supersede() override;

private:
	void Abandon();

private:
	Session session;
	Bound bound;
	//! For diagnosing a column mismatch: a value-less rebind of this copy tells a catalog change apart
	//! from a parameter that settled on another type. Kept only when the statement has parameters.
	unique_ptr<SQLStatement> diagnosis_statement;
	identifier_map_t<BoundParameterData> values;
	ClientProperties client_properties;
	idx_t generation;
	bool may_write;
	atomic<bool> superseded;
	atomic<bool> done;
};

} // namespace engine
} // namespace duckdb
