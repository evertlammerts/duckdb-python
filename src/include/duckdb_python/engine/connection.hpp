//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/engine/connection.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb_python/engine/result.hpp"

#include "duckdb/common/atomic.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/client_context_state.hpp"
#include "duckdb/main/client_context_wrapper.hpp"
#include "duckdb/main/prepared_statement.hpp"
#include "duckdb/main/statement_iterator.hpp"
#include "duckdb/parser/sql_statement.hpp"
#include "duckdb/parser/tableref.hpp"

#include <thread>

namespace duckdb {
class Relation;

namespace engine {

//! The tables a replacement scan resolved, by the name exactly as the statement used it, which is how the
//! client's scan looks it up
using ResolvedTables = unordered_map<string, unique_ptr<TableRef>>;

//! Marks the thread as running Python code for a query of the context. Calls from that code into the same
//! connection are refused: the query waits for the code, so code that waited for the connection would wait on itself.
class CallbackScope {
public:
	explicit CallbackScope(const ClientContext &context);
	~CallbackScope();

	CallbackScope(const CallbackScope &) = delete;
	CallbackScope &operator=(const CallbackScope &) = delete;

	//! Whether this thread is inside a callback of a query of the context, at any depth
	static bool Inside(const ClientContext &context);

private:
	const ClientContext &context;
	CallbackScope *outer;
};

//! Client state kept on the engine's connection. It never owns a result: a result holds the context, and
//! the context holds this state.
class ConnectionState : public ClientContextState {
public:
	static shared_ptr<ConnectionState> Get(ClientContext &context);

public:
	//! Throws the refusal a claim would get, without claiming
	void ThrowIfBusy(const ClientContext &context) const;
	//! The result of the DB-API's execute(). Only the DB-API can end it, so a refusal it causes says how.
	void SetDbapiResult(const shared_ptr<Result> &result);
	//! Throws the refusal naming the DB-API's unfinished result, when the error is the engine refusing work because
	//! that result holds the connection
	void NameRefusal(const ErrorData &error);

	//! Between the two calls, every table the client's replacement scan resolves is recorded
	void BeginResolving();
	ResolvedTables EndResolving();
	void RecordResolved(const string &name, TableRef &table);
	//! While pinned, the replacement scan answers a recorded name with the recorded table; an exclusive pin answers
	//! no other name
	void Pin(optional_ptr<const ResolvedTables> tables, bool exclusive = false);
	unique_ptr<TableRef> RecallResolved(const string &name);
	bool PinnedExclusively();

private:
	friend class ConnectionClaim;

	atomic<bool> busy {false};
	mutex lock;
	weak_ptr<Result> dbapi_result;
	//! Both belong to the thread binding the statement, so another thread's bind on the context does not see them
	unique_ptr<ResolvedTables> resolving;
	std::thread::id resolving_thread;
	optional_ptr<const ResolvedTables> pinned;
	bool pinned_exclusively = false;
	std::thread::id pinned_thread;
};

//! Holds the connection for one stretch of engine work. A second claim, and any claim from a callback of the
//! connection's queries, is refused instead of waiting: the work it would wait for may be waiting for it.
class ConnectionClaim {
public:
	ConnectionClaim(ConnectionState &state, const ClientContext &context);
	explicit ConnectionClaim(ClientContext &context);
	~ConnectionClaim();

	ConnectionClaim(const ConnectionClaim &) = delete;
	ConnectionClaim &operator=(const ConnectionClaim &) = delete;

private:
	ConnectionState &state;
};

//! The engine's check that no result holds the connection, with its refusal named as ConnectionState::NameRefusal does
void VerifyNoOpenResult(ClientContext &context);
//! Names the refusal in an error from engine work that runs outside a Session, as ConnectionState::NameRefusal does
void NameRefusal(ClientContext &context, const std::exception &error);

//! Claims the connection while a relation binds, so a bind is refused rather than blocked; relations derived
//! from one that has it inherit it
class ClaimingContextWrapper : public ClientContextWrapper {
public:
	explicit ClaimingContextWrapper(shared_ptr<ClientContextWrapper> inner);

	void TryBindRelation(Relation &relation, vector<ColumnDefinition> &columns) override;

	static void Install(Relation &relation);

private:
	shared_ptr<ClientContextWrapper> inner;
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
	Prepared Prepare(unique_ptr<SQLStatement> statement);
	shared_ptr<Result> Submit(unique_ptr<SQLStatement> statement, identifier_map_t<BoundParameterData> &values,
	                          const Format &format);
	//! Submit for a caller that already holds the connection's claim
	shared_ptr<Result> SubmitClaimed(unique_ptr<SQLStatement> statement, identifier_map_t<BoundParameterData> &values,
	                                 const Format &format);
	shared_ptr<Result> Submit(Prepared &prepared, identifier_map_t<BoundParameterData> &values, const Format &format);
	//! A relation that writes is submitted as itself; one that reads as a statement over its query
	shared_ptr<Result> Submit(const shared_ptr<Relation> &relation, const Format &format);
	//! Binds a statement without running it, keeping the tables its replacement scans resolved in tables
	StatementSignature Bind(unique_ptr<SQLStatement> statement, ResolvedTables &tables);
	//! Runs a statement to its end and drops its rows
	void Run(unique_ptr<SQLStatement> statement, const InterruptCheck &check);
	//! Runs a statement to its end and drops its rows, its replacement scans answered from tables and nothing else
	void Run(unique_ptr<SQLStatement> statement, const ResolvedTables &tables, const InterruptCheck &check);
	ConnectionClaim Claim();

private:
	shared_ptr<ClientContext> context;
	shared_ptr<ConnectionState> state;
};

//! The statements of one SQL string, run in order as the engine's Query runs them: each is parsed only once the ones
//! before it have run, and the engine expands it, a PRAGMA for example, when it is submitted
class StatementGroup {
public:
	//! aborts_transaction false only parses, as the engine's ExtractStatements does: a parse error leaves an explicit
	//! transaction valid
	StatementGroup(Session session, const string &sql, bool aborts_transaction = true);
	//! A group of one statement parsed earlier
	StatementGroup(Session session, unique_ptr<SQLStatement> statement);

	StatementGroup(const StatementGroup &) = delete;
	StatementGroup &operator=(const StatementGroup &) = delete;
	StatementGroup(StatementGroup &&) = default;

public:
	//! The first statement, parsed but not run; null for a group without statements. Only before Next. Inside an
	//! explicit transaction, an error parsing it aborts the transaction, as in the engine.
	optional_ptr<const SQLStatement> First();
	//! The next statement, parsed but not run; null when no statement is left. Errors abort an explicit transaction
	//! as First's do.
	unique_ptr<SQLStatement> Next();
	//! Runs the statement and every one after it but the last, which it returns unrun; null when the text after the
	//! statements it ran holds no statement
	unique_ptr<SQLStatement> RunAllButLast(unique_ptr<SQLStatement> statement, const InterruptCheck &check);
	//! Called once a statement is submitted and before it runs: one known to return rows replaces the kept result,
	//! which is dropped now so that two results are never held at once
	void ReleaseIfReplaced(const Result &result);
	//! The result of the group, given the last statement's: that of the last statement that returns rows, or of the
	//! last statement when none does
	shared_ptr<Result> Choose(shared_ptr<Result> last);
	//! Throws for parameters given to a group whose text after the statements that ran held no statement
	[[noreturn]] void RefuseParameters();

private:
	bool HasMore();
	void Keep(shared_ptr<Result> result);
	//! Throws an error that ends the group before the engine ran the statement it concerns, as the engine's Query
	//! reports it. Requires the claim.
	[[noreturn]] void Fail(ErrorData error);

private:
	Session session;
	//! The text the statements came from, which an error's location refers to
	string query;
	bool aborts_transaction = true;
	//! Null for a group of one statement parsed earlier, which is held in upcoming
	unique_ptr<StatementIterator> iterator;
	//! Parsed by First, not yet returned by Next
	unique_ptr<SQLStatement> upcoming;
	shared_ptr<Result> kept;
};

} // namespace engine
} // namespace duckdb
