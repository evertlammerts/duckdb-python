#include "duckdb_python/engine/connection.hpp"

#include "duckdb/main/relation.hpp"
#include "duckdb/main/statement_iterator.hpp"
#include "duckdb/main/valid_checker.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref/column_data_ref.hpp"
#include "duckdb/transaction/meta_transaction.hpp"

namespace duckdb {
namespace engine {

shared_ptr<ConnectionState> ConnectionState::Get(ClientContext &context) {
	return context.registered_state->GetOrCreate<ConnectionState>("python_connection_state");
}

void ConnectionState::QueryBegin(ClientContext &) {
	++generation;
}

void ConnectionState::SetOpen(const shared_ptr<Supersedable> &result) {
	lock_guard<mutex> guard(lock);
	open = result;
}

void ConnectionState::ClearOpen(const Supersedable &result) {
	lock_guard<mutex> guard(lock);
	if (open.lock().get() == &result) {
		open.reset();
	}
}

void ConnectionState::SupersedeOpen() {
	shared_ptr<Supersedable> result;
	{
		lock_guard<mutex> guard(lock);
		result = open.lock();
		open.reset();
	}
	if (result) {
		result->Supersede();
	}
}

void ConnectionState::BeginResolving() {
	lock_guard<mutex> guard(lock);
	resolving = make_uniq<ResolvedTables>();
	resolving_thread = std::this_thread::get_id();
}

ResolvedTables ConnectionState::EndResolving() {
	lock_guard<mutex> guard(lock);
	ResolvedTables tables;
	if (resolving) {
		tables = std::move(*resolving);
		resolving.reset();
	}
	return tables;
}

void ConnectionState::RecordResolved(const string &name, TableRef &table) {
	lock_guard<mutex> guard(lock);
	if (resolving && resolving_thread == std::this_thread::get_id() && resolving->find(name) == resolving->end()) {
		(*resolving)[name] = table.Copy();
	}
}

void ConnectionState::Pin(optional_ptr<const ResolvedTables> tables) {
	lock_guard<mutex> guard(lock);
	pinned = tables;
	pinned_thread = std::this_thread::get_id();
}

unique_ptr<TableRef> ConnectionState::RecallResolved(const string &name) {
	lock_guard<mutex> guard(lock);
	if (!pinned || pinned_thread != std::this_thread::get_id()) {
		return nullptr;
	}
	auto entry = pinned->find(name);
	if (entry == pinned->end()) {
		return nullptr;
	}
	return entry->second->Copy();
}

namespace {

class ResolvingGuard {
public:
	explicit ResolvingGuard(ConnectionState &state) : state(state) {
		state.BeginResolving();
	}
	~ResolvingGuard() {
		if (!ended) {
			state.EndResolving();
		}
	}
	ResolvedTables End() {
		ended = true;
		return state.EndResolving();
	}

private:
	ConnectionState &state;
	bool ended = false;
};

class PinGuard {
public:
	PinGuard(ConnectionState &state, const ResolvedTables &tables) : state(state) {
		state.Pin(&tables);
	}
	~PinGuard() {
		state.Pin(nullptr);
	}

private:
	ConnectionState &state;
};

} // namespace

Session::Session(shared_ptr<ClientContext> context_p)
    : context(std::move(context_p)), state(ConnectionState::Get(*context)) {
}

vector<unique_ptr<SQLStatement>> Session::Parse(const string &sql) {
	// The iterator also runs the preprocessor, without which a PRAGMA cannot be prepared
	auto iterator = context->IterateStatements(sql);
	vector<unique_ptr<SQLStatement>> statements;
	while (iterator.Peek()) {
		if (auto statement = iterator.GetStatement()) {
			statements.push_back(std::move(statement));
		}
	}
	return statements;
}

Prepared Session::Prepare(unique_ptr<SQLStatement> statement) {
	state->SupersedeOpen();
	Prepared prepared;
	{
		ResolvingGuard guard(*state);
		prepared.statement = context->Prepare(std::move(statement));
		prepared.tables = guard.End();
	}
	if (prepared.statement->HasError()) {
		prepared.statement->GetErrorObject().Throw();
	}
	return prepared;
}

Bound Session::Bind(unique_ptr<SQLStatement> statement) {
	Bound bound;
	{
		ResolvingGuard guard(*state);
		try {
			bound.signature = context->BindStatement(statement->Copy());
		} catch (std::exception &ex) {
			// Rendered like an error from any other entry point: with its location, or as JSON
			ErrorData error(ex);
			context->ProcessError(error, statement->query);
			error.Throw();
		}
		bound.tables = guard.End();
	}
	bound.statement = std::move(statement);
	return bound;
}

shared_ptr<Result> Session::Track(unique_ptr<QueryResult> handle, const Format &format, shared_ptr<Relation> relation) {
	auto result = make_shared_ptr<Result>(std::move(handle), format, std::move(relation));
	state->SetOpen(result);
	return result;
}

shared_ptr<Result> Session::Submit(unique_ptr<SQLStatement> statement, identifier_map_t<BoundParameterData> &values,
                                   const Format &format) {
	state->SupersedeOpen();
	auto handle = context->Submit(std::move(statement), values, QueryParameters(format.EngineFormat()));
	return Track(std::move(handle), format, nullptr);
}

shared_ptr<Result> Session::Submit(Prepared &prepared, identifier_map_t<BoundParameterData> &values,
                                   const Format &format) {
	state->SupersedeOpen();
	unique_ptr<QueryResult> handle;
	{
		PinGuard guard(*state, prepared.tables);
		handle = prepared.statement->Submit(values, QueryParameters(format.EngineFormat()));
	}
	return Track(std::move(handle), format, nullptr);
}

shared_ptr<Result> Session::Submit(Bound &bound, identifier_map_t<BoundParameterData> &values, const Format &format) {
	state->SupersedeOpen();
	unique_ptr<QueryResult> handle;
	{
		PinGuard guard(*state, bound.tables);
		handle = context->Submit(std::move(bound.statement), values, QueryParameters(format.EngineFormat()));
	}
	return Track(std::move(handle), format, nullptr);
}

shared_ptr<Result> Session::Submit(const shared_ptr<Relation> &relation, const Format &format) {
	state->SupersedeOpen();
	QueryParameters parameters(format.EngineFormat());
	if (!relation->IsReadOnly()) {
		auto handle = context->Submit(relation, parameters);
		return Track(std::move(handle), format, relation);
	}
	auto select = make_uniq<SelectStatement>();
	select->node = relation->GetQueryNode();
	select->query = relation->GetQuery();
	auto handle = context->Submit(std::move(select), parameters);
	return Track(std::move(handle), format, relation);
}

shared_ptr<Result> Session::Submit(unique_ptr<ColumnDataCollection> collection, const vector<Identifier> &names,
                                   const Format &format) {
	// The binder rejects duplicate column names; the caller keeps the originals
	auto deduplicated_names = names;
	QueryResult::DeduplicateColumns(deduplicated_names);
	auto table_ref = make_uniq<ColumnDataRef>(std::move(collection), std::move(deduplicated_names));
	// Binding asserts on an unset alias
	table_ref->alias = "materialized";
	auto select_node = make_uniq<SelectNode>();
	select_node->select_list.push_back(make_uniq<StarExpression>());
	select_node->from_table = std::move(table_ref);
	auto select = make_uniq<SelectStatement>();
	select->node = std::move(select_node);
	identifier_map_t<BoundParameterData> no_values;
	return Submit(std::move(select), no_values, format);
}

void Session::Run(unique_ptr<SQLStatement> statement, const InterruptCheck &check) {
	state->SupersedeOpen();
	auto handle = context->Submit(std::move(statement), QueryParameters());
	Complete(*handle, check);
}

bool Session::TransactionFailed() const {
	auto &transaction = context->transaction;
	return transaction.HasActiveTransaction() && ValidChecker::IsInvalidated(transaction.ActiveTransaction());
}

shared_ptr<Deferred> Session::Defer(Bound bound, identifier_map_t<BoundParameterData> values) {
	identifier_map_t<idx_t> parameters;
	for (auto &parameter : bound.signature.parameters) {
		parameters[parameter.identifier] = parameter.index;
	}
	PreparedStatement::VerifyParameters(values, parameters, context.get());
	// execute() ends whatever else is reading on the connection, as a statement that ran would
	state->SupersedeOpen();
	auto deferred = make_shared_ptr<Deferred>(*this, std::move(bound), std::move(values));
	state->SetOpen(deferred);
	return deferred;
}

bool Deferred::MayDefer(StatementType type) {
	switch (type) {
	case StatementType::SELECT_STATEMENT:
	case StatementType::INSERT_STATEMENT:
	case StatementType::UPDATE_STATEMENT:
	case StatementType::DELETE_STATEMENT:
	case StatementType::MERGE_INTO_STATEMENT:
		return true;
	default:
		return false;
	}
}

bool Deferred::CanDefer(const Bound &bound) {
	auto &properties = bound.signature.properties;
	// A statement whose types are only known once its values are bound cannot be described before it runs
	return MayDefer(bound.statement->type) && properties.return_type == StatementReturnType::QUERY_RESULT &&
	       properties.bound_all_parameters;
}

Deferred::Deferred(Session session_p, Bound bound_p, identifier_map_t<BoundParameterData> values_p)
    : session(std::move(session_p)), bound(std::move(bound_p)), values(std::move(values_p)),
      client_properties(session.Context().GetClientProperties()), generation(session.state->Generation()),
      may_write(!bound.signature.properties.modified_databases.empty()), superseded(false), done(false) {
}

Deferred::~Deferred() {
	try {
		Close();
	} catch (...) { // NOLINT: a destructor must not throw
	}
}

const vector<Identifier> &Deferred::Names() const {
	return bound.signature.names;
}

const vector<LogicalType> &Deferred::Types() const {
	return bound.signature.types;
}

shared_ptr<Result> Deferred::Start(const Format &format) {
	if (!superseded && session.state->Generation() != generation) {
		Supersede();
	}
	if (done.exchange(true)) {
		if (superseded) {
			throw InterruptException(
			    "The result was cancelled because another statement ran on its connection before it was read");
		}
		throw InternalException("engine::Deferred started twice");
	}
	if (superseded) {
		throw InterruptException(
		    "The result was cancelled because another statement ran on its connection before it was read");
	}
	session.state->ClearOpen(*this);
	auto result = session.Submit(bound, values, format);
	// The statement binds again here; its consumer already holds the columns it was described with
	if (result->Names() != bound.signature.names || result->Types() != bound.signature.types) {
		result->Close();
		throw InvalidInputException("The result's columns changed after execute() described them; execute the query "
		                            "again");
	}
	return result;
}

void Deferred::Close() {
	Abandon();
}

void Deferred::Supersede() {
	superseded = true;
	Abandon();
}

void Deferred::Abandon() {
	if (done.exchange(true) || !may_write) {
		return;
	}
	auto &context = session.Context();
	if (context.transaction.IsAutoCommit() || !context.transaction.HasActiveTransaction()) {
		return;
	}
	// The engine fails the transaction of an unfinished statement that may write, so a write that never ran
	// does the same rather than letting the transaction commit without it
	context.RunFunctionInTransaction(
	    [&]() {
		    ValidChecker::Invalidate(context.transaction.ActiveTransaction(),
		                             "a statement that writes was abandoned before its result was read");
	    },
	    false);
}

} // namespace engine
} // namespace duckdb
