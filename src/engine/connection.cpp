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

void ConnectionState::SetOpen(const shared_ptr<Supersedable> &result) {
	lock_guard<mutex> guard(lock);
	open = result;
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

} // namespace engine
} // namespace duckdb
