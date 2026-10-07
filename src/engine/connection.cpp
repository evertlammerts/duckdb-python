#include "duckdb_python/engine/connection.hpp"

#include "duckdb/main/relation.hpp"
#include "duckdb/main/statement_iterator.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref/column_data_ref.hpp"

namespace duckdb {
namespace engine {

shared_ptr<ConnectionState> ConnectionState::Get(ClientContext &context) {
	return context.registered_state->GetOrCreate<ConnectionState>("python_connection_state");
}

void ConnectionState::SetOpen(const shared_ptr<Result> &result) {
	lock_guard<mutex> guard(lock);
	open = result;
}

void ConnectionState::SupersedeOpen() {
	shared_ptr<Result> result;
	{
		lock_guard<mutex> guard(lock);
		result = open.lock();
		open.reset();
	}
	if (result) {
		result->Supersede();
	}
}

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

unique_ptr<PreparedStatement> Session::Prepare(unique_ptr<SQLStatement> statement) {
	state->SupersedeOpen();
	auto prepared = context->Prepare(std::move(statement));
	if (prepared->HasError()) {
		prepared->GetErrorObject().Throw();
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

shared_ptr<Result> Session::Submit(PreparedStatement &prepared, identifier_map_t<BoundParameterData> &values,
                                   const Format &format) {
	state->SupersedeOpen();
	auto handle = prepared.Submit(values, QueryParameters(format.EngineFormat()));
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
