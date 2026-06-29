#include "duckdb_python/pyconnection/pyconnection.hpp"

namespace duckdb {

std::unique_ptr<DuckDBPyType> DuckDBPyConnection::MapType(const DuckDBPyType &key_type,
                                                          const DuckDBPyType &value_type) {
	auto map_type = LogicalType::MAP(key_type.Type(), value_type.Type());
	return make_uniq<DuckDBPyType>(map_type);
}

std::unique_ptr<DuckDBPyType> DuckDBPyConnection::ListType(const DuckDBPyType &type) {
	auto array_type = LogicalType::LIST(type.Type());
	return make_uniq<DuckDBPyType>(array_type);
}

std::unique_ptr<DuckDBPyType> DuckDBPyConnection::ArrayType(const DuckDBPyType &type, idx_t size) {
	auto array_type = LogicalType::ARRAY(type.Type(), size);
	return make_uniq<DuckDBPyType>(array_type);
}

static child_list_t<LogicalType> GetChildList(const py::object &container) {
	child_list_t<LogicalType> types;
	if (py::isinstance<py::list>(container)) {
		py::list fields = py::cast<py::list>(container);
		idx_t i = 1;
		for (auto item : fields) {
			std::unique_ptr<DuckDBPyType> pytype;
			if (!DuckDBPyType::TryConvert(py::borrow<py::object>(item), pytype)) {
				string actual_type = py::cast<std::string>(py::str((item).type()));
				throw InvalidInputException("object has to be a list of DuckDBPyType's, not '%s'", actual_type);
			}
			types.push_back(std::make_pair(Identifier(StringUtil::Format("v%d", i++)), pytype->Type()));
		}
		return types;
	} else if (py::isinstance<py::dict>(container)) {
		py::dict fields = py::cast<py::dict>(container);
		for (auto item : fields) {
			auto name_p = item.first;
			auto type_p = item.second;
			auto name = Identifier(py::cast<std::string>(name_p));
			std::unique_ptr<DuckDBPyType> pytype;
			if (!DuckDBPyType::TryConvert(py::borrow<py::object>(type_p), pytype)) {
				string actual_type = py::cast<std::string>(py::str((type_p).type()));
				throw InvalidInputException("object has to be a list of DuckDBPyType's, not '%s'", actual_type);
			}
			types.push_back(std::make_pair(name, pytype->Type()));
		}
		return types;
	} else {
		string actual_type = py::cast<std::string>(py::str((container).type()));
		throw InvalidInputException(
		    "Can not construct a child list from object of type '%s', only dict/list is supported", actual_type);
	}
}

std::unique_ptr<DuckDBPyType> DuckDBPyConnection::StructType(const py::object &fields) {
	child_list_t<LogicalType> types = GetChildList(fields);
	if (types.empty()) {
		throw InvalidInputException("Can not create an empty struct type!");
	}
	auto struct_type = LogicalType::STRUCT(std::move(types));
	return make_uniq<DuckDBPyType>(struct_type);
}

std::unique_ptr<DuckDBPyType> DuckDBPyConnection::UnionType(const py::object &members) {
	child_list_t<LogicalType> types = GetChildList(members);

	if (types.empty()) {
		throw InvalidInputException("Can not create an empty union type!");
	}
	auto union_type = LogicalType::UNION(std::move(types));
	return make_uniq<DuckDBPyType>(union_type);
}

std::unique_ptr<DuckDBPyType> DuckDBPyConnection::EnumType(const string &name, const DuckDBPyType &type,
                                                           const py::list &values_p) {
	throw NotImplementedException("enum_type creation method is not implemented yet");
}

std::unique_ptr<DuckDBPyType> DuckDBPyConnection::DecimalType(int width, int scale) {
	auto decimal_type = LogicalType::DECIMAL(width, scale);
	return make_uniq<DuckDBPyType>(decimal_type);
}

std::unique_ptr<DuckDBPyType> DuckDBPyConnection::StringType(const string &collation) {
	LogicalType type;
	if (collation.empty()) {
		type = LogicalType::VARCHAR;
	} else {
		type = LogicalType::VARCHAR_COLLATION(collation);
	}
	return make_uniq<DuckDBPyType>(type);
}

std::unique_ptr<DuckDBPyType> DuckDBPyConnection::Type(const string &type_str) {
	auto &connection = con.GetConnection();
	auto &context = *connection.context;
	std::unique_ptr<DuckDBPyType> result;
	context.RunFunctionInTransaction([&result, &type_str, &context]() {
		result = make_uniq<DuckDBPyType>(TransformStringToLogicalType(type_str, context));
	});
	return result;
}

} // namespace duckdb
