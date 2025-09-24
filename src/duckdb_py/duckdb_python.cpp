#include "duckdb_python/pybind11/pybind_wrapper.hpp"

#include "duckdb/common/atomic.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/parser/parser.hpp"

#include "duckdb_python/python_objects.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/pystatement.hpp"
#include "duckdb_python/pyrelation.hpp"
#include "duckdb_python/expression/pyexpression.hpp"
#include "duckdb_python/pybind11/exceptions.hpp"
#include "duckdb_python/typing.hpp"
#include "duckdb_python/functional.hpp"
#include "duckdb_python/pybind11/conversions/pyconnection_default.hpp"
#include "duckdb/common/box_renderer.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb_python/pybind11/conversions/exception_handling_enum.hpp"
#include "duckdb_python/pybind11/conversions/python_udf_type_enum.hpp"
#include "duckdb_python/pybind11/conversions/python_csv_line_terminator_enum.hpp"
#include "duckdb/common/enums/statement_type.hpp"

#include "duckdb.hpp"

#ifndef DUCKDB_PYTHON_LIB_NAME
#define DUCKDB_PYTHON_LIB_NAME _duckdb
#endif

namespace py = pybind11;

namespace duckdb {

enum PySQLTokenType : uint8_t {
	PY_SQL_TOKEN_IDENTIFIER = 0,
	PY_SQL_TOKEN_NUMERIC_CONSTANT,
	PY_SQL_TOKEN_STRING_CONSTANT,
	PY_SQL_TOKEN_OPERATOR,
	PY_SQL_TOKEN_KEYWORD,
	PY_SQL_TOKEN_COMMENT
};

static py::list PyTokenize(const string &query) {
	auto tokens = Parser::Tokenize(query);
	py::list result;
	for (auto &token : tokens) {
		auto tuple = py::tuple(2);
		tuple[0] = token.start;
		switch (token.type) {
		case SimplifiedTokenType::SIMPLIFIED_TOKEN_IDENTIFIER:
			tuple[1] = PY_SQL_TOKEN_IDENTIFIER;
			break;
		case SimplifiedTokenType::SIMPLIFIED_TOKEN_NUMERIC_CONSTANT:
			tuple[1] = PY_SQL_TOKEN_NUMERIC_CONSTANT;
			break;
		case SimplifiedTokenType::SIMPLIFIED_TOKEN_STRING_CONSTANT:
			tuple[1] = PY_SQL_TOKEN_STRING_CONSTANT;
			break;
		case SimplifiedTokenType::SIMPLIFIED_TOKEN_OPERATOR:
			tuple[1] = PY_SQL_TOKEN_OPERATOR;
			break;
		case SimplifiedTokenType::SIMPLIFIED_TOKEN_KEYWORD:
			tuple[1] = PY_SQL_TOKEN_KEYWORD;
			break;
		case SimplifiedTokenType::SIMPLIFIED_TOKEN_COMMENT:
			tuple[1] = PY_SQL_TOKEN_COMMENT;
			break;
		default:
			break;
		}
		result.append(tuple);
	}
	return result;
}

static void InitializeConnectionMethods(py::module_ &m) {

	// START_OF_CONNECTION_METHODS
	m.def(
	    "cursor",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Cursor();
	    },
	    "Create a duplicate of the current connection", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "register_filesystem",
	    [](AbstractFileSystem filesystem, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    conn->RegisterFilesystem(filesystem);
	    },
	    "Register a fsspec compliant filesystem", py::arg("filesystem"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "unregister_filesystem",
	    [](const py::str &name, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    conn->UnregisterFilesystem(name);
	    },
	    "Unregister a filesystem", py::arg("name"), py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "list_filesystems",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->ListFilesystems();
	    },
	    "List registered filesystems, including builtin ones", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "filesystem_is_registered",
	    [](const string &name, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FileSystemIsRegistered(name);
	    },
	    "Check if a filesystem with the provided name is currently registered", py::arg("name"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "create_function",
	    [](const string &name, const py::function &udf, const py::object &arguments = py::none(),
	       const shared_ptr<DuckDBPyType> &return_type = nullptr, PythonUDFType type = PythonUDFType::NATIVE,
	       FunctionNullHandling null_handling = FunctionNullHandling::DEFAULT_NULL_HANDLING,
	       PythonExceptionHandling exception_handling = PythonExceptionHandling::FORWARD_ERROR,
	       bool side_effects = false, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->RegisterScalarUDF(name, udf, arguments, return_type, type, null_handling, exception_handling,
		                                   side_effects);
	    },
	    "Create a DuckDB function out of the passing in Python function so it can be used in queries", py::arg("name"),
	    py::arg("function"), py::arg("parameters") = py::none(), py::arg("return_type") = py::none(), py::kw_only(),
	    py::arg("type") = PythonUDFType::NATIVE, py::arg("null_handling") = FunctionNullHandling::DEFAULT_NULL_HANDLING,
	    py::arg("exception_handling") = PythonExceptionHandling::FORWARD_ERROR, py::arg("side_effects") = false,
	    py::arg("connection") = py::none());
	m.def(
	    "remove_function",
	    [](const string &name, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->UnregisterUDF(name);
	    },
	    "Remove a previously created function", py::arg("name"), py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "sqltype",
	    [](const string &type_str, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Type(type_str);
	    },
	    "Create a type object by parsing the 'type_str' string", py::arg("type_str"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "dtype",
	    [](const string &type_str, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Type(type_str);
	    },
	    "Create a type object by parsing the 'type_str' string", py::arg("type_str"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "type",
	    [](const string &type_str, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Type(type_str);
	    },
	    "Create a type object by parsing the 'type_str' string", py::arg("type_str"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "array_type",
	    [](const shared_ptr<DuckDBPyType> &type, idx_t size, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->ArrayType(type, size);
	    },
	    "Create an array type object of 'type'", py::arg("type").none(false), py::arg("size"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "list_type",
	    [](const shared_ptr<DuckDBPyType> &type, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->ListType(type);
	    },
	    "Create a list type object of 'type'", py::arg("type").none(false), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "union_type",
	    [](const py::object &members, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->UnionType(members);
	    },
	    "Create a union type object from 'members'", py::arg("members").none(false), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "string_type",
	    [](const string &collation = string(), shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->StringType(collation);
	    },
	    "Create a string type with an optional collation", py::arg("collation") = "", py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "enum_type",
	    [](const string &name, const shared_ptr<DuckDBPyType> &type, const py::list &values_p,
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->EnumType(name, type, values_p);
	    },
	    "Create an enum type of underlying 'type', consisting of the list of 'values'", py::arg("name"),
	    py::arg("type"), py::arg("values"), py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "decimal_type",
	    [](int width, int scale, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->DecimalType(width, scale);
	    },
	    "Create a decimal type with 'width' and 'scale'", py::arg("width"), py::arg("scale"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "struct_type",
	    [](const py::object &fields, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->StructType(fields);
	    },
	    "Create a struct type object from 'fields'", py::arg("fields"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "row_type",
	    [](const py::object &fields, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->StructType(fields);
	    },
	    "Create a struct type object from 'fields'", py::arg("fields"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "map_type",
	    [](const shared_ptr<DuckDBPyType> &key_type, const shared_ptr<DuckDBPyType> &value_type,
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->MapType(key_type, value_type);
	    },
	    "Create a map type object from 'key_type' and 'value_type'", py::arg("key").none(false),
	    py::arg("value").none(false), py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "duplicate",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Cursor();
	    },
	    "Create a duplicate of the current connection", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "execute",
	    [](const py::object &query, py::object params = py::list(), shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Execute(query, params);
	    },
	    "Execute the given SQL query, optionally using prepared statements with parameters set", py::arg("query"),
	    py::arg("parameters") = py::none(), py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "executemany",
	    [](const py::object &query, py::object params = py::list(), shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->ExecuteMany(query, params);
	    },
	    "Execute the given prepared statement multiple times using the list of parameter sets in parameters",
	    py::arg("query"), py::arg("parameters") = py::none(), py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "close",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    conn->Close();
	    },
	    "Close the connection", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "interrupt",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    conn->Interrupt();
	    },
	    "Interrupt pending operations", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "query_progress",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->QueryProgress();
	    },
	    "Query progress of pending operation", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "fetchone",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchOne();
	    },
	    "Fetch a single row from a result following execute", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "fetchmany",
	    [](idx_t size, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchMany(size);
	    },
	    "Fetch the next set of rows from a result following execute", py::arg("size") = 1, py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "fetchall",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchAll();
	    },
	    "Fetch all rows from a result following execute", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "fetchnumpy",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchNumpy();
	    },
	    "Fetch a result as list of NumPy arrays following execute", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "fetchdf",
	    [](bool date_as_object, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchDF(date_as_object);
	    },
	    "Fetch a result as DataFrame following execute()", py::kw_only(), py::arg("date_as_object") = false,
	    py::arg("connection") = py::none());
	m.def(
	    "fetch_df",
	    [](bool date_as_object, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchDF(date_as_object);
	    },
	    "Fetch a result as DataFrame following execute()", py::kw_only(), py::arg("date_as_object") = false,
	    py::arg("connection") = py::none());
	m.def(
	    "df",
	    [](bool date_as_object, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchDF(date_as_object);
	    },
	    "Fetch a result as DataFrame following execute()", py::kw_only(), py::arg("date_as_object") = false,
	    py::arg("connection") = py::none());
	m.def(
	    "fetch_df_chunk",
	    [](const idx_t vectors_per_chunk = 1, bool date_as_object = false,
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchDFChunk(vectors_per_chunk, date_as_object);
	    },
	    "Fetch a chunk of the result as DataFrame following execute()", py::arg("vectors_per_chunk") = 1, py::kw_only(),
	    py::arg("date_as_object") = false, py::arg("connection") = py::none());
	m.def(
	    "pl",
	    [](idx_t rows_per_batch, bool lazy, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchPolars(rows_per_batch, lazy);
	    },
	    "Fetch a result as Polars DataFrame following execute()", py::arg("rows_per_batch") = 1000000, py::kw_only(),
	    py::arg("lazy") = false, py::arg("connection") = py::none());
	m.def(
	    "fetch_arrow_table",
	    [](idx_t rows_per_batch, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchArrow(rows_per_batch);
	    },
	    "Fetch a result as Arrow table following execute()", py::arg("rows_per_batch") = 1000000, py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "fetch_record_batch",
	    [](const idx_t rows_per_batch, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchRecordBatchReader(rows_per_batch);
	    },
	    "Fetch an Arrow RecordBatchReader following execute()", py::arg("rows_per_batch") = 1000000, py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "arrow",
	    [](const idx_t rows_per_batch, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchRecordBatchReader(rows_per_batch);
	    },
	    "Fetch an Arrow RecordBatchReader following execute()", py::arg("rows_per_batch") = 1000000, py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "torch",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchPyTorch();
	    },
	    "Fetch a result as dict of PyTorch Tensors following execute()", py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "tf",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchTF();
	    },
	    "Fetch a result as dict of TensorFlow Tensors following execute()", py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "begin",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Begin();
	    },
	    "Start a new transaction", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "commit",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Commit();
	    },
	    "Commit changes performed within a transaction", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "rollback",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Rollback();
	    },
	    "Roll back changes performed within a transaction", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "checkpoint",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Checkpoint();
	    },
	    "Synchronizes data in the write-ahead log (WAL) to the database data file (no-op for in-memory connections)",
	    py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "append",
	    [](const string &name, const PandasDataFrame &value, bool by_name,
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Append(name, value, by_name);
	    },
	    "Append the passed DataFrame to the named table", py::arg("table_name"), py::arg("df"), py::kw_only(),
	    py::arg("by_name") = false, py::arg("connection") = py::none());
	m.def(
	    "register",
	    [](const string &name, const py::object &python_object, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->RegisterPythonObject(name, python_object);
	    },
	    "Register the passed Python Object value for querying with a view", py::arg("view_name"),
	    py::arg("python_object"), py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "unregister",
	    [](const string &name, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->UnregisterPythonObject(name);
	    },
	    "Unregister the view name", py::arg("view_name"), py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "table",
	    [](const string &tname, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Table(tname);
	    },
	    "Create a relation object for the named table", py::arg("table_name"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "view",
	    [](const string &vname, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->View(vname);
	    },
	    "Create a relation object for the named view", py::arg("view_name"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "values",
	    [](const py::args &params, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Values(params);
	    },
	    "Create a relation object from the passed values", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "table_function",
	    [](const string &fname, py::object params = py::list(), shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->TableFunction(fname, params);
	    },
	    "Create a relation object from the named table function with given parameters", py::arg("name"),
	    py::arg("parameters") = py::none(), py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "read_json",
	    [](const py::object &name, const Optional<py::object> &columns = py::none(),
	       const Optional<py::object> &sample_size = py::none(), const Optional<py::object> &maximum_depth = py::none(),
	       const Optional<py::str> &records = py::none(), const Optional<py::str> &format = py::none(),
	       const Optional<py::object> &date_format = py::none(),
	       const Optional<py::object> &timestamp_format = py::none(),
	       const Optional<py::object> &compression = py::none(),
	       const Optional<py::object> &maximum_object_size = py::none(),
	       const Optional<py::object> &ignore_errors = py::none(),
	       const Optional<py::object> &convert_strings_to_integers = py::none(),
	       const Optional<py::object> &field_appearance_threshold = py::none(),
	       const Optional<py::object> &map_inference_threshold = py::none(),
	       const Optional<py::object> &maximum_sample_files = py::none(),
	       const Optional<py::object> &filename = py::none(),
	       const Optional<py::object> &hive_partitioning = py::none(),
	       const Optional<py::object> &union_by_name = py::none(), const Optional<py::object> &hive_types = py::none(),
	       const Optional<py::object> &hive_types_autocast = py::none(),
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->ReadJSON(name, columns, sample_size, maximum_depth, records, format, date_format,
		                          timestamp_format, compression, maximum_object_size, ignore_errors,
		                          convert_strings_to_integers, field_appearance_threshold, map_inference_threshold,
		                          maximum_sample_files, filename, hive_partitioning, union_by_name, hive_types,
		                          hive_types_autocast);
	    },
	    "Create a relation object from the JSON file in 'name'", py::arg("path_or_buffer"), py::kw_only(),
	    py::arg("columns") = py::none(), py::arg("sample_size") = py::none(), py::arg("maximum_depth") = py::none(),
	    py::arg("records") = py::none(), py::arg("format") = py::none(), py::arg("date_format") = py::none(),
	    py::arg("timestamp_format") = py::none(), py::arg("compression") = py::none(),
	    py::arg("maximum_object_size") = py::none(), py::arg("ignore_errors") = py::none(),
	    py::arg("convert_strings_to_integers") = py::none(), py::arg("field_appearance_threshold") = py::none(),
	    py::arg("map_inference_threshold") = py::none(), py::arg("maximum_sample_files") = py::none(),
	    py::arg("filename") = py::none(), py::arg("hive_partitioning") = py::none(),
	    py::arg("union_by_name") = py::none(), py::arg("hive_types") = py::none(),
	    py::arg("hive_types_autocast") = py::none(), py::arg("connection") = py::none());
	m.def(
	    "extract_statements",
	    [](const string &query, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->ExtractStatements(query);
	    },
	    "Parse the query string and extract the Statement object(s) produced", py::arg("query"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "sql",
	    [](const py::object &query, string alias = "", py::object params = py::list(),
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->RunQuery(query, alias, params);
	    },
	    "Run a SQL query. If it is a SELECT statement, create a relation object from the given SQL query, otherwise "
	    "run the query as-is.",
	    py::arg("query"), py::kw_only(), py::arg("alias") = "", py::arg("params") = py::none(),
	    py::arg("connection") = py::none());
	m.def(
	    "query",
	    [](const py::object &query, string alias = "", py::object params = py::list(),
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->RunQuery(query, alias, params);
	    },
	    "Run a SQL query. If it is a SELECT statement, create a relation object from the given SQL query, otherwise "
	    "run the query as-is.",
	    py::arg("query"), py::kw_only(), py::arg("alias") = "", py::arg("params") = py::none(),
	    py::arg("connection") = py::none());
	m.def(
	    "from_query",
	    [](const py::object &query, string alias = "", py::object params = py::list(),
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->RunQuery(query, alias, params);
	    },
	    "Run a SQL query. If it is a SELECT statement, create a relation object from the given SQL query, otherwise "
	    "run the query as-is.",
	    py::arg("query"), py::kw_only(), py::arg("alias") = "", py::arg("params") = py::none(),
	    py::arg("connection") = py::none());
	m.def(
	    "read_csv",
	    [](const py::object &name, py::kwargs &kwargs) {
		    auto connection_arg = kwargs.contains("conn") ? kwargs["conn"] : py::none();
		    auto conn = py::cast<shared_ptr<DuckDBPyConnection>>(connection_arg);

		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->ReadCSV(name, kwargs);
	    },
	    "Create a relation object from the CSV file in 'name'", py::arg("path_or_buffer"), py::kw_only());
	m.def(
	    "from_csv_auto",
	    [](const py::object &name, py::kwargs &kwargs) {
		    auto connection_arg = kwargs.contains("conn") ? kwargs["conn"] : py::none();
		    auto conn = py::cast<shared_ptr<DuckDBPyConnection>>(connection_arg);

		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->ReadCSV(name, kwargs);
	    },
	    "Create a relation object from the CSV file in 'name'", py::arg("path_or_buffer"), py::kw_only());
	m.def(
	    "from_df",
	    [](const PandasDataFrame &value, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromDF(value);
	    },
	    "Create a relation object from the DataFrame in df", py::arg("df"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "from_arrow",
	    [](py::object &arrow_object, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromArrow(arrow_object);
	    },
	    "Create a relation object from an Arrow object", py::arg("arrow_object"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "from_parquet",
	    [](const string &file_glob, bool binary_as_string, bool file_row_number, bool filename, bool hive_partitioning,
	       bool union_by_name, const py::object &compression = py::none(),
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromParquet(file_glob, binary_as_string, file_row_number, filename, hive_partitioning,
		                             union_by_name, compression);
	    },
	    "Create a relation object from the Parquet files in file_glob", py::arg("file_glob"),
	    py::arg("binary_as_string") = false, py::kw_only(), py::arg("file_row_number") = false,
	    py::arg("filename") = false, py::arg("hive_partitioning") = false, py::arg("union_by_name") = false,
	    py::arg("compression") = py::none(), py::arg("connection") = py::none());
	m.def(
	    "read_parquet",
	    [](const string &file_glob, bool binary_as_string, bool file_row_number, bool filename, bool hive_partitioning,
	       bool union_by_name, const py::object &compression = py::none(),
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromParquet(file_glob, binary_as_string, file_row_number, filename, hive_partitioning,
		                             union_by_name, compression);
	    },
	    "Create a relation object from the Parquet files in file_glob", py::arg("file_glob"),
	    py::arg("binary_as_string") = false, py::kw_only(), py::arg("file_row_number") = false,
	    py::arg("filename") = false, py::arg("hive_partitioning") = false, py::arg("union_by_name") = false,
	    py::arg("compression") = py::none(), py::arg("connection") = py::none());
	m.def(
	    "from_parquet",
	    [](const vector<string> &file_globs, bool binary_as_string, bool file_row_number, bool filename,
	       bool hive_partitioning, bool union_by_name, const py::object &compression = py::none(),
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromParquets(file_globs, binary_as_string, file_row_number, filename, hive_partitioning,
		                              union_by_name, compression);
	    },
	    "Create a relation object from the Parquet files in file_globs", py::arg("file_globs"),
	    py::arg("binary_as_string") = false, py::kw_only(), py::arg("file_row_number") = false,
	    py::arg("filename") = false, py::arg("hive_partitioning") = false, py::arg("union_by_name") = false,
	    py::arg("compression") = py::none(), py::arg("connection") = py::none());
	m.def(
	    "read_parquet",
	    [](const vector<string> &file_globs, bool binary_as_string, bool file_row_number, bool filename,
	       bool hive_partitioning, bool union_by_name, const py::object &compression = py::none(),
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromParquets(file_globs, binary_as_string, file_row_number, filename, hive_partitioning,
		                              union_by_name, compression);
	    },
	    "Create a relation object from the Parquet files in file_globs", py::arg("file_globs"),
	    py::arg("binary_as_string") = false, py::kw_only(), py::arg("file_row_number") = false,
	    py::arg("filename") = false, py::arg("hive_partitioning") = false, py::arg("union_by_name") = false,
	    py::arg("compression") = py::none(), py::arg("connection") = py::none());
	m.def(
	    "get_table_names",
	    [](const string &query, bool qualified, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->GetTableNames(query, qualified);
	    },
	    "Extract the required table names from a query", py::arg("query"), py::kw_only(), py::arg("qualified") = false,
	    py::arg("connection") = py::none());
	m.def(
	    "install_extension",
	    [](const string &extension, bool force_install = false, const py::object &repository = py::none(),
	       const py::object &repository_url = py::none(), const py::object &version = py::none(),
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    conn->InstallExtension(extension, force_install, repository, repository_url, version);
	    },
	    "Install an extension by name, with an optional version and/or repository to get the extension from",
	    py::arg("extension"), py::kw_only(), py::arg("force_install") = false, py::arg("repository") = py::none(),
	    py::arg("repository_url") = py::none(), py::arg("version") = py::none(), py::arg("connection") = py::none());
	m.def(
	    "load_extension",
	    [](const string &extension, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    conn->LoadExtension(extension);
	    },
	    "Load an installed extension", py::arg("extension"), py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "project",
	    [](const PandasDataFrame &df, const py::args &args, const string &groups = "",
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromDF(df)->Project(args, groups);
	    },
	    "Project the relation object by the projection in project_expr", py::arg("df"), py::kw_only(),
	    py::arg("groups") = "", py::arg("connection") = py::none());
	m.def(
	    "distinct",
	    [](const PandasDataFrame &df, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromDF(df)->Distinct();
	    },
	    "Retrieve distinct rows from this relation object", py::arg("df"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "write_csv",
	    [](const PandasDataFrame &df, const string &filename, const py::object &sep = py::none(),
	       const py::object &na_rep = py::none(), const py::object &header = py::none(),
	       const py::object &quotechar = py::none(), const py::object &escapechar = py::none(),
	       const py::object &date_format = py::none(), const py::object &timestamp_format = py::none(),
	       const py::object &quoting = py::none(), const py::object &encoding = py::none(),
	       const py::object &compression = py::none(), const py::object &overwrite = py::none(),
	       const py::object &per_thread_output = py::none(), const py::object &use_tmp_file = py::none(),
	       const py::object &partition_by = py::none(), const py::object &write_partition_columns = py::none(),
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    conn->FromDF(df)->ToCSV(filename, sep, na_rep, header, quotechar, escapechar, date_format, timestamp_format,
		                            quoting, encoding, compression, overwrite, per_thread_output, use_tmp_file,
		                            partition_by, write_partition_columns);
	    },
	    "Write the relation object to a CSV file in 'file_name'", py::arg("df"), py::arg("filename"), py::kw_only(),
	    py::arg("sep") = py::none(), py::arg("na_rep") = py::none(), py::arg("header") = py::none(),
	    py::arg("quotechar") = py::none(), py::arg("escapechar") = py::none(), py::arg("date_format") = py::none(),
	    py::arg("timestamp_format") = py::none(), py::arg("quoting") = py::none(), py::arg("encoding") = py::none(),
	    py::arg("compression") = py::none(), py::arg("overwrite") = py::none(),
	    py::arg("per_thread_output") = py::none(), py::arg("use_tmp_file") = py::none(),
	    py::arg("partition_by") = py::none(), py::arg("write_partition_columns") = py::none(),
	    py::arg("connection") = py::none());
	m.def(
	    "aggregate",
	    [](const PandasDataFrame &df, const py::object &expr, const string &groups = "",
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromDF(df)->Aggregate(expr, groups);
	    },
	    "Compute the aggregate aggr_expr by the optional groups group_expr on the relation", py::arg("df"),
	    py::arg("aggr_expr"), py::arg("group_expr") = "", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "alias",
	    [](const PandasDataFrame &df, const string &expr, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromDF(df)->SetAlias(expr);
	    },
	    "Rename the relation object to new alias", py::arg("df"), py::arg("alias"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "filter",
	    [](const PandasDataFrame &df, const py::object &expr, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromDF(df)->Filter(expr);
	    },
	    "Filter the relation object by the filter in filter_expr", py::arg("df"), py::arg("filter_expr"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "limit",
	    [](const PandasDataFrame &df, int64_t n, int64_t offset = 0, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromDF(df)->Limit(n, offset);
	    },
	    "Only retrieve the first n rows from this relation object, starting at offset", py::arg("df"), py::arg("n"),
	    py::arg("offset") = 0, py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "order",
	    [](const PandasDataFrame &df, const string &expr, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromDF(df)->Order(expr);
	    },
	    "Reorder the relation object by order_expr", py::arg("df"), py::arg("order_expr"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "query_df",
	    [](const PandasDataFrame &df, const string &view_name, const string &sql_query,
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromDF(df)->Query(view_name, sql_query);
	    },
	    "Run the given SQL query in sql_query on the view named virtual_table_name that refers to the relation object",
	    py::arg("df"), py::arg("virtual_table_name"), py::arg("sql_query"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "description",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->GetDescription();
	    },
	    "Get result set attributes, mainly column names", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "rowcount",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->GetRowcount();
	    },
	    "Get result set row count", py::kw_only(), py::arg("connection") = py::none());

	// END_OF_CONNECTION_METHODS

	// We define these "wrapper" methods manually because they are overloaded to return relations
	m.def(
	    "arrow",
	    [](py::object &arrow_object, shared_ptr<DuckDBPyConnection> conn) -> unique_ptr<DuckDBPyRelation> {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromArrow(arrow_object);
	    },
	    "Create a relation object from an Arrow object", py::arg("arrow_object"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "df",
	    [](const PandasDataFrame &value, shared_ptr<DuckDBPyConnection> conn) -> unique_ptr<DuckDBPyRelation> {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromDF(value);
	    },
	    "Create a relation object from the DataFrame df", py::arg("df"), py::kw_only(),
	    py::arg("connection") = py::none());
}

PYBIND11_MODULE(DUCKDB_PYTHON_LIB_NAME, m) { // NOLINT
	py::enum_<duckdb::ExplainType>(m, "ExplainType",
	                               "Enumeration for different explain types.\n\n"
	                               "This enum lists the available execution plan explanation types.\n"
	                               "Different types provide different levels of detail about query execution.")
	    .value("STANDARD", duckdb::ExplainType::EXPLAIN_STANDARD,
	           "Standard explain output showing the logical query plan.")
	    .value("ANALYZE", duckdb::ExplainType::EXPLAIN_ANALYZE,
	           "Analyze explain with execution statistics and timing information.");

	py::enum_<duckdb::StatementReturnType>(m, "ExpectedResultType",
	                                       "Enumeration for different result types.\n\n"
	                                       "This enum lists the possible result types from queries.")
	    .value("QUERY_RESULT", duckdb::StatementReturnType::QUERY_RESULT)
	    .value("CHANGED_ROWS", duckdb::StatementReturnType::CHANGED_ROWS)
	    .value("NOTHING", duckdb::StatementReturnType::NOTHING);

	py::enum_<duckdb::StatementType>(m, "StatementType",
	                                 "Enumeration for different SQL statement types.\n\n"
	                                 "Identifies the type of SQL statement being executed, which is useful\n"
	                                 "for query analysis, logging, performance monitoring, and implementing\n"
	                                 "different execution strategies based on statement characteristics.")
	    .value("INVALID_STATEMENT", duckdb::StatementType::INVALID_STATEMENT,
	           "Invalid or unrecognized statement.\n"
	           "Represents a statement that could not be parsed or is malformed.")
	    .value("SELECT_STATEMENT", duckdb::StatementType::SELECT_STATEMENT,
	           "Data retrieval statement (SELECT).\n"
	           "Queries that read data from tables, views, or other relations\n"
	           "without modifying the database state.")
	    .value("INSERT_STATEMENT", duckdb::StatementType::INSERT_STATEMENT,
	           "Data insertion statement (INSERT).\n"
	           "Statements that add new rows to tables, including INSERT INTO\n"
	           "and INSERT ... SELECT operations.")
	    .value("UPDATE_STATEMENT", duckdb::StatementType::UPDATE_STATEMENT,
	           "Data modification statement (UPDATE).\n"
	           "Statements that modify existing rows in tables based on\n"
	           "specified conditions and SET clauses.")
	    .value("CREATE_STATEMENT", duckdb::StatementType::CREATE_STATEMENT,
	           "Object creation statement (CREATE).\n"
	           "Statements that create new database objects such as tables,\n"
	           "views, indexes, schemas, or other database structures.")
	    .value("DELETE_STATEMENT", duckdb::StatementType::DELETE_STATEMENT,
	           "Data deletion statement (DELETE).\n"
	           "Statements that remove rows from tables based on\n"
	           "specified WHERE conditions.")
	    .value("PREPARE_STATEMENT", duckdb::StatementType::PREPARE_STATEMENT,
	           "Statement preparation (PREPARE).\n"
	           "Statements that prepare SQL statements for repeated execution\n"
	           "with different parameter values.")
	    .value("EXECUTE_STATEMENT", duckdb::StatementType::EXECUTE_STATEMENT,
	           "Prepared statement execution (EXECUTE).\n"
	           "Statements that execute previously prepared statements\n"
	           "with specific parameter bindings.")
	    .value("ALTER_STATEMENT", duckdb::StatementType::ALTER_STATEMENT,
	           "Object alteration statement (ALTER).\n"
	           "Statements that modify the structure or properties of existing\n"
	           "database objects like tables, columns, or constraints.")
	    .value("TRANSACTION_STATEMENT", duckdb::StatementType::TRANSACTION_STATEMENT,
	           "Transaction control statement.\n"
	           "Statements that control transaction boundaries such as\n"
	           "BEGIN, COMMIT, ROLLBACK, and SAVEPOINT operations.")
	    .value("COPY_STATEMENT", duckdb::StatementType::COPY_STATEMENT,
	           "Data import/export statement (COPY).\n"
	           "Statements that bulk load data from files into tables\n"
	           "or export table data to files.")
	    .value("ANALYZE_STATEMENT", duckdb::StatementType::ANALYZE_STATEMENT,
	           "Statistics collection statement (ANALYZE).\n"
	           "Statements that gather or update table statistics used\n"
	           "by the query optimizer for better execution plans.")
	    .value("VARIABLE_SET_STATEMENT", duckdb::StatementType::VARIABLE_SET_STATEMENT,
	           "Variable assignment statement.\n"
	           "Statements that set or modify session or global variables\n"
	           "that affect database behavior or query execution.")
	    .value("CREATE_FUNC_STATEMENT", duckdb::StatementType::CREATE_FUNC_STATEMENT,
	           "Function creation statement (CREATE FUNCTION).\n"
	           "Statements that create user-defined functions, including\n"
	           "scalar functions, aggregate functions, and table functions.")
	    .value("EXPLAIN_STATEMENT", duckdb::StatementType::EXPLAIN_STATEMENT,
	           "Query plan explanation statement (EXPLAIN).\n"
	           "Statements that show the execution plan for other SQL statements\n"
	           "without actually executing them.")
	    .value("DROP_STATEMENT", duckdb::StatementType::DROP_STATEMENT,
	           "Object deletion statement (DROP).\n"
	           "Statements that remove database objects such as tables,\n"
	           "views, indexes, functions, or schemas from the database.")
	    .value("EXPORT_STATEMENT", duckdb::StatementType::EXPORT_STATEMENT,
	           "Database export statement (EXPORT).\n"
	           "Statements that export entire databases or large portions\n"
	           "of database content to external formats.")
	    .value("PRAGMA_STATEMENT", duckdb::StatementType::PRAGMA_STATEMENT,
	           "System configuration statement (PRAGMA).\n"
	           "Statements that query or modify database engine settings,\n"
	           "configuration options, and system parameters.")
	    .value("VACUUM_STATEMENT", duckdb::StatementType::VACUUM_STATEMENT,
	           "Database maintenance statement (VACUUM).\n"
	           "Statements that perform database cleanup, optimization,\n"
	           "and space reclamation operations.")
	    .value("CALL_STATEMENT", duckdb::StatementType::CALL_STATEMENT,
	           "Procedure or function call statement (CALL).\n"
	           "Statements that invoke stored procedures, functions,\n"
	           "or other callable database objects.")
	    .value("SET_STATEMENT", duckdb::StatementType::SET_STATEMENT,
	           "Configuration setting statement (SET).\n"
	           "Statements that modify session-level or connection-level\n"
	           "configuration parameters and options.")
	    .value("LOAD_STATEMENT", duckdb::StatementType::LOAD_STATEMENT,
	           "Extension or module loading statement (LOAD).\n"
	           "Statements that load extensions, plugins, or external\n"
	           "modules to extend database functionality.")
	    .value("RELATION_STATEMENT", duckdb::StatementType::RELATION_STATEMENT,
	           "Relation-based statement.\n"
	           "Statements that operate on relation objects or perform\n"
	           "relational algebra operations programmatically.")
	    .value("EXTENSION_STATEMENT", duckdb::StatementType::EXTENSION_STATEMENT,
	           "Extension management statement.\n"
	           "Statements that install, configure, or manage database\n"
	           "extensions and their associated functionality.")
	    .value("LOGICAL_PLAN_STATEMENT", duckdb::StatementType::LOGICAL_PLAN_STATEMENT,
	           "Logical plan examination statement.\n"
	           "Statements that expose or manipulate the logical query plan\n"
	           "representation used internally by the query engine.")
	    .value("ATTACH_STATEMENT", duckdb::StatementType::ATTACH_STATEMENT,
	           "Database attachment statement (ATTACH).\n"
	           "Statements that attach external databases or data sources\n"
	           "to the current database connection for cross-database queries.")
	    .value("DETACH_STATEMENT", duckdb::StatementType::DETACH_STATEMENT,
	           "Database detachment statement (DETACH).\n"
	           "Statements that detach previously attached databases\n"
	           "or data sources from the current connection.")
	    .value("MULTI_STATEMENT", duckdb::StatementType::MULTI_STATEMENT,
	           "Multiple statement batch.\n"
	           "Represents a batch containing multiple SQL statements\n"
	           "that are executed together as a single unit.")
	    .value("COPY_DATABASE_STATEMENT", duckdb::StatementType::COPY_DATABASE_STATEMENT,
	           "Database copying statement.\n"
	           "Statements that copy entire databases or significant\n"
	           "portions of database content to another location.")
	    .value("MERGE_INTO_STATEMENT", duckdb::StatementType::MERGE_INTO_STATEMENT,
	           "Data merging statement (MERGE INTO).\n"
	           "Statements that perform conditional insert, update, or delete\n"
	           "operations based on matching conditions between source and target tables.");

	py::enum_<duckdb::PythonCSVLineTerminator::Type>(
	    m, "CSVLineTerminator",
	    "Enumeration for CSV line terminator types.\n\n"
	    "Specifies the character sequence used to terminate lines in CSV files.\n"
	    "Different operating systems and applications use different conventions\n"
	    "for line endings, and this enum allows explicit control over parsing.")
	    .value("LINE_FEED", duckdb::PythonCSVLineTerminator::Type::LINE_FEED,
	           "Unix-style line terminator using only Line Feed (\\n).\n"
	           "This is the standard line ending on Unix-like systems including Linux and macOS.")
	    .value("CARRIAGE_RETURN_LINE_FEED", duckdb::PythonCSVLineTerminator::Type::CARRIAGE_RETURN_LINE_FEED,
	           "Windows-style line terminator using Carriage Return + Line Feed (\\r\\n).\n"
	           "This is the standard line ending on Windows systems and some network protocols.");

	py::enum_<duckdb::PythonExceptionHandling>(m, "PythonExceptionHandling",
	                                           "Enumeration for Python exception handling strategies.\n\n"
	                                           "Controls how exceptions that occur during Python function execution\n"
	                                           "are handled within DuckDB queries. This affects the behavior when\n"
	                                           "user-defined functions or Python expressions encounter errors.")
	    .value("DEFAULT", duckdb::PythonExceptionHandling::FORWARD_ERROR,
	           "Forward exceptions to the caller (default behavior).\n"
	           "When a Python function raises an exception, it will be propagated\n"
	           "and cause the entire query to fail with an error message.")
	    .value("RETURN_NULL", duckdb::PythonExceptionHandling::RETURN_NULL,
	           "Return NULL when Python functions raise exceptions.\n"
	           "Instead of failing the query, exceptions in Python functions\n"
	           "will result in NULL values, allowing the query to continue execution.");

	py::enum_<duckdb::RenderMode>(m, "RenderMode",
	                              "Enumeration for result rendering modes.\n\n"
	                              "Controls how query results are formatted and displayed.\n"
	                              "Different modes optimize for different use cases, such as\n"
	                              "human readability versus programmatic processing.")
	    .value("ROWS", duckdb::RenderMode::ROWS,
	           "Row-oriented rendering mode.\n"
	           "Results are displayed with each row on a separate line,\n"
	           "which is the traditional tabular format optimized for readability.")
	    .value("COLUMNS", duckdb::RenderMode::COLUMNS,
	           "Column-oriented rendering mode.\n"
	           "Results are displayed with columns grouped together,\n"
	           "which can be more efficient for wide tables or analytical workflows.");

	DuckDBPyTyping::Initialize(m);
	DuckDBPyFunctional::Initialize(m);
	DuckDBPyExpression::Initialize(m);
	DuckDBPyStatement::Initialize(m);
	DuckDBPyRelation::Initialize(m);
	DuckDBPyConnection::Initialize(m);
	PythonObject::Initialize();

	py::options pybind_opts;

	m.doc() = "DuckDB is an embeddable SQL OLAP Database Management System.";
	m.attr("__package__") = "duckdb";
	m.attr("__version__") = std::string(DuckDB::LibraryVersion()).substr(1);
	m.attr("__standard_vector_size__") = DuckDB::StandardVectorSize();
	m.attr("__git_revision__") = DuckDB::SourceID();
	m.attr("__interactive__") = DuckDBPyConnection::DetectAndGetEnvironment();
	m.attr("__jupyter__") = DuckDBPyConnection::IsJupyter();
	m.attr("__formatted_python_version__") = DuckDBPyConnection::FormattedPythonVersion();
	m.def("default_connection", &DuckDBPyConnection::DefaultConnection,
	      "Retrieve the connection currently registered as the default to be used by the module");
	m.def("set_default_connection", &DuckDBPyConnection::SetDefaultConnection,
	      "Register the provided connection as the default to be used by the module",
	      py::arg("connection").none(false));
	m.attr("apilevel") = "2.0";
	m.attr("threadsafety") = 1;
	m.attr("paramstyle") = "qmark";

	InitializeConnectionMethods(m);

	RegisterExceptions(m);

	m.def("connect", &DuckDBPyConnection::Connect,
	      "Create a DuckDB database instance. Can take a database file name to read/write persistent data and a "
	      "read_only flag if no changes are desired",
	      py::arg("database") = ":memory:", py::arg("read_only") = false, py::arg_v("config", py::dict(), "None"));
	m.def("tokenize", PyTokenize,
	      "Tokenizes a SQL string, returning a list of (position, type) tuples that can be "
	      "used for e.g., syntax highlighting",
	      py::arg("query"));
	py::enum_<PySQLTokenType>(m, "token_type", py::module_local(),
	                          "Enumeration for SQL token types used in lexical analysis.\n\n"
	                          "Represents the different categories of tokens that can be identified\n"
	                          "when parsing or tokenizing SQL statements. This is useful for syntax\n"
	                          "highlighting, query analysis, and building SQL parsing tools.")
	    .value("identifier", PySQLTokenType::PY_SQL_TOKEN_IDENTIFIER,
	           "SQL identifier token.\n"
	           "Represents names of database objects such as table names, column names,\n"
	           "function names, aliases, and other user-defined identifiers.")
	    .value("numeric_const", PySQLTokenType::PY_SQL_TOKEN_NUMERIC_CONSTANT,
	           "Numeric constant token.\n"
	           "Represents literal numeric values including integers, floating-point numbers,\n"
	           "decimal numbers, and scientific notation in SQL statements.")
	    .value("string_const", PySQLTokenType::PY_SQL_TOKEN_STRING_CONSTANT,
	           "String constant token.\n"
	           "Represents literal string values enclosed in single or double quotes,\n"
	           "including escape sequences and multi-line string literals.")
	    .value("operator", PySQLTokenType::PY_SQL_TOKEN_OPERATOR,
	           "SQL operator token.\n"
	           "Represents operators such as arithmetic (+, -, *, /), comparison (=, <, >),\n"
	           "logical (AND, OR, NOT), and other SQL operators including assignment.")
	    .value("keyword", PySQLTokenType::PY_SQL_TOKEN_KEYWORD,
	           "SQL keyword token.\n"
	           "Represents reserved SQL keywords such as SELECT, FROM, WHERE, INSERT,\n"
	           "CREATE, ALTER, and other language constructs defined by the SQL standard.")
	    .value("comment", PySQLTokenType::PY_SQL_TOKEN_COMMENT,
	           "SQL comment token.\n"
	           "Represents comment text in SQL statements, including single-line comments\n"
	           "(-- comment) and multi-line comments (/* comment */) used for documentation.")
	    .export_values();

	// we need this because otherwise we try to remove registered_dfs on shutdown when python is already dead
	auto clean_default_connection = []() {
		DuckDBPyConnection::Cleanup();
	};
	m.add_object("_clean_default_connection", py::capsule(clean_default_connection));
}

} // namespace duckdb
