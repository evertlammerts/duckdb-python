//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/pyresult.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb_python/numpy/numpy_result_conversion.hpp"
#include "duckdb.hpp"
#include "duckdb_python/engine/result.hpp"
#include "duckdb_python/nb/casters.hpp"
#include "duckdb_python/python_objects.hpp"
#include "duckdb_python/dataframe.hpp"

namespace duckdb {

struct DuckDBPyResult {
public:
	static constexpr idx_t DEFAULT_ARROW_BATCH_SIZE = 1000000;

public:
	explicit DuckDBPyResult(shared_ptr<engine::Result> result);
	~DuckDBPyResult();

public:
	Optional<nb::tuple> Fetchone();

	nb::list Fetchmany(idx_t size);

	nb::list Fetchall();

	nb::dict FetchNumpy();

	nb::dict FetchNumpyInternal(bool chunked = false, idx_t vectors_per_chunk = 1,
	                            std::unique_ptr<NumpyResultConversion> conversion = nullptr);

	PandasDataFrame FetchDF(bool date_as_object);

	PandasDataFrame FetchDFChunk(const idx_t vectors_per_chunk = 1, bool date_as_object = false);

	nb::dict FetchPyTorch();

	nb::dict FetchTF();

	duckdb::pyarrow::Table FetchArrowTable(idx_t rows_per_batch, bool to_polars);
	duckdb::pyarrow::RecordBatchReader FetchRecordBatchReader(idx_t rows_per_batch = DEFAULT_ARROW_BATCH_SIZE);
	nb::object FetchArrowCapsule(idx_t rows_per_batch = DEFAULT_ARROW_BATCH_SIZE);
	static void CheckBatchSize(idx_t rows_per_batch);

	static nb::list GetDescription(const vector<string> &names, const vector<LogicalType> &types);

	void Close();

	unique_ptr<DataChunk> FetchChunk();

	vector<string> GetNames();
	const vector<LogicalType> &GetTypes() const;

	const ClientProperties &GetClientProperties() const;

private:
	void FillNumpy(nb::dict &res, idx_t col_idx, NumpyResultConversion &conversion, const char *name);

	PandasDataFrame FrameFromNumpy(bool date_as_object, const nb::handle &o);

	void ConvertDateTimeTypes(PandasDataFrame &df, bool date_as_object) const;
	//! The names the Python layer reports, see the definition for why this is not always the result's own.
	const vector<Identifier> &ResultNames() const;
	bool Empty() const {
		return !result;
	}
	//! Flat vectors, for the row fetch
	unique_ptr<DataChunk> FetchNext();
	unique_ptr<DataChunk> FetchNextRaw();
	//! Rows a row fetch popped but has not returned yet. Once the stream's query has ended underneath them
	//! they are dropped, so that the next fetch reports why.
	unique_ptr<DataChunk> TakeBufferedRows();
	std::unique_ptr<NumpyResultConversion> InitializeNumpyConversion(bool pandas = false);

	//! A result in the chunk format is scanned again by the engine in the Arrow format
	void PromoteToArrow(idx_t batch_size);

	template <typename T>
	T RunWithArrowSchema(const std::function<T(const ArrowSchema &)> &fun, bool dedup_col_names);
	//! The stream's private data owns the result, and its callbacks run without the GIL.
	ArrowArrayStream FetchArrowArrayStream(idx_t rows_per_batch);

private:
	shared_ptr<engine::Result> result;
	idx_t chunk_offset = 0;
	//! Set only when the result was re-bound (promotion to Arrow de-duplicates column names
	//! and core exposes no setter), so the original names survive. Empty means "use result's".
	vector<Identifier> names_override;
	unique_ptr<DataChunk> current_chunk;
	// Holds the categories of Categorical/ENUM types
	unordered_map<idx_t, nb::list> categories;
	// Holds the categorical type of Categorical/ENUM types
	unordered_map<idx_t, nb::object> categories_type;
};

} // namespace duckdb
