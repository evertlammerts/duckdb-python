#pragma once

#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/function/table/arrow/arrow_duck_schema.hpp"

namespace duckdb {

//! Converts Arrow record batches into DataChunks of at most STANDARD_VECTOR_SIZE rows against one resolved schema, one
//! batch at a time, referencing the batch's buffers where the engine's conversion allows. The batch's own validity
//! bitmap makes a whole row NULL. Not thread safe: each scanning thread owns one.
class ArrowBatchImporter {
public:
	//! Output column i is read from batch column columns[i]
	ArrowBatchImporter(ClientContext &context, shared_ptr<ArrowTableSchema> schema, vector<idx_t> columns);

	//! Takes the batch over, clearing its release, after checking its shape against the schema; the previous batch
	//! is dropped, drained or not
	void Append(ArrowArray &batch);
	//! Fills `output`, whose columns match the requested ones, with the next rows of the current batch; false once
	//! the batch is drained
	bool Next(DataChunk &output);

private:
	ArrowArrayScanState &StateOf(idx_t column);

	ClientContext &context;
	const shared_ptr<ArrowTableSchema> schema;
	const arrow_column_map_t &arrow_columns;
	const vector<idx_t> columns;
	shared_ptr<ArrowArrayWrapper> batch;
	idx_t offset = 0;
	idx_t length = 0;
	//! Kept while one batch is drained, so its dictionaries are decoded once rather than once per chunk
	unordered_map<idx_t, unique_ptr<ArrowArrayScanState>> states;
};

} // namespace duckdb
