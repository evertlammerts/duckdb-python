//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/registry.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nanobind/nanobind.h>

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "lifetime.hpp"

namespace duckdb_python {

/// A Python object registered under a table name, read through the scan function by that name.
///
/// The registry owns the object for as long as the name is registered; a plan bound over the name holds its own
/// reference, so an unregister cannot pull the object from under a query already bound.
struct Registered {
	Registered(std::string name, nb::object object, bool one_shot);
	/// Freed from engine threads too, so the Python reference is dropped under the GIL.
	~Registered();

	std::string name;
	nb::object object;
	/// A stream: it can be read once, so a second scan is refused rather than silently empty.
	bool one_shot;
	std::mutex read_lock;
	bool read = false;
};

/// The objects registered on one database, by name.
class Registry {
public:
	/// Register `object` as `name`, replacing an earlier registration of that name. Names compare case-insensitively,
	/// as SQL identifiers do.
	void Add(const std::string &name, nb::object object, bool one_shot);
	/// Forget `name`; false when it was not registered.
	bool Remove(const std::string &name);
	std::shared_ptr<Registered> ByName(const std::string &name);
	/// The registered objects, for the garbage collector's visit.
	std::vector<nb::handle> Objects();
	void Clear();

private:
	std::mutex lock;
	std::unordered_map<std::string, std::shared_ptr<Registered>> by_name;
};

/// Add the scan function and the replacement scan that resolves registered names to the database; called once,
/// before any other connection exists, since an instance-wide replacement scan may not be added while queries bind.
void InstallRegistryScan(cxx::Instance &instance, std::shared_ptr<Registry> registry);

} // namespace duckdb_python
