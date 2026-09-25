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

/// A Python object registered under a table name, read by that name through the table function in scan.hpp.
///
/// The registry owns the object for as long as the name is registered; a plan bound over the name holds its own
/// reference, so an unregister cannot pull the object from under a query already bound.
struct Registered {
	Registered(std::string name, nb::object object, bool one_shot, bool native);
	/// Freed from engine threads too, so the Python reference is dropped under the GIL.
	~Registered();

	std::string name;
	nb::object object;
	/// A stream: it can be read once, so a second query over it is refused rather than silently empty.
	bool one_shot;
	/// Whether the replacement scan resolves this name to the native pandas scan rather than the object scan.
	bool native;
	std::mutex read_lock;
	bool read = false;
};

/// The objects registered on one database, by name.
class Registry {
public:
	/// Register `object` as `name`, replacing an earlier registration of that name. Names compare case-insensitively,
	/// as SQL identifiers do.
	void Add(const std::string &name, nb::object object, bool one_shot, bool native);
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

/// Add the table function of scan.hpp and the replacement scan that resolves registered names to it; called once,
/// before any other connection exists, since an instance-wide replacement scan may not be added while queries bind.
void InstallRegistryScan(cxx::Instance &instance, std::shared_ptr<Registry> registry,
                         std::shared_ptr<ModuleState> module);

} // namespace duckdb_python
