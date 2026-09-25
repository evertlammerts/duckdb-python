//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/registry.cpp
//
//
//===----------------------------------------------------------------------===//

#include "registry.hpp"

#include "arrow_scan.hpp"
#include "numpy_scan.hpp"

#include <string>
#include <utility>

namespace duckdb_python {
namespace {

/// ASCII letters only, as the engine and the Python side fold identifiers; a locale-aware fold would rewrite the
/// bytes of a non-ASCII name.
std::string Fold(const std::string &name) {
	std::string folded = name;
	for (auto &c : folded) {
		if (c >= 'A' && c <= 'Z') {
			c = static_cast<char>(c - 'A' + 'a');
		}
	}
	return folded;
}

struct ReplacementUserData {
	std::shared_ptr<Registry> registry;
};

void ReplaceRegisteredName(cxx::ReplacementScan::Input &input) {
	auto &registry = *input.GetUserData<ReplacementUserData>().registry;
	const auto name = input.GetName();
	if (name.GetPartCount() != 1) {
		return;
	}
	auto entry = registry.ByName(std::string(name.GetPart(0)));
	if (!entry) {
		return;
	}
	input.SetFunctionName(entry->native ? kNumpyScanFunction : kArrowScanFunction);
	auto context = input.GetContext();
	input.AddArgument(context.CreateValue(cxx::varchar_t(name.GetPart(0))));
}

} // namespace

Registered::Registered(std::string name, nb::object object, bool one_shot, bool native)
    : name(std::move(name)), object(std::move(object)), one_shot(one_shot), native(native) {
}

Registered::~Registered() {
	nb::gil_scoped_acquire gil;
	object.reset();
}

void Registry::Add(const std::string &name, nb::object object, bool one_shot, bool native) {
	// The replaced entry, if any, is dropped outside the lock: its destructor takes the GIL, which a thread waiting
	// for this lock may hold.
	std::shared_ptr<Registered> replaced;
	{
		std::lock_guard<std::mutex> guard(lock);
		auto &slot = by_name[Fold(name)];
		replaced = std::move(slot);
		slot = std::make_shared<Registered>(name, std::move(object), one_shot, native);
	}
}

bool Registry::Remove(const std::string &name) {
	std::shared_ptr<Registered> removed;
	{
		std::lock_guard<std::mutex> guard(lock);
		auto it = by_name.find(Fold(name));
		if (it == by_name.end()) {
			return false;
		}
		removed = std::move(it->second);
		by_name.erase(it);
	}
	return true;
}

std::shared_ptr<Registered> Registry::ByName(const std::string &name) {
	std::lock_guard<std::mutex> guard(lock);
	auto it = by_name.find(Fold(name));
	return it == by_name.end() ? nullptr : it->second;
}

std::vector<nb::handle> Registry::Objects() {
	std::lock_guard<std::mutex> guard(lock);
	std::vector<nb::handle> objects;
	objects.reserve(by_name.size());
	for (const auto &[_, entry] : by_name) {
		objects.push_back(entry->object);
	}
	return objects;
}

void Registry::Clear() {
	std::vector<std::shared_ptr<Registered>> dropped;
	{
		std::lock_guard<std::mutex> guard(lock);
		dropped.reserve(by_name.size());
		for (auto &[_, entry] : by_name) {
			dropped.push_back(std::move(entry));
		}
		by_name.clear();
	}
}

void InstallRegistryScan(cxx::Instance &instance, std::shared_ptr<Registry> registry,
                         std::shared_ptr<ModuleState> module) {
	const auto batch_rows =
	    static_cast<cxx::idx_t>(std::stoull(std::string(instance.GetOption("standard_vector_size").GetValue())));
	auto connection = instance.Connect();
	RegisterArrowScan(connection, registry, module, batch_rows);
	RegisterNumpyScan(connection, registry, std::move(module), batch_rows);

	auto scan = cxx::ReplacementScan::Create(instance);
	scan.SetUserData<ReplacementUserData>(ReplacementUserData {std::move(registry)});
	scan.SetCallback(&ReplaceRegisteredName);
	scan.Register();
}

} // namespace duckdb_python
