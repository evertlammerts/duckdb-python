//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/predicate.cpp
//
//
//===----------------------------------------------------------------------===//

#include "predicate.hpp"

namespace duckdb_python {
namespace {

bool Temporal(cxx::LogicalTypeId type) {
	switch (type) {
	case cxx::LogicalTypeId::DATE:
	case cxx::LogicalTypeId::TIMESTAMP:
	case cxx::LogicalTypeId::TIMESTAMP_SEC:
	case cxx::LogicalTypeId::TIMESTAMP_MS:
	case cxx::LogicalTypeId::TIMESTAMP_TZ:
		return true;
	default:
		return false;
	}
}

/// The comparison's operator as the frame spells it.
const char *Operator(cxx::ExpressionType comparison) {
	switch (comparison) {
	case cxx::ExpressionType::COMPARE_EQUAL:
		return "=";
	case cxx::ExpressionType::COMPARE_NOTEQUAL:
		return "!=";
	case cxx::ExpressionType::COMPARE_LESSTHAN:
		return "<";
	case cxx::ExpressionType::COMPARE_GREATERTHAN:
		return ">";
	case cxx::ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return "<=";
	case cxx::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return ">=";
	default:
		throw Refused {};
	}
}

/// The comparison that holds with its two sides swapped.
cxx::ExpressionType Converse(cxx::ExpressionType comparison) {
	switch (comparison) {
	case cxx::ExpressionType::COMPARE_LESSTHAN:
		return cxx::ExpressionType::COMPARE_GREATERTHAN;
	case cxx::ExpressionType::COMPARE_GREATERTHAN:
		return cxx::ExpressionType::COMPARE_LESSTHAN;
	case cxx::ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return cxx::ExpressionType::COMPARE_GREATERTHANOREQUALTO;
	case cxx::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return cxx::ExpressionType::COMPARE_LESSTHANOREQUALTO;
	default:
		return comparison;
	}
}

/// Builds the nodes through the `duckdb._expressions` package, imported once per predicate.
class Walker {
public:
	Walker(const ColumnResolver &resolve, ConversionContext &conversion)
	    : resolve(resolve), conversion(conversion), nodes(nb::module_::import_("duckdb._expressions")) {
	}

	nb::object Predicate(const cxx::Expression &node) {
		// A BETWEEN node does not say whether its bounds are inclusive, and the optimizer folds `a < x AND x < b`
		// into one with exclusive bounds, so it is refused with everything else outside the set.
		switch (node.GetType()) {
		case cxx::ExpressionType::CONJUNCTION_AND:
			return Conjunction(node, "AND");
		case cxx::ExpressionType::CONJUNCTION_OR:
			return Conjunction(node, "OR");
		case cxx::ExpressionType::OPERATOR_NOT:
			return nodes.attr("Unary")("NOT", Predicate(node.GetChild(0)));
		case cxx::ExpressionType::OPERATOR_IS_NULL:
			return nodes.attr("Postfix")("IS NULL", Column(node.GetChild(0)).reference);
		case cxx::ExpressionType::OPERATOR_IS_NOT_NULL:
			return nodes.attr("Postfix")("IS NOT NULL", Column(node.GetChild(0)).reference);
		case cxx::ExpressionType::COMPARE_EQUAL:
		case cxx::ExpressionType::COMPARE_NOTEQUAL:
		case cxx::ExpressionType::COMPARE_LESSTHAN:
		case cxx::ExpressionType::COMPARE_GREATERTHAN:
		case cxx::ExpressionType::COMPARE_LESSTHANOREQUALTO:
		case cxx::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			return Comparison(node);
		case cxx::ExpressionType::COMPARE_IN: {
			auto column = Column(node.GetChild(0));
			nb::list candidates;
			for (cxx::idx_t i = 1; i < node.GetChildCount(); i++) {
				candidates.append(Constant(node.GetChild(i), column.type));
			}
			return nodes.attr("In")(column.reference, candidates);
		}
		default:
			throw Refused {};
		}
	}

private:
	struct Resolved {
		nb::object reference;
		cxx::LogicalTypeId type;
	};

	nb::object Conjunction(const cxx::Expression &node, const char *op) {
		nb::object folded = Predicate(node.GetChild(0));
		for (cxx::idx_t i = 1; i < node.GetChildCount(); i++) {
			folded = nodes.attr("Binary")(op, folded, Predicate(node.GetChild(i)));
		}
		return folded;
	}

	nb::object Comparison(const cxx::Expression &node) {
		auto left = node.GetChild(0);
		auto right = node.GetChild(1);
		const bool column_on_right = left.GetType() != cxx::ExpressionType::BOUND_COLUMN_REF;
		auto column = Column(column_on_right ? right : left);
		auto constant = Constant(column_on_right ? left : right, column.type);
		const auto comparison = column_on_right ? Converse(node.GetType()) : node.GetType();
		return nodes.attr("Binary")(Operator(comparison), column.reference, constant);
	}

	Resolved Column(const cxx::Expression &node) {
		if (node.GetType() != cxx::ExpressionType::BOUND_COLUMN_REF) {
			throw Refused {};
		}
		const auto column = resolve(node.GetColumnIndex());
		nb::str name(column.name.c_str(), column.name.size());
		return Resolved {nodes.attr("Col")(nb::make_tuple(name)), column.type};
	}

	nb::object Constant(const cxx::Expression &node, cxx::LogicalTypeId type) {
		if (node.GetType() != cxx::ExpressionType::VALUE_CONSTANT) {
			throw Refused {};
		}
		auto value = node.GetConstantValue();
		if (value.IsNull() || value.GetLogicalType().GetTypeId() != type || !ConvertsLossless(type)) {
			throw Refused {};
		}
		if (type == cxx::LogicalTypeId::FLOAT || type == cxx::LogicalTypeId::DOUBLE) {
			// NaN orders differently in every library; a predicate naming it stays with the engine.
			const double number = type == cxx::LogicalTypeId::FLOAT ? value.Get<float>() : value.Get<double>();
			if (number != number) {
				throw Refused {};
			}
		}
		if (Temporal(type)) {
			// An infinite date or timestamp converts to Python's largest or smallest value, which a real row can hold.
			const auto text = value.ToText();
			if (text == "infinity" || text == "-infinity") {
				throw Refused {};
			}
		}
		try {
			return nodes.attr("Lit")(ValueToPython(value, conversion));
		} catch (const cxx::Exception &) {
			throw Refused {};
		} catch (nb::python_error &) {
			throw Refused {};
		}
	}

	const ColumnResolver &resolve;
	ConversionContext &conversion;
	nb::module_ nodes;
};

} // namespace

nb::object TranslatePredicate(const cxx::Expression &node, const ColumnResolver &resolve,
                              ConversionContext &conversion) {
	return Walker(resolve, conversion).Predicate(node);
}

} // namespace duckdb_python
