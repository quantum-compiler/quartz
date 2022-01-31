#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"
#include "quartz/context/context.h"
#include "quartz/parser/qasm_parser.h"

using namespace quartz;

int main() {
	Context ctx({GateType::input_qubit, GateType::input_param, GateType::h,
	             GateType::cx});

	// Construct circuit graph from qasm file
	QASMParser qasm_parser(&ctx);
	DAG *dag = nullptr;
	if (!qasm_parser.load_qasm("circuit/example-circuits/h_h.qasm", dag)) {
		std::cout << "Parser failed" << std::endl;
		return 0;
	}
	Graph graph(&ctx, dag);

	EquivalenceSet eqs;
	// Load equivalent dags from file
	if (!eqs.load_json(&ctx, "bfs_verified_simplified.json")) {
		std::cout << "Failed to load equivalence file." << std::endl;
		assert(false);
	}

	// Get xfer from the equivalent set
	auto all_xfers = eqs.get_all_equivalence_sets();
	auto num_equivalent_classes = eqs.num_equivalence_classes();
	std::cout << num_equivalent_classes << std::endl;
	GraphXfer *xfer =
	    GraphXfer::create_GraphXfer(&ctx, all_xfers[0][1], all_xfers[0][0]);

	std::vector< Op > ops;
	graph.all_ops(ops);
	for (auto it = ops.begin(); it != ops.end(); ++it) {
		std::cout << gate_type_name(it->ptr->tp) << std::endl;
	}
	for (auto it = ops.begin(); it != ops.end(); ++it) {
		bool xfer_ok = graph.xfer_appliable(xfer, &(*it));
		std::cout << (int)xfer_ok << std::endl;
	}
}