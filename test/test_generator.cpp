#include "test_generator.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/parser/qasm_parser.h"

using namespace quartz;

int main() {
	Context ctx({GateType::x, GateType::y, GateType::cx, GateType::h});
	Generator gen(&ctx);
	Dataset dataset;
	gen.generate_dfs(3 /*num_qubits*/, 3 /*max_num_input_parameters*/,
	                 3 /*max_num_gates*/, 1 /*max_num_param_gates*/, dataset,
	                 true /*restrict_search_space*/);
	for (const auto &it : dataset.dataset) {
		bool is_first = true;
		DAG *first_dag = NULL;
		for (auto &dag : it.second) {
			if (is_first) {
				first_dag = dag.get();
				is_first = false;
			}
			else {
				GraphXfer xfer(&ctx, first_dag, dag.get());
			}
		}
	}

	QASMParser parser(&ctx);
	DAG *dag = NULL;
	parser.load_qasm("test.qasm", dag);
	return 0;
}
