#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"

#include <chrono>
#include <iostream>
#include <stdexcept>

using namespace quartz;

// GraphXfer currently has no owning destructor for its pattern operations.
// Keep fixture allocations out of leak checks and benchmark measurements.
struct DeleteTestXfer {
  void operator()(GraphXfer *xfer) const {
    if (!xfer)
      return;
    for (auto op : xfer->srcOps)
      delete op;
    for (auto op : xfer->dstOps)
      delete op;
    delete xfer;
  }
};

// Run from the repository root, once per mode in a fresh process:
//   ./build/benchmark_compressed_search reference|compressed
//       circuit/nam_circs/barenco_tof_4.qasm
//       eccset/Nam_5_3_complete_ECC_set.json 300
// The final arguments are an expansion budget (0 means unlimited), an optional
// timeout in seconds, and an optional step-file prefix. Compare equal budgets
// for equal work, or use budget 0 for equal-time measurements. Repeat runs and
// alternate mode order; use an external tool (e.g. /usr/bin/time -v on Linux)
// for peak process memory. peak_packed_queue_bytes counts only snapshot
// storage, not allocator/queue overhead, retained history, or other process
// memory.
int main(int argc, char **argv) {
  try {
    if (argc < 5) {
      std::cerr << "Usage: benchmark_compressed_search reference|compressed "
                   "circuit.qasm ecc.json max_expansions [timeout_seconds] "
                   "[step_prefix]\n";
      return 1;
    }
    const std::string mode = argv[1];
    if (mode != "reference" && mode != "compressed") {
      throw std::runtime_error("Unknown search mode");
    }
    ParamInfo params;
    Context ctx({GateType::input_qubit, GateType::input_param, GateType::cx,
                 GateType::h, GateType::rz, GateType::x, GateType::add},
                &params);
    EquivalenceSet eqs;
    if (!eqs.load_json(&ctx, argv[3], false)) {
      throw std::runtime_error("Could not load ECC set");
    }
    auto graph = Graph::from_qasm_file(&ctx, argv[2]);
    if (!graph) {
      throw std::runtime_error("Could not load circuit");
    }
    auto xfers = GraphXfer::get_all_xfers_from_eqs(&ctx, eqs);
    std::vector<std::unique_ptr<GraphXfer, DeleteTestXfer>> owned;
    for (auto xfer : xfers)
      owned.emplace_back(xfer);
    OptimizerSearchStats stats;
    OptimizerSearchOptions options;
    options.compress_candidates = mode == "compressed";
    options.max_expansions = std::stoull(argv[4]);
    options.stats = &stats;
    const double timeout = argc > 5 ? std::stod(argv[5]) : 3600;
    const std::string prefix = argc > 6 ? argv[6] : "";
    auto start = std::chrono::steady_clock::now();
    auto optimized =
        graph->optimize(xfers, graph->total_cost() * 1.05, argv[2], "", false,
                        nullptr, timeout, prefix, false, options);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    std::cout << "mode=" << mode << " expanded=" << stats.expanded
              << " accepted=" << stats.accepted
              << " peak_candidates=" << stats.peak_candidates
              << " shrinks=" << stats.queue_shrinks
              << " peak_packed_queue_bytes=" << stats.peak_packed_queue_bytes
              << " best_cost=" << optimized->total_cost()
              << " result_hash=" << optimized->hash() << " seconds=" << seconds
              << '\n';
    if (!prefix.empty())
      optimized->to_qasm(prefix + "result.qasm", false, false);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
