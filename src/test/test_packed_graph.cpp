#include "quartz/tasograph/packed_graph.h"
#include "quartz/tasograph/substitution.h"

#include <filesystem>
#include <fstream>
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

namespace {
void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void compare_edges(const Graph &lhs, const Graph &rhs) {
  auto compare = [](const auto &a, const auto &b) {
    require(a.size() == b.size(), "adjacency map size changed");
    auto j = b.begin();
    for (const auto &entry : a) {
      require(entry.first == j->first, "operation identity changed");
      require(entry.second.size() == j->second.size(), "edge count changed");
      auto k = j->second.begin();
      for (const auto &edge : entry.second) {
        require(edge.srcOp == k->srcOp && edge.dstOp == k->dstOp &&
                    edge.srcIdx == k->srcIdx && edge.dstIdx == k->dstIdx,
                "connection changed");
        ++k;
      }
      ++j;
    }
  };
  compare(lhs.inEdges, rhs.inEdges);
  compare(lhs.outEdges, rhs.outEdges);
}

void round_trip(Graph &graph) {
  PackedGraph packed(graph);
  auto restored = packed.unpack();
  compare_edges(graph, *restored);
  require(graph.special_op_guid == restored->special_op_guid,
          "special operation counter changed");
  require(graph.input_qubit_op_2_qubit_idx ==
              restored->input_qubit_op_2_qubit_idx,
          "qubit indices changed");
  require(graph.param_idx == restored->param_idx, "parameter IDs changed");
  require(graph.pos_2_logical_qubit == restored->pos_2_logical_qubit,
          "logical qubit mapping changed");
  require(graph.hash() == restored->hash(), "hash changed");
  require(graph.total_cost() == restored->total_cost(), "cost changed");
  std::vector<Op> before, after;
  graph.topology_order_ops(before);
  restored->topology_order_ops(after);
  require(before == after, "traversal order changed");
  // A second reconstruction must not advance the context's ID allocator.
  auto id = graph.context->next_global_unique_id();
  auto second = packed.unpack();
  compare_edges(graph, *second);
  require(graph.context->next_global_unique_id() == id + 1,
          "unpacking allocated new operation IDs");
}

void test_round_trips() {
  ParamInfo params;
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::h,
               GateType::x, GateType::cx, GateType::rz, GateType::add,
               GateType::mult, GateType::pi},
              &params);
  Graph empty(&ctx);
  round_trip(empty);
  CircuitSeq seq(4);  // Includes an isolated input qubit.
  require(seq.add_gate({2}, {}, ctx.get_gate(GateType::h), &ctx), "add h");
  require(seq.add_gate({2, 0}, {}, ctx.get_gate(GateType::cx), &ctx), "add cx");
  int symbol = ctx.get_new_param_id();
  int expression = ctx.get_new_param_expression_id({symbol, symbol},
                                                   ctx.get_gate(GateType::add));
  int constant = ctx.get_new_param_id(ParamType(1));
  require(seq.add_gate({0}, {expression}, ctx.get_gate(GateType::rz), &ctx),
          "add symbolic rz");
  require(seq.add_gate({1}, {expression}, ctx.get_gate(GateType::rz), &ctx),
          "add repeated symbolic rz");
  require(seq.add_gate({2}, {constant}, ctx.get_gate(GateType::rz), &ctx),
          "add constant rz");
  require(seq.add_gate({0, 1}, {},
                       ctx.get_general_controlled_gate(GateType::cx, {false}),
                       &ctx),
          "add controlled variant");
  Graph graph(&ctx, &seq);
  round_trip(graph);
  // The snapshot owns no source graph, and survives its destruction.
  auto packed = [&]() {
    Graph temporary(graph);
    return PackedGraph(temporary);
  }();
  compare_edges(graph, *packed.unpack());
}

std::string read_file(const std::filesystem::path &path) {
  std::ifstream input(path);
  require(input.good(), "missing exported step");
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

struct Result {
  OptimizerSearchStats stats;
  std::string qasm;
  int steps;
};

Result run_search(bool compressed, const std::filesystem::path &prefix,
                  bool custom_cost, bool continue_steps = false,
                  bool stress = false) {
  ParamInfo params;
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::h,
               GateType::x, GateType::cx},
              &params);
  const std::string header =
      "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\n";
  std::string input = "h q[0]; h q[0]; x q[1]; x q[1]; "
                      "cx q[0],q[2]; cx q[0],q[2];";
  if (stress) {
    input.clear();
    for (int i = 0; i < 30; ++i) {
      input += i % 2 ? "h q[0]; h q[0];" : "x q[0]; x q[0];";
    }
    input += "cx q[1],q[2]; cx q[1],q[2];";
  }
  auto source = CircuitSeq::from_qasm_style_string(&ctx, header + input);
  require(source != nullptr, "parse search input");
  Graph graph(&ctx, source.get());
  std::vector<std::unique_ptr<GraphXfer, DeleteTestXfer>> owned;
  for (const auto &gate :
       {std::string("h q[0]; h q[0];"), std::string("x q[1]; x q[1];"),
        std::string("cx q[0],q[2]; cx q[0],q[2];")}) {
    if (stress && gate.find("cx") != 0) {
      continue;
    }
    auto from = CircuitSeq::from_qasm_style_string(&ctx, header + gate);
    CircuitSeq identity(3);
    owned.emplace_back(
        GraphXfer::create_GraphXfer(&ctx, from.get(), &identity));
    require(owned.back() != nullptr, "create cancellation rewrite");
  }
  if (stress) {
    auto hh =
        CircuitSeq::from_qasm_style_string(&ctx, header + "h q[0]; h q[0];");
    auto xx =
        CircuitSeq::from_qasm_style_string(&ctx, header + "x q[0]; x q[0];");
    owned.emplace_back(GraphXfer::create_GraphXfer(&ctx, hh.get(), xx.get()));
    owned.emplace_back(GraphXfer::create_GraphXfer(&ctx, xx.get(), hh.get()));
  }
  std::vector<GraphXfer *> xfers;
  for (const auto &xfer : owned)
    xfers.push_back(xfer.get());
  Result result;
  OptimizerSearchOptions options;
  options.compress_candidates = compressed;
  options.stats = &result.stats;
  options.max_expansions = stress ? 250 : 0;
  std::function<float(Graph *)> cost;
  if (custom_cost) {
    cost = [](Graph *g) { return g->total_cost() + 2 * g->circuit_depth(); };
  }
  if (continue_steps) {
    graph.to_qasm(prefix.string() + "0.qasm", false, false);
    std::ofstream(prefix.string() + ".txt") << 7;
  }
  auto optimized = graph.optimize(xfers, 100, "packed-test", "", false, cost,
                                  60, prefix.string(), continue_steps, options);
  if (!stress) {
    require(optimized->gate_count() == 0,
            "cancellations did not reach identity");
  }
  // Check semantics on every basis state, independently of hashes and cost.
  auto input_matrix = source->get_matrix(&ctx);
  auto output_matrix = optimized->to_circuit_sequence()->get_matrix(&ctx);
  require(input_matrix.size() == output_matrix.size(), "matrix size changed");
  for (size_t column = 0; column < input_matrix.size(); ++column) {
    for (int row = 0; row < input_matrix[column].size(); ++row) {
      require(std::abs(input_matrix[column][row] - output_matrix[column][row]) <
                  1e-12,
              "optimized circuit matrix changed");
    }
  }
  result.qasm = optimized->to_qasm();
  std::ifstream(prefix.string() + ".txt") >> result.steps;
  if (!stress) {
    require(result.steps == (continue_steps ? 10 : 3),
            "incorrect history length");
    // Exactly three transformations, with the root written only at index 0.
    require(read_file(prefix.string() + "0.qasm") == graph.to_qasm(),
            "initial graph export changed");
    const int first_step = continue_steps ? 8 : 1;
    require(read_file(prefix.string() + std::to_string(first_step) + ".qasm") !=
                graph.to_qasm(),
            "initial graph was exported again as a transformation step");
  }
  // Also round-trip a graph produced by actual rewrites.
  round_trip(*optimized);
  return result;
}

void test_search_boundaries(const std::filesystem::path &directory) {
  for (bool compressed : {false, true}) {
    ParamInfo params;
    Context ctx({GateType::input_qubit, GateType::input_param, GateType::h},
                &params);
    CircuitSeq source(2), identity(2);
    require(source.add_gate({0}, {}, ctx.get_gate(GateType::h), &ctx), "add h");
    require(source.add_gate({0}, {}, ctx.get_gate(GateType::h), &ctx), "add h");
    Graph graph(&ctx, &source);
    OptimizerSearchStats stats;
    OptimizerSearchOptions options;
    options.compress_candidates = compressed;
    options.stats = &stats;
    std::unique_ptr<GraphXfer, DeleteTestXfer> xfer(
        GraphXfer::create_GraphXfer(&ctx, &source, &identity));
    require(xfer != nullptr, "create timeout rewrite");
    const auto prefix =
        (directory / (compressed ? "timeout-packed" : "timeout-reference"))
            .string();
    auto result = graph.optimize({xfer.get()}, 10, "timeout", "", false,
                                 nullptr, -1, prefix, false, options);
    require(result->to_qasm() == graph.to_qasm() && stats.accepted == 0,
            "timeout changed the incumbent");
    require(read_file(prefix + ".txt") == "0\n",
            "timeout history is not empty");
    EquivalenceSet empty_eqs;
    result = graph.optimize(&ctx, empty_eqs, "empty-eqs", false, nullptr, -1,
                            60, "", options);
    require(result->to_qasm() == graph.to_qasm() && stats.expanded == 1 &&
                stats.accepted == 0,
            "greedy wrapper did not preserve empty search");
    require((stats.peak_packed_queue_bytes > 0) == compressed,
            "greedy wrapper did not forward compression option");
  }
}

void test_search(const std::filesystem::path &directory) {
  auto reference =
      run_search(false, directory / "stress-reference", false, false, true);
  auto compressed =
      run_search(true, directory / "stress-compressed", false, false, true);
  require(reference.stats.queue_shrinks > 0,
          "stress case did not shrink queue");
  require(reference.stats.expanded == compressed.stats.expanded &&
              reference.stats.accepted == compressed.stats.accepted &&
              reference.stats.queue_shrinks == compressed.stats.queue_shrinks &&
              reference.qasm == compressed.qasm &&
              reference.steps == compressed.steps,
          "queue pruning changed search results or work counts");
  for (int i = 0; i <= reference.steps; ++i) {
    require(read_file((directory / "stress-reference").string() +
                      std::to_string(i) + ".qasm") ==
                read_file((directory / "stress-compressed").string() +
                          std::to_string(i) + ".qasm"),
            "pruned search history changed");
  }

  for (bool custom : {false, true}) {
    for (bool continued : {false, true}) {
      auto a = directory / "reference";
      auto b = directory / "compressed";
      auto reference = run_search(false, a, custom, continued);
      auto compressed = run_search(true, b, custom, continued);
      require(reference.qasm == compressed.qasm, "search output changed");
      require(reference.stats.expanded == compressed.stats.expanded &&
                  reference.stats.accepted == compressed.stats.accepted,
              "search results or work counts changed");
      for (int i = continued ? 8 : 0; i <= reference.steps; ++i) {
        require(read_file(a.string() + std::to_string(i) + ".qasm") ==
                    read_file(b.string() + std::to_string(i) + ".qasm"),
                "exported history changed");
      }
      require(reference.stats.peak_packed_queue_bytes == 0 &&
                  compressed.stats.peak_packed_queue_bytes > 0,
              "compression statistics incorrect");
    }
  }
}
}  // namespace

int main() {
  try {
    test_round_trips();
    // Each invocation gets a separate directory; remove only files we created.
    auto directory =
        std::filesystem::temp_directory_path() /
        ("quartz-packed-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    test_search_boundaries(directory);
    test_search(directory);
    std::filesystem::remove_all(directory);
    std::cout << "Packed graph round-trip and search tests passed.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
