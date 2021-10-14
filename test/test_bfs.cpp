#include "test_generator.h"

int main() {
  Context ctx
      ({GateType::x, GateType::y, GateType::rx, GateType::cx, GateType::add});
  Generator gen(&ctx);

  Dataset dataset1;
  auto start = std::chrono::steady_clock::now();
  gen.generate_dfs(3/*num_qubits*/,
                   2/*max_num_input_parameters*/,
                   5/*max_num_gates*/,
                   dataset1);
  auto end = std::chrono::steady_clock::now();
  std::cout << std::dec << "DFS: " << dataset1.num_total_dags()
            << " Circuits with " << dataset1.num_hash_values()
            << " different hash values are found in "
            << (double) std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start).count() / 1000.0 << " seconds."
            << std::endl;
  dataset1.save_json("dfs.json");

  Dataset dataset2;
  start = std::chrono::steady_clock::now();
  gen.generate(3/*num_qubits*/,
               2/*num_input_parameters*/,
               5/*max_num_gates*/,
               dataset2);
  end = std::chrono::steady_clock::now();
  std::cout << std::dec << "BFS: " << dataset2.num_total_dags()
            << " Circuits with " << dataset2.num_hash_values()
            << " different hash values are found in "
            << (double) std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start).count() / 1000.0 << " seconds."
            << std::endl;
  dataset2.save_json("bfs.json");
  return 0;
}
