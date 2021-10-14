#include "test_generator.h"

int main() {
  Context ctx({GateType::x, GateType::y, GateType::cx, GateType::add});
  Generator gen(&ctx);

  Dataset dataset1;
  auto start = std::chrono::steady_clock::now();
  gen.generate_dfs(3/*num_qubits*/,
                   3/*max_num_input_parameters*/,
                   3/*max_num_gates*/,
                   dataset1);
  auto end = std::chrono::steady_clock::now();
  std::cout << std::dec << "DFS: Circuits with " << dataset1.dataset.size()
            << " different hash values are found in "
            << (double) std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start).count() / 1000.0 << " seconds."
            << std::endl;
  dataset1.save_json("dfs.json");

  Dataset dataset2;
  start = std::chrono::steady_clock::now();
  gen.generate(3/*num_qubits*/,
                   3/*num_input_parameters*/,
                   3/*max_num_gates*/,
                   dataset2);
  end = std::chrono::steady_clock::now();
  std::cout << std::dec << "BFS: Circuits with " << dataset1.dataset.size()
            << " different hash values are found in "
            << (double) std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start).count() / 1000.0 << " seconds."
            << std::endl;
  dataset2.save_json("bfs.json");
  return 0;
}
