#include "test_generator.h"

int main() {
  Context ctx
      ({GateType::x, GateType::y, GateType::rx, GateType::cx, GateType::add});
  Generator gen(&ctx);

  const int num_qubits = 3;
  const int num_input_parameters = 2;
  const int max_num_gates = 5;

  Dataset dataset1;
  auto start = std::chrono::steady_clock::now();
  gen.generate_dfs(num_qubits, num_input_parameters, max_num_gates,
                   dataset1);
  auto end = std::chrono::steady_clock::now();
  std::cout << std::dec << "DFS: " << dataset1.num_total_dags()
            << " Circuits with " << dataset1.num_hash_values()
            << " different hash values are found in "
            << (double) std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start).count() / 1000.0 << " seconds."
            << std::endl;
  dataset1.save_json("dfs.json");

  ctx.clear_representatives();

  Dataset dataset2;
  start = std::chrono::steady_clock::now();
  gen.generate(num_qubits, num_input_parameters, max_num_gates,
               &dataset2, /*verify_equivalences=*/false, nullptr);
  end = std::chrono::steady_clock::now();
  std::cout << std::dec << "BFS: " << dataset2.num_total_dags()
            << " Circuits with " << dataset2.num_hash_values()
            << " different hash values are found in "
            << (double) std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start).count() / 1000.0 << " seconds."
            << std::endl;
  dataset2.save_json("bfs.json");

  ctx.clear_representatives();

  Dataset dataset3;
  EquivalenceSet equiv_set;
  start = std::chrono::steady_clock::now();
  gen.generate(num_qubits, num_input_parameters, max_num_gates,
               &dataset3, /*verify_equivalences=*/true, &equiv_set);
  end = std::chrono::steady_clock::now();
  std::cout << std::dec << "BFS verified: " << dataset3.num_total_dags()
            << " Circuits with " << dataset3.num_hash_values()
            << " different hash values are found in "
            << (double) std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start).count() / 1000.0 << " seconds."
            << std::endl;
  dataset3.save_json("bfs_verified.json");
  return 0;
}
