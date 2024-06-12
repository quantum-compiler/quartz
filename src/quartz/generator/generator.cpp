#include "generator.h"

#include <cassert>
#include <filesystem>

namespace quartz {

bool Generator::generate(
    int num_qubits, int max_num_quantum_gates, Dataset *dataset,
    bool invoke_python_verifier, EquivalenceSet *equiv_set,
    bool unique_parameters, bool verbose,
    std::chrono::steady_clock::duration *record_verification_time) {
  auto empty_dag = std::make_unique<CircuitSeq>(num_qubits);
  empty_dag->hash(ctx_);  // generate other hash values
  std::vector<CircuitSeq *> dags_to_search(1, empty_dag.get());
  if (invoke_python_verifier) {
    assert(equiv_set);
    auto equiv_class = std::make_unique<EquivalenceClass>();
    equiv_class->insert(std::make_unique<CircuitSeq>(*empty_dag));
    equiv_set->insert_class(ctx_, std::move(equiv_class));
  } else {
    ctx_->set_representative(std::make_unique<CircuitSeq>(*empty_dag));
  }
  dataset->insert(ctx_, std::move(empty_dag));
  std::vector<std::vector<CircuitSeq *>> dags(1, dags_to_search);

  initialize_supported_quantum_gates();

  // We need this even if |unique_parameters| is false because we use its
  // size as the number of parameters.
  input_param_masks_ = ctx_->get_param_masks();

  // To avoid EquivalenceSet deleting the DAGs in |dags| when calling
  // clear().
  std::vector<std::unique_ptr<CircuitSeq>> dag_holder;

  for (int num_gates = 1; num_gates <= max_num_quantum_gates; num_gates++) {
    if (verbose) {
      std::cout << "BFS: " << dags_to_search.size()
                << " representative DAGs to search with " << num_gates - 1
                << " gates." << std::endl;
    }
    if (!invoke_python_verifier) {
      assert(dataset);
      dags_to_search.clear();
      bfs(dags, *dataset, &dags_to_search, invoke_python_verifier, nullptr,
          unique_parameters);
      dags.push_back(dags_to_search);
    } else {
      assert(dataset);
      assert(equiv_set);
      bfs(dags, *dataset, nullptr, invoke_python_verifier, equiv_set,
          unique_parameters);
      // Do not verify when |num_gates == max_num_quantum_gates|.
      // This is to make the behavior the same when
      // |invoke_python_verifier| is true or false.
      if (num_gates == max_num_quantum_gates) {
        break;
      }
      bool ret = dataset->save_json(ctx_, kQuartzRootPath.string() +
                                              "/tmp_before_verify.json");
      assert(ret);

      decltype(std::chrono::steady_clock::now()) start;
      if (record_verification_time) {
        start = std::chrono::steady_clock::now();
      }
      std::string command_string =
          std::string("python ") + kQuartzRootPath.string() +
          "/src/python/verifier/verify_equivalences.py " +
          kQuartzRootPath.string() + "/tmp_before_verify.json " +
          kQuartzRootPath.string() + "/tmp_after_verify.json";
      system(command_string.c_str());
      if (record_verification_time) {
        auto end = std::chrono::steady_clock::now();
        *record_verification_time += end - start;
      }

      dags_to_search.clear();
      ret = equiv_set->load_json(
          ctx_, kQuartzRootPath.string() + "/tmp_after_verify.json",
          /*from_verifier=*/true, &dags_to_search);
      assert(ret);
      for (auto &dag : dags_to_search) {
        auto new_dag = std::make_unique<CircuitSeq>(*dag);
        dag = new_dag.get();
        dag_holder.push_back(std::move(new_dag));
      }

      dags.push_back(dags_to_search);
      /* Seems problematic
      equiv_set->remove_common_first_or_last_gates(context);
      std::vector<CircuitSeq *> simplified_dags_to_search;
      simplified_dags_to_search.reserve(dags_to_search.size());
      for (auto &circuitseq : dags_to_search) {
        if (equiv_set->contains(context, circuitseq)) {
          simplified_dags_to_search.push_back(circuitseq);
        }
      }
      dags.push_back(simplified_dags_to_search);
      */
    }
  }
  return true;
}

void Generator::initialize_supported_quantum_gates() {
  supported_quantum_gates_.clear();
  auto gates = ctx_->get_supported_quantum_gates();
  for (auto &gate : gates) {
    while ((int)supported_quantum_gates_.size() <=
           ctx_->get_gate(gate)->get_num_qubits()) {
      supported_quantum_gates_.emplace_back();
    }
    supported_quantum_gates_[ctx_->get_gate(gate)->get_num_qubits()].push_back(
        gate);
  }
}

void Generator::bfs(const std::vector<std::vector<CircuitSeq *>> &dags,
                    Dataset &dataset,
                    std::vector<CircuitSeq *> *new_representatives,
                    bool invoke_python_verifier,
                    const EquivalenceSet *equiv_set, bool unique_parameters) {
  auto try_to_add_to_result = [&](CircuitSeq *new_dag) {
    // A new CircuitSeq with |current_max_num_gates| + 1 gates.
    if (invoke_python_verifier) {
      // We will verify the equivalence later in Python.
      assert(equiv_set);
      if (!verifier_.redundant(ctx_, equiv_set, new_dag)) {
        auto new_new_dag = std::make_unique<CircuitSeq>(*new_dag);
        auto new_new_dag_ptr = new_new_dag.get();
        dataset.insert(ctx_, std::move(new_new_dag));
        if (new_representatives) {
          // Warning: this is not the new representatives -- only
          // the new DAGs.
          new_representatives->push_back(new_new_dag_ptr);
        }
      }
    } else {
      // If we will not verify the equivalence later, we should update
      // the representatives in the context now.
      if (verifier_.redundant(ctx_, new_dag)) {
        return;
      }
      // Try to insert to a set with hash value differing no more than 1
      // (see documentation at the function signature of generate()).
      bool ret = dataset.insert_to_nearby_set_if_exists(
          ctx_, std::make_unique<CircuitSeq>(*new_dag));
      if (ret) {
        // The CircuitSeq's hash value is new to the dataset.
        // Note: this is the second instance of CircuitSeq we create in
        // this function.
        auto rep = std::make_unique<CircuitSeq>(*new_dag);
        auto rep_ptr = rep.get();
        ctx_->set_representative(std::move(rep));
        if (new_representatives) {
          new_representatives->push_back(rep_ptr);
        }
      }
    }
  };
  for (auto &old_dag : dags.back()) {
    // Create a new CircuitSeq to avoid editing the old one.
    auto new_dag = std::make_unique<CircuitSeq>(*old_dag);
    auto dag = new_dag.get();
    InputParamMaskType input_param_usage_mask;
    if (unique_parameters) {
      input_param_usage_mask =
          dag->get_input_param_usage_mask(input_param_masks_);
    }
    std::vector<bool> last_gate_used_qubit_index(dag->get_num_qubits(), false);
    int last_gate_min_qubit_index = -1;
    if (dag->get_num_gates() > 0) {
      last_gate_min_qubit_index = dag->gates.back()->get_min_qubit_index();
      for (auto &input_node : dag->gates.back()->input_wires) {
        if (input_node->is_qubit()) {
          last_gate_used_qubit_index[input_node->index] = true;
        }
      }
    }
    std::vector<int> qubit_indices, parameter_indices;
    Gate *gate;

    auto search_parameters = [this, &dag, &gate, &qubit_indices,
                              &parameter_indices, &try_to_add_to_result,
                              &unique_parameters](int num_remaining_parameters,
                                                  const InputParamMaskType
                                                      &current_usage_mask,
                                                  auto
                                                      &search_parameters_ref /*feed in the lambda implementation to itself as a parameter*/) {
      if (num_remaining_parameters == 0) {
        bool ret = dag->add_gate(qubit_indices, parameter_indices, gate, ctx_);
        assert(ret);
        try_to_add_to_result(dag);
        ret = dag->remove_last_gate();
        assert(ret);
        if (gate->get_num_qubits() > 1 && !gate->is_symmetric()) {
          while (std::next_permutation(qubit_indices.begin(),
                                       qubit_indices.end())) {
            ret = dag->add_gate(qubit_indices, parameter_indices, gate, ctx_);
            assert(ret);
            try_to_add_to_result(dag);
            ret = dag->remove_last_gate();
            assert(ret);
          }
        }
        return;
      }

      for (int p1 = 0; p1 < (int)input_param_masks_.size(); p1++) {
        if (unique_parameters) {
          if (current_usage_mask & input_param_masks_[p1]) {
            // p1 contains an already used input parameter.
            continue;
          }
          parameter_indices.push_back(p1);
          search_parameters_ref(num_remaining_parameters - 1,
                                current_usage_mask | input_param_masks_[p1],
                                search_parameters_ref);
          parameter_indices.pop_back();
        } else {
          parameter_indices.push_back(p1);
          search_parameters_ref(num_remaining_parameters - 1,
                                /*unused*/ 0, search_parameters_ref);
          parameter_indices.pop_back();
        }
      }
    };
    // Add 1 quantum gate according to the qubit index order.

    for (int q1 = 0; q1 < dag->get_num_qubits(); q1++) {
      qubit_indices.push_back(q1);

      // Case: 1-qubit operators. We need the new gate to operate on a qubit
      // with index at least |last_gate_min_qubit_index| to form a canonical
      // sequence.
      if (supported_quantum_gates_.size() > 1) {
        if (q1 >= last_gate_min_qubit_index) {
          for (const auto &idx : supported_quantum_gates_[1]) {
            gate = ctx_->get_gate(idx);
            search_parameters(gate->get_num_parameters(),
                              input_param_usage_mask, search_parameters);
          }
        }
      }

      // Case: 2-qubit operators. We need the new gate to
      // operate on a qubit with index at least |last_gate_min_qubit_index|
      // or a qubit that is used by the last gate to form a canonical sequence.
      if (supported_quantum_gates_.size() > 2) {
        for (int q2 = q1 + 1; q2 < dag->get_num_qubits(); q2++) {
          if (q1 < last_gate_min_qubit_index &&
              !last_gate_used_qubit_index[q2]) {
            continue;
          }
          qubit_indices.push_back(q2);
          for (const auto &idx : supported_quantum_gates_[2]) {
            gate = ctx_->get_gate(idx);
            search_parameters(gate->get_num_parameters(),
                              input_param_usage_mask, search_parameters);
          }
          qubit_indices.pop_back();
        }
      }

      // Case: 3-qubit operators. We need the new gate to
      // operate on a qubit with index at least |last_gate_min_qubit_index|
      // or a qubit that is used by the last gate to form a canonical sequence.
      if (supported_quantum_gates_.size() > 3) {
        for (int q2 = q1 + 1; q2 < dag->get_num_qubits(); q2++) {
          qubit_indices.push_back(q2);
          for (int q3 = q2 + 1; q3 < dag->get_num_qubits(); q3++) {
            if (q1 < last_gate_min_qubit_index &&
                !last_gate_used_qubit_index[q2] &&
                !last_gate_used_qubit_index[q3]) {
              continue;
            }
            qubit_indices.push_back(q3);
            for (const auto &idx : supported_quantum_gates_[3]) {
              gate = ctx_->get_gate(idx);
              search_parameters(gate->get_num_parameters(),
                                input_param_usage_mask, search_parameters);
            }
            qubit_indices.pop_back();
          }
          qubit_indices.pop_back();
        }
      }

      assert(supported_quantum_gates_.size() <= 4);

      qubit_indices.pop_back();
    }
  }
}

}  // namespace quartz
