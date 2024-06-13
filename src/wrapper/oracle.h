#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"

#include <memory>
#include <string>
#include <vector>
using namespace quartz;

class SuperContext {
 public:
  ParamInfo param_info;
  Context ctx;
  std::vector<GraphXfer *> xfers;
  std::vector<GraphXfer *> xfers_greedy_gate;

  static Context createContext(const std::string &gate_set, int n_qubits,
                               ParamInfo &param_info) {
    std::vector<GateType> gates;
    if (gate_set == "Nam") {
      gates = {GateType::input_qubit, GateType::input_param, GateType::cx,
               GateType::h,           GateType::rz,          GateType::x,
               GateType::add};
    } else if (gate_set == "CliffordT") {
      gates = {GateType::input_qubit, GateType::input_param, GateType::h,
               GateType::x,           GateType::t,           GateType::tdg,
               GateType::s,           GateType::sdg,         GateType::z,
               GateType::cx,          GateType::add};
    } else if (gate_set == "Nam_B") {
      gates = {
          GateType::input_qubit, GateType::input_param, GateType::cx,
          GateType::h,           GateType::rz,          GateType::x,
          GateType::add,         GateType::b2,
      };
    } else {
      std::cerr << "Invalid gate set." << std::endl;
      assert(false);
    }

    return Context(gates, n_qubits, &param_info);
  }
  SuperContext(const std::string &gate_set, int n_qubits,
               const std::string &ecc_path)
      : param_info(0), ctx(createContext(gate_set, n_qubits, param_info)) {
    xfers = std::vector<GraphXfer *>();
    xfers_greedy_gate = std::vector<GraphXfer *>();
    EquivalenceSet eqs;
    if (!eqs.load_json(&ctx, ecc_path, false)) {
      std::cout << "Failed to load equivalence file." << std::endl;
      assert(false);
    }
    int n_xfer_1q = 0;
    auto ecc = eqs.get_all_equivalence_sets();
    for (auto &eqcs : ecc) {
      for (auto &circ_0 : eqcs) {
        for (auto &circ_1 : eqcs) {
          if (circ_0 != circ_1) {
            if (circ_0->get_num_qubits() == 1 &&
                circ_1->get_num_qubits() == 1) {
              n_xfer_1q++;
            }
            auto xfer = GraphXfer::create_GraphXfer(&ctx, circ_0, circ_1, true);
            if (xfer != nullptr) {
              xfers.push_back(xfer);
            }
          }
        }
      }
    }
    // Only representatives.
    // std::cout << "Number of 1q xfers: " << n_xfer_1q << std::endl;
    // std::cout << "Number of xfers: " << xfers.size() << std::endl;

    for (auto &eqcs : ecc) {
      for (auto &circ_0 : eqcs) {
        for (auto &circ_1 : eqcs) {
          if (circ_0->get_num_gates() < circ_1->get_num_gates()) {
            auto xfer = GraphXfer::create_GraphXfer(&ctx, circ_1, circ_0, true);
            if (xfer != nullptr) {
              xfers_greedy_gate.push_back(xfer);
            }
          }
        }
      }
    }
  };

  ~SuperContext() = default;
};

std::shared_ptr<SuperContext> get_context_(const std::string gate_set,
                                           int n_qubits,
                                           const std::string ecc_path);
std::string optimize_(std::string circ_string, std::string cost_func,
                      std::string timeout_type, float timeout_value,
                      std::shared_ptr<SuperContext> super_context

);
std::string rotation_merging_(std::string circ_string);
std::string clifford_decomposition_(std::string circ);