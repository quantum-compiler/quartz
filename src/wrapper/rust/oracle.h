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
  static Context createContext(const std::string &gate_set, int n_qubits,
                               ParamInfo &param_info) {
    std::vector<GateType> gates;
    if (gate_set == "Nam") {
      gates = {GateType::input_qubit, GateType::input_param, GateType::cx,
               GateType::h,           GateType::rz,          GateType::x,
               GateType::add};
    } else if (gate_set == "CliffordT") {
      std::cerr << "Not implemented yet." << std::endl;
      assert(false);
    } else {
      std::cerr << "Invalid gate set." << std::endl;
      assert(false);
    }

    return Context(gates, n_qubits, &param_info);
  }
  SuperContext(const std::string &gate_set, int n_qubits,
               const std::string &ecc_path)
      : param_info(0),
        ctx(createContext(gate_set, n_qubits, param_info)) {
    xfers = std::vector<GraphXfer *>();
    EquivalenceSet eqs;
    if (!eqs.load_json(&ctx, ecc_path, false)) {
      std::cout << "Failed to load equivalence file." << std::endl;
      assert(false);
    }

    auto ecc = eqs.get_all_equivalence_sets();
    for (auto &eqcs : ecc) {
      for (auto &circ_0 : eqcs) {
        for (auto &circ_1 : eqcs) {
          if (circ_0 != circ_1) {
            auto xfer = GraphXfer::create_GraphXfer(&ctx, circ_0, circ_1, true);
            if (xfer != nullptr) {
              xfers.push_back(xfer);
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
                      float timeout,
                      std::shared_ptr<SuperContext> super_context);