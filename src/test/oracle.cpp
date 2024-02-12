#include "test/oracle.h"

#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"
#include "test/gen_ecc_set.h"

using namespace quartz;

std::string optimize_(std::string circ_string, std::string cost_func,
                      std::string ecc_path, std::string gate_set, int timeout)
{
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::cx,
               GateType::h, GateType::rz, GateType::x, GateType::add});
  auto graph =
      Graph::from_qasm_str(&ctx, circ_string);

  EquivalenceSet eqs;
  // Load equivalent dags from file
  if (!eqs.load_json(&ctx, ecc_path))
  {
    std::cout << "Failed to load equivalence file." << std::endl;
    assert(false);
  }

  // Get xfer from the equivalent set
  auto ecc = eqs.get_all_equivalence_sets();
  std::vector<GraphXfer *> xfers;
  for (auto eqcs : ecc)
  {
    for (auto circ_0 : eqcs)
    {
      for (auto circ_1 : eqcs)
      {
        if (circ_0 != circ_1)
        {
          auto xfer = GraphXfer::create_GraphXfer(&ctx, circ_0, circ_1, true);
          if (xfer != nullptr)
          {
            xfers.push_back(xfer);
          }
        }
      }
    }
  }
  // std::cout << "number of xfers: " << xfers.size() << std::endl;

  auto newgraph = graph->optimize(xfers, graph->gate_count() * 1.05, "barenco_tof_3", "", false, nullptr, timeout);
  return newgraph->to_qasm(false, false);
}
