#include "quartz/context/context.h"
#include "quartz/dataset/dataset.h"
#include "quartz/gate/gate.h"
#include "quartz/parser/qasm_parser.h"

#include <cassert>

using namespace quartz;

bool add_circ(QASMParser &parser, Context &ctx, Dataset &data, std::string f) {
  // Attempts to parse the circuit.
  CircuitSeq *circ = nullptr;
  if (!parser.load_qasm(f, circ)) {
    return false;
  }

  // Passes ownership of the circuit to the dataset.
  data.insert(&ctx, std::unique_ptr<CircuitSeq>(circ));
  return true;
}

int main(int argc, char **argv) {
  // Sets up an OpenQASM 3 context.
  ParamInfo param_info;
  Context ctx({GateType::pi, GateType::add, GateType::mult, GateType::neg,
               GateType::x, GateType::y, GateType::z, GateType::p, GateType::h,
               GateType::s, GateType::t, GateType::sx, GateType::pdg,
               GateType::sdg, GateType::tdg, GateType::rx, GateType::ry,
               GateType::rz, GateType::cx, GateType::cz, GateType::cp,
               GateType::ch, GateType::swap, GateType::ccx, GateType::ccz,
               GateType::u1, GateType::u2, GateType::u3, GateType::cu1},
              &param_info);

  // Sets up a symbolic OpenQASM 3 parser, for use by both files.
  QASMParser parser(&ctx);
  parser.use_symbolic_pi(true);

  // Processes command-line arguments.
  std::string tmpfile = "tmp.json";
  std::string outfile = "res.json";
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: " << argv[0] << " circ1 circ2 [tmpdir]" << std::endl;
    return -1;
  } else if (argc == 4) {
    std::string dir = argv[3];
    tmpfile = dir + "/" + tmpfile;
    outfile = dir + "/" + outfile;
  }

  // Parses and validates the circuits.
  Dataset data;
  if (!add_circ(parser, ctx, data, argv[1])) {
    std::cerr << "Failed to parse the first circuit." << std::endl;
    return -1;
  }
  if (!add_circ(parser, ctx, data, argv[2])) {
    std::cerr << "Failed to parse the first circuit." << std::endl;
    return -1;
  }

  // Generates a json file for the Python-based equivalence checker.
  if (!data.save_json(&ctx, tmpfile)) {
    std::cerr << "Failed to generate the json file." << std::endl;
    return -1;
  }

  // Command-line arguments for the equivalence checker.
  std::string script = "../src/python/verifier/verify_equivalences.py";
  std::string arglst = tmpfile + " " + outfile + " " + "True True True True";

  // Applies the equivalence checker to the json file.
  std::string command = "python " + script + " " + " " + arglst;
  system(command.c_str());
  return 1;
}
