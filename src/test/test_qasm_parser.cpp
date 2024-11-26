#include "quartz/context/context.h"
#include "quartz/gate/gate.h"
#include "quartz/parser/qasm_parser.h"

#include <cassert>

using namespace quartz;

bool has_exprs(Context &ctx, CircuitSeq *seq) {
  for (auto id : seq->get_directly_used_param_indices()) {
    if (ctx.param_is_expression(id)) {
      return true;
    }
  }
  return false;
}

void test_symbolic_exprs() {
  ParamInfo param_info(0);
  Context ctx({GateType::rx, GateType::ry, GateType::rz, GateType::cx,
               GateType::mult, GateType::pi},
              2, &param_info);

  QASMParser parser(&ctx);

  std::string str = "OPENQASM 2.0;\n"
                    "include \"qelib1.inc\";\n"
                    "qreg q[2];\n"
                    "rx(pi) q[0];\n"
                    "cx q[0], q[1];\n"
                    "ry(3*pi) q[0];\n"
                    "rz(pi/3) q[1];\n"
                    "cx q[1], q[0];\n"
                    "rx(2*pi/3) q[1];\n";

  CircuitSeq *seq1 = nullptr;
  parser.use_symbolic_pi(true);
  bool res1 = parser.load_qasm_str(str, seq1);
  if (!res1) {
    std::cout << "Parsing failed with symbolic pi." << std::endl;
    assert(false);
  }

  CircuitSeq *seq2 = nullptr;
  parser.use_symbolic_pi(false);
  bool res2 = parser.load_qasm_str(str, seq2);
  if (!res2) {
    std::cout << "Parsing failed with constant pi." << std::endl;
    assert(false);
  }

  int pnum = ctx.get_num_parameters();
  if (pnum != 11) {
    // Expected caching.
    // - Constants: 1, 2, 3, pi, 3*pi, pi/3, 2*pi/3
    // - Symbols: pi/1, 3*pi, pi/3, 2*pi/3
    std::cout << "Failed to cache all intermediate values." << std::endl;
    std::cout << "Number of parameters: " << pnum << std::endl;
    assert(false);
  }

  auto mat1 = seq1->get_matrix(&ctx);
  auto mat2 = seq2->get_matrix(&ctx);

  assert(mat1.size() == mat2.size());
  for (int i = 0; i < mat1.size(); ++i) {
    assert(mat1[i].size() == mat2[i].size());
    for (int j = 0; j < mat1[i].size(); ++j) {
      if (mat1[i][j] != mat2[i][j]) {
        std::cout << "Disagree at " << i << ", " << j << "." << std::endl;
        assert(false);
      }
    }
  }

  if (!has_exprs(ctx, seq1)) {
    std::cout << "Parser did not use symbolic pi values." << std::endl;
    assert(false);
  }

  if (has_exprs(ctx, seq2)) {
    std::cout << "Parser failed to disable symbolic pi values." << std::endl;
    assert(false);
  }
}

void test_qasm2_qubits() {
  ParamInfo param_info(0);
  Context ctx({GateType::cx}, 5, &param_info);

  QASMParser parser(&ctx);

  std::string str = "OPENQASM 2.0;\n"
                    "include \"qelib1.inc\";\n"
                    "qreg q[2]   ;\n"
                    "qreg r[3];\n"
                    "cx q[0], q[1];\n"
                    "cx q[0], r[0];\n"
                    "cx r[1], r[2];\n";

  CircuitSeq *seq = nullptr;
  bool res = parser.load_qasm_str(str, seq);
  if (!res) {
    std::cout << "Parsing failed with many qubit declarations." << std::endl;
    assert(false);
  }

  int qnum = seq->get_num_qubits();
  if (qnum != 5) {
    std::cout << "Unexpected qubit total: " << qnum << "." << std::endl;
    assert(false);
  }
}

void test_qasm3_qubits() {
  ParamInfo param_info(0);
  Context ctx({GateType::cx}, 5, &param_info);

  QASMParser parser(&ctx);

  std::string str = "OPENQASM 2.0;\n"
                    "include \"qelib1.inc\";\n"
                    "qubit[2] q;\n"
                    "qubit  [3] r   ;\n"
                    "qreg s    [2]   ;\n"
                    "cx q[0], q[1];\n"
                    "cx q[0], r[0];\n"
                    "cx r[1], r[2];\n"
                    "cx s[0], s[1];\n";

  CircuitSeq *seq = nullptr;
  bool res = parser.load_qasm_str(str, seq);
  if (!res) {
    std::cout << "Parsing failed with many qubit declarations." << std::endl;
    assert(false);
  }

  int qnum = seq->get_num_qubits();
  if (qnum != 7) {
    std::cout << "Unexpected qubit total: " << qnum << "." << std::endl;
    assert(false);
  }
}

int main() {
  std::cout << "[Symbolic Expression Tests]" << std::endl;
  test_symbolic_exprs();
  std::cout << "[OpenQASM 2 Parsing Tests]" << std::endl;
  test_qasm2_qubits();
  std::cout << "[OpenQASM 3 Parsing Tests]" << std::endl;
  test_qasm3_qubits();
}
