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
    return;
  }

  CircuitSeq *seq2 = nullptr;
  parser.use_symbolic_pi(false);
  bool res2 = parser.load_qasm_str(str, seq2);
  if (!res2) {
    std::cout << "Parsing failed with constant pi." << std::endl;
    assert(false);
    return;
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
  for (size_t i = 0; i < mat1.size(); ++i) {
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
    return;
  }

  int qnum = seq->get_num_qubits();
  if (qnum != 5) {
    std::cout << "Unexpected qubit total: " << qnum << "." << std::endl;
    assert(false);
  }
}

void test_qasm3_qubits() {
  ParamInfo param_info(0);
  Context ctx({GateType::cx}, 7, &param_info);

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
    return;
  }

  int qnum = seq->get_num_qubits();
  if (qnum != 7) {
    std::cout << "Unexpected qubit total: " << qnum << "." << std::endl;
    assert(false);
  }
}

void test_param_parsing() {
  ParamInfo param_info(0);
  Context ctx({GateType::cx, GateType::rx}, 2, &param_info);

  QASMParser parser(&ctx);

  // Tests parsing a first file.
  std::string str1 = "OPENQASM 3;\n"
                     "include \"stdgates.inc\";\n"
                     "qubit[2] q;\n"
                     "input array[angle,2] ps;\n"
                     "input array[float,3] params;\n"
                     "cx q[0], q[1];\n"
                     "rx(ps[0]) q[0];\n"
                     "rx(ps[1]) q[1];\n"
                     "rx(params[0]) q[0];\n"
                     "rx(params[1]) q[1];\n";

  CircuitSeq *seq1 = nullptr;
  bool res1 = parser.load_qasm_str(str1, seq1);
  if (!res1) {
    std::cout << "Parsing failed with parameter variables." << std::endl;
    assert(false);
    return;
  }

  int pnum1 = ctx.get_num_parameters();
  if (pnum1 != 5) {
    std::cout << "Unexpected parameter total: " << pnum1 << "." << std::endl;
    assert(false);
  }

  int input_num1 = seq1->get_input_param_indices(&ctx).size();
  if (input_num1 != 4) {
    std::cout << "Unexpected input count: " << input_num1 << "." << std::endl;
    assert(false);
  }

  // Tests parsing a second file.
  std::string str2 = "OPENQASM 3;\n"
                     "include \"stdgates.inc\";\n"
                     "qubit[5] q;\n"
                     "input array[angle,2] ps;\n"
                     "cx q[0], q[1];\n"
                     "rx(ps[0]) q[0];\n"
                     "rx(ps[1]) q[1];\n";

  CircuitSeq *seq2 = nullptr;
  bool res2 = parser.load_qasm_str(str2, seq2);
  if (!res2) {
    std::cout << "Parsing failed with parameter variables." << std::endl;
    assert(false);
    return;
  }

  int pnum2 = ctx.get_num_parameters();
  if (pnum2 != 5) {
    std::cout << "Unexpected parameter total: " << pnum2 << "." << std::endl;
    assert(false);
  }

  int input_num2 = seq2->get_input_param_indices(&ctx).size();
  if (input_num2 != 2) {
    std::cout << "Unexpected input count: " << input_num2 << "." << std::endl;
    assert(false);
  }

  // Checks that parameters were reused.
  auto all_indices = seq1->get_input_param_indices(&ctx);
  for (auto j : seq2->get_input_param_indices(&ctx)) {
    if (std::count(all_indices.begin(), all_indices.end(), j) == 0) {
      std::cout << "Unexpected parameter: " << j << "." << std::endl;
      assert(false);
    }
  }
}

void test_sum_parsing() {
  ParamInfo param_info(0);
  Context ctx({GateType::rx, GateType::mult, GateType::add, GateType::pi}, 2,
              &param_info);

  QASMParser parser(&ctx);
  parser.use_symbolic_pi(true);

  std::string str1 = "OPENQASM 2.0;\n"
                     "include \"qelib1.inc\";\n"
                     "qubit[1] q;\n"
                     "rx(pi/5) q[1];\n"
                     "rx(3*pi/2) q[1];\n"
                     "rx(-0.32) q[1];\n";

  CircuitSeq *seq1 = nullptr;
  bool res1 = parser.load_qasm_str(str1, seq1);
  if (!res1) {
    std::cout << "Unexpected parsing failure." << std::endl;
    assert(false);
    return;
  }

  std::string str2 = "OPENQASM 3;\n"
                     "include \"stdgates.inc\";\n"
                     "qubit[1] q;\n"
                     "rx(pi/5+3*pi/2-0.32) q[1];\n";

  CircuitSeq *seq2 = nullptr;
  bool res2 = parser.load_qasm_str(str2, seq2);
  if (!res2) {
    std::cout << "Parsing failed with sums of terms." << std::endl;
    assert(false);
    return;
  }

  int pnum = ctx.get_num_parameters();
  if (pnum != 9) {
    // Expected caching.
    // - Terms: 2, 3, 5, pi/2, pi/5, 3*pi/2, -0.32
    // - Exprs: 3*pi.2-0.32, pi/5+3*pi/2-0.32
    std::cout << "Failed to cache all intermediate values." << std::endl;
    std::cout << "Number of parameters: " << pnum << std::endl;
    assert(false);
  }

  auto mat1 = seq1->get_matrix(&ctx);
  auto mat2 = seq2->get_matrix(&ctx);

  assert(mat1.size() == mat2.size());
  for (size_t i = 0; i < mat1.size(); ++i) {
    assert(mat1[i].size() == mat2[i].size());
    for (int j = 0; j < mat1[i].size(); ++j) {
      if (mat1[i][j] != mat2[i][j]) {
        std::cout << "Disagree at " << i << ", " << j << "." << std::endl;
        assert(false);
      }
    }
  }
}

int main() {
  std::cout << "[Symbolic Expression Tests]" << std::endl;
  test_symbolic_exprs();
  std::cout << "[OpenQASM 2 Parsing Tests]" << std::endl;
  test_qasm2_qubits();
  std::cout << "[OpenQASM 3 Parsing Tests]" << std::endl;
  test_qasm3_qubits();
  std::cout << "[Sybmolic Parameter Parsing Tests]" << std::endl;
  test_param_parsing();
  std::cout << "[Sybmolic Summation Parsing Tests]" << std::endl;
  test_sum_parsing();
}
