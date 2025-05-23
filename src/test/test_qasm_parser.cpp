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

bool eq_mats(std::vector<Vector> lhs, std::vector<Vector> rhs, float err) {
  assert(lhs.size() == rhs.size());
  for (size_t i = 0; i < lhs.size(); ++i) {
    assert(lhs[i].size() == rhs[i].size());
    for (int j = 0; j < lhs[i].size(); ++j) {
      auto diff = abs(lhs[i][j] - rhs[i][j]);
      if (diff > err) {
        std::cout << "Disagree at " << i << ", " << j << "." << std::endl;
        std::cout << lhs[i][j] << " != " << rhs[i][j] << std::endl;
        return false;
      }
    }
  }
  return true;
}

//
// Tests for use_symbolic_pi.
//
void test_symbolic_exprs() {
  ParamInfo param_info;
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

#ifndef USE_RATIONAL
  if (eq_mats(seq1->get_matrix(&ctx), seq2->get_matrix(&ctx), 0)) {
    assert(false);
  }
#endif

  if (!has_exprs(ctx, seq1)) {
    std::cout << "Parser did not use symbolic pi values." << std::endl;
    assert(false);
  }

  if (has_exprs(ctx, seq2)) {
    std::cout << "Parser failed to disable symbolic pi values." << std::endl;
    assert(false);
  }
}

//
// Regression tests for qubit parsing in OpenQASM 2.
//
void test_qasm2_qubits() {
  ParamInfo param_info;
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

//
// Tests for qubit parsing in OpenQASM 3.
//
void test_qasm3_qubits() {
  ParamInfo param_info;
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

//
// Test for OpenQASM 3 input parameter variable parsing.
//
void test_param_parsing() {
  ParamInfo param_info;
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

//
// Test for parsing sums used within parameter expressions.
//
void test_sum_parsing() {
  ParamInfo param_info;
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

//
// Tests for identifying halved parameter expressions.
//
bool test_halved_param_context(Context &ctx, bool is_halved) {
  auto g = ctx.get_gate(GateType::neg);

  auto param_id = ctx.get_new_param_id();
  auto const_id = ctx.get_new_param_id((ParamType)1);
  auto exprs_id = ctx.get_new_param_expression_id({param_id}, g);

  if (ctx.param_is_halved(param_id) != is_halved) {
    std::cout << "is_param_halved returned wrong val for symb." << std::endl;
    return false;
  }

  if (ctx.param_is_halved(const_id)) {
    std::cout << "is_param_halved returned wrong val for const." << std::endl;
    return false;
  }

  if (ctx.param_is_halved(exprs_id)) {
    std::cout << "is_param_halved returned wrong val for expr." << std::endl;
    return false;
  }

  return true;
}

//
// Tests for mixing halved parameter gates with standard parameter gates.
//
void test_halved_param_ids() {
  //
  // Default parameters constructed by ParamInfo.
  //
  ParamInfo param_info_1(4, false);
  if (param_info_1.param_is_halved(2)) {
    std::cout << "ParamInfo(4,false): param_is_halved(2) == true" << std::endl;
    assert(false);
  }

  //
  // Halved parameters constructed by ParamInfo.
  //
  ParamInfo param_info_2(4, true);
  if (!param_info_2.param_is_halved(2)) {
    std::cout << "ParamInfo(4,true): param_is_halved(2) == false" << std::endl;
    assert(false);
  }

  //
  // Default parameters constructed by a Context.
  //
  ParamInfo param_info_3;
  Context ctx_3({GateType::x, GateType::ry, GateType::neg}, 2, &param_info_3);
  if (!test_halved_param_context(ctx_3, true)) {
    std::cout << "Context failed to handle halved param gate." << std::endl;
    assert(false);
  }

  //
  // Halved parameters constructed by a Context.
  //
  ParamInfo param_info_4;
  Context ctx_4({GateType::x, GateType::y, GateType::neg}, 2, &param_info_4);
  if (!test_halved_param_context(ctx_4, false)) {
    std::cout << "Context failed to handle standard gates." << std::endl;
    assert(false);
  }

  //
  // Halved symbolic parameters work when parsing a circuit.
  //
  ParamInfo param_info_5;
  Context ctx_5({GateType::p, GateType::rz, GateType::add, GateType::neg,
                 GateType::x, GateType::mult, GateType::pi},
                &param_info_5);

  QASMParser parser_5(&ctx_5);

  // Implements [[exp(i*2*theta), 1] [0, 1]] with halved parameter theta.
  std::string str1 = "OPENQASM 3;\n"
                     "include \"stdgates.inc\";\n"
                     "qubit[1] q;\n"
                     "input array[angle,1] ps;\n"
                     "rz(ps[0]+ps[0]) q[0];\n"
                     "p(-ps[0]) q[0];\n";

  CircuitSeq *seq1 = nullptr;
  if (!parser_5.load_qasm_str(str1, seq1)) {
    std::cout << "Unexpected parsing failure (1)." << std::endl;
    assert(false);
    return;
  }

  // Implements [[exp(i*2*theta), 1] [0, 1]] after reparameterization.
  std::string str2 = "OPENQASM 3;\n"
                     "include \"stdgates.inc\";\n"
                     "qubit[1] q;\n"
                     "input array[angle,1] ps;\n"
                     "x q[0];\n"
                     "p(-ps[0]) q[0];\n"
                     "x q[0];\n";

  CircuitSeq *seq2 = nullptr;
  if (!parser_5.load_qasm_str(str2, seq2)) {
    std::cout << "Unexpected parsing failure (2)." << std::endl;
    assert(false);
    return;
  }

  if (!eq_mats(seq1->get_matrix(&ctx_5), seq2->get_matrix(&ctx_5), 1e-7)) {
    std::cout << "Disagreement on symbolic parameters." << std::endl;
    assert(false);
  }

  //
  // Halved constant parameters work when parsing a circuit.
  //
  ParamInfo param_info_6;
  Context ctx_6({GateType::x, GateType::s, GateType::rx, GateType::p,
                 GateType::pi, GateType::add, GateType::neg, GateType::mult},
                &param_info_6);

  QASMParser parser_6(&ctx_6);
  parser_6.use_symbolic_pi(true);

  std::string str3 = "OPENQASM 3;\n"
                     "include \"stdgates.inc\";\n"
                     "qubit[1] q;\n"
                     "x q[0];\n";

  CircuitSeq *seq3 = nullptr;
  if (!parser_6.load_qasm_str(str3, seq3)) {
    std::cout << "Unexpected parsing failure (3)." << std::endl;
    assert(false);
    return;
  }

  std::string str4 = "OPENQASM 3;\n"
                     "include \"stdgates.inc\";\n"
                     "qubit[1] q;\n"
                     "rx(pi) q[0];\n"
                     "s q[0];\n"
                     "x q[0];\n"
                     "p(pi/2) q[0];\n"
                     "x q[0];\n";

  CircuitSeq *seq4 = nullptr;
  if (!parser_6.load_qasm_str(str4, seq4)) {
    std::cout << "Unexpected parsing failure (4)." << std::endl;
    assert(false);
    return;
  }

  if (!eq_mats(seq3->get_matrix(&ctx_6), seq4->get_matrix(&ctx_6), 1e-16)) {
    std::cout << "Disagreement on constant parameters." << std::endl;
    assert(false);
  }
}

//
// Tests to_qasm_style_string with mixed constants.
//
void test_printing_halved_params() {
  ParamInfo param_info;
  Context ctx({GateType::p, GateType::rx, GateType::add, GateType::mult,
               GateType::neg, GateType::pi},
              &param_info);

  QASMParser parser(&ctx);
  parser.use_symbolic_pi(true);

#ifdef USE_RATIONAL
  std::string str = "OPENQASM 2.0;\n"
                    "include \"qelib1.inc\";\n"
                    "qreg q[1];\n"
                    "rx(2*pi/3) q[0];\n"
                    "rx(pi/6) q[0];\n"
                    "p(pi/4) q[0];\n";
#else
  std::string str = "OPENQASM 2.0;\n"
                    "include \"qelib1.inc\";\n"
                    "qreg q[1];\n"
                    "rx(2) q[0];\n"
                    "p(2) q[0];\n";
#endif

  CircuitSeq *seq = nullptr;
  if (!parser.load_qasm_str(str, seq)) {
    std::cout << "Unexpected parsing failure." << std::endl;
    assert(false);
    return;
  }

  std::string act = seq->to_qasm_style_string(&ctx, 1);
  if (act != str) {
    std::cout << "to_qasm_style_string: failed to handle halved parameters."
              << std::endl
              << act << std::endl;
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
  std::cout << "[Sybmolic Parameter Parsing Tests]" << std::endl;
  test_param_parsing();
  std::cout << "[Symbolic Summation Parsing Tests]" << std::endl;
  test_sum_parsing();
  std::cout << "[Halved Symbolic Parameters]" << std::endl;
  test_halved_param_ids();
  std::cout << "[Printing Halved Constant Parameters]" << std::endl;
  test_printing_halved_params();
}
