#include "oracle.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"
using namespace quartz;

int main() {
  std::string my_circ = std::string(R"(OPENQASM 2.0;
  include "qelib1.inc";
  qreg q[24];
  cx q[5], q[8];
  rz(-0.123) q[11];
  rz(-0.234) q[12];
  rz(-0.7853981633974483) q[18];
  rz(-0.7853981633974483) q[19];
  cx q[5], q[6];
  rz(0.7853981633974483) q[8];
  cx q[12], q[11];
  cx q[19], q[18];
  rz(-0.7853981633974483) q[6];
  h q[8];
  cx q[12], q[14];
  cx q[19], q[21];
  cx q[5], q[6];
  cx q[8], q[10];
  cx q[9], q[12];
  h q[14];
  cx q[15], q[19]; 
  h q[21];
  h q[21];)");
  auto supercontext = get_context_("Nam", 24,
                                   "/home/pengyul/quicr/soam/resources/"
                                   "Nam_4_3_complete_ECC_set.json");
  std::cout << optimize_(my_circ, "Gate", 1, supercontext) << std::endl;
  std::cout << optimize_(my_circ, "Gate", 1, supercontext) << std::endl;
  return 0;
}