#include "oracle.h"

#include <iostream>
#include <stdio.h>
#include <string>
int main() {
  //   std::string my_circ = std::string(R"(OPENQASM 2.0;
  // include "qelib1.inc";
  // qreg q[24];
  // cx q[5], q[8];
  // rz(-0.7853981633974483) q[11];
  // rz(-0.7853981633974483) q[12];
  // rz(-0.7853981633974483) q[18];
  // rz(-0.7853981633974483) q[19];
  // cx q[5], q[6];
  // rz(0.7853981633974483) q[8];
  // cx q[12], q[11];
  // cx q[19], q[18];
  // rz(-0.7853981633974483) q[6];
  // h q[8];
  // cx q[12], q[14];
  // cx q[19], q[21];
  // cx q[5], q[6];
  // cx q[8], q[10];
  // cx q[9], q[12];
  // h q[14];
  // cx q[15], q[19];
  // h q[21];
  // h q[21];)");
  std::string my_circ = std::string(R"(OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
tdg q[8];
t q[8];)");
  std::cout << optimize_(my_circ, "gate",
                         //  "/home/pengyul/quicr/soam/resources/oracles/"
                         //  "Nam_3_3_complete_ECC_set.json",
                         "/home/pengyul/quicr/soam/resources/oracles/"
                         "quartz_src/build/Clifford_4_3_complete_ECC_set.json",
                         "Nam", 5);
  return 0;
}