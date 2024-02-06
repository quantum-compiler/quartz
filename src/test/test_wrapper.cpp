#include "oracle.h"

#include <iostream>
#include <stdio.h>
#include <string>
int main() {
  std::string my_circ = std::string("OPENQASM 2.0;\
include \"qelib1.inc\";\
qreg q[1];\
h q[0];\
h q[0];\
h q[0];\
h q[0];\
h q[0];\
h q[0];\
h q[0];\
h q[0];\
h q[0];");
  std::cout << optimize_(my_circ, "depth",
                         "/home/pengyul/quicr/soam/resources/oracles/"
                         "Nam_3_3_complete_ECC_set.json",
                         1);
  return 0;
}