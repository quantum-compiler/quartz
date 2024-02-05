#include "test_wrapper.h"

#include "oracle.h"

#include <iostream>
#include <stdio.h>
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
  std::cout << optimize_(my_circ);
  return 0;
}