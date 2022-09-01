OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
t q[1];
t q[1];
h q[1];
cx q[0],q[1];
x q[1];
