OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
x q[0];
h q[0];
cx q[0],q[1];
t q[0];
t q[0];
