OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
x q[1];
cx q[2],q[0];
cx q[1],q[0];
t q[1];
cx q[2],q[1];
