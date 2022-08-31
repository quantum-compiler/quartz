OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
cx q[0],q[2];
x q[0];
t q[2];
cx q[0],q[1];
x q[2];
