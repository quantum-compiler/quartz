OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
cx q[1],q[0];
tdg q[0];
x q[1];
cx q[2],q[0];
x q[0];
