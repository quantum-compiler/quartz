OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
x q[1];
cx q[1],q[2];
cx q[0],q[1];
tdg q[2];
x q[1];
