OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[1];
cx q[1],q[0];
tdg q[1];
tdg q[1];
tdg q[1];
