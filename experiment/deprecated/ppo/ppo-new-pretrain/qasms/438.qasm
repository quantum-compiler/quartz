OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
tdg q[0];
cx q[2],q[1];
tdg q[0];
cx q[2],q[0];
cx q[0],q[1];
