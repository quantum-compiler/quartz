OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
cx q[1],q[0];
cx q[0],q[1];
cx q[0],q[2];
t q[1];
cx q[2],q[1];
