OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
ccz q[0], q[1], q[2];
x q[0];
ccz q[0], q[1], q[2];
