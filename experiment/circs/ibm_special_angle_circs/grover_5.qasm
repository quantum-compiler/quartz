OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
tdg q[0];
tdg q[0];
sx q[0];
t q[0];
t q[0];
tdg q[1];
tdg q[1];
sx q[1];
t q[1];
t q[1];
t q[2];
t q[2];
sx q[2];
t q[2];
t q[2];
t q[3];
t q[3];
sx q[3];
t q[3];
t q[3];
tdg q[4];
tdg q[4];
sx q[4];
t q[4];
t q[4];
x q[5];
t q[6];
t q[6];
sx q[6];
t q[6];
t q[6];
cx q[1],q[6];
tdg q[6];
cx q[0],q[6];
t q[6];
cx q[1],q[6];
tdg q[6];
cx q[0],q[6];
cx q[0],q[1];
tdg q[1];
cx q[0],q[1];
t q[0];
t q[1];
t q[6];
t q[6];
t q[6];
sx q[6];
t q[6];
t q[6];
t q[7];
t q[7];
sx q[7];
t q[7];
t q[7];
cx q[6],q[7];
tdg q[7];
cx q[2],q[7];
t q[7];
cx q[6],q[7];
tdg q[7];
cx q[2],q[7];
cx q[2],q[6];
tdg q[6];
cx q[2],q[6];
t q[2];
t q[6];
t q[7];
t q[7];
t q[7];
sx q[7];
t q[7];
t q[7];
t q[8];
t q[8];
sx q[8];
t q[8];
t q[8];
cx q[7],q[8];
tdg q[8];
cx q[3],q[8];
t q[8];
cx q[7],q[8];
tdg q[8];
cx q[3],q[8];
cx q[3],q[7];
tdg q[7];
cx q[3],q[7];
t q[3];
t q[7];
t q[8];
t q[8];
t q[8];
sx q[8];
t q[8];
t q[8];
cx q[8],q[5];
tdg q[5];
cx q[4],q[5];
t q[5];
cx q[8],q[5];
tdg q[5];
cx q[4],q[5];
cx q[4],q[8];
t q[5];
tdg q[8];
cx q[4],q[8];
x q[4];
t q[4];
t q[4];
t q[4];
t q[8];
t q[8];
t q[8];
sx q[8];
t q[8];
t q[8];
cx q[7],q[8];
t q[8];
cx q[3],q[8];
tdg q[8];
cx q[7],q[8];
t q[8];
cx q[3],q[8];
cx q[3],q[7];
t q[7];
cx q[3],q[7];
tdg q[3];
tdg q[3];
tdg q[3];
sx q[3];
t q[3];
t q[3];
t q[7];
sx q[7];
t q[7];
t q[7];
cx q[6],q[7];
t q[7];
cx q[2],q[7];
tdg q[7];
cx q[6],q[7];
t q[7];
cx q[2],q[7];
cx q[2],q[6];
t q[6];
cx q[2],q[6];
tdg q[2];
tdg q[2];
tdg q[2];
sx q[2];
t q[2];
t q[2];
t q[6];
sx q[6];
t q[6];
t q[6];
cx q[1],q[6];
t q[6];
cx q[0],q[6];
tdg q[6];
cx q[1],q[6];
t q[6];
cx q[0],q[6];
cx q[0],q[1];
t q[1];
cx q[0],q[1];
tdg q[0];
tdg q[0];
tdg q[0];
sx q[0];
tdg q[0];
tdg q[0];
tdg q[1];
tdg q[1];
tdg q[1];
sx q[1];
tdg q[1];
tdg q[1];
tdg q[6];
cx q[1],q[6];
tdg q[6];
cx q[0],q[6];
t q[6];
cx q[1],q[6];
tdg q[6];
cx q[0],q[6];
cx q[0],q[1];
tdg q[1];
cx q[0],q[1];
t q[0];
t q[1];
t q[6];
t q[6];
t q[6];
sx q[6];
t q[6];
t q[6];
tdg q[7];
cx q[6],q[7];
tdg q[7];
cx q[2],q[7];
t q[7];
cx q[6],q[7];
tdg q[7];
cx q[2],q[7];
cx q[2],q[6];
tdg q[6];
cx q[2],q[6];
t q[2];
t q[6];
t q[7];
t q[7];
t q[7];
sx q[7];
t q[7];
t q[7];
cx q[7],q[4];
tdg q[4];
cx q[3],q[4];
t q[4];
cx q[7],q[4];
tdg q[4];
cx q[3],q[4];
cx q[3],q[7];
x q[4];
t q[4];
t q[4];
t q[4];
tdg q[7];
cx q[3],q[7];
t q[3];
t q[3];
t q[3];
sx q[3];
tdg q[3];
tdg q[3];
t q[7];
t q[7];
t q[7];
sx q[7];
t q[7];
t q[7];
cx q[6],q[7];
t q[7];
cx q[2],q[7];
tdg q[7];
cx q[6],q[7];
t q[7];
cx q[2],q[7];
cx q[2],q[6];
t q[6];
cx q[2],q[6];
t q[2];
sx q[2];
tdg q[2];
tdg q[2];
t q[6];
sx q[6];
t q[6];
t q[6];
cx q[1],q[6];
t q[6];
cx q[0],q[6];
tdg q[6];
cx q[1],q[6];
t q[6];
cx q[0],q[6];
cx q[0],q[1];
t q[1];
cx q[0],q[1];
tdg q[0];
tdg q[0];
tdg q[0];
sx q[0];
tdg q[0];
tdg q[0];
tdg q[1];
tdg q[1];
tdg q[1];
sx q[1];
tdg q[1];
tdg q[1];
tdg q[6];
cx q[1],q[6];
tdg q[6];
cx q[0],q[6];
t q[6];
cx q[1],q[6];
tdg q[6];
cx q[0],q[6];
cx q[0],q[1];
tdg q[1];
cx q[0],q[1];
t q[0];
t q[1];
t q[6];
t q[6];
t q[6];
sx q[6];
t q[6];
t q[6];
tdg q[7];
cx q[6],q[7];
tdg q[7];
cx q[2],q[7];
t q[7];
cx q[6],q[7];
tdg q[7];
cx q[2],q[7];
cx q[2],q[6];
tdg q[6];
cx q[2],q[6];
t q[2];
t q[6];
t q[7];
t q[7];
t q[7];
sx q[7];
t q[7];
t q[7];
tdg q[8];
cx q[7],q[8];
tdg q[8];
cx q[3],q[8];
t q[8];
cx q[7],q[8];
tdg q[8];
cx q[3],q[8];
cx q[3],q[7];
tdg q[7];
cx q[3],q[7];
t q[3];
t q[7];
t q[8];
t q[8];
t q[8];
sx q[8];
t q[8];
t q[8];
cx q[8],q[5];
tdg q[5];
cx q[4],q[5];
t q[5];
cx q[8],q[5];
tdg q[5];
cx q[4],q[5];
cx q[4],q[8];
t q[5];
tdg q[8];
cx q[4],q[8];
x q[4];
t q[4];
t q[4];
t q[4];
t q[8];
t q[8];
t q[8];
sx q[8];
t q[8];
t q[8];
cx q[7],q[8];
t q[8];
cx q[3],q[8];
tdg q[8];
cx q[7],q[8];
t q[8];
cx q[3],q[8];
cx q[3],q[7];
t q[7];
cx q[3],q[7];
tdg q[3];
tdg q[3];
tdg q[3];
sx q[3];
t q[3];
t q[3];
t q[7];
sx q[7];
t q[7];
t q[7];
cx q[6],q[7];
t q[7];
cx q[2],q[7];
tdg q[7];
cx q[6],q[7];
t q[7];
cx q[2],q[7];
cx q[2],q[6];
t q[6];
cx q[2],q[6];
tdg q[2];
tdg q[2];
tdg q[2];
sx q[2];
t q[2];
t q[2];
t q[6];
sx q[6];
t q[6];
t q[6];
cx q[1],q[6];
t q[6];
cx q[0],q[6];
tdg q[6];
cx q[1],q[6];
t q[6];
cx q[0],q[6];
cx q[0],q[1];
t q[1];
cx q[0],q[1];
tdg q[0];
tdg q[0];
tdg q[0];
sx q[0];
tdg q[0];
tdg q[0];
tdg q[1];
tdg q[1];
tdg q[1];
sx q[1];
tdg q[1];
tdg q[1];
tdg q[6];
cx q[1],q[6];
tdg q[6];
cx q[0],q[6];
t q[6];
cx q[1],q[6];
tdg q[6];
cx q[0],q[6];
cx q[0],q[1];
tdg q[1];
cx q[0],q[1];
t q[0];
t q[1];
t q[6];
t q[6];
t q[6];
sx q[6];
t q[6];
t q[6];
tdg q[7];
cx q[6],q[7];
tdg q[7];
cx q[2],q[7];
t q[7];
cx q[6],q[7];
tdg q[7];
cx q[2],q[7];
cx q[2],q[6];
tdg q[6];
cx q[2],q[6];
t q[2];
t q[6];
t q[7];
t q[7];
t q[7];
sx q[7];
t q[7];
t q[7];
cx q[7],q[4];
tdg q[4];
cx q[3],q[4];
t q[4];
cx q[7],q[4];
tdg q[4];
cx q[3],q[4];
cx q[3],q[7];
x q[4];
t q[4];
t q[4];
t q[4];
tdg q[7];
cx q[3],q[7];
t q[3];
t q[3];
t q[3];
sx q[3];
tdg q[3];
tdg q[3];
t q[7];
t q[7];
t q[7];
sx q[7];
t q[7];
t q[7];
cx q[6],q[7];
t q[7];
cx q[2],q[7];
tdg q[7];
cx q[6],q[7];
t q[7];
cx q[2],q[7];
cx q[2],q[6];
t q[6];
cx q[2],q[6];
t q[2];
sx q[2];
tdg q[2];
tdg q[2];
t q[6];
sx q[6];
t q[6];
t q[6];
cx q[1],q[6];
t q[6];
cx q[0],q[6];
tdg q[6];
cx q[1],q[6];
t q[6];
cx q[0],q[6];
cx q[0],q[1];
t q[1];
cx q[0],q[1];
tdg q[0];
tdg q[0];
tdg q[0];
sx q[0];
tdg q[0];
tdg q[0];
tdg q[1];
tdg q[1];
tdg q[1];
sx q[1];
tdg q[1];
tdg q[1];
tdg q[6];
cx q[1],q[6];
tdg q[6];
cx q[0],q[6];
t q[6];
cx q[1],q[6];
tdg q[6];
cx q[0],q[6];
cx q[0],q[1];
tdg q[1];
cx q[0],q[1];
t q[0];
t q[1];
t q[6];
t q[6];
t q[6];
sx q[6];
t q[6];
t q[6];
tdg q[7];
cx q[6],q[7];
tdg q[7];
cx q[2],q[7];
t q[7];
cx q[6],q[7];
tdg q[7];
cx q[2],q[7];
cx q[2],q[6];
tdg q[6];
cx q[2],q[6];
t q[2];
t q[6];
t q[7];
t q[7];
t q[7];
sx q[7];
t q[7];
t q[7];
tdg q[8];
cx q[7],q[8];
tdg q[8];
cx q[3],q[8];
t q[8];
cx q[7],q[8];
tdg q[8];
cx q[3],q[8];
cx q[3],q[7];
tdg q[7];
cx q[3],q[7];
t q[3];
t q[7];
t q[8];
t q[8];
t q[8];
sx q[8];
t q[8];
t q[8];
cx q[8],q[5];
tdg q[5];
cx q[4],q[5];
t q[5];
cx q[8],q[5];
tdg q[5];
cx q[4],q[5];
cx q[4],q[8];
t q[5];
tdg q[8];
cx q[4],q[8];
x q[4];
t q[4];
t q[4];
t q[4];
t q[8];
t q[8];
t q[8];
sx q[8];
t q[8];
t q[8];
cx q[7],q[8];
t q[8];
cx q[3],q[8];
tdg q[8];
cx q[7],q[8];
t q[8];
cx q[3],q[8];
cx q[3],q[7];
t q[7];
cx q[3],q[7];
tdg q[3];
tdg q[3];
tdg q[3];
sx q[3];
t q[3];
t q[3];
t q[7];
sx q[7];
t q[7];
t q[7];
cx q[6],q[7];
t q[7];
cx q[2],q[7];
tdg q[7];
cx q[6],q[7];
t q[7];
cx q[2],q[7];
cx q[2],q[6];
t q[6];
cx q[2],q[6];
tdg q[2];
tdg q[2];
tdg q[2];
sx q[2];
t q[2];
t q[2];
t q[6];
sx q[6];
t q[6];
t q[6];
cx q[1],q[6];
t q[6];
cx q[0],q[6];
tdg q[6];
cx q[1],q[6];
t q[6];
cx q[0],q[6];
cx q[0],q[1];
t q[1];
cx q[0],q[1];
tdg q[0];
tdg q[0];
tdg q[0];
sx q[0];
tdg q[0];
tdg q[0];
tdg q[1];
tdg q[1];
tdg q[1];
sx q[1];
tdg q[1];
tdg q[1];
tdg q[6];
cx q[1],q[6];
tdg q[6];
cx q[0],q[6];
t q[6];
cx q[1],q[6];
tdg q[6];
cx q[0],q[6];
cx q[0],q[1];
tdg q[1];
cx q[0],q[1];
t q[0];
t q[1];
t q[6];
t q[6];
t q[6];
sx q[6];
t q[6];
t q[6];
tdg q[7];
cx q[6],q[7];
tdg q[7];
cx q[2],q[7];
t q[7];
cx q[6],q[7];
tdg q[7];
cx q[2],q[7];
cx q[2],q[6];
tdg q[6];
cx q[2],q[6];
t q[2];
t q[6];
t q[7];
t q[7];
t q[7];
sx q[7];
t q[7];
t q[7];
cx q[7],q[4];
tdg q[4];
cx q[3],q[4];
t q[4];
cx q[7],q[4];
tdg q[4];
cx q[3],q[4];
cx q[3],q[7];
x q[4];
t q[4];
t q[4];
t q[4];
tdg q[7];
cx q[3],q[7];
t q[3];
t q[3];
t q[3];
sx q[3];
tdg q[3];
tdg q[3];
t q[7];
t q[7];
t q[7];
sx q[7];
t q[7];
t q[7];
cx q[6],q[7];
t q[7];
cx q[2],q[7];
tdg q[7];
cx q[6],q[7];
t q[7];
cx q[2],q[7];
cx q[2],q[6];
t q[6];
cx q[2],q[6];
t q[2];
sx q[2];
tdg q[2];
tdg q[2];
t q[6];
sx q[6];
t q[6];
t q[6];
cx q[1],q[6];
t q[6];
cx q[0],q[6];
tdg q[6];
cx q[1],q[6];
t q[6];
cx q[0],q[6];
cx q[0],q[1];
t q[1];
cx q[0],q[1];
tdg q[0];
tdg q[0];
tdg q[0];
sx q[0];
tdg q[0];
tdg q[0];
tdg q[1];
tdg q[1];
tdg q[1];
sx q[1];
tdg q[1];
tdg q[1];
tdg q[6];
cx q[1],q[6];
tdg q[6];
cx q[0],q[6];
t q[6];
cx q[1],q[6];
tdg q[6];
cx q[0],q[6];
cx q[0],q[1];
tdg q[1];
cx q[0],q[1];
t q[0];
t q[1];
t q[6];
t q[6];
t q[6];
sx q[6];
t q[6];
t q[6];
tdg q[7];
cx q[6],q[7];
tdg q[7];
cx q[2],q[7];
t q[7];
cx q[6],q[7];
tdg q[7];
cx q[2],q[7];
cx q[2],q[6];
tdg q[6];
cx q[2],q[6];
t q[2];
t q[6];
t q[7];
t q[7];
t q[7];
sx q[7];
t q[7];
t q[7];
tdg q[8];
cx q[7],q[8];
tdg q[8];
cx q[3],q[8];
t q[8];
cx q[7],q[8];
tdg q[8];
cx q[3],q[8];
cx q[3],q[7];
tdg q[7];
cx q[3],q[7];
t q[3];
t q[7];
t q[8];
t q[8];
t q[8];
sx q[8];
t q[8];
t q[8];
cx q[8],q[5];
tdg q[5];
cx q[4],q[5];
t q[5];
cx q[8],q[5];
tdg q[5];
cx q[4],q[5];
cx q[4],q[8];
t q[5];
t q[5];
t q[5];
sx q[5];
t q[5];
t q[5];
tdg q[8];
cx q[4],q[8];
x q[4];
t q[4];
t q[4];
t q[4];
t q[8];
t q[8];
t q[8];
sx q[8];
t q[8];
t q[8];
cx q[7],q[8];
t q[8];
cx q[3],q[8];
tdg q[8];
cx q[7],q[8];
t q[8];
cx q[3],q[8];
cx q[3],q[7];
t q[7];
cx q[3],q[7];
tdg q[3];
tdg q[3];
tdg q[3];
sx q[3];
t q[3];
t q[3];
t q[7];
sx q[7];
t q[7];
t q[7];
cx q[6],q[7];
t q[7];
cx q[2],q[7];
tdg q[7];
cx q[6],q[7];
t q[7];
cx q[2],q[7];
cx q[2],q[6];
t q[6];
cx q[2],q[6];
tdg q[2];
tdg q[2];
tdg q[2];
sx q[2];
t q[2];
t q[2];
t q[6];
sx q[6];
t q[6];
t q[6];
cx q[1],q[6];
t q[6];
cx q[0],q[6];
tdg q[6];
cx q[1],q[6];
t q[6];
cx q[0],q[6];
cx q[0],q[1];
t q[1];
cx q[0],q[1];
tdg q[0];
tdg q[0];
tdg q[0];
sx q[0];
tdg q[0];
tdg q[0];
tdg q[1];
tdg q[1];
tdg q[1];
sx q[1];
tdg q[1];
tdg q[1];
tdg q[6];
cx q[1],q[6];
tdg q[6];
cx q[0],q[6];
t q[6];
cx q[1],q[6];
tdg q[6];
cx q[0],q[6];
cx q[0],q[1];
tdg q[1];
cx q[0],q[1];
t q[0];
t q[1];
t q[6];
t q[6];
t q[6];
sx q[6];
t q[6];
t q[6];
tdg q[7];
cx q[6],q[7];
tdg q[7];
cx q[2],q[7];
t q[7];
cx q[6],q[7];
tdg q[7];
cx q[2],q[7];
cx q[2],q[6];
tdg q[6];
cx q[2],q[6];
t q[2];
t q[6];
t q[7];
t q[7];
t q[7];
sx q[7];
t q[7];
t q[7];
cx q[7],q[4];
tdg q[4];
cx q[3],q[4];
t q[4];
cx q[7],q[4];
tdg q[4];
cx q[3],q[4];
cx q[3],q[7];
tdg q[4];
tdg q[4];
tdg q[4];
tdg q[7];
cx q[3],q[7];
t q[3];
t q[3];
t q[3];
sx q[3];
tdg q[3];
tdg q[3];
t q[7];
t q[7];
t q[7];
sx q[7];
t q[7];
t q[7];
cx q[6],q[7];
t q[7];
cx q[2],q[7];
tdg q[7];
cx q[6],q[7];
t q[7];
cx q[2],q[7];
cx q[2],q[6];
t q[6];
cx q[2],q[6];
t q[2];
sx q[2];
tdg q[2];
tdg q[2];
t q[6];
sx q[6];
t q[6];
t q[6];
cx q[1],q[6];
t q[6];
cx q[0],q[6];
tdg q[6];
cx q[1],q[6];
t q[6];
cx q[0],q[6];
cx q[0],q[1];
t q[1];
cx q[0],q[1];
t q[0];
sx q[0];
tdg q[0];
tdg q[0];
t q[1];
sx q[1];
tdg q[1];
tdg q[1];
t q[6];
sx q[6];
t q[6];
t q[6];
t q[7];
sx q[7];
t q[7];
t q[7];
t q[8];
sx q[8];
t q[8];
t q[8];
