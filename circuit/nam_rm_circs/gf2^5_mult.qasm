OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
h q[10];
h q[11];
h q[12];
h q[13];
cx q[6],q[10];
rz(pi*-0.250000) q[10];
cx q[4],q[10];
rz(pi*0.250000) q[10];
cx q[6],q[10];
rz(pi*-0.250000) q[10];
cx q[4],q[10];
cx q[4],q[6];
cx q[7],q[10];
rz(pi*-0.250000) q[6];
rz(pi*0.250000) q[10];
cx q[4],q[6];
cx q[3],q[10];
rz(pi*1.250000) q[4];
rz(pi*-0.250000) q[10];
cx q[7],q[10];
rz(pi*0.250000) q[10];
cx q[3],q[10];
cx q[3],q[7];
cx q[8],q[10];
rz(pi*0.250000) q[7];
rz(pi*-0.250000) q[10];
cx q[3],q[7];
cx q[2],q[10];
cx q[7],q[11];
rz(pi*0.250000) q[10];
rz(pi*-0.250000) q[11];
cx q[8],q[10];
cx q[4],q[11];
rz(pi*-0.250000) q[10];
rz(pi*0.250000) q[11];
cx q[2],q[10];
cx q[7],q[11];
cx q[2],q[8];
cx q[9],q[10];
rz(pi*-0.250000) q[11];
rz(pi*-0.250000) q[8];
rz(pi*0.250000) q[10];
cx q[4],q[11];
cx q[2],q[8];
cx q[1],q[10];
cx q[4],q[7];
cx q[8],q[11];
rz(pi*-0.250000) q[10];
rz(pi*-0.250000) q[7];
rz(pi*0.250000) q[11];
cx q[9],q[10];
cx q[4],q[7];
cx q[3],q[11];
rz(pi*0.250000) q[10];
rz(pi*-0.250000) q[11];
cx q[1],q[10];
cx q[8],q[11];
h q[10];
cx q[1],q[9];
rz(pi*0.250000) q[11];
rz(pi*0.250000) q[9];
cx q[3],q[11];
cx q[1],q[9];
cx q[3],q[8];
cx q[9],q[11];
rz(pi*0.250000) q[8];
rz(pi*-0.250000) q[11];
cx q[3],q[8];
cx q[2],q[11];
cx q[8],q[12];
rz(pi*0.250000) q[11];
rz(pi*-0.250000) q[12];
cx q[9],q[11];
cx q[4],q[12];
rz(pi*-0.250000) q[11];
rz(pi*0.250000) q[12];
cx q[2],q[11];
cx q[8],q[12];
cx q[2],q[9];
rz(pi*0.250000) q[11];
rz(pi*-0.250000) q[12];
rz(pi*-0.250000) q[9];
h q[11];
cx q[4],q[12];
cx q[2],q[9];
cx q[4],q[8];
cx q[9],q[12];
rz(pi*-0.250000) q[8];
rz(pi*0.250000) q[12];
cx q[4],q[8];
cx q[3],q[12];
rz(pi*-0.250000) q[12];
cx q[9],q[12];
rz(pi*0.250000) q[12];
cx q[3],q[12];
h q[12];
cx q[3],q[9];
cx q[12],q[14];
rz(pi*0.250000) q[9];
h q[14];
cx q[3],q[9];
cx q[5],q[14];
cx q[9],q[13];
rz(pi*-0.250000) q[14];
rz(pi*-0.250000) q[13];
cx q[4],q[13];
rz(pi*0.250000) q[13];
cx q[9],q[13];
rz(pi*-0.250000) q[13];
cx q[4],q[13];
cx q[4],q[9];
rz(pi*0.250000) q[13];
rz(pi*-0.250000) q[9];
h q[13];
cx q[4],q[9];
cx q[13],q[10];
cx q[4],q[14];
cx q[11],q[13];
cx q[10],q[12];
rz(pi*0.250000) q[14];
h q[13];
h q[11];
h q[12];
h q[10];
cx q[5],q[14];
rz(pi*-0.250000) q[14];
cx q[4],q[14];
cx q[4],q[5];
cx q[6],q[14];
rz(pi*-0.250000) q[5];
rz(pi*0.250000) q[14];
cx q[4],q[5];
cx q[3],q[14];
cx q[5],q[13];
rz(pi*-0.250000) q[14];
rz(pi*0.250000) q[13];
cx q[6],q[14];
rz(pi*0.250000) q[14];
cx q[3],q[14];
cx q[3],q[6];
cx q[7],q[14];
rz(pi*0.250000) q[6];
rz(pi*-0.250000) q[14];
cx q[3],q[6];
cx q[2],q[14];
cx q[3],q[13];
rz(pi*0.250000) q[14];
rz(pi*-0.250000) q[13];
cx q[7],q[14];
cx q[5],q[13];
rz(pi*-0.250000) q[14];
rz(pi*0.250000) q[13];
cx q[2],q[14];
cx q[3],q[13];
cx q[2],q[7];
cx q[8],q[14];
cx q[3],q[5];
cx q[6],q[13];
rz(pi*-0.250000) q[7];
rz(pi*0.250000) q[14];
rz(pi*0.250000) q[5];
rz(pi*-0.250000) q[13];
cx q[2],q[7];
cx q[1],q[14];
cx q[3],q[5];
cx q[2],q[13];
rz(pi*-0.250000) q[14];
rz(pi*-1.250000) q[3];
cx q[5],q[12];
rz(pi*0.250000) q[13];
cx q[8],q[14];
rz(pi*-0.250000) q[12];
cx q[6],q[13];
rz(pi*0.250000) q[14];
rz(pi*-0.250000) q[13];
cx q[1],q[14];
cx q[2],q[13];
cx q[1],q[8];
cx q[9],q[14];
cx q[2],q[6];
cx q[7],q[13];
rz(pi*0.250000) q[8];
rz(pi*-0.250000) q[14];
rz(pi*-0.250000) q[6];
rz(pi*0.250000) q[13];
cx q[1],q[8];
cx q[0],q[14];
cx q[2],q[6];
cx q[1],q[13];
rz(pi*0.250000) q[14];
cx q[2],q[12];
rz(pi*-0.250000) q[13];
cx q[9],q[14];
rz(pi*0.250000) q[12];
cx q[7],q[13];
rz(pi*-0.250000) q[14];
cx q[5],q[12];
rz(pi*0.250000) q[13];
cx q[0],q[14];
rz(pi*-0.250000) q[12];
cx q[1],q[13];
cx q[0],q[9];
rz(pi*0.250000) q[14];
cx q[2],q[12];
cx q[1],q[7];
cx q[8],q[13];
rz(pi*-0.250000) q[9];
h q[14];
cx q[2],q[5];
cx q[6],q[12];
rz(pi*0.250000) q[7];
rz(pi*-0.250000) q[13];
cx q[0],q[9];
rz(pi*-0.250000) q[5];
rz(pi*0.250000) q[12];
cx q[1],q[7];
rz(pi*0.250000) q[9];
cx q[0],q[13];
cx q[2],q[5];
cx q[1],q[12];
rz(pi*0.250000) q[13];
rz(pi*1.250000) q[2];
cx q[5],q[11];
rz(pi*-0.250000) q[12];
cx q[8],q[13];
rz(pi*0.250000) q[11];
cx q[6],q[12];
rz(pi*-0.250000) q[13];
rz(pi*0.250000) q[12];
cx q[0],q[13];
cx q[1],q[12];
h q[13];
cx q[0],q[8];
cx q[1],q[6];
cx q[7],q[12];
rz(pi*-0.250000) q[8];
rz(pi*0.250000) q[6];
rz(pi*-0.250000) q[12];
cx q[0],q[8];
cx q[1],q[6];
rz(pi*1.250000) q[0];
rz(pi*0.250000) q[8];
rz(pi*-1.250000) q[1];
cx q[0],q[12];
cx q[1],q[11];
rz(pi*0.250000) q[12];
rz(pi*-0.250000) q[11];
cx q[7],q[12];
cx q[5],q[11];
rz(pi*-0.250000) q[12];
rz(pi*0.250000) q[11];
cx q[0],q[12];
cx q[1],q[11];
cx q[0],q[7];
rz(pi*0.250000) q[12];
cx q[1],q[5];
cx q[6],q[11];
rz(pi*-0.250000) q[7];
h q[12];
rz(pi*0.250000) q[5];
rz(pi*-0.250000) q[11];
cx q[0],q[7];
cx q[1],q[5];
rz(pi*0.250000) q[7];
cx q[0],q[11];
cx q[5],q[10];
rz(pi*0.250000) q[11];
rz(pi*-0.250000) q[10];
cx q[6],q[11];
rz(pi*-0.250000) q[11];
cx q[0],q[11];
h q[11];
cx q[0],q[6];
rz(pi*-0.250000) q[6];
cx q[0],q[6];
rz(pi*0.250000) q[6];
cx q[0],q[10];
rz(pi*0.250000) q[10];
cx q[5],q[10];
rz(pi*-0.250000) q[10];
cx q[0],q[10];
cx q[0],q[5];
rz(pi*0.250000) q[10];
rz(pi*-0.250000) q[5];
h q[10];
cx q[0],q[5];
rz(pi*0.250000) q[5];