OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[9],q[16];
rz(-pi/4) q[16];
cx q[7],q[16];
rz(pi/4) q[16];
cx q[9],q[16];
rz(-pi/4) q[16];
cx q[7],q[16];
rz(pi/4) q[16];
cx q[10],q[16];
rz(pi/4) q[16];
cx q[6],q[16];
rz(-pi/4) q[16];
cx q[10],q[16];
rz(pi/4) q[16];
cx q[6],q[16];
rz(-pi/4) q[16];
cx q[11],q[16];
rz(-pi/4) q[16];
cx q[5],q[16];
rz(pi/4) q[16];
cx q[11],q[16];
rz(-pi/4) q[16];
cx q[5],q[16];
rz(pi/4) q[16];
cx q[12],q[16];
rz(pi/4) q[16];
cx q[4],q[16];
rz(-pi/4) q[16];
cx q[12],q[16];
rz(pi/4) q[16];
cx q[4],q[16];
rz(-pi/4) q[16];
cx q[13],q[16];
rz(-pi/4) q[16];
cx q[3],q[16];
rz(pi/4) q[16];
cx q[13],q[16];
rz(-pi/4) q[16];
cx q[3],q[16];
rz(pi/4) q[16];
cx q[14],q[16];
rz(pi/4) q[16];
cx q[2],q[16];
rz(-pi/4) q[16];
cx q[14],q[16];
rz(pi/4) q[16];
cx q[2],q[16];
rz(-pi/4) q[16];
cx q[15],q[16];
rz(-pi/4) q[16];
cx q[1],q[16];
rz(pi/4) q[16];
cx q[15],q[16];
rz(-pi/4) q[16];
cx q[1],q[16];
cx q[1],q[15];
rz(-pi/4) q[15];
cx q[1],q[15];
rz(pi/4) q[1];
rz(pi/4) q[15];
rz(3*pi/4) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[2],q[14];
rz(pi/4) q[14];
cx q[2],q[14];
rz(-pi/4) q[14];
rz(-pi/4) q[2];
cx q[3],q[13];
rz(-pi/4) q[13];
cx q[3],q[13];
rz(pi/4) q[13];
rz(pi/4) q[3];
cx q[4],q[12];
rz(pi/4) q[12];
cx q[4],q[12];
rz(-pi/4) q[12];
rz(-pi/4) q[4];
cx q[5],q[11];
rz(-pi/4) q[11];
cx q[5],q[11];
rz(pi/4) q[11];
rz(pi/4) q[5];
cx q[6],q[10];
rz(pi/4) q[10];
cx q[6],q[10];
rz(-pi/4) q[10];
rz(-pi/4) q[6];
cx q[7],q[9];
rz(-pi/4) q[9];
cx q[7],q[9];
rz(pi/4) q[7];
rz(pi/4) q[9];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[10],q[17];
rz(-pi/4) q[17];
cx q[7],q[17];
rz(pi/4) q[17];
cx q[10],q[17];
rz(-pi/4) q[17];
cx q[7],q[17];
rz(pi/4) q[17];
cx q[11],q[17];
rz(pi/4) q[17];
cx q[6],q[17];
rz(-pi/4) q[17];
cx q[11],q[17];
rz(pi/4) q[17];
cx q[6],q[17];
rz(-pi/4) q[17];
cx q[12],q[17];
rz(-pi/4) q[17];
cx q[5],q[17];
rz(pi/4) q[17];
cx q[12],q[17];
rz(-pi/4) q[17];
cx q[5],q[17];
rz(pi/4) q[17];
cx q[13],q[17];
rz(pi/4) q[17];
cx q[4],q[17];
rz(-pi/4) q[17];
cx q[13],q[17];
rz(pi/4) q[17];
cx q[4],q[17];
rz(-pi/4) q[17];
cx q[14],q[17];
rz(-pi/4) q[17];
cx q[3],q[17];
rz(pi/4) q[17];
cx q[14],q[17];
rz(-pi/4) q[17];
cx q[3],q[17];
rz(pi/4) q[17];
cx q[15],q[17];
rz(pi/4) q[17];
cx q[2],q[17];
rz(-pi/4) q[17];
cx q[15],q[17];
rz(pi/4) q[17];
cx q[2],q[17];
rz(pi/4) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[2],q[15];
rz(pi/4) q[15];
cx q[2],q[15];
rz(-pi/4) q[15];
rz(-pi/4) q[2];
cx q[3],q[14];
rz(-pi/4) q[14];
cx q[3],q[14];
rz(pi/4) q[14];
rz(pi/4) q[3];
cx q[4],q[13];
rz(pi/4) q[13];
cx q[4],q[13];
rz(-pi/4) q[13];
rz(-pi/4) q[4];
cx q[5],q[12];
rz(-pi/4) q[12];
cx q[5],q[12];
rz(pi/4) q[12];
rz(pi/4) q[5];
cx q[6],q[11];
rz(pi/4) q[11];
cx q[6],q[11];
rz(-pi/4) q[11];
rz(-pi/4) q[6];
cx q[7],q[10];
rz(-pi/4) q[10];
cx q[7],q[10];
rz(pi/4) q[10];
rz(pi/4) q[7];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[11],q[18];
rz(-pi/4) q[18];
cx q[7],q[18];
rz(pi/4) q[18];
cx q[11],q[18];
rz(-pi/4) q[18];
cx q[7],q[18];
rz(pi/4) q[18];
cx q[12],q[18];
rz(pi/4) q[18];
cx q[6],q[18];
rz(-pi/4) q[18];
cx q[12],q[18];
rz(pi/4) q[18];
cx q[6],q[18];
rz(-pi/4) q[18];
cx q[13],q[18];
rz(-pi/4) q[18];
cx q[5],q[18];
rz(pi/4) q[18];
cx q[13],q[18];
rz(-pi/4) q[18];
cx q[5],q[18];
rz(pi/4) q[18];
cx q[14],q[18];
rz(pi/4) q[18];
cx q[4],q[18];
rz(-pi/4) q[18];
cx q[14],q[18];
rz(pi/4) q[18];
cx q[4],q[18];
rz(-pi/4) q[18];
cx q[15],q[18];
rz(-pi/4) q[18];
cx q[3],q[18];
rz(pi/4) q[18];
cx q[15],q[18];
rz(-pi/4) q[18];
cx q[3],q[18];
rz(3*pi/4) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[3],q[15];
rz(-pi/4) q[15];
cx q[3],q[15];
rz(pi/4) q[15];
rz(pi/4) q[3];
cx q[4],q[14];
rz(pi/4) q[14];
cx q[4],q[14];
rz(-pi/4) q[14];
rz(-pi/4) q[4];
cx q[5],q[13];
rz(-pi/4) q[13];
cx q[5],q[13];
rz(pi/4) q[13];
rz(pi/4) q[5];
cx q[6],q[12];
rz(pi/4) q[12];
cx q[6],q[12];
rz(-pi/4) q[12];
rz(-pi/4) q[6];
cx q[7],q[11];
rz(-pi/4) q[11];
cx q[7],q[11];
rz(pi/4) q[11];
rz(pi/4) q[7];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[12],q[19];
rz(-pi/4) q[19];
cx q[7],q[19];
rz(pi/4) q[19];
cx q[12],q[19];
rz(-pi/4) q[19];
cx q[7],q[19];
rz(pi/4) q[19];
cx q[13],q[19];
rz(pi/4) q[19];
cx q[6],q[19];
rz(-pi/4) q[19];
cx q[13],q[19];
rz(pi/4) q[19];
cx q[6],q[19];
rz(-pi/4) q[19];
cx q[14],q[19];
rz(-pi/4) q[19];
cx q[5],q[19];
rz(pi/4) q[19];
cx q[14],q[19];
rz(-pi/4) q[19];
cx q[5],q[19];
rz(pi/4) q[19];
cx q[15],q[19];
rz(pi/4) q[19];
cx q[4],q[19];
rz(-pi/4) q[19];
cx q[15],q[19];
rz(pi/4) q[19];
cx q[4],q[19];
rz(pi/4) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[4],q[15];
rz(pi/4) q[15];
cx q[4],q[15];
rz(-pi/4) q[15];
rz(-pi/4) q[4];
cx q[5],q[14];
rz(-pi/4) q[14];
cx q[5],q[14];
rz(pi/4) q[14];
rz(pi/4) q[5];
cx q[6],q[13];
rz(pi/4) q[13];
cx q[6],q[13];
rz(-pi/4) q[13];
rz(-pi/4) q[6];
cx q[7],q[12];
rz(-pi/4) q[12];
cx q[7],q[12];
rz(pi/4) q[12];
rz(pi/4) q[7];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[13],q[20];
rz(-pi/4) q[20];
cx q[7],q[20];
rz(pi/4) q[20];
cx q[13],q[20];
rz(-pi/4) q[20];
cx q[7],q[20];
rz(pi/4) q[20];
cx q[14],q[20];
rz(pi/4) q[20];
cx q[6],q[20];
rz(-pi/4) q[20];
cx q[14],q[20];
rz(pi/4) q[20];
cx q[6],q[20];
rz(-pi/4) q[20];
cx q[15],q[20];
rz(-pi/4) q[20];
cx q[5],q[20];
rz(pi/4) q[20];
cx q[15],q[20];
rz(-pi/4) q[20];
cx q[5],q[20];
rz(3*pi/4) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[5],q[15];
rz(-pi/4) q[15];
cx q[5],q[15];
rz(pi/4) q[15];
rz(pi/4) q[5];
cx q[6],q[14];
rz(pi/4) q[14];
cx q[6],q[14];
rz(-pi/4) q[14];
rz(-pi/4) q[6];
cx q[7],q[13];
rz(-pi/4) q[13];
cx q[7],q[13];
rz(pi/4) q[13];
rz(pi/4) q[7];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[14],q[21];
rz(-pi/4) q[21];
cx q[7],q[21];
rz(pi/4) q[21];
cx q[14],q[21];
rz(-pi/4) q[21];
cx q[7],q[21];
rz(pi/4) q[21];
cx q[15],q[21];
rz(pi/4) q[21];
cx q[6],q[21];
rz(-pi/4) q[21];
cx q[15],q[21];
rz(pi/4) q[21];
cx q[6],q[21];
rz(pi/4) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[6],q[15];
rz(pi/4) q[15];
cx q[6],q[15];
rz(-pi/4) q[15];
rz(-pi/4) q[6];
cx q[7],q[14];
rz(-pi/4) q[14];
cx q[7],q[14];
rz(pi/4) q[14];
rz(pi/4) q[7];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[15],q[22];
rz(-pi/4) q[22];
cx q[7],q[22];
rz(pi/4) q[22];
cx q[15],q[22];
rz(-pi/4) q[22];
cx q[7],q[22];
rz(3*pi/4) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[22],q[18];
cx q[22],q[17];
cx q[21],q[17];
cx q[22],q[16];
cx q[21],q[16];
cx q[20],q[16];
cx q[7],q[15];
rz(-pi/4) q[15];
cx q[7],q[15];
rz(pi/4) q[15];
rz(pi/4) q[7];
cx q[21],q[23];
cx q[20],q[23];
cx q[19],q[23];
cx q[20],q[22];
cx q[19],q[22];
cx q[18],q[22];
cx q[19],q[21];
cx q[18],q[21];
cx q[17],q[21];
cx q[18],q[20];
cx q[17],q[20];
cx q[16],q[20];
cx q[17],q[19];
cx q[16],q[19];
cx q[16],q[18];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[8],q[23];
rz(-pi/4) q[23];
cx q[7],q[23];
rz(pi/4) q[23];
cx q[8],q[23];
rz(-pi/4) q[23];
cx q[7],q[23];
rz(pi/4) q[23];
cx q[7],q[8];
rz(-pi/4) q[8];
cx q[7],q[8];
rz(pi/4) q[7];
rz(pi/4) q[8];
cx q[8],q[22];
rz(pi/4) q[22];
cx q[9],q[23];
rz(pi/4) q[23];
cx q[6],q[23];
rz(-pi/4) q[23];
cx q[9],q[23];
rz(pi/4) q[23];
cx q[6],q[23];
rz(-pi/4) q[23];
cx q[10],q[23];
rz(-pi/4) q[23];
cx q[5],q[23];
rz(pi/4) q[23];
cx q[10],q[23];
rz(-pi/4) q[23];
cx q[5],q[23];
rz(pi/4) q[23];
cx q[11],q[23];
rz(pi/4) q[23];
cx q[4],q[23];
rz(-pi/4) q[23];
cx q[11],q[23];
rz(pi/4) q[23];
cx q[4],q[23];
rz(-pi/4) q[23];
cx q[12],q[23];
rz(-pi/4) q[23];
cx q[3],q[23];
rz(pi/4) q[23];
cx q[12],q[23];
rz(-pi/4) q[23];
cx q[3],q[23];
rz(pi/4) q[23];
cx q[13],q[23];
rz(pi/4) q[23];
cx q[2],q[23];
rz(-pi/4) q[23];
cx q[13],q[23];
rz(pi/4) q[23];
cx q[2],q[23];
cx q[2],q[13];
rz(pi/4) q[13];
cx q[2],q[13];
rz(-pi/4) q[13];
rz(-pi/4) q[2];
rz(-pi/4) q[23];
cx q[14],q[23];
rz(pi/4) q[23];
cx q[1],q[23];
rz(-pi/4) q[23];
cx q[14],q[23];
rz(pi/4) q[23];
cx q[1],q[23];
cx q[1],q[14];
rz(pi/4) q[14];
cx q[1],q[14];
rz(-pi/4) q[1];
rz(-pi/4) q[14];
rz(-pi/4) q[23];
cx q[15],q[23];
rz(-pi/4) q[23];
cx q[0],q[23];
rz(pi/4) q[23];
cx q[15],q[23];
rz(-pi/4) q[23];
cx q[0],q[23];
cx q[0],q[15];
rz(-pi/4) q[15];
cx q[0],q[15];
rz(pi/4) q[0];
rz(pi/4) q[15];
rz(3*pi/4) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[3],q[12];
rz(-pi/4) q[12];
cx q[3],q[12];
rz(pi/4) q[12];
rz(pi/4) q[3];
cx q[4],q[11];
rz(pi/4) q[11];
cx q[4],q[11];
rz(-pi/4) q[11];
rz(-pi/4) q[4];
cx q[5],q[10];
rz(-pi/4) q[10];
cx q[5],q[10];
rz(pi/4) q[10];
rz(pi/4) q[5];
cx q[6],q[9];
rz(pi/4) q[9];
cx q[6],q[9];
rz(-pi/4) q[6];
cx q[6],q[22];
rz(-pi/4) q[22];
cx q[8],q[22];
rz(pi/4) q[22];
cx q[6],q[22];
rz(-pi/4) q[22];
cx q[6],q[8];
rz(pi/4) q[8];
cx q[6],q[8];
rz(-pi/4) q[6];
rz(-pi/4) q[8];
cx q[8],q[21];
rz(-pi/4) q[21];
rz(-pi/4) q[9];
cx q[9],q[22];
rz(-pi/4) q[22];
cx q[5],q[22];
rz(pi/4) q[22];
cx q[9],q[22];
rz(-pi/4) q[22];
cx q[5],q[22];
rz(pi/4) q[22];
cx q[10],q[22];
rz(pi/4) q[22];
cx q[4],q[22];
rz(-pi/4) q[22];
cx q[10],q[22];
rz(pi/4) q[22];
cx q[4],q[22];
rz(-pi/4) q[22];
cx q[11],q[22];
rz(-pi/4) q[22];
cx q[3],q[22];
rz(pi/4) q[22];
cx q[11],q[22];
rz(-pi/4) q[22];
cx q[3],q[22];
rz(pi/4) q[22];
cx q[12],q[22];
rz(pi/4) q[22];
cx q[2],q[22];
rz(-pi/4) q[22];
cx q[12],q[22];
rz(pi/4) q[22];
cx q[2],q[22];
cx q[2],q[12];
rz(pi/4) q[12];
cx q[2],q[12];
rz(-pi/4) q[12];
rz(-pi/4) q[2];
rz(-pi/4) q[22];
cx q[13],q[22];
rz(-pi/4) q[22];
cx q[1],q[22];
rz(pi/4) q[22];
cx q[13],q[22];
rz(-pi/4) q[22];
cx q[1],q[22];
cx q[1],q[13];
rz(-pi/4) q[13];
cx q[1],q[13];
rz(pi/4) q[1];
rz(pi/4) q[13];
rz(pi/4) q[22];
cx q[14],q[22];
rz(-pi/4) q[22];
cx q[0],q[22];
rz(pi/4) q[22];
cx q[14],q[22];
rz(-pi/4) q[22];
cx q[0],q[22];
cx q[0],q[14];
rz(-pi/4) q[14];
cx q[0],q[14];
rz(pi/4) q[0];
rz(pi/4) q[14];
rz(3*pi/4) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[3],q[11];
rz(-pi/4) q[11];
cx q[3],q[11];
rz(pi/4) q[11];
rz(pi/4) q[3];
cx q[4],q[10];
rz(pi/4) q[10];
cx q[4],q[10];
rz(-pi/4) q[10];
rz(-pi/4) q[4];
cx q[5],q[9];
rz(-pi/4) q[9];
cx q[5],q[9];
rz(pi/4) q[5];
cx q[5],q[21];
rz(pi/4) q[21];
cx q[8],q[21];
rz(-pi/4) q[21];
cx q[5],q[21];
rz(pi/4) q[21];
cx q[5],q[8];
rz(-pi/4) q[8];
cx q[5],q[8];
rz(pi/4) q[5];
rz(pi/4) q[8];
cx q[8],q[20];
rz(pi/4) q[20];
rz(pi/4) q[9];
cx q[9],q[21];
rz(pi/4) q[21];
cx q[4],q[21];
rz(-pi/4) q[21];
cx q[9],q[21];
rz(pi/4) q[21];
cx q[4],q[21];
rz(-pi/4) q[21];
cx q[10],q[21];
rz(-pi/4) q[21];
cx q[3],q[21];
rz(pi/4) q[21];
cx q[10],q[21];
rz(-pi/4) q[21];
cx q[3],q[21];
rz(pi/4) q[21];
cx q[11],q[21];
rz(pi/4) q[21];
cx q[2],q[21];
rz(-pi/4) q[21];
cx q[11],q[21];
rz(pi/4) q[21];
cx q[2],q[21];
cx q[2],q[11];
rz(pi/4) q[11];
cx q[2],q[11];
rz(-pi/4) q[11];
rz(-pi/4) q[2];
rz(-pi/4) q[21];
cx q[12],q[21];
rz(pi/4) q[21];
cx q[1],q[21];
rz(-pi/4) q[21];
cx q[12],q[21];
rz(pi/4) q[21];
cx q[1],q[21];
cx q[1],q[12];
rz(pi/4) q[12];
cx q[1],q[12];
rz(-pi/4) q[1];
rz(-pi/4) q[12];
rz(-pi/4) q[21];
cx q[13],q[21];
rz(-pi/4) q[21];
cx q[0],q[21];
rz(pi/4) q[21];
cx q[13],q[21];
rz(-pi/4) q[21];
cx q[0],q[21];
cx q[0],q[13];
rz(-pi/4) q[13];
cx q[0],q[13];
rz(pi/4) q[0];
rz(pi/4) q[13];
rz(3*pi/4) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[3],q[10];
rz(-pi/4) q[10];
cx q[3],q[10];
rz(pi/4) q[10];
rz(pi/4) q[3];
cx q[4],q[9];
rz(pi/4) q[9];
cx q[4],q[9];
rz(-pi/4) q[4];
cx q[4],q[20];
rz(-pi/4) q[20];
cx q[8],q[20];
rz(pi/4) q[20];
cx q[4],q[20];
rz(-pi/4) q[20];
cx q[4],q[8];
rz(pi/4) q[8];
cx q[4],q[8];
rz(-pi/4) q[4];
rz(-pi/4) q[8];
cx q[8],q[19];
rz(-pi/4) q[19];
rz(-pi/4) q[9];
cx q[9],q[20];
rz(-pi/4) q[20];
cx q[3],q[20];
rz(pi/4) q[20];
cx q[9],q[20];
rz(-pi/4) q[20];
cx q[3],q[20];
rz(pi/4) q[20];
cx q[10],q[20];
rz(pi/4) q[20];
cx q[2],q[20];
rz(-pi/4) q[20];
cx q[10],q[20];
rz(pi/4) q[20];
cx q[2],q[20];
cx q[2],q[10];
rz(pi/4) q[10];
cx q[2],q[10];
rz(-pi/4) q[10];
rz(-pi/4) q[2];
rz(-pi/4) q[20];
cx q[11],q[20];
rz(-pi/4) q[20];
cx q[1],q[20];
rz(pi/4) q[20];
cx q[11],q[20];
rz(-pi/4) q[20];
cx q[1],q[20];
cx q[1],q[11];
rz(-pi/4) q[11];
cx q[1],q[11];
rz(pi/4) q[1];
rz(pi/4) q[11];
rz(pi/4) q[20];
cx q[12],q[20];
rz(-pi/4) q[20];
cx q[0],q[20];
rz(pi/4) q[20];
cx q[12],q[20];
rz(-pi/4) q[20];
cx q[0],q[20];
cx q[0],q[12];
rz(-pi/4) q[12];
cx q[0],q[12];
rz(pi/4) q[0];
rz(pi/4) q[12];
rz(3*pi/4) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[3],q[9];
rz(-pi/4) q[9];
cx q[3],q[9];
rz(pi/4) q[3];
cx q[3],q[19];
rz(pi/4) q[19];
cx q[8],q[19];
rz(-pi/4) q[19];
cx q[3],q[19];
rz(pi/4) q[19];
cx q[3],q[8];
rz(-pi/4) q[8];
cx q[3],q[8];
rz(pi/4) q[3];
rz(pi/4) q[8];
cx q[8],q[18];
rz(pi/4) q[18];
rz(pi/4) q[9];
cx q[9],q[19];
rz(pi/4) q[19];
cx q[2],q[19];
rz(-pi/4) q[19];
cx q[9],q[19];
rz(pi/4) q[19];
cx q[2],q[19];
rz(-pi/4) q[19];
cx q[10],q[19];
rz(pi/4) q[19];
cx q[1],q[19];
rz(-pi/4) q[19];
cx q[10],q[19];
rz(pi/4) q[19];
cx q[1],q[19];
cx q[1],q[10];
rz(pi/4) q[10];
cx q[1],q[10];
rz(-pi/4) q[1];
rz(-pi/4) q[10];
rz(-pi/4) q[19];
cx q[11],q[19];
rz(-pi/4) q[19];
cx q[0],q[19];
rz(pi/4) q[19];
cx q[11],q[19];
rz(-pi/4) q[19];
cx q[0],q[19];
cx q[0],q[11];
rz(-pi/4) q[11];
cx q[0],q[11];
rz(pi/4) q[0];
rz(pi/4) q[11];
rz(3*pi/4) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[2],q[9];
rz(pi/4) q[9];
cx q[2],q[9];
rz(-pi/4) q[2];
cx q[2],q[18];
rz(-pi/4) q[18];
cx q[8],q[18];
rz(pi/4) q[18];
cx q[2],q[18];
rz(-pi/4) q[18];
cx q[2],q[8];
rz(pi/4) q[8];
cx q[2],q[8];
rz(-pi/4) q[2];
rz(-pi/4) q[8];
cx q[8],q[17];
rz(pi/4) q[17];
rz(-pi/4) q[9];
cx q[9],q[18];
rz(-pi/4) q[18];
cx q[1],q[18];
rz(pi/4) q[18];
cx q[9],q[18];
rz(-pi/4) q[18];
cx q[1],q[18];
cx q[1],q[9];
rz(pi/4) q[18];
cx q[10],q[18];
rz(-pi/4) q[18];
cx q[0],q[18];
rz(pi/4) q[18];
cx q[10],q[18];
rz(-pi/4) q[18];
cx q[0],q[18];
cx q[0],q[10];
rz(-pi/4) q[10];
cx q[0],q[10];
rz(pi/4) q[0];
rz(pi/4) q[10];
rz(3*pi/4) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi/4) q[9];
cx q[1],q[9];
rz(pi/4) q[1];
cx q[1],q[17];
rz(-pi/4) q[17];
cx q[8],q[17];
rz(pi/4) q[17];
cx q[1],q[17];
cx q[1],q[8];
rz(-pi/4) q[17];
rz(pi/4) q[8];
cx q[1],q[8];
rz(-pi/4) q[1];
rz(-pi/4) q[8];
cx q[8],q[16];
rz(-pi/4) q[16];
rz(pi/4) q[9];
cx q[9],q[17];
rz(-pi/4) q[17];
cx q[0],q[17];
rz(pi/4) q[17];
cx q[9],q[17];
rz(-pi/4) q[17];
cx q[0],q[17];
cx q[0],q[9];
rz(3*pi/4) q[17];
sx q[17];
rz(pi/2) q[17];
rz(-pi/4) q[9];
cx q[0],q[9];
rz(pi/4) q[0];
cx q[0],q[16];
rz(pi/4) q[16];
cx q[8],q[16];
rz(-pi/4) q[16];
cx q[0],q[16];
cx q[0],q[8];
rz(3*pi/4) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-pi/4) q[8];
cx q[0],q[8];
rz(pi/4) q[0];
rz(pi/4) q[8];
rz(pi/4) q[9];
