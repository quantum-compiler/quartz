OPENQASM 2.0;
include "qelib1.inc";
qreg q[36];
t q[2];
t q[2];
sx q[2];
t q[2];
t q[2];
cx q[1],q[2];
tdg q[2];
cx q[0],q[2];
t q[2];
cx q[1],q[2];
tdg q[2];
cx q[0],q[2];
cx q[0],q[1];
tdg q[1];
cx q[0],q[1];
t q[0];
t q[1];
t q[2];
t q[2];
t q[2];
sx q[2];
t q[2];
t q[2];
t q[5];
t q[5];
sx q[5];
t q[5];
t q[5];
cx q[4],q[5];
tdg q[5];
cx q[3],q[5];
t q[5];
cx q[4],q[5];
tdg q[5];
cx q[3],q[5];
cx q[3],q[4];
tdg q[4];
cx q[3],q[4];
t q[3];
t q[4];
cx q[3],q[4];
t q[5];
cx q[4],q[5];
tdg q[5];
cx q[2],q[5];
t q[5];
cx q[4],q[5];
tdg q[5];
cx q[2],q[5];
cx q[2],q[4];
tdg q[4];
cx q[2],q[4];
t q[2];
t q[4];
cx q[4],q[2];
cx q[3],q[4];
t q[5];
t q[5];
t q[5];
sx q[5];
t q[5];
t q[5];
t q[8];
t q[8];
sx q[8];
t q[8];
t q[8];
t q[9];
t q[9];
sx q[9];
t q[9];
t q[9];
cx q[7],q[9];
tdg q[9];
cx q[6],q[9];
t q[9];
cx q[7],q[9];
tdg q[9];
cx q[6],q[9];
cx q[6],q[7];
tdg q[7];
cx q[6],q[7];
t q[6];
t q[7];
cx q[6],q[7];
t q[9];
t q[9];
t q[9];
sx q[9];
t q[9];
t q[9];
t q[12];
t q[12];
sx q[12];
t q[12];
t q[12];
cx q[11],q[12];
tdg q[12];
cx q[10],q[12];
t q[12];
cx q[11],q[12];
tdg q[12];
cx q[10],q[12];
cx q[10],q[11];
tdg q[11];
cx q[10],q[11];
t q[10];
t q[11];
cx q[10],q[11];
cx q[11],q[8];
t q[12];
tdg q[8];
cx q[7],q[8];
t q[8];
cx q[11],q[8];
tdg q[8];
cx q[7],q[8];
cx q[7],q[11];
tdg q[11];
cx q[7],q[11];
t q[11];
cx q[11],q[12];
tdg q[12];
t q[7];
t q[8];
t q[8];
t q[8];
sx q[8];
t q[8];
t q[8];
cx q[9],q[12];
t q[12];
cx q[11],q[12];
tdg q[12];
cx q[9],q[12];
t q[12];
cx q[8],q[12];
tdg q[12];
cx q[5],q[12];
t q[12];
cx q[8],q[12];
tdg q[12];
cx q[5],q[12];
t q[12];
t q[12];
t q[12];
sx q[12];
t q[12];
t q[12];
cx q[5],q[8];
tdg q[8];
cx q[5],q[8];
t q[5];
t q[8];
t q[8];
t q[8];
sx q[8];
t q[8];
t q[8];
cx q[9],q[11];
tdg q[11];
cx q[9],q[11];
t q[11];
cx q[11],q[8];
t q[8];
t q[9];
t q[9];
t q[9];
sx q[9];
t q[9];
t q[9];
cx q[7],q[9];
t q[9];
cx q[5],q[9];
tdg q[9];
cx q[7],q[9];
t q[9];
cx q[5],q[9];
cx q[5],q[7];
t q[7];
cx q[5],q[7];
tdg q[5];
tdg q[7];
cx q[7],q[8];
tdg q[8];
cx q[11],q[8];
t q[8];
cx q[7],q[8];
cx q[7],q[11];
t q[11];
cx q[7],q[11];
tdg q[11];
tdg q[7];
cx q[7],q[5];
cx q[6],q[7];
t q[8];
sx q[8];
t q[8];
t q[8];
t q[9];
sx q[9];
t q[9];
t q[9];
cx q[11],q[9];
cx q[10],q[11];
t q[15];
t q[15];
sx q[15];
t q[15];
t q[15];
t q[16];
t q[16];
sx q[16];
t q[16];
t q[16];
cx q[14],q[16];
tdg q[16];
cx q[13],q[16];
t q[16];
cx q[14],q[16];
tdg q[16];
cx q[13],q[16];
cx q[13],q[14];
tdg q[14];
cx q[13],q[14];
t q[13];
t q[14];
cx q[13],q[14];
t q[16];
t q[16];
t q[16];
sx q[16];
t q[16];
t q[16];
t q[19];
t q[19];
sx q[19];
t q[19];
t q[19];
t q[20];
t q[20];
sx q[20];
t q[20];
t q[20];
cx q[18],q[20];
tdg q[20];
cx q[17],q[20];
t q[20];
cx q[18],q[20];
tdg q[20];
cx q[17],q[20];
cx q[17],q[18];
tdg q[18];
cx q[17],q[18];
t q[17];
t q[18];
cx q[17],q[18];
cx q[18],q[15];
tdg q[15];
cx q[14],q[15];
t q[15];
cx q[18],q[15];
tdg q[15];
cx q[14],q[15];
cx q[14],q[18];
t q[15];
t q[15];
t q[15];
sx q[15];
t q[15];
t q[15];
tdg q[18];
cx q[14],q[18];
t q[14];
t q[18];
t q[20];
cx q[18],q[20];
tdg q[20];
cx q[16],q[20];
t q[20];
cx q[18],q[20];
tdg q[20];
cx q[16],q[20];
cx q[16],q[18];
tdg q[18];
cx q[16],q[18];
t q[16];
t q[16];
t q[16];
sx q[16];
t q[16];
t q[16];
cx q[14],q[16];
tdg q[16];
t q[18];
t q[20];
t q[20];
t q[20];
sx q[20];
t q[20];
t q[20];
t q[23];
t q[23];
sx q[23];
t q[23];
t q[23];
t q[24];
t q[24];
sx q[24];
t q[24];
t q[24];
cx q[22],q[24];
tdg q[24];
cx q[21],q[24];
t q[24];
cx q[22],q[24];
tdg q[24];
cx q[21],q[24];
cx q[21],q[22];
tdg q[22];
cx q[21],q[22];
t q[21];
t q[22];
cx q[21],q[22];
t q[24];
t q[24];
t q[24];
sx q[24];
t q[24];
t q[24];
t q[27];
t q[27];
sx q[27];
t q[27];
t q[27];
cx q[26],q[27];
tdg q[27];
cx q[25],q[27];
t q[27];
cx q[26],q[27];
tdg q[27];
cx q[25],q[27];
cx q[25],q[26];
tdg q[26];
cx q[25],q[26];
t q[25];
t q[26];
cx q[25],q[26];
cx q[26],q[23];
tdg q[23];
cx q[22],q[23];
t q[23];
cx q[26],q[23];
tdg q[23];
cx q[22],q[23];
cx q[22],q[26];
t q[23];
t q[23];
t q[23];
sx q[23];
t q[23];
t q[23];
cx q[23],q[19];
tdg q[19];
cx q[15],q[19];
t q[19];
cx q[23],q[19];
tdg q[19];
cx q[15],q[19];
cx q[15],q[23];
t q[19];
t q[19];
t q[19];
sx q[19];
t q[19];
t q[19];
tdg q[23];
cx q[15],q[23];
t q[15];
t q[23];
tdg q[26];
cx q[22],q[26];
t q[22];
t q[26];
t q[27];
cx q[26],q[27];
tdg q[27];
cx q[24],q[27];
t q[27];
cx q[26],q[27];
tdg q[27];
cx q[24],q[27];
cx q[24],q[26];
tdg q[26];
cx q[24],q[26];
t q[24];
t q[24];
t q[24];
sx q[24];
t q[24];
t q[24];
cx q[22],q[24];
tdg q[24];
t q[26];
t q[27];
cx q[23],q[27];
t q[27];
cx q[20],q[27];
tdg q[27];
cx q[23],q[27];
t q[27];
cx q[20],q[27];
cx q[20],q[23];
t q[23];
cx q[20],q[23];
t q[20];
sx q[20];
t q[20];
t q[20];
cx q[15],q[20];
tdg q[20];
tdg q[23];
tdg q[27];
cx q[19],q[27];
t q[27];
cx q[12],q[27];
tdg q[27];
cx q[19],q[27];
t q[27];
cx q[12],q[27];
cx q[12],q[19];
t q[19];
cx q[12],q[19];
tdg q[12];
cx q[12],q[20];
t q[19];
sx q[19];
t q[19];
t q[19];
t q[20];
cx q[15],q[20];
tdg q[20];
cx q[12],q[20];
cx q[12],q[15];
tdg q[15];
cx q[12],q[15];
t q[12];
cx q[12],q[16];
t q[15];
t q[16];
cx q[14],q[16];
tdg q[16];
cx q[12],q[16];
cx q[12],q[14];
tdg q[14];
cx q[12],q[14];
t q[12];
t q[14];
t q[16];
t q[16];
t q[16];
sx q[16];
t q[16];
t q[16];
t q[20];
t q[20];
t q[20];
sx q[20];
t q[20];
t q[20];
cx q[20],q[24];
cx q[23],q[19];
t q[19];
cx q[15],q[19];
tdg q[19];
cx q[23],q[19];
t q[19];
cx q[15],q[19];
cx q[15],q[23];
t q[19];
sx q[19];
t q[19];
t q[19];
t q[23];
cx q[15],q[23];
t q[15];
sx q[15];
t q[15];
t q[15];
cx q[18],q[15];
t q[15];
cx q[14],q[15];
tdg q[15];
cx q[18],q[15];
t q[15];
cx q[14],q[15];
cx q[14],q[18];
t q[15];
sx q[15];
t q[15];
t q[15];
t q[18];
cx q[14],q[18];
tdg q[14];
cx q[14],q[12];
cx q[13],q[14];
tdg q[18];
cx q[18],q[16];
cx q[17],q[18];
t q[23];
sx q[23];
t q[23];
t q[23];
t q[24];
cx q[22],q[24];
tdg q[24];
cx q[20],q[24];
cx q[20],q[22];
tdg q[22];
cx q[20],q[22];
t q[20];
t q[22];
t q[24];
t q[24];
t q[24];
sx q[24];
t q[24];
t q[24];
cx q[26],q[23];
t q[23];
cx q[22],q[23];
tdg q[23];
cx q[26],q[23];
t q[23];
cx q[22],q[23];
cx q[22],q[26];
t q[23];
sx q[23];
t q[23];
t q[23];
t q[26];
cx q[22],q[26];
tdg q[22];
cx q[22],q[20];
cx q[21],q[22];
tdg q[26];
cx q[26],q[24];
cx q[25],q[26];
t q[27];
sx q[27];
t q[27];
t q[27];
t q[30];
t q[30];
sx q[30];
t q[30];
t q[30];
t q[31];
t q[31];
sx q[31];
t q[31];
t q[31];
cx q[29],q[31];
tdg q[31];
cx q[28],q[31];
t q[31];
cx q[29],q[31];
tdg q[31];
cx q[28],q[31];
cx q[28],q[29];
tdg q[29];
cx q[28],q[29];
t q[28];
t q[29];
cx q[28],q[29];
t q[31];
t q[31];
t q[31];
sx q[31];
t q[31];
t q[31];
t q[34];
t q[34];
sx q[34];
t q[34];
t q[34];
cx q[33],q[34];
tdg q[34];
cx q[32],q[34];
t q[34];
cx q[33],q[34];
tdg q[34];
cx q[32],q[34];
cx q[32],q[33];
tdg q[33];
cx q[32],q[33];
t q[32];
t q[33];
cx q[32],q[33];
cx q[33],q[30];
tdg q[30];
cx q[29],q[30];
t q[30];
cx q[33],q[30];
tdg q[30];
cx q[29],q[30];
cx q[29],q[33];
t q[30];
t q[30];
t q[30];
sx q[30];
t q[30];
t q[30];
tdg q[33];
cx q[29],q[33];
t q[29];
t q[33];
t q[34];
cx q[33],q[34];
tdg q[34];
cx q[31],q[34];
t q[34];
cx q[33],q[34];
tdg q[34];
cx q[31],q[34];
cx q[31],q[33];
tdg q[33];
cx q[31],q[33];
t q[31];
t q[31];
t q[31];
sx q[31];
t q[31];
t q[31];
cx q[29],q[31];
tdg q[31];
cx q[27],q[31];
t q[31];
cx q[29],q[31];
tdg q[31];
cx q[27],q[31];
cx q[27],q[29];
tdg q[29];
cx q[27],q[29];
t q[27];
t q[29];
t q[31];
t q[31];
t q[31];
sx q[31];
t q[31];
t q[31];
t q[33];
t q[34];
cx q[30],q[34];
t q[34];
cx q[27],q[34];
tdg q[34];
cx q[30],q[34];
t q[34];
cx q[27],q[34];
cx q[27],q[30];
t q[30];
cx q[27],q[30];
tdg q[27];
t q[30];
sx q[30];
t q[30];
t q[30];
cx q[33],q[30];
t q[30];
cx q[29],q[30];
tdg q[30];
cx q[33],q[30];
t q[30];
cx q[29],q[30];
cx q[29],q[33];
tdg q[30];
t q[33];
cx q[29],q[33];
tdg q[29];
cx q[29],q[27];
cx q[28],q[29];
tdg q[33];
cx q[33],q[30];
t q[30];
t q[30];
sx q[30];
t q[30];
t q[30];
cx q[32],q[33];
t q[34];
sx q[34];
t q[34];
t q[34];
cx q[1],q[35];
cx q[0],q[35];
