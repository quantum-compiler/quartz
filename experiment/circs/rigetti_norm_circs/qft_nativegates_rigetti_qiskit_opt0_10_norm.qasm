OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
creg meas[10];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
rx1 q[4];
rz(pi/2) q[4];
rz(pi/2) q[5];
rx1 q[5];
rz(pi/2) q[5];
rz(pi/2) q[6];
rx1 q[6];
rz(pi/2) q[6];
rz(pi/2) q[7];
rx1 q[7];
rz(pi/2) q[7];
rz(pi/2) q[8];
rx1 q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
rx1 q[9];
rz(pi/2) q[9];
rz(pi/4) q[9];
cz q[9],q[8];
rz(pi/2) q[8];
rx1 q[8];
rz(pi/2) q[8];
rz(-pi/4) q[8];
rz(pi/2) q[8];
rx1 q[8];
rz(pi/2) q[8];
cz q[9],q[8];
rz(pi/2) q[8];
rx1 q[8];
rz(pi/2) q[8];
rz(pi/4) q[8];
rz(pi/2) q[8];
rx1 q[8];
rz(pi/2) q[8];
rz(pi/4) q[8];
rz(pi/8) q[9];
cz q[9],q[7];
rz(pi/2) q[7];
rx1 q[7];
rz(pi/2) q[7];
rz(-pi/8) q[7];
rz(pi/2) q[7];
rx1 q[7];
rz(pi/2) q[7];
cz q[9],q[7];
rz(pi/2) q[7];
rx1 q[7];
rz(pi/2) q[7];
rz(pi/8) q[7];
rz(pi/2) q[7];
rx1 q[7];
rz(pi/2) q[7];
cz q[8],q[7];
rz(pi/2) q[7];
rx1 q[7];
rz(pi/2) q[7];
rz(-pi/4) q[7];
rz(pi/2) q[7];
rx1 q[7];
rz(pi/2) q[7];
cz q[8],q[7];
rz(pi/2) q[7];
rx1 q[7];
rz(pi/2) q[7];
rz(pi/4) q[7];
rz(pi/2) q[7];
rx1 q[7];
rz(pi/2) q[7];
rz(pi/4) q[7];
rz(pi/8) q[8];
rz(pi/16) q[9];
cz q[9],q[6];
rz(pi/2) q[6];
rx1 q[6];
rz(pi/2) q[6];
rz(-pi/16) q[6];
rz(pi/2) q[6];
rx1 q[6];
rz(pi/2) q[6];
cz q[9],q[6];
rz(pi/2) q[6];
rx1 q[6];
rz(pi/2) q[6];
rz(pi/16) q[6];
rz(pi/2) q[6];
rx1 q[6];
rz(pi/2) q[6];
cz q[8],q[6];
rz(pi/2) q[6];
rx1 q[6];
rz(pi/2) q[6];
rz(-pi/8) q[6];
rz(pi/2) q[6];
rx1 q[6];
rz(pi/2) q[6];
cz q[8],q[6];
rz(pi/2) q[6];
rx1 q[6];
rz(pi/2) q[6];
rz(pi/8) q[6];
rz(pi/2) q[6];
rx1 q[6];
rz(pi/2) q[6];
cz q[7],q[6];
rz(pi/2) q[6];
rx1 q[6];
rz(pi/2) q[6];
rz(-pi/4) q[6];
rz(pi/2) q[6];
rx1 q[6];
rz(pi/2) q[6];
cz q[7],q[6];
rz(pi/2) q[6];
rx1 q[6];
rz(pi/2) q[6];
rz(pi/4) q[6];
rz(pi/2) q[6];
rx1 q[6];
rz(pi/2) q[6];
rz(pi/4) q[6];
rz(pi/8) q[7];
rz(pi/16) q[8];
rz(pi/32) q[9];
cz q[9],q[5];
rz(pi/2) q[5];
rx1 q[5];
rz(pi/2) q[5];
rz(-pi/32) q[5];
rz(pi/2) q[5];
rx1 q[5];
rz(pi/2) q[5];
cz q[9],q[5];
rz(pi/2) q[5];
rx1 q[5];
rz(pi/2) q[5];
rz(pi/32) q[5];
rz(pi/2) q[5];
rx1 q[5];
rz(pi/2) q[5];
cz q[8],q[5];
rz(pi/2) q[5];
rx1 q[5];
rz(pi/2) q[5];
rz(-pi/16) q[5];
rz(pi/2) q[5];
rx1 q[5];
rz(pi/2) q[5];
cz q[8],q[5];
rz(pi/2) q[5];
rx1 q[5];
rz(pi/2) q[5];
rz(pi/16) q[5];
rz(pi/2) q[5];
rx1 q[5];
rz(pi/2) q[5];
cz q[7],q[5];
rz(pi/2) q[5];
rx1 q[5];
rz(pi/2) q[5];
rz(-pi/8) q[5];
rz(pi/2) q[5];
rx1 q[5];
rz(pi/2) q[5];
cz q[7],q[5];
rz(pi/2) q[5];
rx1 q[5];
rz(pi/2) q[5];
rz(pi/8) q[5];
rz(pi/2) q[5];
rx1 q[5];
rz(pi/2) q[5];
cz q[6],q[5];
rz(pi/2) q[5];
rx1 q[5];
rz(pi/2) q[5];
rz(-pi/4) q[5];
rz(pi/2) q[5];
rx1 q[5];
rz(pi/2) q[5];
cz q[6],q[5];
rz(pi/2) q[5];
rx1 q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
rz(pi/2) q[5];
rx1 q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
rz(pi/8) q[6];
rz(pi/16) q[7];
rz(pi/32) q[8];
rz(pi/64) q[9];
cz q[9],q[4];
rz(pi/2) q[4];
rx1 q[4];
rz(pi/2) q[4];
rz(-pi/64) q[4];
rz(pi/2) q[4];
rx1 q[4];
rz(pi/2) q[4];
cz q[9],q[4];
rz(pi/2) q[4];
rx1 q[4];
rz(pi/2) q[4];
rz(pi/64) q[4];
rz(pi/2) q[4];
rx1 q[4];
rz(pi/2) q[4];
cz q[8],q[4];
rz(pi/2) q[4];
rx1 q[4];
rz(pi/2) q[4];
rz(-pi/32) q[4];
rz(pi/2) q[4];
rx1 q[4];
rz(pi/2) q[4];
cz q[8],q[4];
rz(pi/2) q[4];
rx1 q[4];
rz(pi/2) q[4];
rz(pi/32) q[4];
rz(pi/2) q[4];
rx1 q[4];
rz(pi/2) q[4];
cz q[7],q[4];
rz(pi/2) q[4];
rx1 q[4];
rz(pi/2) q[4];
rz(-pi/16) q[4];
rz(pi/2) q[4];
rx1 q[4];
rz(pi/2) q[4];
cz q[7],q[4];
rz(pi/2) q[4];
rx1 q[4];
rz(pi/2) q[4];
rz(pi/16) q[4];
rz(pi/2) q[4];
rx1 q[4];
rz(pi/2) q[4];
cz q[6],q[4];
rz(pi/2) q[4];
rx1 q[4];
rz(pi/2) q[4];
rz(-pi/8) q[4];
rz(pi/2) q[4];
rx1 q[4];
rz(pi/2) q[4];
cz q[6],q[4];
rz(pi/2) q[4];
rx1 q[4];
rz(pi/2) q[4];
rz(pi/8) q[4];
rz(pi/2) q[4];
rx1 q[4];
rz(pi/2) q[4];
cz q[5],q[4];
rz(pi/2) q[4];
rx1 q[4];
rz(pi/2) q[4];
rz(-pi/4) q[4];
rz(pi/2) q[4];
rx1 q[4];
rz(pi/2) q[4];
cz q[5],q[4];
rz(pi/2) q[4];
rx1 q[4];
rz(pi/2) q[4];
rz(pi/4) q[4];
rz(pi/2) q[4];
rx1 q[4];
rz(pi/2) q[4];
rz(pi/4) q[4];
rz(pi/8) q[5];
rz(pi/16) q[6];
rz(pi/32) q[7];
rz(pi/64) q[8];
rz(pi/128) q[9];
cz q[9],q[3];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
rz(-pi/128) q[3];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
cz q[9],q[3];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
rz(pi/128) q[3];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
cz q[8],q[3];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
rz(-pi/64) q[3];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
cz q[8],q[3];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
rz(pi/64) q[3];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
cz q[7],q[3];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
rz(-pi/32) q[3];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
cz q[7],q[3];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
rz(pi/32) q[3];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
cz q[6],q[3];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
rz(-pi/16) q[3];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
cz q[6],q[3];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
rz(pi/16) q[3];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
cz q[5],q[3];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
rz(-pi/8) q[3];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
cz q[5],q[3];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
rz(pi/8) q[3];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
cz q[4],q[3];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
rz(-pi/4) q[3];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
cz q[4],q[3];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
rz(pi/4) q[3];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
rz(pi/4) q[3];
rz(pi/8) q[4];
rz(pi/16) q[5];
rz(pi/32) q[6];
rz(pi/64) q[7];
rz(pi/128) q[8];
rz(pi/256) q[9];
cz q[9],q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
rz(-pi/256) q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
cz q[9],q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
rz(pi/256) q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
cz q[8],q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
rz(-pi/128) q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
cz q[8],q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
rz(pi/128) q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
cz q[7],q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
rz(-pi/64) q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
cz q[7],q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
rz(pi/64) q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
cz q[6],q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
rz(-pi/32) q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
cz q[6],q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
rz(pi/32) q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
cz q[5],q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
rz(-pi/16) q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
cz q[5],q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
rz(pi/16) q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
cz q[4],q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
rz(-pi/8) q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
cz q[4],q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
rz(pi/8) q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
cz q[3],q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
rz(-pi/4) q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
cz q[3],q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
rz(pi/4) q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
rz(pi/4) q[2];
rz(pi/8) q[3];
rz(pi/16) q[4];
rz(pi/32) q[5];
rz(pi/64) q[6];
rz(pi/128) q[7];
rz(pi/256) q[8];
rz(pi/512) q[9];
cz q[9],q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
rz(-pi/512) q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
cz q[9],q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
rz(pi/512) q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
cz q[8],q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
rz(-pi/256) q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
cz q[8],q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
rz(pi/256) q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
cz q[7],q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
rz(-pi/128) q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
cz q[7],q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
rz(pi/128) q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
cz q[6],q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
rz(-pi/64) q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
cz q[6],q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
rz(pi/64) q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
cz q[5],q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
rz(-pi/32) q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
cz q[5],q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
rz(pi/32) q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
cz q[4],q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
rz(-pi/16) q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
cz q[4],q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
rz(pi/16) q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
cz q[3],q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
rz(-pi/8) q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
cz q[3],q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
rz(pi/8) q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
cz q[2],q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
rz(-pi/4) q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
cz q[2],q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
rz(pi/4) q[1];
rz(pi/8) q[2];
rz(pi/16) q[3];
rz(pi/32) q[4];
rz(pi/64) q[5];
rz(pi/128) q[6];
rz(pi/256) q[7];
rz(pi/512) q[8];
rz(pi/1024) q[9];
cz q[9],q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
rz(-pi/1024) q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
cz q[9],q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
rz(pi/1024) q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
cz q[8],q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
rz(-pi/512) q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
cz q[8],q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
rz(pi/512) q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
cz q[7],q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
rz(-pi/256) q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
cz q[7],q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
rz(pi/256) q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
cz q[6],q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
rz(-pi/128) q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
cz q[6],q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
rz(pi/128) q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
cz q[5],q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
rz(-pi/64) q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
cz q[5],q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
rz(pi/64) q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
cz q[4],q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
rz(-pi/32) q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
cz q[4],q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
rz(pi/32) q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
cz q[3],q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
rz(-pi/16) q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
cz q[3],q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
rz(pi/16) q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
cz q[2],q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
rz(-pi/8) q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
cz q[2],q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
rz(pi/8) q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
cz q[1],q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
rz(-pi/4) q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
cz q[1],q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
rz(pi/2) q[5];
rx1 q[5];
rz(pi/2) q[5];
cz q[4],q[5];
rz(pi/2) q[4];
rx1 q[4];
rz(pi/2) q[4];
rz(pi/2) q[5];
rx1 q[5];
rz(pi/2) q[5];
cz q[5],q[4];
rz(pi/2) q[4];
rx1 q[4];
rz(pi/2) q[4];
rz(pi/2) q[5];
rx1 q[5];
rz(pi/2) q[5];
cz q[4],q[5];
rz(pi/2) q[5];
rx1 q[5];
rz(pi/2) q[5];
rz(pi/2) q[6];
rx1 q[6];
rz(pi/2) q[6];
cz q[3],q[6];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
rz(pi/2) q[6];
rx1 q[6];
rz(pi/2) q[6];
cz q[6],q[3];
rz(pi/2) q[3];
rx1 q[3];
rz(pi/2) q[3];
rz(pi/2) q[6];
rx1 q[6];
rz(pi/2) q[6];
cz q[3],q[6];
rz(pi/2) q[6];
rx1 q[6];
rz(pi/2) q[6];
rz(pi/2) q[7];
rx1 q[7];
rz(pi/2) q[7];
cz q[2],q[7];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
rz(pi/2) q[7];
rx1 q[7];
rz(pi/2) q[7];
cz q[7],q[2];
rz(pi/2) q[2];
rx1 q[2];
rz(pi/2) q[2];
rz(pi/2) q[7];
rx1 q[7];
rz(pi/2) q[7];
cz q[2],q[7];
rz(pi/2) q[7];
rx1 q[7];
rz(pi/2) q[7];
rz(pi/2) q[8];
rx1 q[8];
rz(pi/2) q[8];
cz q[1],q[8];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
rz(pi/2) q[8];
rx1 q[8];
rz(pi/2) q[8];
cz q[8],q[1];
rz(pi/2) q[1];
rx1 q[1];
rz(pi/2) q[1];
rz(pi/2) q[8];
rx1 q[8];
rz(pi/2) q[8];
cz q[1],q[8];
rz(pi/2) q[8];
rx1 q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
rx1 q[9];
rz(pi/2) q[9];
cz q[0],q[9];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
rz(pi/2) q[9];
rx1 q[9];
rz(pi/2) q[9];
cz q[9],q[0];
rz(pi/2) q[0];
rx1 q[0];
rz(pi/2) q[0];
rz(pi/2) q[9];
rx1 q[9];
rz(pi/2) q[9];
cz q[0],q[9];
rz(pi/2) q[9];
rx1 q[9];
rz(pi/2) q[9];
