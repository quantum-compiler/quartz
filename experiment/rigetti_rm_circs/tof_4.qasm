OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
cz q[1],q[4];
x q[4];
rz(pi*0.500000) q[4];
rx1 q[4];
rz(pi*-0.500000) q[4];
rz(pi*-0.250000) q[4];
x q[4];
rz(pi*0.500000) q[4];
rx1 q[4];
rz(pi*-0.500000) q[4];
cz q[0],q[4];
x q[4];
rz(pi*0.500000) q[4];
rx1 q[4];
rz(pi*-0.500000) q[4];
rz(pi*0.250000) q[4];
x q[4];
rz(pi*0.500000) q[4];
rx1 q[4];
rz(pi*-0.500000) q[4];
cz q[1],q[4];
x q[4];
rz(pi*0.500000) q[4];
rx1 q[4];
rz(pi*-0.500000) q[4];
rz(pi*-0.250000) q[4];
x q[4];
rz(pi*0.500000) q[4];
rx1 q[4];
rz(pi*-0.500000) q[4];
cz q[0],q[4];
x q[4];
rz(pi*0.500000) q[4];
rx1 q[4];
rz(pi*-0.500000) q[4];
rz(pi*0.250000) q[4];
x q[4];
rz(pi*0.500000) q[4];
rx1 q[4];
rz(pi*-0.500000) q[4];
cz q[4],q[5];
x q[5];
rz(pi*0.500000) q[5];
rx1 q[5];
rz(pi*-0.500000) q[5];
rz(pi*-0.250000) q[5];
x q[5];
rz(pi*0.500000) q[5];
rx1 q[5];
rz(pi*-0.500000) q[5];
cz q[2],q[5];
x q[5];
rz(pi*0.500000) q[5];
rx1 q[5];
rz(pi*-0.500000) q[5];
rz(pi*0.250000) q[5];
x q[5];
rz(pi*0.500000) q[5];
rx1 q[5];
rz(pi*-0.500000) q[5];
cz q[4],q[5];
x q[5];
rz(pi*0.500000) q[5];
rx1 q[5];
rz(pi*-0.500000) q[5];
rz(pi*-0.250000) q[5];
x q[5];
rz(pi*0.500000) q[5];
rx1 q[5];
rz(pi*-0.500000) q[5];
cz q[2],q[5];
x q[5];
rz(pi*0.500000) q[5];
rx1 q[5];
rz(pi*-0.500000) q[5];
rz(pi*0.250000) q[5];
x q[5];
rz(pi*0.500000) q[5];
rx1 q[5];
rz(pi*-0.500000) q[5];
cz q[5],q[6];
x q[6];
rz(pi*0.500000) q[6];
rx1 q[6];
rz(pi*-0.500000) q[6];
rz(pi*-0.250000) q[6];
x q[6];
rz(pi*0.500000) q[6];
rx1 q[6];
rz(pi*-0.500000) q[6];
cz q[3],q[6];
x q[6];
rz(pi*0.500000) q[6];
rx1 q[6];
rz(pi*-0.500000) q[6];
rz(pi*0.250000) q[6];
x q[6];
rz(pi*0.500000) q[6];
rx1 q[6];
rz(pi*-0.500000) q[6];
cz q[5],q[6];
x q[6];
x q[5];
rz(pi*0.500000) q[6];
rz(pi*0.500000) q[5];
rx1 q[6];
rx1 q[5];
rz(pi*-0.500000) q[6];
rz(pi*-0.500000) q[5];
rz(pi*-0.250000) q[6];
x q[6];
rz(pi*0.500000) q[6];
rx1 q[6];
rz(pi*-0.500000) q[6];
cz q[3],q[6];
cz q[3],q[5];
x q[6];
x q[5];
rz(pi*0.500000) q[6];
rz(pi*0.500000) q[5];
rx1 q[6];
rx1 q[5];
rz(pi*-0.500000) q[6];
rz(pi*-0.500000) q[5];
rz(pi*0.250000) q[6];
rz(pi*-0.250000) q[5];
x q[6];
x q[5];
rz(pi*0.500000) q[6];
rz(pi*0.500000) q[5];
rx1 q[6];
rx1 q[5];
rz(pi*-0.500000) q[6];
rz(pi*-0.500000) q[5];
cz q[3],q[5];
rz(pi*0.250000) q[3];
x q[5];
rz(pi*0.500000) q[5];
rx1 q[5];
rz(pi*-0.500000) q[5];
rz(pi*0.250000) q[5];
cz q[4],q[5];
x q[5];
rz(pi*0.500000) q[5];
rx1 q[5];
rz(pi*-0.500000) q[5];
rz(pi*0.250000) q[5];
x q[5];
rz(pi*0.500000) q[5];
rx1 q[5];
rz(pi*-0.500000) q[5];
cz q[2],q[5];
x q[5];
rz(pi*0.500000) q[5];
rx1 q[5];
rz(pi*-0.500000) q[5];
rz(pi*-0.250000) q[5];
x q[5];
rz(pi*0.500000) q[5];
rx1 q[5];
rz(pi*-0.500000) q[5];
cz q[4],q[5];
cz q[1],q[4];
x q[5];
x q[4];
rz(pi*0.500000) q[5];
rz(pi*0.500000) q[4];
rx1 q[5];
rx1 q[4];
rz(pi*-0.500000) q[5];
rz(pi*-0.500000) q[4];
rz(pi*0.250000) q[5];
rz(pi*0.250000) q[4];
x q[5];
x q[4];
rz(pi*0.500000) q[5];
rz(pi*0.500000) q[4];
rx1 q[5];
rx1 q[4];
rz(pi*-0.500000) q[5];
rz(pi*-0.500000) q[4];
cz q[2],q[5];
cz q[0],q[4];
x q[5];
x q[4];
rz(pi*0.500000) q[5];
rz(pi*0.500000) q[4];
rx1 q[5];
rx1 q[4];
rz(pi*-0.500000) q[5];
rz(pi*-0.500000) q[4];
rz(pi*-0.250000) q[5];
rz(pi*-0.250000) q[4];
x q[5];
x q[4];
rz(pi*0.500000) q[5];
rz(pi*0.500000) q[4];
rx1 q[5];
rx1 q[4];
rz(pi*-0.500000) q[5];
rz(pi*-0.500000) q[4];
cz q[1],q[4];
x q[4];
rz(pi*0.500000) q[4];
rx1 q[4];
rz(pi*-0.500000) q[4];
rz(pi*0.250000) q[4];
x q[4];
rz(pi*0.500000) q[4];
rx1 q[4];
rz(pi*-0.500000) q[4];
cz q[0],q[4];
x q[4];
rz(pi*0.500000) q[4];
rx1 q[4];
rz(pi*-0.500000) q[4];
rz(pi*-0.250000) q[4];
x q[4];
rz(pi*0.500000) q[4];
rx1 q[4];
rz(pi*-0.500000) q[4];
