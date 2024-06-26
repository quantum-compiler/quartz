// Benchmark was created by MQT Bench on 2022-08-30
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 0.1.0
// Qiskit version: {'qiskit-terra': '0.20.0', 'qiskit-aer': '0.10.4', 'qiskit-ignis': '0.7.0', 'qiskit-ibmq-provider': '0.19.0', 'qiskit-aqua': None, 'qiskit': '0.36.0', 'qiskit-nature': '0.3.1', 'qiskit-finance': '0.3.1', 'qiskit-optimization': '0.3.2', 'qiskit-machine-learning': '0.4.0'}
// Used Gate Set: ['rxx', 'rz', 'ry', 'rx', 'measure']

OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg meas[10];
ry(-1.0252989334081) q[0];
ry(pi/2) q[0];
ry(0.868724734484185) q[1];
ry(pi/2) q[1];
rx(pi) q[1];
rxx(pi/2) q[0],q[1];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[1];
ry(pi/2) q[1];
rx(pi) q[1];
ry(pi/2) q[1];
ry(-3.67740483888856) q[2];
ry(pi/2) q[2];
rx(pi) q[2];
rxx(pi/2) q[0],q[2];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[2];
ry(pi/2) q[2];
rx(pi) q[2];
ry(pi/2) q[2];
rx(pi) q[2];
rxx(pi/2) q[1],q[2];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[2];
ry(pi/2) q[2];
rx(pi) q[2];
ry(pi/2) q[2];
ry(0.909401312595822) q[3];
ry(pi/2) q[3];
rx(pi) q[3];
rxx(pi/2) q[0],q[3];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[3];
ry(pi/2) q[3];
rx(pi) q[3];
ry(pi/2) q[3];
rx(pi) q[3];
rxx(pi/2) q[1],q[3];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[3];
ry(pi/2) q[3];
rx(pi) q[3];
ry(pi/2) q[3];
rx(pi) q[3];
rxx(pi/2) q[2],q[3];
rx(-pi/2) q[2];
ry(-pi/2) q[2];
ry(pi/2) q[2];
rx(-pi/2) q[3];
ry(pi/2) q[3];
rx(pi) q[3];
ry(pi/2) q[3];
ry(3.09196939117687) q[4];
ry(pi/2) q[4];
rx(pi) q[4];
rxx(pi/2) q[0],q[4];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[4];
ry(pi/2) q[4];
rx(pi) q[4];
ry(pi/2) q[4];
rx(pi) q[4];
rxx(pi/2) q[1],q[4];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[4];
ry(pi/2) q[4];
rx(pi) q[4];
ry(pi/2) q[4];
rx(pi) q[4];
rxx(pi/2) q[2],q[4];
rx(-pi/2) q[2];
ry(-pi/2) q[2];
ry(pi/2) q[2];
rx(-pi/2) q[4];
ry(pi/2) q[4];
rx(pi) q[4];
ry(pi/2) q[4];
rx(pi) q[4];
rxx(pi/2) q[3],q[4];
rx(-pi/2) q[3];
ry(-pi/2) q[3];
ry(pi/2) q[3];
rx(-pi/2) q[4];
ry(pi/2) q[4];
rx(pi) q[4];
ry(pi/2) q[4];
ry(-4.86416512723816) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
rxx(pi/2) q[0],q[5];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
rxx(pi/2) q[1],q[5];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
rxx(pi/2) q[2],q[5];
rx(-pi/2) q[2];
ry(-pi/2) q[2];
ry(pi/2) q[2];
rx(-pi/2) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
rxx(pi/2) q[3],q[5];
rx(-pi/2) q[3];
ry(-pi/2) q[3];
ry(pi/2) q[3];
rx(-pi/2) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
rxx(pi/2) q[4],q[5];
rx(-pi/2) q[4];
ry(-pi/2) q[4];
ry(pi/2) q[4];
rx(-pi/2) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
ry(pi/2) q[5];
ry(4.37319860789649) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
rxx(pi/2) q[0],q[6];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
rxx(pi/2) q[1],q[6];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
rxx(pi/2) q[2],q[6];
rx(-pi/2) q[2];
ry(-pi/2) q[2];
ry(pi/2) q[2];
rx(-pi/2) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
rxx(pi/2) q[3],q[6];
rx(-pi/2) q[3];
ry(-pi/2) q[3];
ry(pi/2) q[3];
rx(-pi/2) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
rxx(pi/2) q[4],q[6];
rx(-pi/2) q[4];
ry(-pi/2) q[4];
ry(pi/2) q[4];
rx(-pi/2) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
rxx(pi/2) q[5],q[6];
rx(-pi/2) q[5];
ry(-pi/2) q[5];
ry(pi/2) q[5];
rx(-pi/2) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
ry(pi/2) q[6];
ry(1.27124881428196) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
rxx(pi/2) q[0],q[7];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
rxx(pi/2) q[1],q[7];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
rxx(pi/2) q[2],q[7];
rx(-pi/2) q[2];
ry(-pi/2) q[2];
ry(pi/2) q[2];
rx(-pi/2) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
rxx(pi/2) q[3],q[7];
rx(-pi/2) q[3];
ry(-pi/2) q[3];
ry(pi/2) q[3];
rx(-pi/2) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
rxx(pi/2) q[4],q[7];
rx(-pi/2) q[4];
ry(-pi/2) q[4];
ry(pi/2) q[4];
rx(-pi/2) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
rxx(pi/2) q[5],q[7];
rx(-pi/2) q[5];
ry(-pi/2) q[5];
ry(pi/2) q[5];
rx(-pi/2) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
rxx(pi/2) q[6],q[7];
rx(-pi/2) q[6];
ry(-pi/2) q[6];
ry(pi/2) q[6];
rx(-pi/2) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
ry(pi/2) q[7];
ry(-0.73797353180387) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
rxx(pi/2) q[0],q[8];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
rxx(pi/2) q[1],q[8];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
rxx(pi/2) q[2],q[8];
rx(-pi/2) q[2];
ry(-pi/2) q[2];
ry(pi/2) q[2];
rx(-pi/2) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
rxx(pi/2) q[3],q[8];
rx(-pi/2) q[3];
ry(-pi/2) q[3];
ry(pi/2) q[3];
rx(-pi/2) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
rxx(pi/2) q[4],q[8];
rx(-pi/2) q[4];
ry(-pi/2) q[4];
ry(pi/2) q[4];
rx(-pi/2) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
rxx(pi/2) q[5],q[8];
rx(-pi/2) q[5];
ry(-pi/2) q[5];
ry(pi/2) q[5];
rx(-pi/2) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
rxx(pi/2) q[6],q[8];
rx(-pi/2) q[6];
ry(-pi/2) q[6];
ry(pi/2) q[6];
rx(-pi/2) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
rxx(pi/2) q[7],q[8];
rx(-pi/2) q[7];
ry(-pi/2) q[7];
ry(pi/2) q[7];
rx(-pi/2) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
ry(pi/2) q[8];
ry(-3.58956003996621) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[0],q[9];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(-0.92625884582298) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[1],q[9];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
ry(-0.612467590426359) q[1];
ry(pi/2) q[1];
rx(pi) q[1];
rxx(pi/2) q[0],q[1];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[1];
ry(pi/2) q[1];
rx(pi) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[2],q[9];
rx(-pi/2) q[2];
ry(-pi/2) q[2];
ry(-0.118840722526631) q[2];
ry(pi/2) q[2];
rx(pi) q[2];
rxx(pi/2) q[0],q[2];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[2];
ry(pi/2) q[2];
rx(pi) q[2];
ry(pi/2) q[2];
rx(pi) q[2];
rxx(pi/2) q[1],q[2];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[2];
ry(pi/2) q[2];
rx(pi) q[2];
ry(pi/2) q[2];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[3],q[9];
rx(-pi/2) q[3];
ry(-pi/2) q[3];
ry(6.0851680307623) q[3];
ry(pi/2) q[3];
rx(pi) q[3];
rxx(pi/2) q[0],q[3];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[3];
ry(pi/2) q[3];
rx(pi) q[3];
ry(pi/2) q[3];
rx(pi) q[3];
rxx(pi/2) q[1],q[3];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[3];
ry(pi/2) q[3];
rx(pi) q[3];
ry(pi/2) q[3];
rx(pi) q[3];
rxx(pi/2) q[2],q[3];
rx(-pi/2) q[2];
ry(-pi/2) q[2];
ry(pi/2) q[2];
rx(-pi/2) q[3];
ry(pi/2) q[3];
rx(pi) q[3];
ry(pi/2) q[3];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[4],q[9];
rx(-pi/2) q[4];
ry(-pi/2) q[4];
ry(0.47236403868833) q[4];
ry(pi/2) q[4];
rx(pi) q[4];
rxx(pi/2) q[0],q[4];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[4];
ry(pi/2) q[4];
rx(pi) q[4];
ry(pi/2) q[4];
rx(pi) q[4];
rxx(pi/2) q[1],q[4];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[4];
ry(pi/2) q[4];
rx(pi) q[4];
ry(pi/2) q[4];
rx(pi) q[4];
rxx(pi/2) q[2],q[4];
rx(-pi/2) q[2];
ry(-pi/2) q[2];
ry(pi/2) q[2];
rx(-pi/2) q[4];
ry(pi/2) q[4];
rx(pi) q[4];
ry(pi/2) q[4];
rx(pi) q[4];
rxx(pi/2) q[3],q[4];
rx(-pi/2) q[3];
ry(-pi/2) q[3];
ry(pi/2) q[3];
rx(-pi/2) q[4];
ry(pi/2) q[4];
rx(pi) q[4];
ry(pi/2) q[4];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[5],q[9];
rx(-pi/2) q[5];
ry(-pi/2) q[5];
ry(1.94276400477115) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
rxx(pi/2) q[0],q[5];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
rxx(pi/2) q[1],q[5];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
rxx(pi/2) q[2],q[5];
rx(-pi/2) q[2];
ry(-pi/2) q[2];
ry(pi/2) q[2];
rx(-pi/2) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
rxx(pi/2) q[3],q[5];
rx(-pi/2) q[3];
ry(-pi/2) q[3];
ry(pi/2) q[3];
rx(-pi/2) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
rxx(pi/2) q[4],q[5];
rx(-pi/2) q[4];
ry(-pi/2) q[4];
ry(pi/2) q[4];
rx(-pi/2) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
ry(pi/2) q[5];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[6],q[9];
rx(-pi/2) q[6];
ry(-pi/2) q[6];
ry(2.78827321663537) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
rxx(pi/2) q[0],q[6];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
rxx(pi/2) q[1],q[6];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
rxx(pi/2) q[2],q[6];
rx(-pi/2) q[2];
ry(-pi/2) q[2];
ry(pi/2) q[2];
rx(-pi/2) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
rxx(pi/2) q[3],q[6];
rx(-pi/2) q[3];
ry(-pi/2) q[3];
ry(pi/2) q[3];
rx(-pi/2) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
rxx(pi/2) q[4],q[6];
rx(-pi/2) q[4];
ry(-pi/2) q[4];
ry(pi/2) q[4];
rx(-pi/2) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
rxx(pi/2) q[5],q[6];
rx(-pi/2) q[5];
ry(-pi/2) q[5];
ry(pi/2) q[5];
rx(-pi/2) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
ry(pi/2) q[6];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[7],q[9];
rx(-pi/2) q[7];
ry(-pi/2) q[7];
ry(-4.57238597345447) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
rxx(pi/2) q[0],q[7];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
rxx(pi/2) q[1],q[7];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
rxx(pi/2) q[2],q[7];
rx(-pi/2) q[2];
ry(-pi/2) q[2];
ry(pi/2) q[2];
rx(-pi/2) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
rxx(pi/2) q[3],q[7];
rx(-pi/2) q[3];
ry(-pi/2) q[3];
ry(pi/2) q[3];
rx(-pi/2) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
rxx(pi/2) q[4],q[7];
rx(-pi/2) q[4];
ry(-pi/2) q[4];
ry(pi/2) q[4];
rx(-pi/2) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
rxx(pi/2) q[5],q[7];
rx(-pi/2) q[5];
ry(-pi/2) q[5];
ry(pi/2) q[5];
rx(-pi/2) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
rxx(pi/2) q[6],q[7];
rx(-pi/2) q[6];
ry(-pi/2) q[6];
ry(pi/2) q[6];
rx(-pi/2) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
ry(pi/2) q[7];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[8],q[9];
rx(-pi/2) q[8];
ry(-pi/2) q[8];
ry(-6.04359985059162) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
rxx(pi/2) q[0],q[8];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
rxx(pi/2) q[1],q[8];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
rxx(pi/2) q[2],q[8];
rx(-pi/2) q[2];
ry(-pi/2) q[2];
ry(pi/2) q[2];
rx(-pi/2) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
rxx(pi/2) q[3],q[8];
rx(-pi/2) q[3];
ry(-pi/2) q[3];
ry(pi/2) q[3];
rx(-pi/2) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
rxx(pi/2) q[4],q[8];
rx(-pi/2) q[4];
ry(-pi/2) q[4];
ry(pi/2) q[4];
rx(-pi/2) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
rxx(pi/2) q[5],q[8];
rx(-pi/2) q[5];
ry(-pi/2) q[5];
ry(pi/2) q[5];
rx(-pi/2) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
rxx(pi/2) q[6],q[8];
rx(-pi/2) q[6];
ry(-pi/2) q[6];
ry(pi/2) q[6];
rx(-pi/2) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
rxx(pi/2) q[7],q[8];
rx(-pi/2) q[7];
ry(-pi/2) q[7];
ry(pi/2) q[7];
rx(-pi/2) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
ry(pi/2) q[8];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(6.13523223485188) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[0],q[9];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(0.551621890117284) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[1],q[9];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
ry(4.8661943564414) q[1];
ry(pi/2) q[1];
rx(pi) q[1];
rxx(pi/2) q[0],q[1];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[1];
ry(pi/2) q[1];
rx(pi) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[2],q[9];
rx(-pi/2) q[2];
ry(-pi/2) q[2];
ry(4.97399984004705) q[2];
ry(pi/2) q[2];
rx(pi) q[2];
rxx(pi/2) q[0],q[2];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[2];
ry(pi/2) q[2];
rx(pi) q[2];
ry(pi/2) q[2];
rx(pi) q[2];
rxx(pi/2) q[1],q[2];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[2];
ry(pi/2) q[2];
rx(pi) q[2];
ry(pi/2) q[2];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[3],q[9];
rx(-pi/2) q[3];
ry(-pi/2) q[3];
ry(1.51590298635271) q[3];
ry(pi/2) q[3];
rx(pi) q[3];
rxx(pi/2) q[0],q[3];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[3];
ry(pi/2) q[3];
rx(pi) q[3];
ry(pi/2) q[3];
rx(pi) q[3];
rxx(pi/2) q[1],q[3];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[3];
ry(pi/2) q[3];
rx(pi) q[3];
ry(pi/2) q[3];
rx(pi) q[3];
rxx(pi/2) q[2],q[3];
rx(-pi/2) q[2];
ry(-pi/2) q[2];
ry(pi/2) q[2];
rx(-pi/2) q[3];
ry(pi/2) q[3];
rx(pi) q[3];
ry(pi/2) q[3];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[4],q[9];
rx(-pi/2) q[4];
ry(-pi/2) q[4];
ry(-1.02599024201347) q[4];
ry(pi/2) q[4];
rx(pi) q[4];
rxx(pi/2) q[0],q[4];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[4];
ry(pi/2) q[4];
rx(pi) q[4];
ry(pi/2) q[4];
rx(pi) q[4];
rxx(pi/2) q[1],q[4];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[4];
ry(pi/2) q[4];
rx(pi) q[4];
ry(pi/2) q[4];
rx(pi) q[4];
rxx(pi/2) q[2],q[4];
rx(-pi/2) q[2];
ry(-pi/2) q[2];
ry(pi/2) q[2];
rx(-pi/2) q[4];
ry(pi/2) q[4];
rx(pi) q[4];
ry(pi/2) q[4];
rx(pi) q[4];
rxx(pi/2) q[3],q[4];
rx(-pi/2) q[3];
ry(-pi/2) q[3];
ry(pi/2) q[3];
rx(-pi/2) q[4];
ry(pi/2) q[4];
rx(pi) q[4];
ry(pi/2) q[4];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[5],q[9];
rx(-pi/2) q[5];
ry(-pi/2) q[5];
ry(1.4703088183747) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
rxx(pi/2) q[0],q[5];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
rxx(pi/2) q[1],q[5];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
rxx(pi/2) q[2],q[5];
rx(-pi/2) q[2];
ry(-pi/2) q[2];
ry(pi/2) q[2];
rx(-pi/2) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
rxx(pi/2) q[3],q[5];
rx(-pi/2) q[3];
ry(-pi/2) q[3];
ry(pi/2) q[3];
rx(-pi/2) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
rxx(pi/2) q[4],q[5];
rx(-pi/2) q[4];
ry(-pi/2) q[4];
ry(pi/2) q[4];
rx(-pi/2) q[5];
ry(pi/2) q[5];
rx(pi) q[5];
ry(pi/2) q[5];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[6],q[9];
rx(-pi/2) q[6];
ry(-pi/2) q[6];
ry(-0.294065572333419) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
rxx(pi/2) q[0],q[6];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
rxx(pi/2) q[1],q[6];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
rxx(pi/2) q[2],q[6];
rx(-pi/2) q[2];
ry(-pi/2) q[2];
ry(pi/2) q[2];
rx(-pi/2) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
rxx(pi/2) q[3],q[6];
rx(-pi/2) q[3];
ry(-pi/2) q[3];
ry(pi/2) q[3];
rx(-pi/2) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
rxx(pi/2) q[4],q[6];
rx(-pi/2) q[4];
ry(-pi/2) q[4];
ry(pi/2) q[4];
rx(-pi/2) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
rxx(pi/2) q[5],q[6];
rx(-pi/2) q[5];
ry(-pi/2) q[5];
ry(pi/2) q[5];
rx(-pi/2) q[6];
ry(pi/2) q[6];
rx(pi) q[6];
ry(pi/2) q[6];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[7],q[9];
rx(-pi/2) q[7];
ry(-pi/2) q[7];
ry(1.13205614915729) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
rxx(pi/2) q[0],q[7];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
rxx(pi/2) q[1],q[7];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
rxx(pi/2) q[2],q[7];
rx(-pi/2) q[2];
ry(-pi/2) q[2];
ry(pi/2) q[2];
rx(-pi/2) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
rxx(pi/2) q[3],q[7];
rx(-pi/2) q[3];
ry(-pi/2) q[3];
ry(pi/2) q[3];
rx(-pi/2) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
rxx(pi/2) q[4],q[7];
rx(-pi/2) q[4];
ry(-pi/2) q[4];
ry(pi/2) q[4];
rx(-pi/2) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
rxx(pi/2) q[5],q[7];
rx(-pi/2) q[5];
ry(-pi/2) q[5];
ry(pi/2) q[5];
rx(-pi/2) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
rxx(pi/2) q[6],q[7];
rx(-pi/2) q[6];
ry(-pi/2) q[6];
ry(pi/2) q[6];
rx(-pi/2) q[7];
ry(pi/2) q[7];
rx(pi) q[7];
ry(pi/2) q[7];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[8],q[9];
rx(-pi/2) q[8];
ry(-pi/2) q[8];
ry(3.93132602194527) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
rxx(pi/2) q[0],q[8];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
rxx(pi/2) q[1],q[8];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
rxx(pi/2) q[2],q[8];
rx(-pi/2) q[2];
ry(-pi/2) q[2];
ry(pi/2) q[2];
rx(-pi/2) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
rxx(pi/2) q[3],q[8];
rx(-pi/2) q[3];
ry(-pi/2) q[3];
ry(pi/2) q[3];
rx(-pi/2) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
rxx(pi/2) q[4],q[8];
rx(-pi/2) q[4];
ry(-pi/2) q[4];
ry(pi/2) q[4];
rx(-pi/2) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
rxx(pi/2) q[5],q[8];
rx(-pi/2) q[5];
ry(-pi/2) q[5];
ry(pi/2) q[5];
rx(-pi/2) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
rxx(pi/2) q[6],q[8];
rx(-pi/2) q[6];
ry(-pi/2) q[6];
ry(pi/2) q[6];
rx(-pi/2) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
rxx(pi/2) q[7],q[8];
rx(-pi/2) q[7];
ry(-pi/2) q[7];
ry(pi/2) q[7];
rx(-pi/2) q[8];
ry(pi/2) q[8];
rx(pi) q[8];
ry(pi/2) q[8];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(-4.0365811955971) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[0],q[9];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(-3.54537681268794) q[0];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[1],q[9];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
ry(-0.128011789418389) q[1];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[2],q[9];
rx(-pi/2) q[2];
ry(-pi/2) q[2];
ry(5.16019322972049) q[2];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[3],q[9];
rx(-pi/2) q[3];
ry(-pi/2) q[3];
ry(-3.05338918688024) q[3];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[4],q[9];
rx(-pi/2) q[4];
ry(-pi/2) q[4];
ry(5.85715072316761) q[4];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[5],q[9];
rx(-pi/2) q[5];
ry(-pi/2) q[5];
ry(5.5219041057866) q[5];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[6],q[9];
rx(-pi/2) q[6];
ry(-pi/2) q[6];
ry(4.29470511689634) q[6];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[7],q[9];
rx(-pi/2) q[7];
ry(-pi/2) q[7];
ry(1.03103311886819) q[7];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
rxx(pi/2) q[8],q[9];
rx(-pi/2) q[8];
ry(-pi/2) q[8];
ry(2.31383252008268) q[8];
rx(-pi/2) q[9];
ry(pi/2) q[9];
rx(pi) q[9];
ry(4.55101343035376) q[9];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
measure q[8] -> meas[8];
measure q[9] -> meas[9];
