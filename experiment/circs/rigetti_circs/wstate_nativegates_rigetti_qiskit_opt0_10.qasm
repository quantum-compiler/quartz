// Benchmark was created by MQT Bench on 2022-08-31
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 0.1.0
// Qiskit version: {'qiskit-terra': '0.20.0', 'qiskit-aer': '0.10.4', 'qiskit-ignis': '0.7.0', 'qiskit-ibmq-provider': '0.19.0', 'qiskit-aqua': None, 'qiskit': '0.36.0', 'qiskit-nature': '0.3.1', 'qiskit-finance': '0.3.1', 'qiskit-optimization': '0.3.2', 'qiskit-machine-learning': '0.4.0'}
// Used Gate Set: ['rx', 'rz', 'cz', 'measure']

OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg meas[10];
rz(0.0) q[0];
rx(pi/2) q[0];
rz(3*pi/4) q[0];
rx(pi/2) q[0];
rz(3*pi) q[0];
rz(0.0) q[1];
rx(pi/2) q[1];
rz(2.18627603546528) q[1];
rx(pi/2) q[1];
rz(3*pi) q[1];
rz(0.0) q[2];
rx(pi/2) q[2];
rz(2*pi/3) q[2];
rx(pi/2) q[2];
rz(3*pi) q[2];
rz(0.0) q[3];
rx(pi/2) q[3];
rz(2.0344439357957) q[3];
rx(pi/2) q[3];
rz(3*pi) q[3];
rz(0.0) q[4];
rx(pi/2) q[4];
rz(1.99133066207886) q[4];
rx(pi/2) q[4];
rz(3*pi) q[4];
rz(0.0) q[5];
rx(pi/2) q[5];
rz(1.95839301345008) q[5];
rx(pi/2) q[5];
rz(3*pi) q[5];
rz(0.0) q[6];
rx(pi/2) q[6];
rz(1.9321634507016) q[6];
rx(pi/2) q[6];
rz(3*pi) q[6];
rz(0.0) q[7];
rx(pi/2) q[7];
rz(1.91063323624902) q[7];
rx(pi/2) q[7];
rz(3*pi) q[7];
rz(0.0) q[8];
rx(pi/2) q[8];
rz(1.89254688119154) q[8];
rx(pi/2) q[8];
rz(3*pi) q[8];
rx(pi) q[9];
cz q[9],q[8];
rz(0.0) q[8];
rx(pi/2) q[8];
rz(4.39063842598805) q[8];
rx(pi/2) q[8];
rz(3*pi) q[8];
cz q[8],q[7];
rz(0.0) q[7];
rx(pi/2) q[7];
rz(4.37255207093057) q[7];
rx(pi/2) q[7];
rz(3*pi) q[7];
cz q[7],q[6];
rz(0.0) q[6];
rx(pi/2) q[6];
rz(4.35102185647798) q[6];
rx(pi/2) q[6];
rz(3*pi) q[6];
cz q[6],q[5];
rz(0.0) q[5];
rx(pi/2) q[5];
rz(4.32479229372951) q[5];
rx(pi/2) q[5];
rz(3*pi) q[5];
cz q[5],q[4];
rz(0.0) q[4];
rx(pi/2) q[4];
rz(4.29185464510072) q[4];
rx(pi/2) q[4];
rz(3*pi) q[4];
cz q[4],q[3];
rz(0.0) q[3];
rx(pi/2) q[3];
rz(4.24874137138388) q[3];
rx(pi/2) q[3];
rz(3*pi) q[3];
cz q[3],q[2];
rz(0.0) q[2];
rx(pi/2) q[2];
rz(4*pi/3) q[2];
rx(pi/2) q[2];
rz(3*pi) q[2];
cz q[2],q[1];
rz(0.0) q[1];
rx(pi/2) q[1];
rz(4.0969092717143) q[1];
rx(pi/2) q[1];
rz(3*pi) q[1];
cz q[1],q[0];
rz(0.0) q[0];
rx(pi/2) q[0];
rz(5*pi/4) q[0];
rx(pi/2) q[0];
rz(3*pi) q[0];
rz(pi/2) q[9];
rx(pi/2) q[9];
rz(pi/2) q[9];
cz q[8],q[9];
rz(pi/2) q[8];
rx(pi/2) q[8];
rz(pi/2) q[8];
cz q[7],q[8];
rz(pi/2) q[7];
rx(pi/2) q[7];
rz(pi/2) q[7];
cz q[6],q[7];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(pi/2) q[6];
cz q[5],q[6];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
cz q[4],q[5];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(pi/2) q[4];
cz q[3],q[4];
rz(pi/2) q[3];
rx(pi/2) q[3];
rz(pi/2) q[3];
cz q[2],q[3];
rz(pi/2) q[2];
rx(pi/2) q[2];
rz(pi/2) q[2];
cz q[1],q[2];
rz(pi/2) q[1];
rx(pi/2) q[1];
rz(pi/2) q[1];
cz q[0],q[1];
rz(pi/2) q[1];
rx(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[2];
rx(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[3];
rx(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[7];
rx(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[8];
rx(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
rx(pi/2) q[9];
rz(pi/2) q[9];
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
