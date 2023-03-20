// Benchmark was created by MQT Bench on 2022-08-30
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 0.1.0
// Qiskit version: {'qiskit-terra': '0.20.0', 'qiskit-aer': '0.10.4', 'qiskit-ignis': '0.7.0', 'qiskit-ibmq-provider': '0.19.0', 'qiskit-aqua': None, 'qiskit': '0.36.0', 'qiskit-nature': '0.3.1', 'qiskit-finance': '0.3.1', 'qiskit-optimization': '0.3.2', 'qiskit-machine-learning': '0.4.0'}
// Used Gate Set: ['rxx', 'rz', 'ry', 'rx', 'measure']

OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg meas[2];
ry(-0.468040773022127) q[0];
ry(pi/2) q[0];
ry(2.87095075136648) q[1];
rxx(pi/2) q[0],q[1];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(2.044126937339) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[1];
ry(-1.01652317160523) q[1];
rxx(pi/2) q[0],q[1];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(0.476017561708842) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[1];
ry(1.59154255461749) q[1];
rxx(pi/2) q[0],q[1];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
ry(2.05525465103479) q[0];
rx(-pi/2) q[1];
ry(2.72337423162104) q[1];
barrier q[0],q[1];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
