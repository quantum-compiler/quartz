OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg meas[8];
sx q[0];
sx q[1];
sx q[2];
sx q[3];
sx q[4];
sx q[5];
sx q[6];
rz(1.0*pi) q[7];
rz(2.2473530235284764*pi) q[0];
rz(1.881693988565556*pi) q[1];
rz(3.9561828748577472*pi) q[2];
rz(2.475739297961719*pi) q[3];
rz(1.2874885421665592*pi) q[4];
rz(2.987476349974218*pi) q[5];
rz(2.2099393985189693*pi) q[6];
sx q[7];
sx q[0];
sx q[1];
sx q[2];
sx q[3];
sx q[4];
sx q[5];
sx q[6];
rz(1.9561301142634866*pi) q[7];
rz(2.5*pi) q[0];
sx q[1];
rz(1.0*pi) q[2];
rz(1.0*pi) q[3];
sx q[4];
sx q[5];
rz(1.0*pi) q[6];
sx q[7];
sx q[0];
sx q[2];
sx q[3];
sx q[6];
rz(1.5*pi) q[0];
cx q[0],q[1];
sx q[0];
rz(3.0*pi) q[1];
rz(0.5*pi) q[0];
x q[1];
rz(2.5*pi) q[0];
rz(2.5*pi) q[1];
sx q[0];
sx q[1];
rz(1.5*pi) q[0];
rz(1.5*pi) q[1];
cx q[0],q[2];
sx q[0];
sx q[2];
rz(0.5*pi) q[0];
cx q[1],q[2];
rz(2.5*pi) q[0];
sx q[1];
rz(1.0*pi) q[2];
sx q[0];
rz(0.5*pi) q[1];
rz(2.5*pi) q[2];
rz(1.5*pi) q[0];
rz(2.5*pi) q[1];
sx q[2];
cx q[0],q[3];
sx q[1];
rz(1.5*pi) q[2];
sx q[0];
rz(1.5*pi) q[1];
sx q[3];
rz(0.5*pi) q[0];
cx q[1],q[3];
rz(2.5*pi) q[0];
sx q[1];
sx q[3];
sx q[0];
rz(0.5*pi) q[1];
cx q[2],q[3];
rz(1.5*pi) q[0];
rz(2.5*pi) q[1];
sx q[2];
rz(1.0*pi) q[3];
cx q[0],q[4];
sx q[1];
rz(0.5*pi) q[2];
rz(2.5*pi) q[3];
sx q[0];
rz(1.5*pi) q[1];
rz(2.5*pi) q[2];
sx q[3];
sx q[4];
rz(0.5*pi) q[0];
cx q[1],q[4];
sx q[2];
rz(1.5*pi) q[3];
rz(2.5*pi) q[0];
sx q[1];
rz(1.5*pi) q[2];
sx q[4];
sx q[0];
rz(0.5*pi) q[1];
cx q[2],q[4];
rz(1.5*pi) q[0];
rz(2.5*pi) q[1];
sx q[2];
sx q[4];
cx q[0],q[5];
sx q[1];
rz(0.5*pi) q[2];
cx q[3],q[4];
sx q[0];
rz(1.5*pi) q[1];
rz(2.5*pi) q[2];
sx q[3];
rz(3.0*pi) q[4];
sx q[5];
rz(0.5*pi) q[0];
cx q[1],q[5];
sx q[2];
rz(0.5*pi) q[3];
x q[4];
rz(2.5*pi) q[0];
sx q[1];
rz(1.5*pi) q[2];
rz(2.5*pi) q[3];
rz(2.5*pi) q[4];
sx q[5];
sx q[0];
rz(0.5*pi) q[1];
cx q[2],q[5];
sx q[3];
sx q[4];
rz(1.5*pi) q[0];
rz(2.5*pi) q[1];
sx q[2];
rz(1.5*pi) q[3];
rz(1.5*pi) q[4];
sx q[5];
cx q[0],q[6];
sx q[1];
rz(0.5*pi) q[2];
cx q[3],q[5];
sx q[0];
rz(1.5*pi) q[1];
rz(2.5*pi) q[2];
sx q[3];
sx q[5];
sx q[6];
rz(0.5*pi) q[0];
cx q[1],q[6];
sx q[2];
rz(0.5*pi) q[3];
cx q[4],q[5];
rz(2.5*pi) q[0];
sx q[1];
rz(1.5*pi) q[2];
rz(2.5*pi) q[3];
sx q[4];
rz(3.0*pi) q[5];
sx q[6];
sx q[0];
rz(0.5*pi) q[1];
cx q[2],q[6];
sx q[3];
rz(0.5*pi) q[4];
x q[5];
rz(1.5*pi) q[0];
rz(2.5*pi) q[1];
sx q[2];
rz(1.5*pi) q[3];
rz(2.5*pi) q[4];
rz(2.5*pi) q[5];
sx q[6];
cx q[0],q[7];
sx q[1];
rz(0.5*pi) q[2];
cx q[3],q[6];
sx q[4];
sx q[5];
sx q[0];
rz(1.5*pi) q[1];
rz(2.5*pi) q[2];
sx q[3];
rz(1.5*pi) q[4];
rz(1.5*pi) q[5];
sx q[6];
sx q[7];
rz(0.5*pi) q[0];
cx q[1],q[7];
sx q[2];
rz(0.5*pi) q[3];
cx q[4],q[6];
rz(1.235964023789605*pi) q[0];
sx q[1];
rz(1.5*pi) q[2];
rz(2.5*pi) q[3];
sx q[4];
sx q[6];
sx q[7];
sx q[0];
rz(0.5*pi) q[1];
cx q[2],q[7];
sx q[3];
rz(0.5*pi) q[4];
cx q[5],q[6];
rz(1.0*pi) q[0];
rz(1.357983238373847*pi) q[1];
sx q[2];
rz(1.5*pi) q[3];
rz(2.5*pi) q[4];
sx q[5];
rz(1.0*pi) q[6];
sx q[7];
sx q[1];
rz(0.5*pi) q[2];
cx q[3],q[7];
sx q[4];
rz(0.5*pi) q[5];
rz(2.5*pi) q[6];
rz(1.0*pi) q[1];
rz(2.475938035288558*pi) q[2];
sx q[3];
rz(1.5*pi) q[4];
rz(2.5*pi) q[5];
sx q[6];
sx q[7];
sx q[2];
rz(0.5*pi) q[3];
cx q[4],q[7];
sx q[5];
rz(1.5*pi) q[6];
rz(1.0*pi) q[2];
rz(2.616129348751537*pi) q[3];
sx q[4];
rz(1.5*pi) q[5];
sx q[7];
sx q[3];
rz(0.5*pi) q[4];
cx q[5],q[7];
rz(1.0*pi) q[3];
rz(2.767848631212374*pi) q[4];
sx q[5];
sx q[7];
sx q[4];
rz(0.5*pi) q[5];
cx q[6],q[7];
rz(1.0*pi) q[4];
rz(2.647471190850327*pi) q[5];
sx q[6];
sx q[7];
sx q[5];
rz(0.5*pi) q[6];
rz(3.219353926517741*pi) q[7];
rz(1.0*pi) q[5];
rz(1.1218845143037632*pi) q[6];
sx q[7];
sx q[6];
rz(1.0*pi) q[6];
