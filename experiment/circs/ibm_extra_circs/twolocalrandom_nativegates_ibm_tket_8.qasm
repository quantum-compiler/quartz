OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg meas[8];
sx q[0];
sx q[1];
sx q[2];
rz(1.0*pi) q[3];
sx q[4];
sx q[5];
sx q[6];
rz(1.0*pi) q[7];
rz(0.5431701715420276*pi) q[0];
rz(1.2633284263505207*pi) q[1];
rz(0.10577199677947968*pi) q[2];
sx q[3];
rz(1.308627497063049*pi) q[4];
rz(3.2394457225565723*pi) q[5];
rz(0.2761622706765129*pi) q[6];
sx q[7];
rz(2.5*pi) q[0];
sx q[1];
sx q[2];
rz(1.8076004064353566*pi) q[3];
sx q[4];
sx q[5];
sx q[6];
rz(1.8808895616280283*pi) q[7];
sx q[0];
rz(1.0*pi) q[2];
sx q[3];
sx q[4];
sx q[6];
sx q[7];
rz(1.5*pi) q[0];
sx q[2];
cx q[0],q[1];
sx q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[0];
sx q[1];
rz(2.5*pi) q[0];
rz(0.5*pi) q[1];
sx q[0];
rz(2.5*pi) q[1];
rz(1.5*pi) q[0];
sx q[1];
cx q[0],q[2];
rz(1.5*pi) q[1];
sx q[0];
sx q[2];
rz(0.5*pi) q[0];
cx q[1],q[2];
rz(2.5*pi) q[0];
sx q[1];
sx q[2];
sx q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(1.5*pi) q[0];
rz(2.5*pi) q[1];
rz(2.5*pi) q[2];
cx q[0],q[3];
sx q[1];
sx q[2];
sx q[0];
rz(1.5*pi) q[1];
rz(1.5*pi) q[2];
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
rz(3.5*pi) q[3];
cx q[0],q[4];
sx q[1];
rz(0.5*pi) q[2];
sx q[3];
sx q[0];
rz(1.5*pi) q[1];
rz(2.5*pi) q[2];
rz(0.5*pi) q[3];
sx q[4];
rz(0.5*pi) q[0];
cx q[1],q[4];
sx q[2];
rz(2.5*pi) q[3];
rz(2.5*pi) q[0];
sx q[1];
rz(1.5*pi) q[2];
sx q[3];
sx q[4];
sx q[0];
rz(0.5*pi) q[1];
cx q[2],q[4];
rz(1.5*pi) q[3];
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
rz(1.0*pi) q[4];
sx q[5];
rz(0.5*pi) q[0];
cx q[1],q[5];
sx q[2];
rz(0.5*pi) q[3];
sx q[4];
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
rz(2.5*pi) q[4];
rz(1.5*pi) q[0];
rz(2.5*pi) q[1];
sx q[2];
rz(1.5*pi) q[3];
sx q[4];
sx q[5];
cx q[0],q[6];
sx q[1];
rz(0.5*pi) q[2];
cx q[3],q[5];
rz(1.5*pi) q[4];
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
rz(0.5*pi) q[5];
sx q[6];
sx q[0];
rz(0.5*pi) q[1];
cx q[2],q[6];
sx q[3];
rz(0.5*pi) q[4];
sx q[5];
rz(1.5*pi) q[0];
rz(2.5*pi) q[1];
sx q[2];
rz(1.5*pi) q[3];
rz(2.5*pi) q[4];
rz(0.5*pi) q[5];
sx q[6];
cx q[0],q[7];
sx q[1];
rz(0.5*pi) q[2];
cx q[3],q[6];
sx q[4];
rz(2.5*pi) q[5];
sx q[0];
rz(1.5*pi) q[1];
rz(2.5*pi) q[2];
sx q[3];
rz(1.5*pi) q[4];
sx q[5];
sx q[6];
sx q[7];
rz(0.5*pi) q[0];
cx q[1],q[7];
sx q[2];
rz(0.5*pi) q[3];
cx q[4],q[6];
rz(1.5*pi) q[5];
sx q[0];
sx q[1];
rz(1.5*pi) q[2];
rz(2.5*pi) q[3];
sx q[4];
sx q[6];
sx q[7];
rz(0.30659918504750205*pi) q[0];
rz(0.5*pi) q[1];
cx q[2],q[7];
sx q[3];
rz(0.5*pi) q[4];
cx q[5],q[6];
rz(2.5*pi) q[0];
sx q[1];
sx q[2];
rz(1.5*pi) q[3];
rz(2.5*pi) q[4];
sx q[5];
sx q[6];
sx q[7];
sx q[0];
rz(0.5967209058493856*pi) q[1];
rz(0.5*pi) q[2];
cx q[3],q[7];
sx q[4];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
rz(1.5*pi) q[0];
sx q[1];
sx q[2];
sx q[3];
rz(1.5*pi) q[4];
rz(2.5*pi) q[5];
rz(2.5*pi) q[6];
sx q[7];
cx q[0],q[1];
rz(3.6132821624684204*pi) q[2];
rz(0.5*pi) q[3];
cx q[4],q[7];
sx q[5];
sx q[6];
sx q[0];
rz(0.5*pi) q[1];
sx q[2];
rz(1.0*pi) q[3];
sx q[4];
rz(1.5*pi) q[5];
rz(1.5*pi) q[6];
sx q[7];
rz(0.5*pi) q[0];
sx q[1];
rz(1.0*pi) q[2];
sx q[3];
rz(0.5*pi) q[4];
cx q[5],q[7];
rz(2.5*pi) q[0];
rz(0.5*pi) q[1];
sx q[2];
rz(0.4552116812082011*pi) q[3];
sx q[4];
sx q[5];
sx q[7];
sx q[0];
rz(2.5*pi) q[1];
sx q[3];
rz(0.6744317144288456*pi) q[4];
rz(0.5*pi) q[5];
cx q[6],q[7];
rz(1.5*pi) q[0];
sx q[1];
sx q[4];
sx q[5];
sx q[6];
sx q[7];
cx q[0],q[2];
rz(1.5*pi) q[1];
sx q[4];
rz(0.5170686309999644*pi) q[5];
rz(0.5*pi) q[6];
rz(3.945551699244001*pi) q[7];
sx q[0];
sx q[2];
sx q[5];
sx q[6];
sx q[7];
rz(0.5*pi) q[0];
cx q[1],q[2];
rz(1.667381205526376*pi) q[6];
rz(2.5*pi) q[0];
sx q[1];
sx q[2];
sx q[6];
sx q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
sx q[6];
rz(1.5*pi) q[0];
rz(2.5*pi) q[1];
rz(2.5*pi) q[2];
cx q[0],q[3];
sx q[1];
sx q[2];
sx q[0];
rz(1.5*pi) q[1];
rz(1.5*pi) q[2];
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
rz(3.5*pi) q[3];
cx q[0],q[4];
sx q[1];
rz(0.5*pi) q[2];
sx q[3];
sx q[0];
rz(1.5*pi) q[1];
rz(2.5*pi) q[2];
rz(0.5*pi) q[3];
sx q[4];
rz(0.5*pi) q[0];
cx q[1],q[4];
sx q[2];
rz(2.5*pi) q[3];
rz(2.5*pi) q[0];
sx q[1];
rz(1.5*pi) q[2];
sx q[3];
sx q[4];
sx q[0];
rz(0.5*pi) q[1];
cx q[2],q[4];
rz(1.5*pi) q[3];
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
rz(1.0*pi) q[4];
sx q[5];
rz(0.5*pi) q[0];
cx q[1],q[5];
sx q[2];
rz(0.5*pi) q[3];
sx q[4];
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
rz(2.5*pi) q[4];
rz(1.5*pi) q[0];
rz(2.5*pi) q[1];
sx q[2];
rz(1.5*pi) q[3];
sx q[4];
sx q[5];
cx q[0],q[6];
sx q[1];
rz(0.5*pi) q[2];
cx q[3],q[5];
rz(1.5*pi) q[4];
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
rz(0.5*pi) q[5];
sx q[6];
sx q[0];
rz(0.5*pi) q[1];
cx q[2],q[6];
sx q[3];
rz(0.5*pi) q[4];
sx q[5];
rz(1.5*pi) q[0];
rz(2.5*pi) q[1];
sx q[2];
rz(1.5*pi) q[3];
rz(2.5*pi) q[4];
rz(0.5*pi) q[5];
sx q[6];
cx q[0],q[7];
sx q[1];
rz(0.5*pi) q[2];
cx q[3],q[6];
sx q[4];
rz(2.5*pi) q[5];
sx q[0];
rz(1.5*pi) q[1];
rz(2.5*pi) q[2];
sx q[3];
rz(1.5*pi) q[4];
sx q[5];
sx q[6];
sx q[7];
rz(0.5*pi) q[0];
cx q[1],q[7];
sx q[2];
rz(0.5*pi) q[3];
cx q[4],q[6];
rz(1.5*pi) q[5];
sx q[0];
sx q[1];
rz(1.5*pi) q[2];
rz(2.5*pi) q[3];
sx q[4];
sx q[6];
sx q[7];
rz(0.0350414812349468*pi) q[0];
rz(0.5*pi) q[1];
cx q[2],q[7];
sx q[3];
rz(0.5*pi) q[4];
cx q[5],q[6];
rz(2.5*pi) q[0];
sx q[1];
sx q[2];
rz(1.5*pi) q[3];
rz(2.5*pi) q[4];
sx q[5];
sx q[6];
sx q[7];
sx q[0];
rz(0.8156831175310444*pi) q[1];
rz(0.5*pi) q[2];
cx q[3],q[7];
sx q[4];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
rz(1.5*pi) q[0];
sx q[1];
sx q[2];
sx q[3];
rz(1.5*pi) q[4];
rz(2.5*pi) q[5];
rz(2.5*pi) q[6];
sx q[7];
cx q[0],q[1];
rz(3.529869694801386*pi) q[2];
rz(0.5*pi) q[3];
cx q[4],q[7];
sx q[5];
sx q[6];
sx q[0];
rz(0.5*pi) q[1];
sx q[2];
rz(1.0*pi) q[3];
sx q[4];
rz(1.5*pi) q[5];
rz(1.5*pi) q[6];
sx q[7];
rz(0.5*pi) q[0];
sx q[1];
rz(1.0*pi) q[2];
sx q[3];
rz(0.5*pi) q[4];
cx q[5],q[7];
rz(2.5*pi) q[0];
rz(0.5*pi) q[1];
sx q[2];
rz(0.412136021946531*pi) q[3];
sx q[4];
sx q[5];
sx q[7];
sx q[0];
rz(2.5*pi) q[1];
sx q[3];
rz(0.6904276334235547*pi) q[4];
rz(0.5*pi) q[5];
cx q[6],q[7];
rz(1.5*pi) q[0];
sx q[1];
sx q[4];
sx q[5];
sx q[6];
sx q[7];
cx q[0],q[2];
rz(1.5*pi) q[1];
sx q[4];
rz(0.7980934015836622*pi) q[5];
rz(0.5*pi) q[6];
rz(3.8248148380460276*pi) q[7];
sx q[0];
sx q[2];
sx q[5];
sx q[6];
sx q[7];
rz(0.5*pi) q[0];
cx q[1],q[2];
rz(1.6943508352812446*pi) q[6];
rz(2.5*pi) q[0];
sx q[1];
sx q[2];
sx q[6];
sx q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
sx q[6];
rz(1.5*pi) q[0];
rz(2.5*pi) q[1];
rz(2.5*pi) q[2];
cx q[0],q[3];
sx q[1];
sx q[2];
sx q[0];
rz(1.5*pi) q[1];
rz(1.5*pi) q[2];
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
rz(3.5*pi) q[3];
cx q[0],q[4];
sx q[1];
rz(0.5*pi) q[2];
sx q[3];
sx q[0];
rz(1.5*pi) q[1];
rz(2.5*pi) q[2];
rz(0.5*pi) q[3];
sx q[4];
rz(0.5*pi) q[0];
cx q[1],q[4];
sx q[2];
rz(2.5*pi) q[3];
rz(2.5*pi) q[0];
sx q[1];
rz(1.5*pi) q[2];
sx q[3];
sx q[4];
sx q[0];
rz(0.5*pi) q[1];
cx q[2],q[4];
rz(1.5*pi) q[3];
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
rz(1.0*pi) q[4];
sx q[5];
rz(0.5*pi) q[0];
cx q[1],q[5];
sx q[2];
rz(0.5*pi) q[3];
sx q[4];
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
rz(2.5*pi) q[4];
rz(1.5*pi) q[0];
rz(2.5*pi) q[1];
sx q[2];
rz(1.5*pi) q[3];
sx q[4];
sx q[5];
cx q[0],q[6];
sx q[1];
rz(0.5*pi) q[2];
cx q[3],q[5];
rz(1.5*pi) q[4];
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
rz(0.5*pi) q[5];
sx q[6];
sx q[0];
rz(0.5*pi) q[1];
cx q[2],q[6];
sx q[3];
rz(0.5*pi) q[4];
sx q[5];
rz(1.5*pi) q[0];
rz(2.5*pi) q[1];
sx q[2];
rz(1.5*pi) q[3];
rz(2.5*pi) q[4];
rz(0.5*pi) q[5];
sx q[6];
cx q[0],q[7];
sx q[1];
rz(0.5*pi) q[2];
cx q[3],q[6];
sx q[4];
rz(2.5*pi) q[5];
sx q[0];
rz(1.5*pi) q[1];
rz(2.5*pi) q[2];
sx q[3];
rz(1.5*pi) q[4];
sx q[5];
sx q[6];
sx q[7];
rz(0.5*pi) q[0];
cx q[1],q[7];
sx q[2];
rz(0.5*pi) q[3];
cx q[4],q[6];
rz(1.5*pi) q[5];
sx q[0];
sx q[1];
rz(1.5*pi) q[2];
rz(2.5*pi) q[3];
sx q[4];
sx q[6];
sx q[7];
rz(2.759763745317097*pi) q[0];
rz(0.5*pi) q[1];
cx q[2],q[7];
sx q[3];
rz(0.5*pi) q[4];
cx q[5],q[6];
sx q[0];
sx q[1];
sx q[2];
rz(1.5*pi) q[3];
rz(2.5*pi) q[4];
sx q[5];
sx q[6];
sx q[7];
rz(1.0*pi) q[0];
rz(2.8173170800474865*pi) q[1];
rz(0.5*pi) q[2];
cx q[3],q[7];
sx q[4];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
sx q[1];
sx q[2];
sx q[3];
rz(1.5*pi) q[4];
rz(2.5*pi) q[5];
rz(2.5*pi) q[6];
sx q[7];
rz(1.0*pi) q[1];
rz(2.5316462554363155*pi) q[2];
rz(0.5*pi) q[3];
cx q[4],q[7];
sx q[5];
sx q[6];
sx q[2];
sx q[3];
sx q[4];
rz(1.5*pi) q[5];
rz(1.5*pi) q[6];
sx q[7];
rz(1.0*pi) q[2];
rz(2.6499118331396616*pi) q[3];
rz(0.5*pi) q[4];
cx q[5],q[7];
sx q[3];
sx q[4];
sx q[5];
sx q[7];
rz(1.0*pi) q[3];
rz(2.757385530482723*pi) q[4];
rz(0.5*pi) q[5];
cx q[6],q[7];
sx q[4];
sx q[5];
sx q[6];
sx q[7];
rz(1.0*pi) q[4];
rz(2.5662728283661744*pi) q[5];
rz(0.5*pi) q[6];
rz(0.9342579895852678*pi) q[7];
sx q[5];
sx q[6];
sx q[7];
rz(1.0*pi) q[5];
rz(2.7168724862638607*pi) q[6];
sx q[6];
rz(1.0*pi) q[6];
