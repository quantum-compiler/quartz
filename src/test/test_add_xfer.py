import quartz

# Build quartz context
qtz: quartz.QuartzContext = quartz.QuartzContext(
    gate_set=["z", "cx", "x", "rz", "sx", "t", "tdg", "add"],
    filename="../experiment/ecc_set/ibm_special_angles_3_2_5_ecc.json",
)

src_str = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[0]; z q[0];'
dst_str = (
    'OPENQASM 2.0; include "qelib1.inc"; qreg q[0]; sx q[0]; rz(pi/2) q[0]; sx q[0];'
)

qtz.add_xfer(src_str=src_str, dst_str=dst_str)
