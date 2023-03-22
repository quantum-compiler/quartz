import quartz

# If z gate is included in the ECC set
# quartz_context = quartz.QuartzContext(
#     gate_set={"cx", "x", "sx", "rz", "z", "add"},
#     filename="ecc_set/ibm_325_ecc.json",  # may need to modify
#     no_increase=False,
#     include_nop=True,
# )

# equivalent_circ_pairs = [
#     (
#         'OpenQASM 2.0; include "qelib1.inc"; qreg q[1]; z q[0];',
#         'OpenQASM 2.0; include "qelib1.inc"; qreg q[1]; rz(pi) q[0];',
#     ),
#     (
#         'OpenQASM 2.0; include "qelib1.inc"; qreg q[1]; sx q[0]; rz(pi/2) q[0]; sx q[0];',
#         'OpenQASM 2.0; include "qelib1.inc"; qreg q[1]; rz(pi/2) q[0]; sx q[0]; rz(pi/2) q[0];',
#     ),
#     (
#         'OpenQASM 2.0; include "qelib1.inc"; qreg q[1]; sx q[0]; rz(3*pi/2) q[0]; sx q[0];',
#         'OpenQASM 2.0; include "qelib1.inc"; qreg q[1]; rz(3*pi/2) q[0]; sx q[0]; rz(3*pi/2) q[0];',
#     ),
# ]

# for circ_pair in equivalent_circ_pairs:
#     quartz_context.add_xfer_from_qasm_str(circ_pair[0], circ_pair[1])
#     quartz_context.add_xfer_from_qasm_str(circ_pair[1], circ_pair[0])


# Else if z gate is not included in the ECC set


def ibm_add_xfer(context: quartz.QuartzContext):
    equivalent_circ_pairs = [
        (
            'OpenQASM 2.0; include "qelib1.inc"; qreg q[1]; rz(pi) q[0];',
            'OpenQASM 2.0; include "qelib1.inc"; qreg q[1]; sx q[0]; rz(pi) q[0]; sx q[0];',
        ),
        (
            'OpenQASM 2.0; include "qelib1.inc"; qreg q[1]; sx q[0]; rz(pi/2) q[0]; sx q[0];',
            'OpenQASM 2.0; include "qelib1.inc"; qreg q[1]; rz(pi/2) q[0]; sx q[0]; rz(pi/2) q[0];',
        ),
        (
            'OpenQASM 2.0; include "qelib1.inc"; qreg q[1]; sx q[0]; rz(3*pi/2) q[0]; sx q[0];',
            'OpenQASM 2.0; include "qelib1.inc"; qreg q[1]; rz(3*pi/2) q[0]; sx q[0]; rz(3*pi/2) q[0];',
        ),
    ]

    for circ_pair in equivalent_circ_pairs:
        print(circ_pair)
        context.add_xfer_from_qasm_str(src_str=circ_pair[0], dst_str=circ_pair[1])
        context.add_xfer_from_qasm_str(src_str=circ_pair[1], dst_str=circ_pair[0])

    return context
