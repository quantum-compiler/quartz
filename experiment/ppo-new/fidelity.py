import csv
import os
import sys
from os.path import isfile, join

import numpy as np
from qiskit import IBMQ, QuantumCircuit, transpile
from qiskit.providers.fake_provider import (
    FakeManhattan,
    FakeManhattanV2,
    FakeMelbourne,
    FakeMelbourneV2,
    FakeMontreal,
    FakeMontrealV2,
    FakeSherbrooke,
    FakeWashington,
    FakeWashingtonV2,
)
from qiskit.providers.models import BackendProperties
from qiskit.transpiler import PassManager
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.passes import BasisTranslator, SabreLayout, VF2PostLayout
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

# class Simulator:
#     def __init__(self, file_path):
#         self.sx_error = {}
#         self.px_error = {}
#         self.cx_error = {}
#         with open(file_path, "r") as csv_file:
#             reader = csv.DictReader(csv_file, delimiter=",")
#             for row in reader:
#                 for key in row:
#                     try:
#                         row[key] = float(row[key])
#                     except ValueError:
#                         pass
#                 self.sx_error[int(row["Qubit"])] = row["√x (sx) error "]
#                 self.px_error[int(row["Qubit"])] = row["Pauli-X error "]
#                 qubit_error_strs = row["CNOT error "].split(";")
#                 for qubit_error in qubit_error_strs:
#                     qubits, error = qubit_error.split(":")
#                     error = float(error)
#                     qubits = qubits.split("_")
#                     qubits = tuple(map(int, qubits))
#                     self.cx_error[qubits] = error

#     def gate_error(self, gate_name: str, qubits: tuple) -> float:
#         if len(qubits) == 1:
#             qubits = qubits[0]
#         if gate_name == "sx":
#             return self.sx_error[qubits]
#         elif gate_name == "cx":
#             return self.cx_error[qubits]
#         elif gate_name == "x":
#             return self.px_error[qubits]
#         elif gate_name == "rz":
#             return 0
#         else:
#             assert False, "Unknown gate"


# def get_properties(backend):
#     try:
#         bp = backend.properties()
#     except:
#         backend._set_props_dict_from_json()
#         bp = BackendProperties.from_dict(backend._props_dict)
#     return bp


# def get_log_fidelity(fn: str, times: int = 20) -> float:
#     circ = QuantumCircuit.from_qasm_file(fn)
#     num_qubits: int = circ.num_qubits
#     backend = None
#     simulator = None
#     if num_qubits < 27:
#         backend = FakeMontrealV2()
#         simulator = Simulator(
#             "/home/zikun/quartz/experiment/calibration_data/ibmq_montreal_calibrations_2023-04-06T22_37_15Z.csv"
#         )
#     elif num_qubits <= 127:
#         backend = FakeWashingtonV2()
#         simulator = Simulator(
#             "/home/zikun/quartz/experiment/calibration_data/ibm_washington_calibrations_2023-04-06T21_24_42Z.csv"
#         )
#     else:
#         assert False, "Too many qubits"

#     bf = get_properties(backend)

#     pm = generate_preset_pass_manager(
#         backend=backend,
#         optimization_level=0,
#         layout_method="sabre",
#         routing_method="sabre",
#     )
#     # pm.append(VF2PostLayout(coupling_map=backend.coupling_map, properties=bf))

#     acc_circ_log_succ_rate: float = 0

#     for _ in range(times):
#         while True:
#             circ_log_succ_rate: float = 0
#             fail = False
#             # transpile to backend, only do mapping and routing
#             # transpiled_circ = transpile(
#             #     circ,
#             #     backend=backend,
#             #     optimization_level=0,
#             #     routing_method="sabre",
#             #     pass_manager=pm,
#             # )
#             transpiled_circ = pm.run(circ)
#             # print(f"transpiled count ops {transpiled_circ.count_ops()}")
#             instructions = transpiled_circ.data
#             for circ_inst in instructions:
#                 gate_name = circ_inst.operation.name
#                 qubits = ()
#                 for qubit in circ_inst.qubits:
#                     qubits += (qubit.index,)
#                 # gate_error = simulator.gate_error(gate_name, qubits)
#                 gate_error = bf.gate_error(gate_name, qubits)
#                 succ_rate = 1 - gate_error
#                 if succ_rate == 0:
#                     print(
#                         f"gate error equals to 1: {gate_name}, {qubits}, {gate_error}"
#                     )
#                     fail = True
#                     break
#                 circ_log_succ_rate += np.log(succ_rate)
#             if not fail:
#                 break
#         acc_circ_log_succ_rate += circ_log_succ_rate

#     return acc_circ_log_succ_rate / times


def get_log_logical_fidelity(fn: str) -> float:
    gate_error_rate = {
        "cx": 1.214e-2,
        "x": 2.77e-4,
        "sx": 2.77e-4,
        "rz": 0,
    }
    circ = QuantumCircuit.from_qasm_file(fn)
    gate_counts = circ.count_ops()
    circ_log_succ_rate = 0
    for gate, count in gate_counts.items():
        gate_error = gate_error_rate[gate]
        succ_rate = 1 - gate_error
        circ_log_succ_rate += count * np.log(succ_rate)
    return circ_log_succ_rate


if __name__ == "__main__":
    dir = sys.argv[1]
    output_order = [
        "tof_3",
        "barenco_tof_3",
        "mod5_4",
        "tof_4",
        "tof_5",
        "barenco_tof_4",
        "mod_mult_55",
        "vbe_adder_3",
        "barenco_tof_5",
        "csla_mux_3",
        "rc_adder_6",
        "gf2^4_mult",
        "hwb6",
        "mod_red_21",
        "tof_10",
        "gf2^5_mult",
        "csum_mux_9",
        "barenco_tof_10",
        "ham15-low",
        "qcla_com_7",
        "gf2^6_mult",
        "qcla_adder_10",
        "gf2^7_mult",
        "gf2^8_mult",
        "qcla_mod_7",
        "adder_8",
        "vqe_nativegates_ibm_tket_8",
        "qgan_nativegates_ibm_tket_8",
        "qaoa_nativegates_ibm_tket_8",
        "ae_nativegates_ibm_tket_8",
        "qpeexact_nativegates_ibm_tket_8",
        "qpeinexact_nativegates_ibm_tket_8",
        "qft_nativegates_ibm_tket_8",
        "qftentangled_nativegates_ibm_tket_8",
        "portfoliovqe_nativegates_ibm_tket_8",
        "portfolioqaoa_nativegates_ibm_tket_8",
    ]
    abs_path = os.path.abspath(dir)
    fns = [
        fn
        for fn in os.listdir(abs_path)
        if isfile(join(abs_path, fn)) and fn[-4:] == "qasm"
    ]
    # print(fns)
    log_fidelity_dict = {}
    for fn in fns:
        # print(fn)
        full_path = join(abs_path, fn)
        log_fidelity_dict[fn] = get_log_logical_fidelity(dir + fn)
        print(f"{fn} has log fidelity {log_fidelity_dict[fn]}")

    print("Ordered output:")
    for name in output_order:
        try:
            print(f"{log_fidelity_dict[str(name) + '.qasm']}")
        except:
            print(f"{log_fidelity_dict[str(name) + '_optimized.qasm']}")
