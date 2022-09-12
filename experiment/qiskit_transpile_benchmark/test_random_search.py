from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import SabreLayout, SabreSwap
from qiskit.transpiler.passmanager_config import PassManagerConfig
from customized_sabre_transpiler import sabre_pass_manager


def parse_gate_type(tp, circuit):
    if tp == "cx":
        return circuit.cx
    elif tp == "h":
        return circuit.h
    elif tp == "t":
        return circuit.t
    elif tp == "tdg":
        return circuit.tdg
    elif tp == "s":
        return circuit.s
    elif tp == "sdg":
        return circuit.sdg
    else:
        raise NotImplementedError


def IBM_Q20_Tokyo():
    # identical to IBM Q20 Tokyo
    coupling = [
        # rows
        [0, 1], [1, 2], [2, 3], [3, 4],
        [5, 6], [6, 7], [7, 8], [8, 9],
        [10, 11], [11, 12], [12, 13], [13, 14],
        [15, 16], [16, 17], [17, 18], [18, 19],
        # cols
        [0, 5], [5, 10], [10, 15],
        [1, 6], [6, 11], [11, 16],
        [2, 7], [7, 12], [12, 17],
        [3, 8], [8, 13], [13, 18],
        [4, 9], [9, 14], [14, 19],
        # crossings
        [1, 7], [2, 6],
        [3, 9], [4, 8],
        [5, 11], [6, 10],
        [8, 12], [7, 13],
        [11, 17], [12, 16],
        [13, 19], [14, 18]
    ]
    reversed_coupling = []
    for pair in coupling:
        reversed_coupling.append([pair[1], pair[0]])
    coupling_map = CouplingMap(couplinglist=coupling + reversed_coupling)
    return coupling_map


def IBM_Q27_Falcon():
    # identical to IBM Q27 Falcon
    coupling = [
        # 1st row
        [0, 1], [1, 4], [4, 7], [7, 10], [10, 12], [12, 15], [15, 18],
        [18, 21], [21, 23],
        # 2nd row
        [3, 5], [5, 8], [8, 11], [11, 14], [14, 16], [16, 19], [19, 22],
        [22, 25], [25, 26],
        # cols
        [6, 7], [17, 18], [1, 2], [2, 3], [12, 13], [13, 14], [23, 24],
        [24, 25], [8, 9], [19, 20]
    ]
    reversed_coupling = []
    for pair in coupling:
        reversed_coupling.append([pair[1], pair[0]])
    coupling_map = CouplingMap(couplinglist=coupling + reversed_coupling)
    return coupling_map


def IBM_Q65_Hummingbird():
    # identical to IBM Q65 Hummingbird
    coupling = [
        # first row
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9],
        # first layer columns
        [0, 10], [10, 13], [4, 11], [11, 17], [8, 12], [12, 21],
        # second row
        [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23],
        # second layer columns
        [15, 24], [24, 29], [19, 25], [25, 33], [23, 26], [26, 37],
        # third layer
        [27, 28], [28, 29], [29, 30], [30, 31], [31, 32], [32, 33], [33, 34], [34, 35], [35, 36], [36, 37],
        # third layer columns
        [27, 38], [38, 41], [31, 39], [39, 45], [35, 40], [40, 49],
        # fourth layer
        [41, 42], [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49], [49, 50], [50, 51],
        # fourth layer columns
        [43, 52], [52, 56], [47, 53], [53, 60], [51, 54], [54, 64],
        # fifth layer
        [55, 56], [56, 57], [57, 58], [58, 59], [59, 60], [60, 61], [61, 62], [62, 63], [63, 64],
    ]
    reversed_coupling = []
    for pair in coupling:
        reversed_coupling.append([pair[1], pair[0]])
    coupling_map = CouplingMap(couplinglist=coupling + reversed_coupling)
    return coupling_map


def run_benchmark(qasm_file_name, device_name, best_cost, show_mapping):
    # get device coupling map
    if device_name == "IBM_Q20_Tokyo":
        coupling_map, num_regs = IBM_Q20_Tokyo(), 20
    elif device_name == "IBM_Q27_Falcon":
        coupling_map, num_regs = IBM_Q27_Falcon(), 27
    elif device_name == "IBM_Q65_Hummingbird":
        coupling_map, num_regs = IBM_Q65_Hummingbird(), 65
    else:
        raise NotImplementedError

    # parse qasm file into a circuit
    circuit = QuantumCircuit(num_regs)
    with open("../../circuit/nam-circuits/qasm_files/" + qasm_file_name) as file:
        # omit the header
        file.readline()
        file.readline()
        line = file.readline()
        num_qubits = int(line.split(' ')[1].split(']')[0].split('[')[1])
        # parse the rest
        line = file.readline()
        while line != '':
            # add to circuit
            arg_list = line.split(' ')
            if arg_list[0] == '':
                arg_list = arg_list[1:]
            if len(arg_list) == 3:
                # gate type
                tp = arg_list[0]
                # two qubits gate
                qubit1 = int(arg_list[1].split(']')[0].split('[')[1])
                qubit2 = int(arg_list[2].split(']')[0].split('[')[1])
                parse_gate_type(tp=tp, circuit=circuit)(qubit1, qubit2)
            elif len(arg_list) == 2:
                # gate type
                tp = arg_list[0]
                # single qubit gate
                qubit1 = int(arg_list[1].split(']')[0].split('[')[1])
                parse_gate_type(tp=tp, circuit=circuit)(qubit1)
            else:
                assert False
            # read another line
            line = file.readline()

    # run sabre layout and sabre swap
    sabre_manager = sabre_pass_manager(PassManagerConfig(coupling_map=coupling_map, layout_method="sabre",
                                                         routing_method="sabre"))
    sabre_circuit = sabre_manager.run(circuit)

    # original gate count
    ori_circuit_op_list = dict(circuit.count_ops())
    ori_gate_count = 0
    for key in ori_circuit_op_list:
        if key == "swap":
            assert False, "swap in original circuit!"
        else:
            ori_gate_count += ori_circuit_op_list[key]

    # get gate count of sabre
    sabre_circuit_op_list = dict(sabre_circuit.count_ops())
    sabre_gate_count = 0
    sabre_swap_count = 0
    for key in sabre_circuit_op_list:
        if key == "swap":
            sabre_gate_count += 3 * sabre_circuit_op_list[key]
            sabre_swap_count = sabre_circuit_op_list[key]
        else:
            assert sabre_circuit_op_list[key] == ori_circuit_op_list[key]
            sabre_gate_count += sabre_circuit_op_list[key]
    assert sabre_gate_count - 3 * sabre_swap_count == ori_gate_count
    if sabre_swap_count < best_cost and show_mapping:
        print(f"{sabre_swap_count=}")
        cur_layout = sabre_manager.passes()[0]["passes"][0].property_set["layout"]
        layout_dict = cur_layout.get_virtual_bits()
        logical_id_list, physical_id_list = [], []
        for key in layout_dict:
            logical_id_list.append(key.index)
            physical_id_list.append(layout_dict[key])
        print(f"swap count {sabre_swap_count}")
        print(f"logical: {logical_id_list}")
        print(f"physical: {physical_id_list}")

    return ori_gate_count, sabre_gate_count, sabre_swap_count


def main():
    # parameters
    # gf2^E5_mult_after_heavy.qasm, barenco_tof_10_before.qasm, csla_mux_3_after_heavy.qasm, qcla_adder_10_before.qasm
    # IBM_Q20_Tokyo, IBM_Q27_Falcon, IBM_Q65_Hummingbird
    qasm_file_name = "qcla_adder_10_before.qasm"
    device_name = "IBM_Q65_Hummingbird"
    num_runs = 100 * 1000
    show_mapping = False

    # run benchmark
    print(f"circuit name: {qasm_file_name}.")
    print(f"device name: {device_name}.")
    min_swap_count = 10000
    step_count = 0
    original_gate_count = 0
    while step_count < num_runs:
        step_count += 1
        data = run_benchmark(qasm_file_name=qasm_file_name, device_name=device_name, best_cost=min_swap_count,
                             show_mapping=show_mapping)
        ori_gate_count, sabre_gate_count, sabre_swap_count = data
        if sabre_swap_count < min_swap_count:
            min_swap_count = sabre_swap_count
            original_gate_count = ori_gate_count    # actually this is only set once
            print(f"Best Implementation found: {min_swap_count} swaps. ({step_count} runs)")
    print(f"Best Implementation found in {step_count} runs: {min_swap_count} swaps."
          f" (original: {original_gate_count}, after: {original_gate_count + 3 * min_swap_count})")


if __name__ == '__main__':
    main()
