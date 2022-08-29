from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import SabreLayout, SabreSwap
from qiskit.transpiler.preset_passmanagers import level_0_pass_manager, level_1_pass_manager, \
    level_2_pass_manager, level_3_pass_manager
from qiskit.transpiler.passmanager_config import PassManagerConfig


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


def run_benchmark(qasm_file_name, device_name, enable_lv2=True, enable_lv3=True):
    # get device coupling map
    if device_name == "IBM_Q20_Tokyo":
        coupling_map = IBM_Q20_Tokyo()
    elif device_name == "IBM_Q27_Falcon":
        coupling_map = IBM_Q27_Falcon()
    else:
        raise NotImplementedError

    # parse qasm file into a circuit
    circuit = QuantumCircuit(20)
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
    level_0_manager = level_0_pass_manager(PassManagerConfig(coupling_map=coupling_map, layout_method="sabre",
                                                             routing_method="sabre"))
    level_1_manager = level_1_pass_manager(PassManagerConfig(coupling_map=coupling_map, layout_method="sabre",
                                                             routing_method="sabre"))
    level_2_manager = level_2_pass_manager(PassManagerConfig(coupling_map=coupling_map, layout_method="sabre",
                                                             routing_method="sabre"))
    level_3_manager = level_3_pass_manager(PassManagerConfig(coupling_map=coupling_map, layout_method="sabre",
                                                             routing_method="sabre"))
    result_circuit_0 = level_0_manager.run(circuit)
    result_circuit_1 = level_1_manager.run(circuit)
    result_circuit_2 = level_2_manager.run(circuit)
    result_circuit_3 = level_3_manager.run(circuit)

    # original gate count
    ori_circuit_op_list = dict(circuit.count_ops())
    ori_gate_count = 0
    for key in ori_circuit_op_list:
        if key == "swap":
            ori_gate_count += 3 * ori_circuit_op_list[key]
        else:
            ori_gate_count += ori_circuit_op_list[key]

    # level 0 gate count
    level0_circuit_op_list = dict(result_circuit_0.count_ops())
    level0_gate_count = 0
    level0_swap_count = 0
    for key in level0_circuit_op_list:
        if key == "swap":
            level0_gate_count += 3 * level0_circuit_op_list[key]
            level0_swap_count = level0_circuit_op_list[key]
        else:
            assert level0_circuit_op_list[key] == ori_circuit_op_list[key]
            level0_gate_count += level0_circuit_op_list[key]
    assert level0_gate_count - 3 * level0_swap_count == ori_gate_count

    # level 1 gate count
    level1_circuit_op_list = dict(result_circuit_1.count_ops())
    level1_gate_count = 0
    level1_swap_count = 0
    for key in level1_circuit_op_list:
        if key == "swap":
            level1_gate_count += 3 * level1_circuit_op_list[key]
            level1_swap_count = level1_circuit_op_list[key]
        else:
            assert level1_circuit_op_list[key] == ori_circuit_op_list[key]
            level1_gate_count += level1_circuit_op_list[key]
    assert level1_gate_count - 3 * level1_swap_count == ori_gate_count

    # level 2 gate count
    level2_circuit_op_list = dict(result_circuit_2.count_ops())
    level2_gate_count = 0
    level2_swap_count = 0
    if enable_lv2:
        for key in level2_circuit_op_list:
            if key == "swap":
                level2_gate_count += 3 * level2_circuit_op_list[key]
                level2_swap_count = level2_circuit_op_list[key]
            else:
                assert level2_circuit_op_list[key] == ori_circuit_op_list[key]
                level2_gate_count += level2_circuit_op_list[key]
        assert level2_gate_count - 3 * level2_swap_count == ori_gate_count

    # level 3 gate count
    level3_circuit_op_list = dict(result_circuit_3.count_ops())
    level3_gate_count = 0
    level3_swap_count = 0
    if enable_lv3:
        for key in level3_circuit_op_list:
            if key == "swap":
                level3_gate_count += 3 * level3_circuit_op_list[key]
                level3_swap_count = level3_circuit_op_list[key]
            else:
                assert level3_circuit_op_list[key] == ori_circuit_op_list[key]
                level3_gate_count += level3_circuit_op_list[key]
        assert level3_gate_count - 3 * level3_swap_count == ori_gate_count

    return ori_gate_count, level0_gate_count, level0_swap_count, level1_gate_count, level1_swap_count, \
        level2_gate_count, level2_swap_count, level3_gate_count, level3_swap_count


def main():
    # parameters
    benchmark_runs = 32
    qasm_file_name = "gf2^E5_mult_after_heavy.qasm"
    device_name = "IBM_Q20_Tokyo"
    enable_lv2 = True
    enable_lv3 = True

    # run benchmark
    original_gate_count_list = []
    level0_gate_count_list, level0_swap_count_list = [], []
    level1_gate_count_list, level1_swap_count_list = [], []
    level2_gate_count_list, level2_swap_count_list = [], []
    level3_gate_count_list, level3_swap_count_list = [], []
    for _ in range(benchmark_runs):
        data = run_benchmark(qasm_file_name=qasm_file_name, device_name=device_name,
                             enable_lv2=enable_lv2, enable_lv3=enable_lv3)
        ori_gate_count, level0_gate_count, level0_swap_count, level1_gate_count, level1_swap_count, \
            level2_gate_count, level2_swap_count, level3_gate_count, level3_swap_count = data
        original_gate_count_list.append(ori_gate_count)
        level0_gate_count_list.append(level0_gate_count)
        level0_swap_count_list.append(level0_swap_count)
        level1_gate_count_list.append(level1_gate_count)
        level1_swap_count_list.append(level1_swap_count)
        level2_gate_count_list.append(level2_gate_count)
        level2_swap_count_list.append(level2_swap_count)
        level3_gate_count_list.append(level3_gate_count)
        level3_swap_count_list.append(level3_swap_count)

    # print results
    print(f"circuit name: {qasm_file_name}.")
    print(f"device name: {device_name}.")
    print(f"parallel runs: {benchmark_runs}.")
    print(f"Qiskit Transpile level 0 best: {min(level0_gate_count_list)} gates, {min(level0_swap_count_list)} swaps.")
    print(f"Qiskit Transpile level 0 avg.: {sum(level0_gate_count_list) / benchmark_runs} gates,"
          f" {sum(level0_swap_count_list) / benchmark_runs} swaps.")
    print(f"Qiskit Transpile level 1 best: {min(level1_gate_count_list)} gates, {min(level1_swap_count_list)} swaps.")
    print(f"Qiskit Transpile level 1 avg.: {sum(level1_gate_count_list) / benchmark_runs} gates,"
          f" {sum(level1_swap_count_list) / benchmark_runs} swaps.")
    print(f"Qiskit Transpile level 2 best: {min(level2_gate_count_list)} gates, {min(level2_swap_count_list)} swaps.")
    print(f"Qiskit Transpile level 2 avg.: {sum(level2_gate_count_list) / benchmark_runs} gates,"
          f" {sum(level2_swap_count_list) / benchmark_runs} swaps.")
    print(f"Qiskit Transpile level 3 best: {min(level3_gate_count_list)} gates, {min(level3_swap_count_list)} swaps.")
    print(f"Qiskit Transpile level 3 avg.: {sum(level3_gate_count_list) / benchmark_runs} gates,"
          f" {sum(level3_swap_count_list) / benchmark_runs} swaps.")


if __name__ == '__main__':
    main()
