import pulp
import qiskit
import qiskit.circuit.library.standard_gates as gates


def solve_ilp(
    circuit_gate_qubits,
    circuit_gate_is_sparse,
    out_gate,
    num_qubits,
    num_local_qubits,
    num_iterations,
    print_solution=False,
):
    print(
        f"Solving ILP for num_qubits={num_qubits}, num_local_qubits={num_local_qubits}, num_iterations={num_iterations}..."
    )
    # Check if the ILP is feasible in M rounds.
    prob = pulp.LpProblem(
        f"{num_qubits}_{num_local_qubits}_{num_iterations}", pulp.LpMinimize
    )
    num_gates = len(circuit_gate_qubits)

    # a[i, j] = 1 iff the i-th qubit is a local qubit in the j-th iteration
    a = pulp.LpVariable.dicts(
        "a",
        [(i, j) for i in range(num_qubits) for j in range(num_iterations)],
        0,
        1,
        pulp.LpInteger,
    )

    # b[i, j] = 1 iff the i-th gate is finished before or at the j-th iteration
    b = pulp.LpVariable.dicts(
        "b",
        [(i, j) for i in range(num_gates) for j in range(num_iterations + 1)],
        0,
        1,
        pulp.LpInteger,
    )

    prob += 0  # minimize nothing

    # For each iteration, we have at most k local qubits.
    for j in range(num_iterations):
        prob += sum([a[i, j] for i in range(num_qubits)]) <= num_local_qubits

    # If a gate is executed before or at the j-th iteration,
    # this should also hold in the (j+1)-th iteration.
    for i in range(num_gates):
        for j in range(num_iterations):
            prob += b[i, j] <= b[i, j + 1]

    # If a non-sparse gate is executed at the j-th iteration,
    # its qubits must be local at the j-th iteration.
    for i in range(num_gates):
        if not circuit_gate_is_sparse[i]:
            for qubit_id in circuit_gate_qubits[i]:
                for j in range(num_iterations):
                    prob += b[i, j + 1] - b[i, j] <= a[qubit_id, j]

    # Dependencies
    for i1 in range(num_gates):
        for i2 in out_gate[i1]:
            for j in range(num_iterations + 1):
                prob += b[i1, j] >= b[i2, j]

    # At the beginning, all gates should not be executed.
    for i in range(num_gates):
        prob += b[i, 0] == 0

    # At the end, all gates should be executed.
    for i in range(num_gates):
        prob += b[i, num_iterations] == 1

    print("Available solvers:", pulp.listSolvers(onlyAvailable=True))
    solver = pulp.HiGHS_CMD()
    prob.solve(solver)
    # prob.solve()
    if print_solution:
        print("Status:", pulp.LpStatus[prob.status])
        for v in prob.variables():
            print(v.name, "=", v.varValue)
        executed = set()
        for j in range(1, num_iterations + 1):
            print("Iteration", j)
            for v in prob.variables():
                if v.name.startswith("b") and v.varValue == 1.0:
                    if v.name.endswith(str(j) + ")"):
                        gate_id = int(v.name.split("(")[1].split(",")[0])
                        if gate_id not in executed:
                            print(gate_id)
                            executed.add(gate_id)
        for j in range(num_iterations):
            for v in prob.variables():
                if v.name.startswith("a") and v.varValue == 1.0:
                    if v.name.endswith(str(j) + ")"):
                        print(v.name.split("(")[1].split(",")[0], end=" ")
            print()

    if prob.status is not pulp.LpStatusOptimal:
        return []  # an empty list
    else:
        # return the solution
        result = [[] for _ in range(num_iterations)]
        for j in range(num_iterations):
            for v in prob.variables():
                if v.name.startswith("a") and v.varValue == 1.0:
                    if v.name.endswith(str(j) + ")"):
                        result[j].append(int(v.name.split("(")[1].split(",")[0]))
        return result


def preprocess_circuit_seq(circuit_seq, index_offset, num_qubits):
    sparse_gates = {
        gates.x.XGate,
        gates.x.CXGate,
        gates.z.CZGate,
        gates.p.CPhaseGate,
        gates.swap.SwapGate,
    }
    non_sparse_gates = {
        gates.h.HGate,
        gates.ry.RYGate,
        gates.u.UGate,
        gates.u2.U2Gate,
        gates.u3.U3Gate,
    }

    circuit_gate_qubits = [
        [index_offset[qubit.register] + qubit.index for qubit in gate[1]]
        for gate in circuit_seq
    ]

    for gate in circuit_seq:
        assert type(gate[0]) in sparse_gates or type(gate[0]) in non_sparse_gates
    circuit_gate_is_sparse = [type(gate[0]) in sparse_gates for gate in circuit_seq]

    # Dependencies
    num_gates = len(circuit_seq)
    last_gate = [None for _ in range(num_qubits)]
    in_gate = [set() for _ in range(num_gates)]
    out_gate = [set() for _ in range(num_gates)]
    for i in range(num_gates):
        for qubit_id in circuit_gate_qubits[i]:
            if last_gate[qubit_id] is not None:
                in_gate[i].add(last_gate[qubit_id])
                out_gate[last_gate[qubit_id]].add(i)
            last_gate[qubit_id] = i

    return circuit_gate_qubits, circuit_gate_is_sparse, out_gate


def run(n, circuit_name, repeat_circuit=1):
    print("Start running", circuit_name, "with n =", n)
    circuit = qiskit.circuit.QuantumCircuit.from_qasm_file(
        f"circuit/MQTBench_{n}q/{circuit_name}_indep_qiskit_{n}.qasm"
    )
    # Compute |index_offset| by sorting quantum registers by their names.
    qregs = sorted(circuit.qregs, key=lambda qreg: qreg.name)
    index_offset = {}
    num_qubits = 0
    for qreg in qregs:
        index_offset[qreg] = num_qubits
        num_qubits += qreg.size

    circuit_seq_raw = circuit.data
    circuit_seq = []
    for gate in circuit_seq_raw:
        if not isinstance(gate[0], qiskit.circuit.barrier.Barrier) and not isinstance(
            gate[0], qiskit.circuit.measure.Measure
        ):
            circuit_seq.append(gate)

    if repeat_circuit > 1:
        # Repeat the whole circuit |repeat_circuit| times
        circuit_seq *= repeat_circuit

    # Contract the sparse gates.
    # Commented to not contract them now.
    # for i in range(G):
    #     if type(circuit_seq[i][0]) in sparse_gates:
    #         for g1 in in_gate[i]:
    #             out_gate[g1].remove(i)
    #         for g2 in out_gate[i]:
    #             in_gate[g2].remove(i)
    #         for g1 in in_gate[i]:
    #             for g2 in out_gate[i]:
    #                 out_gate[g1].add(g2)
    #                 in_gate[g2].add(g1)
    #         in_gate[i].clear()
    #         out_gate[i].clear()

    print("Start solving ILP...")
    # print(n, end=" ", file=log_file, flush=True)
    for k in range(28, 29):
        for M in range(1, 100):
            if (
                len(
                    solve_ilp(
                        *preprocess_circuit_seq(circuit_seq, index_offset, n),
                        num_qubits=n,
                        num_local_qubits=k,
                        num_iterations=M,
                    )
                )
                != 0
            ):
                print(M, end=" ", flush=True)
                break
    print("Done!")
    # if circuit_name == "wstate":
    #     solve_ilp(circuit_seq, out_gate, index_offset, n, 30, 2, print_solution=True)


# TODO: Add a Python script in another file
# if __name__ == '__main__':
#     log_file = open("../../../result_ilp.txt", "w")
#
#     circuit_names = [
#         "dj",
#         "ghz",
#         "graphstate",
#         "qft",
#         "qftentangled",
#         "realamprandom",
#         "su2random",
#         "twolocalrandom",
#         "wstate",
#         "ae",
#         "qpeexact",
#         "qpeinexact",
#     ]
#
#     start = time.time()
#
#     for circuit_name in circuit_names:
#         print(circuit_name, file=log_file)
#         for n in range(40, 43):
#             run(n, circuit_name, repeat_circuit=2)
#             print(file=log_file)
#
#     log_file.close()
#     end = time.time()
#     print(end - start)
