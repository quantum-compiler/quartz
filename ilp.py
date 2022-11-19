import time

import pulp
import qiskit
import qiskit.circuit.library.standard_gates as gates

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
log_file = open("result_ilp.txt", "w")


def solve_ilp(circuit_seq, out_gate, n, k, M, print_solution=False):
    print(f"Solving ILP for n={n}, k={k}, M={M}...")
    # Check if the ILP is feasible in M rounds.
    prob = pulp.LpProblem(f"{n}_{k}_{M}", pulp.LpMinimize)
    G = len(circuit_seq)

    # a[i, j] = 1 iff the i-th qubit is a local qubit in the j-th iteration
    a = pulp.LpVariable.dicts(
        "a", [(i, j) for i in range(n) for j in range(M)], 0, 1, pulp.LpInteger
    )

    # b[i, j] = 1 iff the i-th gate is finished before or at the j-th iteration
    b = pulp.LpVariable.dicts(
        "b", [(i, j) for i in range(G) for j in range(M + 1)], 0, 1, pulp.LpInteger
    )

    prob += 0  # minimize nothing

    # For each iteration, we have at most k local qubits.
    for j in range(M):
        prob += sum([a[i, j] for i in range(n)]) <= k

    # If a gate is executed before or at the j-th iteration,
    # this should also hold in the (j+1)-th iteration.
    for i in range(G):
        for j in range(M):
            prob += b[i, j] <= b[i, j + 1]

    # If a non-sparse gate is executed at the j-th iteration,
    # its qubits must be local at the j-th iteration.
    for i in range(G):
        if type(circuit_seq[i][0]) in non_sparse_gates:
            for qubit in circuit_seq[i][1]:
                qubit_id = (
                    qubit.index
                )  # XXX: assume there is only one qreg; similarly hereinafter
                for j in range(M):
                    prob += b[i, j + 1] - b[i, j] <= a[qubit_id, j]

    # Dependencies
    for i1 in range(G):
        for i2 in out_gate[i1]:
            for j in range(M + 1):
                prob += b[i1, j] >= b[i2, j]

    # At the beginning, all gates should not be executed.
    for i in range(G):
        prob += b[i, 0] == 0

    # At the end, all gates should be executed.
    for i in range(G):
        prob += b[i, M] == 1

    print("Available solvers:", pulp.listSolvers(onlyAvailable=True))
    solver = pulp.HiGHS_CMD()
    prob.solve(solver)
    # prob.solve()
    if print_solution:
        print("Status:", pulp.LpStatus[prob.status])
        for v in prob.variables():
            print(v.name, "=", v.varValue)
        for j in range(M):
            for v in prob.variables():
                if v.name.startswith('a') and v.varValue == 1.0:
                    if v.name.endswith(str(j) + ')'):
                        print(v.name.split('(')[1].split(',')[0], end=' ')
            print()
    return prob.status is pulp.LpStatusOptimal


def run(n, circuit_name):
    print("Start running", circuit_name, "with n =", n)
    circuit = qiskit.circuit.QuantumCircuit.from_qasm_file(
        f"circuit/MQTBench_{n}q/{circuit_name}_indep_qiskit_{n}.qasm"
    )
    circuit_seq_raw = circuit.data
    circuit_seq = []
    for gate in circuit_seq_raw:
        if not isinstance(gate[0], qiskit.circuit.barrier.Barrier) and not isinstance(
            gate[0], qiskit.circuit.measure.Measure
        ):
            circuit_seq.append(gate)
            # print(gate)
            assert type(gate[0]) in sparse_gates or type(gate[0]) in non_sparse_gates

    # Dependencies
    G = len(circuit_seq)
    last_gate = [None for _ in range(n)]
    in_gate = [set() for _ in range(G)]
    out_gate = [set() for _ in range(G)]
    for i in range(G):
        for qubit in circuit_seq[i][1]:
            qubit_id = qubit.index
            if last_gate[qubit_id] is not None:
                in_gate[i].add(last_gate[qubit_id])
                out_gate[last_gate[qubit_id]].add(i)
            last_gate[qubit_id] = i

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
    print(n, end=" ", file=log_file, flush=True)
    for k in range(12, 34):
        for M in range(1, 100):
            if solve_ilp(circuit_seq, out_gate, n, k, M):
                print(M, end=" ", file=log_file, flush=True)
                break
    print("Done!")
    if circuit_name == "wstate":
        solve_ilp(circuit_seq, out_gate, n, 14, 3, print_solution=True)


circuit_names = [
    "dj",
    "ghz",
    "graphstate",
    "qft",
    "qftentangled",
    "realamprandom",
    "su2random",
    "twolocalrandom",
    "wstate",
]

for circuit_name in circuit_names:
    print(circuit_name, file=log_file)
    for n in range(40, 43):
        run(n, circuit_name)
        print(file=log_file)

log_file.close()
print(time.process_time())
