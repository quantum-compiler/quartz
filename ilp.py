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


def solve_ilp(circuit_seq, n, k, M, print_solution=False):
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
                qubit_id = qubit.index  # XXX: assume there is only one qreg
                for j in range(M):
                    prob += b[i, j + 1] - b[i, j] <= a[qubit_id, j]

    # TODO: this is ad-hoc
    # Dependencies
    for i1 in range(G):
        if type(circuit_seq[i1][0]) in non_sparse_gates:
            i1_qubits = set([qubit.index for qubit in circuit_seq[i1][1]])
            for i2 in range(G):
                if type(circuit_seq[i2][0]) in non_sparse_gates:
                    i2_qubits = set([qubit.index for qubit in circuit_seq[i2][1]])
                    if i1_qubits & i2_qubits:
                        for j in range(M):
                            prob += b[i1, j] >= b[i2, j]

    # At the beginning, all gates should be executed.
    for i in range(G):
        prob += b[i, 0] == 0

    # At the end, all gates should be executed.
    for i in range(G):
        prob += b[i, M] == 1

    prob.solve()
    if print_solution:
        print("Status:", pulp.LpStatus[prob.status])
        for v in prob.variables():
            print(v.name, "=", v.varValue)
    return prob.status is pulp.LpStatusOptimal


def run(n, circuit_name):
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
    print(n, end=" ", file=log_file, flush=True)
    for k in range(12, 34):
        for M in range(1, 100):
            if solve_ilp(circuit_seq, n, k, M):
                print(M, end=" ", file=log_file, flush=True)
                break
    if circuit_name == "wstate":
        solve_ilp(circuit_seq, n, 14, 3, print_solution=True)


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
