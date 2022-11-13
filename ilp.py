import pulp
import qiskit
import qiskit.circuit.library.standard_gates as gates

sparse_gates = {gates.p.CPhaseGate, gates.swap.SwapGate}
non_sparse_gates = {gates.h.HGate}


def solve_ilp(circuit_seq, n, k, M):
    # Check if the ILP is feasible in M rounds.
    prob = pulp.LpProblem(f'{n}_{k}_{M}', pulp.LpMinimize)
    G = len(circuit_seq)

    # a[i, j] = 1 iff the i-th qubit is a local qubit in the j-th iteration
    a = pulp.LpVariable.dicts(
        'a', [(i, j) for i in range(n) for j in range(M)], 0, 1, pulp.LpInteger
    )

    # b[i, j] = 1 iff the i-th gate is finished before or at the j-th iteration
    b = pulp.LpVariable.dicts(
        'b', [(i, j) for i in range(G) for j in range(M + 1)], 0, 1, pulp.LpInteger
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
    print("Status:", pulp.LpStatus[prob.status])
    # for v in prob.variables():
    #     print(v.name, "=", v.varValue)
    return prob.status is pulp.LpStatusOptimal


circuit = qiskit.circuit.QuantumCircuit.from_qasm_file(
    'circuit/MQTBench_40q/qft_indep_qiskit_40.qasm'
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
n = circuit.num_qubits


print(solve_ilp(circuit_seq, n, 30, 1))
print(solve_ilp(circuit_seq, n, 30, 2))
