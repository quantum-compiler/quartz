#!/usr/bin/python3

"""
We represent a matrix as list of list of pairs, where a pair
is represented as a real and an imaginary part of a complex number.
Angles are represented with two real numbers, s and c, satisfying s*s+c*c=1
"""

import z3
import math
from .gates import get_matrix, compute
from .utils import *


# functions for generating z3 constraints

def eq_matrix(A, B):
    assert len(A) == len(B)
    assert all(len(ra) == len(rb) for ra, rb in zip(A, B))
    assert all(len(z) == 2 for M in (A, B) for r in M for z in r)
    return z3.And([
        x == y
        for ra, rb in zip(A, B)
        for za, zb in zip(ra, rb)
        for x, y in zip(za, zb)
    ])


def eq_vector(A, B):
    assert len(A) == len(B)
    assert all(len(ra) == len(rb) for ra, rb in zip(A, B))
    return z3.And([
        za == zb
        for ra, rb in zip(A, B)
        for za, zb in zip(ra, rb)
    ])


def matmul(A, B):
    n = len(A)
    assert len(B) == n
    assert all(len(ra) == len(rb) == n for ra, rb in zip(A, B))
    assert all(len(z) == 2 for M in (A, B) for r in M for z in r)
    matrix = list()
    for r in range(n):
        row = list()
        for c in range(n):
            val_r = 0
            val_c = 0
            for k in range(n):
                val_r += A[r][k][0] * B[k][c][0] - A[r][k][1] * B[k][c][1]
                val_c += A[r][k][0] * B[k][c][1] + A[r][k][1] * B[k][c][0]
            row.append((val_r, val_c))
        matrix.append(row)
    return matrix


# Apply |mat| at |qubit_indices| on the distribution vector |vec|
def apply_matrix(vec, mat, qubit_indices):
    # See also: Vector::apply_matrix() in math/vector.cpp
    S = len(vec)
    n = len(mat)
    n0 = len(qubit_indices)
    assert 1 <= n0 <= 2
    assert (1 << n0) == n
    assert n <= S
    assert S % n == 0
    assert all(len(row) == n for row in mat)
    assert all(1 <= (1 << index) < S for index in qubit_indices)
    result_vec = [None] * S
    for i in range(S):
        already_applied = False
        for index in qubit_indices:
            if i & (1 << index):
                already_applied = True
        if already_applied:
            continue

        current_indices = list()
        for j in range(n):
            current_index = i
            for k in range(n0):
                if j & (1 << k):
                    current_index ^= (1 << qubit_indices[k])
            current_indices.append(current_index)

        # matrix * vector
        for r in range(n):
            val_real = 0
            val_imag = 0
            for k in range(n):
                val_real += mat[r][k][0] * vec[current_indices[k]][0] - mat[r][k][1] * vec[current_indices[k]][1]
                val_imag += mat[r][k][0] * vec[current_indices[k]][1] + mat[r][k][1] * vec[current_indices[k]][0]
            result_vec[current_indices[r]] = (val_real, val_imag)
    return result_vec


def input_distribution(num_qubits, solver):
    vec_size = 1 << num_qubits
    real_part = z3.RealVector('r', vec_size)
    imag_part = z3.RealVector('i', vec_size)
    # A quantum state requires the sum of modulus of all numbers to be 1.
    sum_modulus = sum([x * x for x in real_part]) + sum([x * x for x in imag_part])
    solver.add(sum_modulus == 1)
    return list(zip(real_part, imag_part))


def angle(s, c):
    return s * s + c * c == 1


def create_parameters(num_parameters, solver):
    param_cos = z3.RealVector('c', num_parameters)
    param_sin = z3.RealVector('s', num_parameters)
    for i in range(num_parameters):
        solver.add(angle(param_cos[i], param_sin[i]))
    return list(zip(param_cos, param_sin))


def evaluate(dag, input_dis, input_parameters):
    dag_meta = dag[0]
    num_input_parameters = dag_meta[meta_index_num_input_parameters]
    num_total_parameters = dag_meta[meta_index_num_total_parameters]
    assert (len(input_parameters) >= num_input_parameters)
    parameters = input_parameters[:num_input_parameters] + [None] * (num_total_parameters - num_input_parameters)

    output_dis = input_dis
    gates = dag[1]
    for gate in gates:
        parameter_values = []
        qubit_indices = []
        for input in gate[2]:
            if input.startswith('P'):
                # parameter input
                parameter_values.append(parameters[int(input[1:])])
            else:
                assert (input.startswith('Q'))
                # qubit input
                qubit_indices.append(int(input[1:]))
        if gate[1][0].startswith('P'):
            # parameter gate
            assert len(gate[1]) == 1
            parameter_index = int(gate[1][0][1:])
            parameters[parameter_index] = compute(gate[0], *parameter_values)
        else:
            assert (gate[1][0].startswith('Q'))
            # quantum gate
            output_dis = apply_matrix(output_dis, get_matrix(gate[0], *parameter_values), qubit_indices)
    return output_dis


def equivalent(dag1, dag2):
    dag1_meta = dag1[0]
    dag2_meta = dag2[0]
    for index in [meta_index_num_qubits]:
        # check num_qubits
        if dag1_meta[index] != dag2_meta[index]:
            return False

    solver = z3.Solver()
    num_qubits = dag1_meta[meta_index_num_qubits]
    vec = input_distribution(num_qubits, solver)
    num_parameters = max(dag1_meta[meta_index_num_input_parameters], dag2_meta[meta_index_num_input_parameters])
    params = create_parameters(num_parameters, solver)
    output_vec1 = evaluate(dag1, vec, params)
    output_vec2 = evaluate(dag2, vec, params)
    solver.add(z3.Not(eq_vector(output_vec1, output_vec2)))
    result = solver.check()
    return result == z3.unsat


def load_json(file_name):
    import json
    with open(file_name, "r") as f:
        data = json.load(f)
    return data


def dump_json(data, file_name):
    import json
    with open(file_name, 'w') as f:
        json.dump(data, f)


def find_equivalences(input_file, output_file, verbose=False):
    data = load_json(input_file)
    output_dict = {}
    equivalent_called = 0
    total_equivalence_found = 0
    num_hashtags = 0
    num_dags = 0
    num_potential_equivalences = 0
    import time
    t_start = time.monotonic()
    for hashtag, dags in data.items():
        num_hashtags += 1
        num_dags += len(dags)
        if len(dags) <= 1:
            continue
        num_potential_equivalences += len(dags) - 1
        different_dags_with_same_hash = []
        for dag in dags:
            equivalence_found = False
            for i in range(len(different_dags_with_same_hash)):
                other_dag = different_dags_with_same_hash[i]
                equivalent_called += 1
                if equivalent(dag, other_dag):
                    current_tag = hashtag + '_' + str(i)
                    if current_tag not in output_dict.keys():
                        output_dict[current_tag] = [other_dag]
                    output_dict[current_tag].append(dag)
                    equivalence_found = True
                    total_equivalence_found += 1
                    break
            if not equivalence_found:
                different_dags_with_same_hash.append(dag)
    t_end = time.monotonic()
    print(f'{total_equivalence_found} equivalences found in {t_end - t_start} seconds'
          f' (solver invoked {equivalent_called} times for {num_dags} DAGs'
          f' with {num_hashtags} different hash values and {num_potential_equivalences} potential equivalences).')
    t_start = time.monotonic()
    dump_json(output_dict, output_file)
    t_end = time.monotonic()
    print(f'Json saved in {t_end - t_start} seconds.')
