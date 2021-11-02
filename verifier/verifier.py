#!/usr/bin/python3

"""
We represent a matrix as list of list of pairs, where a pair
is represented as a real and an imaginary part of a complex number.
Angles are represented with two real numbers, s and c, satisfying s*s+c*c=1
"""

import z3
from .gates import get_matrix, compute
from utils.utils import *


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


def input_distribution(num_qubits, equation_list):
    vec_size = 1 << num_qubits
    real_part = z3.RealVector('r', vec_size)
    imag_part = z3.RealVector('i', vec_size)
    # A quantum state requires the sum of modulus of all numbers to be 1.
    sum_modulus = sum([x * x for x in real_part]) + sum([x * x for x in imag_part])
    equation_list.append(sum_modulus == 1)
    return list(zip(real_part, imag_part))


def angle(c, s):
    return s * s + c * c == 1


def create_parameters(num_parameters, equation_list):
    param_cos = z3.RealVector('c', num_parameters)
    param_sin = z3.RealVector('s', num_parameters)
    for i in range(num_parameters):
        equation_list.append(angle(param_cos[i], param_sin[i]))
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


def phase_shift(vec, lam):
    # shift |vec| by exp(i * lam)
    assert len(lam) == 2  # cos, sin
    shifted_vec = vec
    for i in range(len(shifted_vec)):
        assert len(shifted_vec[i]) == 2
        shifted_vec[i] = (shifted_vec[i][0] * lam[0] - shifted_vec[i][1] * lam[1],
                          shifted_vec[i][1] * lam[0] + shifted_vec[i][0] * lam[1])
    return shifted_vec


def equivalent(dag1, dag2, check_phase_shift=False):
    dag1_meta = dag1[0]
    dag2_meta = dag2[0]
    for index in [meta_index_num_qubits]:
        # check num_qubits
        if dag1_meta[index] != dag2_meta[index]:
            return False

    solver = z3.Solver()
    num_qubits = dag1_meta[meta_index_num_qubits]
    equation_list = []
    vec = input_distribution(num_qubits, equation_list)
    num_parameters = max(dag1_meta[meta_index_num_input_parameters], dag2_meta[meta_index_num_input_parameters])
    params = create_parameters(num_parameters, equation_list)
    output_vec1 = evaluate(dag1, vec, params)
    output_vec2 = evaluate(dag2, vec, params)
    if check_phase_shift:
        cosL = z3.Reals('cosL')
        sinL = z3.Reals('sinL')
        output_vec2 = phase_shift(output_vec2, [cosL, sinL])
        solver.add(z3.ForAll([cosL, sinL], z3.Implies(angle(cosL, sinL),
                                                      z3.Exists(vec + params, z3.And(equation_list,
                                                                                     z3.Not(eq_vector(output_vec1, output_vec2)))))))
    else:
        solver.add(z3.And(equation_list))
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


def find_equivalences(input_file, output_file, print_basic_info=True, verbose=False, keep_classes_with_1_dag=False,
                      check_equivalence_with_different_hash=True, check_phase_shift=False):
    data = load_json(input_file)
    output_dict = {}
    equivalent_called = 0
    total_equivalence_found = 0
    num_hashtags = 0
    num_dags = 0
    num_potential_equivalences = 0
    import time
    t_start = time.monotonic()
    num_different_dags_with_same_hash = {}
    for hashtag, dags in data.items():
        num_hashtags += 1
        num_dags += len(dags)
        num_potential_equivalences += len(dags) - 1
        different_dags_with_same_hash = []
        if verbose:
            print(f'Verifying {len(dags)} DAGs with hash value {hashtag}...')
        for dag in dags:
            equivalence_found = False
            for i in range(len(different_dags_with_same_hash)):
                other_dag = different_dags_with_same_hash[i]
                equivalent_called += 1
                if equivalent(dag, other_dag, check_phase_shift):
                    current_tag = hashtag + '_' + str(i)
                    assert current_tag in output_dict.keys()
                    output_dict[current_tag].append(dag)
                    equivalence_found = True
                    total_equivalence_found += 1
                    break
            if not equivalence_found:
                different_dags_with_same_hash.append(dag)
                # Insert |dag| eagerly
                current_tag = hashtag + '_' + str(len(different_dags_with_same_hash) - 1)
                output_dict[current_tag] = [dag]
        num_different_dags_with_same_hash[hashtag] = len(different_dags_with_same_hash)

    hashtags_in_more_equivalences = set()
    if check_equivalence_with_different_hash:
        more_equivalences = []
        equivalent_called_2 = 0
        for hashtag, dags in output_dict.items():
            other_hashtags = set()
            assert len(dags) > 0
            for dag in dags:
                dag_meta = dag[0]
                other_hashtags.update(dag_meta[meta_index_other_hash_values])
            assert hashtag.split('_')[0] not in other_hashtags
            if len(other_hashtags) == 0:
                print(f'Warning: other hash values unspecified for hash value {hashtag}.'
                      f' Cannot guarantee there are no missing equivalences.')
            possible_equivalent_dags = []
            for other_hashtag in other_hashtags:
                if other_hashtag not in data.keys():
                    # Not equivalent to any other ones
                    continue
                assert other_hashtag in num_different_dags_with_same_hash.keys()
                i_range = num_different_dags_with_same_hash[other_hashtag]
                for i in range(i_range):
                    current_tag = other_hashtag + '_' + str(i)
                    possible_equivalent_dags.append((output_dict[current_tag][0], current_tag))
            if verbose and len(possible_equivalent_dags) > 0:
                print(f'Verifying {len(possible_equivalent_dags)} possible missing equivalences'
                      f' with hash value {hashtag} and {len(other_hashtags)} other hash values...')
            for other_dag in possible_equivalent_dags:
                equivalent_called_2 += 1
                if equivalent(dags[0], other_dag[0], check_phase_shift):
                    more_equivalences.append((hashtag, other_dag[1]))
                    hashtags_in_more_equivalences.update(hashtag)
                    hashtags_in_more_equivalences.update(other_dag[1])
                    if verbose:
                        print(f'Equivalence with hash value {hashtag} and {other_dag[1]}:\n'
                              f'{dags[0]}\n'
                              f'{other_dag[0]}')
        output_dict = [more_equivalences, output_dict]
        if print_basic_info:
            print(f'Solver invoked {equivalent_called_2} times to find {len(more_equivalences)} equivalences'
                  f' with different hash.')
    else:
        # Add a placeholder here
        output_dict = [[], output_dict]

    if not keep_classes_with_1_dag:
        output_dict[1] = {k: v for k, v in output_dict[1].items() if
                          (len(v) >= 2 or k in hashtags_in_more_equivalences)}
    t_end = time.monotonic()
    if print_basic_info:
        print(f'{total_equivalence_found} equivalences found in {t_end - t_start} seconds'
              f' (solver invoked {equivalent_called} times for {num_dags} DAGs'
              f' with {num_hashtags} different hash values and {num_potential_equivalences} potential equivalences),'
              f' output {len(output_dict[1])} equivalence classes.')
    t_start = time.monotonic()
    dump_json(output_dict, output_file)
    t_end = time.monotonic()
    if print_basic_info:
        print(f'Json saved in {t_end - t_start} seconds.')
