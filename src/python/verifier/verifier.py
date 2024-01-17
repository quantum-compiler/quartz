#!/usr/bin/python3

"""
We represent a matrix as list of pairs, where a pair
is represented as a real and an imaginary part of a complex number.
Angles are represented with two real numbers, s and c, satisfying s*s+c*c=1
"""

import copy
import math
import multiprocessing as mp
import os
import sys

import z3
from gates import add, compute, get_matrix, neg  # for searching phase factors

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.utils import *

# functions for generating z3 constraints


def eq_matrix(A, B):
    assert len(A) == len(B)
    assert all(len(ra) == len(rb) for ra, rb in zip(A, B))
    assert all(len(z) == 2 for M in (A, B) for r in M for z in r)
    return z3.And(
        [
            x == y
            for ra, rb in zip(A, B)
            for za, zb in zip(ra, rb)
            for x, y in zip(za, zb)
        ]
    )


def eq_vector(A, B):
    assert len(A) == len(B)
    assert all(len(ra) == len(rb) for ra, rb in zip(A, B))
    return [za == zb for ra, rb in zip(A, B) for za, zb in zip(ra, rb)]


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
                    current_index ^= 1 << qubit_indices[k]
            current_indices.append(current_index)

        # matrix * vector
        for r in range(n):
            val_real = 0
            val_imag = 0
            for k in range(n):
                val_real += (
                    mat[r][k][0] * vec[current_indices[k]][0]
                    - mat[r][k][1] * vec[current_indices[k]][1]
                )
                val_imag += (
                    mat[r][k][0] * vec[current_indices[k]][1]
                    + mat[r][k][1] * vec[current_indices[k]][0]
                )
            result_vec[current_indices[r]] = (val_real, val_imag)
    assert len(result_vec[0]) == 2
    return result_vec


def input_distribution(num_qubits, equation_list):
    vec_size = 1 << num_qubits
    real_part = z3.RealVector("r", vec_size)
    imag_part = z3.RealVector("i", vec_size)
    # A quantum state requires the sum of modulus of all numbers to be 1.
    sum_modulus = sum([x * x for x in real_part]) + sum([x * x for x in imag_part])
    equation_list.append(sum_modulus == 1)
    return list(zip(real_part, imag_part))


def random_input_distribution(num_qubits):
    vec_size = 1 << num_qubits
    import random

    real_part = [random.random() * 2 - 1 for _ in range(vec_size)]
    imag_part = [random.random() * 2 - 1 for _ in range(vec_size)]
    # A quantum state requires the sum of modulus of all numbers to be 1.
    sum_modulus = sum([x * x for x in real_part]) + sum([x * x for x in imag_part])
    real_part = [x / math.sqrt(sum_modulus) for x in real_part]
    imag_part = [x / math.sqrt(sum_modulus) for x in imag_part]
    return list(zip(real_part, imag_part))


def angle(c, s):
    return s * s + c * c == 1


def create_parameters(num_parameters, equation_list):
    param_cos = z3.RealVector("c", num_parameters)
    param_sin = z3.RealVector("s", num_parameters)
    for i in range(num_parameters):
        equation_list.append(angle(param_cos[i], param_sin[i]))
    return list(zip(param_cos, param_sin))


def random_parameters(num_parameters):
    import random

    param_cos = [random.random() * 2 - 1 for _ in range(num_parameters)]
    param_sin = [0 for _ in range(num_parameters)]
    for i in range(num_parameters):
        param_sin[i] = math.sqrt(1 - param_cos[i] * param_cos[i])
        if random.random() > 0.5:
            param_sin[i] = -param_sin[i]
    return list(zip(param_cos, param_sin))


def evaluate(dag, input_dis, params, use_z3=True):
    output_dis = input_dis
    gates = dag[1]
    for gate in gates:
        parameter_values = []
        qubit_indices = []
        for input_wire in gate[2]:
            if input_wire.startswith("P"):
                # parameter input
                parameter_values.append(params[int(input_wire[1:])])
            else:
                assert input_wire.startswith("Q")
                # qubit input
                qubit_indices.append(int(input_wire[1:]))
        assert gate[1][0].startswith("Q")
        # quantum gate
        if use_z3:
            output_dis = apply_matrix(
                output_dis, get_matrix(gate[0], *parameter_values), qubit_indices
            )
        else:
            output_dis = apply_matrix(
                output_dis,
                get_matrix(gate[0], *parameter_values, False),
                qubit_indices,
            )
    return output_dis


def phase_shift(vec, lam):
    # shift |vec| by exp(i * lam)
    import copy

    assert len(lam) == 2  # cos, sin
    shifted_vec = copy.deepcopy(vec)
    for i in range(len(shifted_vec)):
        assert len(shifted_vec[i]) == 2
        shifted_vec[i] = (
            shifted_vec[i][0] * lam[0] - shifted_vec[i][1] * lam[1],
            shifted_vec[i][1] * lam[0] + shifted_vec[i][0] * lam[1],
        )
    return shifted_vec


def phase_shift_by_id(vec, dag, phase_shift_id, all_parameters):
    # Warning: If CircuitSeq::hash() is modified, this function should be modified correspondingly.
    if (
        kCheckPhaseShiftOfPiOver4Index
        < phase_shift_id
        < kCheckPhaseShiftOfPiOver4Index + 8
    ):
        k = phase_shift_id - kCheckPhaseShiftOfPiOver4Index
        phase_shift_lambda = (
            kPhaseFactorConstantCosTable[k],
            kPhaseFactorConstantSinTable[k],
        )
    elif phase_shift_id < len(all_parameters):
        phase_shift_lambda = all_parameters[phase_shift_id]
    else:
        phase_shift_lambda = all_parameters[phase_shift_id - len(all_parameters)]
        # lam -> -lam: (cos, sin) -> (cos, -sin)
        phase_shift_lambda = (phase_shift_lambda[0], -phase_shift_lambda[1])
    return phase_shift(vec, phase_shift_lambda)


def search_phase_factor_to_check_equivalence(
    dag1,
    dag2,
    equations,
    output_vec1,
    output_vec2,
    do_not_invoke_smt_solver,
    parameters_symbolic,
    parameters_for_fingerprint,
    num_parameters,
    goal_phase_factor,
    current_param_id,
    current_phase_factor_symbolic,
    current_phase_factor_for_fingerprint,
):
    if current_param_id == num_parameters:
        # Search for constants
        for const_coeff in range(
            kPhaseFactorConstantCoeffMin, kPhaseFactorConstantCoeffMax + 1
        ):
            const_val = const_coeff * kPhaseFactorConstant
            new_phase_factor_for_fingerprint = (
                current_phase_factor_for_fingerprint + const_val
            )
            new_phase_factor_symbolic = add(
                current_phase_factor_symbolic,
                (
                    kPhaseFactorConstantCosTable[const_coeff],
                    kPhaseFactorConstantSinTable[const_coeff],
                ),
            )
            if search_phase_factor_to_check_equivalence(
                dag1,
                dag2,
                equations,
                output_vec1,
                output_vec2,
                do_not_invoke_smt_solver,
                parameters_symbolic,
                parameters_for_fingerprint,
                num_parameters,
                goal_phase_factor,
                current_param_id + 1,
                new_phase_factor_symbolic,
                new_phase_factor_for_fingerprint,
            ):
                return True
        return False
    if current_param_id == num_parameters + 1:
        # Done searching, check for equivalence
        phase_factor_for_fingerprint = (
            math.cos(current_phase_factor_for_fingerprint),
            math.sin(current_phase_factor_for_fingerprint),
        )
        difference = complex(*phase_factor_for_fingerprint) - goal_phase_factor
        if abs(difference) > kPhaseFactorEpsilon:
            return False
        if do_not_invoke_smt_solver:
            dag1_meta = dag1[0]
            dag2_meta = dag2[0]
            num_qubits = dag1_meta[meta_index_num_qubits]
            num_parameters = len(parameters_for_fingerprint)
            if num_parameters == 0:
                return True
            # Generate a random test to verify it.
            vec = random_input_distribution(num_qubits)
            params = random_parameters(num_parameters)
            output_vec1 = evaluate(dag1, vec, params, use_z3=False)
            output_vec2 = evaluate(dag2, vec, params, use_z3=False)
            current_phase_factor = [
                z3.simplify(
                    z3.substitute(
                        phase_factor_component,
                        [
                            (parameters_symbolic[i], z3.RealVal(params[i]))
                            for i in range(num_parameters)
                        ],
                    )
                )
                for phase_factor_component in current_phase_factor_symbolic
            ]
            output_vec2_shifted = phase_shift(output_vec2, current_phase_factor)
            for i in range(1 << num_qubits):
                a = output_vec1[i]
                b = output_vec2_shifted[i]
                diff_a_b = (a[0] - b[0], a[1] - b[1])
                if z3.simplify(
                    z3.Sqrt(diff_a_b[0] ** 2 + diff_a_b[1] ** 2) > kPhaseFactorEpsilon
                ):
                    return False
            return True
        # Found a possible phase factor
        # print(f'Checking phase factor {current_phase_factor_for_fingerprint}')
        solver = z3.Solver()
        solver.add(equations)
        output_vec2_shifted = phase_shift(output_vec2, current_phase_factor_symbolic)
        solver.add(z3.Not(z3.And(eq_vector(output_vec1, output_vec2_shifted))))
        solver.set("timeout", 30000)  # timeout after 30s
        result = solver.check()
        if result != z3.unsat:
            print(
                f"z3 returns {result} for the following equivalence which passed random testing:"
            )
            print(f"Phase factor for fingerprint is {phase_factor_for_fingerprint}")
            print(f"Goal phase factor is {goal_phase_factor}")
            print(f"Symbolic phase factor is {current_phase_factor_symbolic}")
            print(f"Dags are {dag1} and {dag2}")
            if result == z3.sat:
                print(f"Solver found {solver.model()}")
        return result == z3.unsat

    # Search for the parameter |current_param_id|
    for coeff in kPhaseFactorCoeffs:
        new_phase_factor_for_fingerprint = current_phase_factor_for_fingerprint
        new_phase_factor_symbolic = current_phase_factor_symbolic
        if coeff != 0:
            new_phase_factor_for_fingerprint = (
                current_phase_factor_for_fingerprint
                + coeff * parameters_for_fingerprint[current_param_id]
            )
            if coeff == 1:
                new_phase_factor_symbolic = add(
                    current_phase_factor_symbolic, parameters_symbolic[current_param_id]
                )
            elif coeff == -1:
                new_phase_factor_symbolic = add(
                    current_phase_factor_symbolic,
                    neg(parameters_symbolic[current_param_id]),
                )
            elif coeff == 2:
                new_phase_factor_symbolic = add(
                    current_phase_factor_symbolic,
                    add(
                        parameters_symbolic[current_param_id],
                        parameters_symbolic[current_param_id],
                    ),
                )
            elif coeff == -2:
                new_phase_factor_symbolic = add(
                    current_phase_factor_symbolic,
                    neg(
                        add(
                            parameters_symbolic[current_param_id],
                            parameters_symbolic[current_param_id],
                        )
                    ),
                )
            else:
                raise Exception(f"Unsupported phase factor coefficient {coeff}")
        if search_phase_factor_to_check_equivalence(
            dag1,
            dag2,
            equations,
            output_vec1,
            output_vec2,
            do_not_invoke_smt_solver,
            parameters_symbolic,
            parameters_for_fingerprint,
            num_parameters,
            goal_phase_factor,
            current_param_id + 1,
            new_phase_factor_symbolic,
            new_phase_factor_for_fingerprint,
        ):
            return True
    return False


def equivalent(
    dag1,
    dag2,
    params,
    equation_list_for_params,
    parameters_for_fingerprint,
    do_not_invoke_smt_solver=False,
    check_phase_shift_in_smt_solver=False,
    phase_shift_id=None,
):
    dag1_meta = dag1[0]
    dag2_meta = dag2[0]
    for index in [meta_index_num_qubits]:
        # check num_qubits
        if dag1_meta[index] != dag2_meta[index]:
            return False

    solver = z3.Solver()
    num_qubits = dag1_meta[meta_index_num_qubits]
    equation_list = copy.deepcopy(
        equation_list_for_params
    )  # avoid changing |equation_list_for_params|
    if check_phase_shift_in_smt_solver:
        # Let z3 check phase shift
        cosL = z3.Real("cosL")
        sinL = z3.Real("sinL")
        matrix_equal_list = []
        for S in range(1 << num_qubits):
            # Construct a vector with only the S-th place being 1
            vec_S = [(int(i == S), 0) for i in range(1 << num_qubits)]
            output_vec1_S = evaluate(dag1, vec_S, params)
            output_vec2_S = evaluate(dag2, vec_S, params)
            output_vec2_S_shifted = phase_shift(output_vec2_S, [cosL, sinL])
            matrix_equal_list += eq_vector(output_vec1_S, output_vec2_S_shifted)
        solver.add(z3.And(equation_list))
        solver.add(
            z3.ForAll(
                [cosL, sinL],
                z3.Implies(angle(cosL, sinL), z3.Not(z3.And(matrix_equal_list))),
            )
        )
        result = solver.check()
        assert result != z3.unknown
        return result == z3.unsat
    else:
        # Phase factor is never provided in generator now
        assert phase_shift_id is None

        # Figure out the phase factor here
        num_parameters = len(parameters_for_fingerprint)
        goal_phase_factor = complex(
            *dag1_meta[meta_index_original_fingerprint]
        ) / complex(*dag2_meta[meta_index_original_fingerprint])
        if num_parameters > 0:
            # Construct the input vector as variables
            vec = input_distribution(num_qubits, equation_list)
            output_vec1 = evaluate(dag1, vec, params)
            output_vec2 = evaluate(dag2, vec, params)
            equations = z3.And(equation_list)
            result = search_phase_factor_to_check_equivalence(
                dag1,
                dag2,
                equations,
                output_vec1,
                output_vec2,
                do_not_invoke_smt_solver,
                params,
                parameters_for_fingerprint,
                num_parameters,
                goal_phase_factor,
                current_param_id=0,
                current_phase_factor_symbolic=(z3.RealVal(1), z3.RealVal(0)),
                current_phase_factor_for_fingerprint=0,
            )
            if not result:
                print(
                    f"Cannot find equivalence for dags (vector approach):\n{dag1}\n{dag2}"
                )
            return result
        else:
            # Compare the matrices directly
            output_vec1 = []
            output_vec2 = []
            for S in range(1 << num_qubits):
                # Construct a vector with only the S-th place being 1
                vec_S = [(int(i == S), 0) for i in range(1 << num_qubits)]
                output_vec1_S = evaluate(dag1, vec_S, [])
                output_vec2_S = evaluate(dag2, vec_S, [])
                output_vec1 += output_vec1_S
                output_vec2 += output_vec2_S
            result = search_phase_factor_to_check_equivalence(
                dag1,
                dag2,
                [],
                output_vec1,
                output_vec2,
                do_not_invoke_smt_solver,
                [],
                parameters_for_fingerprint,
                num_parameters,
                goal_phase_factor,
                current_param_id=0,
                current_phase_factor_symbolic=(z3.RealVal(1), z3.RealVal(0)),
                current_phase_factor_for_fingerprint=0,
            )
            if not result:
                print(
                    f"Cannot find equivalence for dags (matrix approach):\n{dag1}\n{dag2}"
                )
            return result


def load_json(file_name):
    import json

    with open(file_name, "r") as f:
        data = json.load(f)
    return data


def dump_json(data, file_name):
    import json

    with open(file_name, "w") as f:
        json.dump(data, f)


# find_equivalence_helper_called_bar = 0
# find_equivalence_helper_called = 0
# total_circuits_verified = 0


def find_equivalences_helper(
    hashtag,
    dags,
    param_info,
    parameters_for_fingerprint,
    check_phase_shift_in_smt_solver,
    verbose,
    do_not_invoke_smt_solver,
):
    output_dict = {}
    equivalent_called = 0
    total_equivalence_found = 0
    different_dags_with_same_hash = []
    if verbose:
        print(f"Verifying {len(dags)} DAGs with hash value {hashtag}...")
    # global find_equivalence_helper_called
    # global find_equivalence_helper_called_bar
    # global total_circuits_verified
    # find_equivalence_helper_called += 1
    # total_circuits_verified += len(dags)
    # if find_equivalence_helper_called >= find_equivalence_helper_called_bar:
    #     print(f'{find_equivalence_helper_called} find_equivalences_helper() called, '
    #           f'{total_circuits_verified} circuits verified', flush=True)
    #     find_equivalence_helper_called_bar += 100
    params, equation_list_for_params = compute_params(param_info)
    for dag in dags:
        for i, other_dag in enumerate(different_dags_with_same_hash):
            equivalent_called += 1
            if equivalent(
                dag,
                other_dag,
                params,
                equation_list_for_params,
                parameters_for_fingerprint,
                do_not_invoke_smt_solver,
                check_phase_shift_in_smt_solver,
            ):
                current_tag = hashtag + "_" + str(i)
                assert current_tag in output_dict.keys()
                output_dict[current_tag].append(dag)
                total_equivalence_found += 1
                break
        else:
            # dag is not equivalent do any of different_dags_with_same_hash
            different_dags_with_same_hash.append(dag)
            # Insert |dag| eagerly
            current_tag = hashtag + "_" + str(len(different_dags_with_same_hash) - 1)
            output_dict[current_tag] = [dag]
    return hashtag, output_dict, equivalent_called, total_equivalence_found


def compute_params(param_info):
    equation_list_for_params = []
    # compute all parameters from |param_info|
    params = []
    num_symbolic_params = 0

    for i in range(len(param_info)):
        if param_info[i] == "":  # symbolic
            num_symbolic_params += 1
    symbolic_params = create_parameters(num_symbolic_params, equation_list_for_params)

    for i in range(len(param_info)):
        if param_info[i] == "":  # symbolic
            params.append(symbolic_params.pop(0))
        elif isinstance(param_info[i], (int, float)):  # concrete
            params.append(param_info[i])
        else:  # expression
            op = param_info[i][0]
            current_inputs = []
            for input_wire in param_info[i][2]:
                assert input_wire.startswith("P")
                # parameter input
                current_inputs.append(params[int(input_wire[1:])])
            params.append(compute(op, *current_inputs))
    return params, equation_list_for_params


def find_equivalences(
    input_file,
    output_file,
    print_basic_info=True,
    verbose=False,
    keep_classes_with_1_dag=False,
    check_equivalence_with_different_hash=True,
    check_phase_shift_in_smt_solver=False,
    do_not_invoke_smt_solver=False,
):
    input_file_data = load_json(input_file)
    data = input_file_data[1]
    param_info = input_file_data[0][0][1:]
    # parameters generated for random testing
    parameters_for_fingerprint = input_file_data[0][1][1:]
    output_dict = {}
    equivalent_called = 0
    total_equivalence_found = 0
    num_hashtags = 0
    num_dags = 0
    num_potential_equivalences = 0
    import time

    t_start = time.monotonic()
    num_different_dags_with_same_hash = {}
    print(
        f"Considering a total of {sum(len(x) for x in data.values())} circuits split into {len(data)} hash values..."
    )

    params, equation_list_for_params = compute_params(param_info)

    if False:
        # sequential version
        for hashtag, dags in data.items():
            num_hashtags += 1
            num_dags += len(dags)
            num_potential_equivalences += len(dags) - 1
            different_dags_with_same_hash = []
            if verbose:
                print(f"Verifying {len(dags)} DAGs with hash value {hashtag}...")
            for dag in dags:
                equivalence_found = False
                for i in range(len(different_dags_with_same_hash)):
                    other_dag = different_dags_with_same_hash[i]
                    equivalent_called += 1
                    if equivalent(
                        dag,
                        other_dag,
                        params,
                        equation_list_for_params,
                        parameters_for_fingerprint,
                        do_not_invoke_smt_solver,
                        check_phase_shift_in_smt_solver,
                    ):
                        current_tag = hashtag + "_" + str(i)
                        assert current_tag in output_dict.keys()
                        output_dict[current_tag].append(dag)
                        equivalence_found = True
                        total_equivalence_found += 1
                        break
                if not equivalence_found:
                    different_dags_with_same_hash.append(dag)
                    # Insert |dag| eagerly
                    current_tag = (
                        hashtag + "_" + str(len(different_dags_with_same_hash) - 1)
                    )
                    output_dict[current_tag] = [dag]
            num_different_dags_with_same_hash[hashtag] = len(
                different_dags_with_same_hash
            )

    else:
        # parallel version
        num_hashtags = len(data)
        num_dags = sum(len(dags) for dags in data.values())
        num_potential_equivalences = num_dags - num_hashtags
        # first process hashtags with only 1 CircuitSeq
        for hashtag, dags in data.items():
            if len(dags) == 1:
                output_dict[hashtag + "_0"] = [dags[0]]
                num_different_dags_with_same_hash[hashtag] = 1
        print(
            f"Processed {len(output_dict)} hash values that had only 1 circuit sequence, now processing the remaining {len(data) - len(output_dict)} ones with 2 or more circuit sequences..."
        )
        # now process hashtags with >1 DAGs
        with mp.Pool() as pool:
            for (
                hashtag,
                output_dict_,
                equivalent_called_,
                total_equivalence_found_,
            ) in pool.starmap(
                find_equivalences_helper,
                (
                    (
                        hashtag,
                        dags,
                        param_info,
                        parameters_for_fingerprint,
                        check_phase_shift_in_smt_solver,
                        verbose,
                        do_not_invoke_smt_solver,
                    )
                    for hashtag, dags in data.items()
                    if len(dags) > 1
                ),
            ):
                output_dict.update(output_dict_)
                equivalent_called += equivalent_called_
                total_equivalence_found += total_equivalence_found_
                num_different_dags_with_same_hash[hashtag] = len(output_dict_)

    hashtags_in_more_equivalences = set()
    if check_equivalence_with_different_hash:
        more_equivalences = []
        equivalent_called_2 = 0
        num_equivalences_under_phase_shift = 0
        possible_num_equivalences_under_phase_shift = 0
        print("Start checking equivalence with different hash...")
        for hashtag, dags in output_dict.items():
            from collections import defaultdict

            # A map from other hashtags to corresponding phase shifts.
            other_hashtags = defaultdict(dict)
            # |other_hashtags[other_hash][None]| indicates that if it's possible that a CircuitSeq with |other_hash|
            #    is equivalent with a CircuitSeq with |hashtag| without phase shifts.
            # |other_hashtags[other_hash][phase_shift_id]| is a list of DAGs with |hashtag| that can be equivalent
            #    to a CircuitSeq with |other_hash| under phase shift |phase_shift_id|.
            assert len(dags) > 0
            for dag in dags:
                dag_meta = dag[0]
                for item in dag_meta[meta_index_other_hash_values]:
                    if isinstance(item, str):
                        # no phase shift
                        other_hashtags[item][None] = None
                    else:
                        # phase shift id is item[1]
                        assert isinstance(item, list)
                        assert len(item) == 2
                        # We need the exact parameter in |dag|, so we cannot use the representative CircuitSeq |dags[0]|.
                        other_hashtags[item[0]][item[1]] = other_hashtags[item[0]].get(
                            item[1], []
                        ) + [dag]
            assert hashtag.split("_")[0] not in other_hashtags
            if len(other_hashtags) == 0:
                print(
                    f"Warning: other hash values unspecified for hash value {hashtag}."
                    f" Cannot guarantee there are no missing equivalences."
                )
            possible_equivalent_dags = []
            for other_hashtag, phase_shift_ids in other_hashtags.items():
                if other_hashtag not in data.keys():
                    # Not equivalent to any other ones
                    continue
                assert other_hashtag in num_different_dags_with_same_hash.keys()
                i_range = num_different_dags_with_same_hash[other_hashtag]
                for i in range(i_range):
                    other_hashtag_full = other_hashtag + "_" + str(i)
                    possible_equivalent_dags.append(
                        (
                            output_dict[other_hashtag_full][0],
                            other_hashtag_full,
                            phase_shift_ids,
                        )
                    )
            if verbose and len(possible_equivalent_dags) > 0:
                print(
                    f"Verifying {len(possible_equivalent_dags)} possible missing equivalences"
                    f" with hash value {hashtag} and {len(other_hashtags)} other hash values..."
                )
            for item in possible_equivalent_dags:
                other_dag = item[0]
                other_hashtag_full = item[1]
                phase_shift_ids = item[2]
                equivalence_verified = False
                phase_shift_id_when_equivalence_verified = None
                dag_when_equivalence_verified = dags[0]
                if None in phase_shift_ids:
                    equivalent_called_2 += 1
                    if equivalent(
                        dags[0],
                        other_dag,
                        params,
                        equation_list_for_params,
                        parameters_for_fingerprint,
                        do_not_invoke_smt_solver,
                        check_phase_shift_in_smt_solver,
                        None,
                    ):
                        equivalence_verified = True
                if not equivalence_verified:
                    for phase_shift_id, dag_list in phase_shift_ids.items():
                        if phase_shift_id is None:
                            continue
                        # Pruning: we only need to try each input parameter once.
                        input_param_tried = False
                        for dag in dag_list:
                            # Warning: If CircuitSeq::hash() is modified,
                            # the expression |is_fixed_for_all_dags| should be modified correspondingly.
                            is_fixed_for_all_dags = (
                                0 <= phase_shift_id < len(param_info)
                                or len(param_info)
                                <= phase_shift_id
                                < len(param_info) * 2
                                or kCheckPhaseShiftOfPiOver4Index
                                < phase_shift_id
                                < kCheckPhaseShiftOfPiOver4Index + 8
                            )
                            if is_fixed_for_all_dags:
                                if input_param_tried:
                                    continue
                                else:
                                    input_param_tried = True
                            equivalent_called_2 += 1
                            possible_num_equivalences_under_phase_shift += 1
                            # |phase_shift_id[0]| is the CircuitSeq generating this phase shift id.
                            if equivalent(
                                dag,
                                other_dag,
                                params,
                                equation_list_for_params,
                                parameters_for_fingerprint,
                                do_not_invoke_smt_solver,
                                check_phase_shift_in_smt_solver,
                                phase_shift_id,
                            ):
                                equivalence_verified = True
                                num_equivalences_under_phase_shift += 1
                                phase_shift_id_when_equivalence_verified = (
                                    phase_shift_id
                                )
                                dag_when_equivalence_verified = dag
                                break
                        if equivalence_verified:
                            break
                if equivalence_verified:
                    more_equivalences.append((hashtag, other_hashtag_full))
                    hashtags_in_more_equivalences.update(hashtag)
                    hashtags_in_more_equivalences.update(other_hashtag_full)
                    if verbose:
                        print(
                            f"Equivalence with hash value {hashtag} and {other_hashtag_full}"
                            f" with phase shift id = {phase_shift_id_when_equivalence_verified}:\n"
                            f"{dag_when_equivalence_verified}\n"
                            f"{other_dag}"
                        )
        output_dict = [more_equivalences, output_dict]
        if print_basic_info:
            print(
                f"Solver invoked {equivalent_called_2} times to find {len(more_equivalences)} equivalences"
                f" with different hash,"
                f" including {num_equivalences_under_phase_shift} out of"
                f" {possible_num_equivalences_under_phase_shift} possible equivalences under phase shift."
            )
    else:
        # Add a placeholder here
        output_dict = [[], output_dict]

    if not keep_classes_with_1_dag:
        output_dict[1] = {
            k: v
            for k, v in output_dict[1].items()
            if (len(v) >= 2 or k in hashtags_in_more_equivalences)
        }
    t_end = time.monotonic()
    if print_basic_info:
        print(
            f"{total_equivalence_found} equivalences found in {t_end - t_start} seconds"
            f" (solver invoked {equivalent_called} times for {num_dags} DAGs"
            f" with {num_hashtags} different hash values and {num_potential_equivalences} potential equivalences),"
            f" output {len(output_dict[1])} equivalence classes."
        )
    t_start = time.monotonic()
    dump_json(output_dict, output_file)
    t_end = time.monotonic()
    if print_basic_info:
        print(f"Json saved in {t_end - t_start} seconds.")
