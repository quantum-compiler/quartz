import ast
import sys

from verifier import *

sys.path.append("..")


if __name__ == "__main__":
    # Usage example: python verify_equivalences.py input.json output.json -n
    # Do not invoke verify solver and assume we can verify it
    if len(sys.argv) == 4 and str(sys.argv[3]) == "-n":
        find_equivalences(
            sys.argv[1],
            sys.argv[2],
            keep_classes_with_1_dag=True,
            do_not_invoke_smt_solver=True,
        )
        exit(0)
    # Usage example: python verify_equivalences.py input.json output.json True True True True
    # The third parameter is printing basic information or not
    # The fourth parameter is verbose or not
    # The fifth parameter is keeping equivalence classes with only 1 DAG or not
    #   (note that the default value is different with find_equivalences for the fifth parameter)
    # The sixth parameter is checking equivalences with different hash values or not
    # The seventh parameter is checking equivalences with a phase shift in the SMT solver or not
    find_equivalences(
        sys.argv[1],
        sys.argv[2],
        print_basic_info=(
            True if len(sys.argv) <= 3 else ast.literal_eval(sys.argv[3])
        ),
        verbose=(False if len(sys.argv) <= 4 else ast.literal_eval(sys.argv[4])),
        keep_classes_with_1_dag=(
            True if len(sys.argv) <= 5 else ast.literal_eval(sys.argv[5])
        ),
        check_equivalence_with_different_hash=(
            True if len(sys.argv) <= 6 else ast.literal_eval(sys.argv[6])
        ),
        check_phase_shift_in_smt_solver=(
            False if len(sys.argv) <= 7 else ast.literal_eval(sys.argv[7])
        ),
    )
