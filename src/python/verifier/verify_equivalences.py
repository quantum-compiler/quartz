import ast
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.utils import use_sympy

if use_sympy:
    from verifier_sympy import *
else:
    from verifier import *


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
    # Usage example: python verify_equivalences.py input.json output.json True True True True False 30000
    # The 3rd parameter is printing basic information or not
    # The 4th parameter is verbose or not
    # The 5th parameter is keeping equivalence classes with only 1 DAG or not
    #   (note that the default value is different with find_equivalences for the fifth parameter)
    # The 6th parameter is checking equivalences with different hash values or not
    # The 7th parameter is checking equivalences with a phase shift in the SMT solver or not
    #   (note that even if this parameter is False, Quartz will check phase shift;
    #    it is recommended to set this to False to reduce the burden for Z3)
    # The 8th parameter is timeout for each Z3 invocation in milliseconds (default is 30s)
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
        timeout=(30000 if len(sys.argv) <= 8 else ast.literal_eval(sys.argv[8])),
    )
