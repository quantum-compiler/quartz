import sys
import ast

sys.path.append("..")

from verifier.verifier import *

if __name__ == '__main__':
    # Usage example: python verify_equivalences.py input.json output.json True True
    # The third parameter is verbose or not
    # The fourth parameter is keeping equivalence classes with only 1 DAG or not
    # The fifth parameter is verifying there are no missing equivalence or not
    find_equivalences(sys.argv[1], sys.argv[2],
                      verbose=(False if len(sys.argv) <= 3 else ast.literal_eval(sys.argv[3])),
                      keep_classes_with_1_dag=(True if len(sys.argv) <= 4 else ast.literal_eval(sys.argv[4])),
                      assert_no_missing_equivalence=(False if len(sys.argv) <= 5 else ast.literal_eval(sys.argv[5])))
