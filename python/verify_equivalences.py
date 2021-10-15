import sys
import ast

sys.path.append("..")

from verifier.verifier import *

if __name__ == '__main__':
    # Usage example: python verify_equivalences.py input.json output.json True
    # The last parameter is verbose or not
    find_equivalences(sys.argv[1], sys.argv[2],
                      verbose=(False if len(sys.argv) <= 3 else ast.literal_eval(sys.argv[3])),
                      keep_classes_with_1_dag=True)
