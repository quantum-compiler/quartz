import json
import sys

sys.path.append("../..")

from utils.draw_circuit import *

if __name__ == '__main__':
    dag_str = input()
    dag = json.loads(dag_str)
    draw_circuit(dag)
