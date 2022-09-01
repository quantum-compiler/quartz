import os
from os.path import isfile, join

qasm_path = os.getcwd() + '/../t_tdg_h_cx_toffoli_flip_dataset/'
qasm_fns = [
    fn
    for fn in os.listdir(qasm_path)
    if isfile(join(qasm_path, fn)) and fn[-23:] == 'after_toffoli_flip.qasm'
]

with open('run_tests.sh', 'w') as f:
    for fn in qasm_fns:
        f.write(f'python test_ppo.py ../t_tdg_h_cx_toffoli_flip_dataset/{fn}\n')
