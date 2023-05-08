import os
import sys
from typing import List, Tuple

from natsort import natsorted
from tqdm import tqdm

import quartz

quartz_context = None


def init_quartz_context(
    gate_set: List[str] = [
        'cx',
        'x',
        'sx',
        'rz',
        'add',
    ],
    ecc_file_path: str = '../../ecc_set/ibm_325_ecc.json',
    no_increase: bool = False,
    include_nop: bool = True,
):
    global quartz_context
    quartz_context = quartz.QuartzContext(
        gate_set=gate_set,
        filename=ecc_file_path,
        no_increase=no_increase,
        include_nop=include_nop,
    )
    return quartz_context


def qasm_to_graph(qasm_str: str) -> quartz.PyGraph:
    graph = quartz.PyGraph.from_qasm_str(context=quartz_context, qasm_str=qasm_str)
    return graph


def file_to_graph(filename: str) -> quartz.PyGraph:
    with open(filename, 'r') as f:
        qasm_str = f.read()
    return qasm_to_graph(qasm_str)


def is_nop(xfer_id: int) -> bool:
    return quartz_context.get_xfer_from_id(id=xfer_id).is_nop


init_quartz_context(
    gate_set=[
        'h',
        'cx',
        'x',
        'sx',
        'rz',
        'add',
        'neg',
    ],
    ecc_file_path='../../ecc_set/ibm_3_2_5_ecc.json',
    no_increase=False,
    include_nop=True,
)

if __name__ == '__main__':
    dir_path = sys.argv[1]
    sort = False
    if len(sys.argv) > 2:
        sort = True

    name_gc_list: List[Tuple[str, int]] = []

    for circ_file in tqdm(natsorted(os.listdir(dir_path))):
        circ_path = os.path.join(dir_path, circ_file)
        circ = file_to_graph(circ_path)
        name_gc_list.append((circ_file[:-5], circ.gate_count))

    if sort:
        name_gc_list = sorted(name_gc_list, key=lambda x: x[1])

    for name, gc in name_gc_list:
        print(f'{name} {gc}')
