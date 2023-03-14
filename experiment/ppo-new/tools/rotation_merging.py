import os
import sys
from typing import List, cast

from natsort import natsorted

import quartz

quartz_context = None


def init_quartz_context(
    gate_set: List[str] = ['h', 'cx', 'x', 'rz', 'add'],
    ecc_file_path: str = '../ecc_set/nam_ecc.json',
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
    gate_set=['h', 'cx', 'x', 'rz', 'add'],
    ecc_file_path='../ecc_set/nam_ecc.json',
    no_increase=False,
    include_nop=True,
)

if __name__ == '__main__':
    output_dir: str | None = None

    dir_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for filename in natsorted(os.listdir(dir_path)):
        filename = cast(str, filename)
        if filename.endswith('.qasm'):
            full_path = os.path.join(dir_path, filename)
            graph = file_to_graph(full_path)
            old_gate_count = graph.gate_count
            graph.rotation_merging('rz')

            print(
                f'{graph.gate_count}  {filename}: {old_gate_count} -> {graph.gate_count}'
            )

            if output_dir:
                rm_filename = os.path.join(output_dir, f'{filename}')
                with open(rm_filename, 'w') as f:
                    f.write(graph.to_qasm_str())
