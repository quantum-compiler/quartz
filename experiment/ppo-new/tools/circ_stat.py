import os
from typing import List

import quartz  # type: ignore

quartz_context = None


def init_quartz_context(
    gate_set: List[str] = ['h', 'cx', 'x', 'rz', 'add'],
    ecc_file_path: str = '../../ecc_set/nam_ecc.json',
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
    gate_set=['h', 'cx', 'rz', 'x', 'add', 't', 'tdg', 'sdg', 's', 'ccx', 'z'],
    ecc_file_path='../../ecc_set/nam_ecc.json',
    no_increase=False,
    include_nop=True,
)

circs = """
tof_3
barenco_tof_3
mod5_4
tof_4
tof_5
barenco_tof_4
mod_mult_55
vbe_adder_3
barenco_tof_5
csla_mux_3
rc_adder_6
gf2^4_mult
tof_10
mod_red_21
gf2^5_mult
csum_mux_9
qcla_com_7
barenco_tof_10
gf2^6_mult
qcla_adder_10
gf2^7_mult
gf2^8_mult
qcla_mod_7
adder_8
gf2^9_mult
gf2^10_mult
gf2^16_mult
"""
circs = circs.split()

# init_quartz_context(gate_set=['h', 'cx', 't', 'tdg', 'x'], ecc_file_path='../../ecc_set/t_tdg_ecc.json')
print(
    f'circ_name\tgraph_before.gate_count\tgraph_before.cx_count\tgraph_before.depth\tgraph_after_heavy.gate_count\tgraph_after_heavy.cx_count\tgraph_after_heavy.depth'
)
nam_circ_dir = '../../../circuit/nam-circuits/qasm_files'
for circ_name in circs:
    if circ_name not in ['']:
        if circ_name.startswith('gf2^'):
            circ_name = 'gf2^E' + circ_name[len('gf2^') :]
        before_file = os.path.join(nam_circ_dir, f'{circ_name}_before.qasm')
        after_heavy_file = os.path.join(nam_circ_dir, f'{circ_name}_after_heavy.qasm')
        assert os.path.isfile(before_file), f'No {before_file}'
        assert os.path.isfile(after_heavy_file), f'No {after_heavy_file}'
        # print(before_file, flush=True)
        graph_before = file_to_graph(before_file)
        # print(after_heavy_file, flush=True)
        graph_after_heavy = file_to_graph(after_heavy_file)
        print(
            f'{circ_name}\t{graph_before.gate_count}\t{graph_before.cx_count}\t{graph_before.depth}\t{graph_after_heavy.gate_count}\t{graph_after_heavy.cx_count}\t{graph_after_heavy.depth}'
        )
