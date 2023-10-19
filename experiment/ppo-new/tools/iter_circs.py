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
    name_cx_list: List[Tuple[str, int]] = []
    name_depth_list: List[Tuple[str, int]] = []

    for circ_file in tqdm(natsorted(os.listdir(dir_path))):
        if not circ_file.endswith('.qasm'):
            continue
        circ_path = os.path.join(dir_path, circ_file)
        circ = file_to_graph(circ_path)
        name_gc_list.append((circ_file[:-5], circ.gate_count))
        name_cx_list.append((circ_file[:-5], circ.cx_count))
        name_depth_list.append((circ_file[:-5], circ.depth))

    if sort:
        # name_gc_list = sorted(name_gc_list, key=lambda x: x[1])
        origs = {'tof_3': 45, 'barenco_tof_3': 58, 'mod5_4': 63, 'tof_4': 75, 'tof_5': 105, 'barenco_tof_4': 114, 'mod_mult_55': 119, 'vbe_adder_3': 150, 'barenco_tof_5': 170, 'csla_mux_3': 170, 'rc_adder_6': 200, 'gf2^4_mult': 225, 'tof_10': 255, 'hwb6': 259, 'mod_red_21': 278, 'gf2^5_mult': 347, 'csum_mux_9': 420, 'qcla_com_7': 443, 'ham15-low': 443+1, 'barenco_tof_10': 450, 'gf2^6_mult': 495, 'qcla_adder_10': 521, 'gf2^7_mult': 669, 'grover_5': 831, 'gf2^8_mult': 883, 'qcla_mod_7': 884, 'adder_8': 900}
        # origs = {'tof_3': 1, 'barenco_tof_3': 2, 'mod5_4': 3, 'tof_4': 4, 'tof_5': 5, 'barenco_tof_4': 6, 'mod_mult_55': 7, 'vbe_adder_3': 8, 'barenco_tof_5': 9, 'csla_mux_3': 10, 'rc_adder_6': 11, 'gf2^4_mult': 12, 'hwb6': 13, 'mod_red_21': 14, 'tof_10': 15, 'gf2^5_mult': 16, 'csum_mux_9': 17, 'barenco_tof_10': 18, 'ham15-low': 19, 'qcla_com_7': 20, 'gf2^6_mult': 21, 'qcla_adder_10': 22, 'gf2^7_mult': 23, 'gf2^8_mult': 24, 'qcla_mod_7': 25, 'adder_8': 26, 'vqe_nativegates_ibm_tket_8': 27, 'qgan_nativegates_ibm_tket_8': 28, 'qaoa_nativegates_ibm_tket_8': 29, 'ae_nativegates_ibm_tket_8': 30, 'qpeexact_nativegates_ibm_tket_8': 31, 'qpeinexact_nativegates_ibm_tket_8': 32, 'qft_nativegates_ibm_tket_8': 33, 'qftentangled_nativegates_ibm_tket_8': 34, 'portfoliovqe_nativegates_ibm_tket_8': 35, 'portfolioqaoa_nativegates_ibm_tket_8': 36}
        name_gc_list = sorted(name_gc_list, key=lambda x: origs[x[0]])
        name_cx_list = sorted(name_cx_list, key=lambda x: origs[x[0]])
        name_depth_list = sorted(name_depth_list, key=lambda x: origs[x[0]])

    for name, gc in name_gc_list:
        print(f'{name} {gc}')

    for name, cx in name_cx_list:
        # print(f"{name} {cx}")
        print(f"{cx}")

    print("")
    for name, depth in name_depth_list:
        print(f'{depth}')

