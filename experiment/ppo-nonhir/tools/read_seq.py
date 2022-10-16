import os
from asyncio.base_subprocess import ReadSubprocessPipeProto
from typing import Dict, List, cast

import hydra
import qiskit
from IPython import embed
from matplotlib.pyplot import text
from natsort import natsorted
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.tools.visualization import circuit_drawer

import quartz

# import qtz
# from config.config import *


def read_seq(dir_path: str) -> None:
    circ_files: List[str] = natsorted(os.listdir(dir_path))
    graph_list: List[quartz.PyGraph] = []
    for circ_file in circ_files:
        # 10_54_47_8323.qasm
        splitted_fname = os.path.splitext(circ_file)[0].split("_")
        assert len(splitted_fname) == 4
        i_step, cost, action_node, action_xfer = list(map(int, splitted_fname))

        with open(os.path.join(dir_path, circ_file)) as f:
            qasm = f.read()
        # graph = qtz.qasm_to_graph(qasm)

        # if len(graph_list) > 0:
        #     pre_graph = graph_list[-1]
        #     this_graph, next_nodes = pre_graph.apply_xfer_with_local_state_tracking(
        #         node=pre_graph.get_node_from_id(id=action_node),
        #         xfer=qtz.quartz_context.get_xfer_from_id(id=action_xfer),
        #         eliminate_rotation=qtz.has_parameterized_gate,
        #     )
        #     assert hash(this_graph) == hash(graph)

        # graph_list.append(graph)
        # print(f"Got i_step = {i_step}, cost = {cost}, graph = {graph}")

        circ = QuantumCircuit.from_qasm_str(qasm)
        circ.draw(output="mpl", filename=f"output/{circ_file[:-5]}.png")
        # print(circ.draw(output='text'))
        # circuit_drawer(circ, output='text')
        # embed()


# @hydra.main(config_path="config", config_name="config")
# def main(config: Config) -> None:
#     output_dir = os.path.abspath(os.curdir)  # get hydra output dir
#     os.chdir(hydra.utils.get_original_cwd())  # set working dir to the original one

#     cfg: BaseConfig = config.c

#     qtz.init_quartz_context(
#         cfg.gate_set,
#         cfg.ecc_file,
#         cfg.no_increase,
#         cfg.include_nop,
#     )

#     read_seq(cfg.full_seq_path)


if __name__ == "__main__":
    # main()
    read_seq(dir_path="../../../../full_seq/barenco_tof_3_fullseq_36")
