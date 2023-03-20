from typing import Any, Callable, Dict, List, Set, Tuple

import quartz  # type: ignore

"""global vars"""
quartz_context: quartz.QuartzContext
quartz_parser: quartz.PyQASMParser

has_parameterized_gate: bool


def init_quartz_context(
    gate_set: List[str],
    ecc_file_path: str,
    no_increase: bool,
    include_nop: bool,
) -> None:
    global quartz_context
    global quartz_parser
    global has_parameterized_gate
    quartz_context = quartz.QuartzContext(
        gate_set=gate_set,
        filename=ecc_file_path,
        no_increase=no_increase,
        include_nop=include_nop,
    )
    quartz_parser = quartz.PyQASMParser(context=quartz_context)
    has_parameterized_gate = quartz_context.has_parameterized_gate()


def qasm_to_graph_th_dag(qasm_str: str) -> quartz.PyGraph:
    global quartz_context
    global quartz_parser
    dag = quartz_parser.load_qasm_str(qasm_str)
    graph = quartz.PyGraph(context=quartz_context, dag=dag)
    return graph


def qasm_to_graph(qasm_str: str) -> quartz.PyGraph:
    global quartz_context
    graph = quartz.PyGraph.from_qasm_str(context=quartz_context, qasm_str=qasm_str)
    return graph


def is_nop(xfer_id: int) -> bool:
    global quartz_context
    return quartz_context.get_xfer_from_id(id=xfer_id).is_nop
