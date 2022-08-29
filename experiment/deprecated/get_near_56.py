import quartz

quartz_context = quartz.QuartzContext(
    gate_set=['h', 'cx', 't', 'tdg'], filename='../bfs_verified_simplified.json'
)
parser = quartz.PyQASMParser(context=quartz_context)
my_dag = parser.load_qasm(filename="barenco_tof_3_opt_path/subst_history_39.qasm")
my_graph = quartz.PyGraph(context=quartz_context, dag=my_dag)
trace_56 = [
    (28, 3223),
    (28, 3223),
    (8, 3430),
    (33, 440),
    (13, 2281),
    (37, 3155),
    (35, 3155),
    (34, 992),
]
graph = my_graph
for node_id, xfer_id in trace_56[:-1]:
    graph = graph.apply_xfer(
        xfer=quartz_context.get_xfer_from_id(id=xfer_id),
        node=graph.get_node_from_id(id=node_id),
    )
    print(graph.gate_count)
graph.to_qasm(filename='near_56.qasm')
