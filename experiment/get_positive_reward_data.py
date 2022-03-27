import quartz

quartz_context = quartz.QuartzContext(
    gate_set=['h', 'cx', 't', 'tdg'],
    filename='../bfs_verified_simplified.json')
parser = quartz.PyQASMParser(context=quartz_context)
my_dag = parser.load_qasm(
    filename="barenco_tof_3_opt_path/subst_history_39.qasm")
my_graph = quartz.PyGraph(context=quartz_context, dag=my_dag)