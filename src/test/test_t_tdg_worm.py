import heapq
import os
import sys
import time
from fileinput import filename

import quartz


def optimize(
    context: quartz.QuartzContext,
    init_circ: quartz.PyGraph,
    circ_name: str,
    upper_limit: float = 1.05,
    print_message: bool = True,
    timeout: int = 1,
) -> tuple[int, quartz.PyGraph]:
    candidate = [(init_circ.t_count, init_circ.to_qasm_str())]
    hash_set = set([init_circ.hash()])
    best_circ: quartz.PyGraph = init_circ
    best_t_cnt: int = init_circ.t_count
    original_cnt = init_circ.t_count

    start = time.time()
    invoke_cnt: int = 0

    while candidate != []:
        _, circ_qasm_str = heapq.heappop(candidate)
        circ = quartz.PyGraph.from_qasm_str(context=context, qasm_str=circ_qasm_str)
        all_nodes = circ.all_nodes()
        for xfer in context.get_xfers():
            for node in all_nodes:
                t = time.time()
                if t - start > timeout:
                    return best_t_cnt, best_circ

                invoke_cnt += 1
                if print_message and invoke_cnt % 100_000_000 == 0:
                    print(
                        f"[{circ_name} (w/o rotation merging)] best t count: {best_t_cnt}, candidate count: {len(candidate)}, API invoke time: {invoke_cnt}, time cost: {t - start:.3f}s"
                    )

                new_circ = circ.apply_xfer(
                    xfer=xfer, node=node, eliminate_rotation=True
                )

                if new_circ == None:
                    continue

                new_hash = new_circ.hash()
                new_cnt = new_circ.t_count
                if new_cnt > upper_limit * original_cnt:
                    continue

                if new_hash not in hash_set:
                    hash_set.add(new_hash)
                    heapq.heappush(candidate, (new_cnt, new_circ.to_qasm_str()))

                    if new_cnt < best_t_cnt:
                        best_t_cnt = new_cnt
                        best_circ = new_circ

    return best_t_cnt, best_circ


if __name__ == '__main__':
    assert len(sys.argv) > 2

    circ_name = sys.argv[1]
    output_dir = sys.argv[2]

    context = quartz.QuartzContext(
        gate_set=['h', 'cx', 'x', 't', 'tdg'],
        filename='../../T_TDG_complete_ECC_set.json',
        no_increase=False,
        include_nop=True,
    )
    circ = quartz.PyGraph.from_qasm(
        context=context, filename=f"../../circuit/t_tdg_circs/{circ_name}.qasm"
    )

    best_t_cnt, best_circ = optimize(context, circ, circ_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    best_circ.to_qasm(filename=f'{output_dir}/{circ_name}_optimized.qasm')

    with open(f'{output_dir}/results.txt', 'a') as f:
        f.write(f"{best_t_cnt}/t/t{circ_name}\n")
