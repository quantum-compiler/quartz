import heapq
import os
import sys
import time

import wandb

import quartz


def optimize(
    context: quartz.QuartzContext,
    init_circ: quartz.PyGraph,
    circ_name: str,
    upper_limit: float = 1.05,
    print_message: bool = True,
    max_candidate_len: int = 1_000_000,
    timeout: int = 86400,
) -> tuple[int, quartz.PyGraph]:
    candidate = [(init_circ.gate_count, init_circ.to_qasm_str())]
    hash_set = set([init_circ.hash()])
    best_circ: quartz.PyGraph = init_circ
    best_gate_cnt: int = init_circ.gate_count
    original_cnt = init_circ.gate_count

    start = time.time()
    invoke_cnt: int = 0

    while candidate != []:
        if len(candidate) > max_candidate_len:
            print(
                f"[{circ_name} (rotation merging)] candidate queue shrink from {len(candidate)} to {len(candidate) // 2}"
            )
            candidate = candidate[: len(candidate) // 2]

        _, circ_qasm_str = heapq.heappop(candidate)
        circ = quartz.PyGraph.from_qasm_str(context=context, qasm_str=circ_qasm_str)
        all_nodes = circ.all_nodes()
        for node in all_nodes:
            av_xfers = circ.available_xfers_parallel(
                context=context,
                node=node,
            )
            for i_xfer in av_xfers:
                xfer = context.get_xfer_from_id(id=i_xfer)
                t = time.time()
                if t - start > timeout:
                    return best_gate_cnt, best_circ

                invoke_cnt += 1
                if print_message and invoke_cnt % 100_000_000 == 0:
                    wandb.log(
                        {'invoke_cnt': invoke_cnt, 'best_gate_cnt': best_gate_cnt}
                    )
                    print(
                        f"[{circ_name} (rotation merging)] best gate count: {best_gate_cnt}, candidate count: {len(candidate)}, API invoke time: {invoke_cnt}, time cost: {t - start:.3f}s",
                        flush=True,
                    )

                new_circ = circ.apply_xfer(
                    xfer=xfer, node=node, eliminate_rotation=True
                )

                if new_circ is None:
                    continue

                new_hash = new_circ.hash()
                new_cnt = new_circ.gate_count
                if new_cnt > original_cnt * upper_limit:
                    continue
                if new_hash not in hash_set:
                    hash_set.add(new_hash)
                    heapq.heappush(candidate, (new_cnt, new_circ.to_qasm_str()))

                    if new_cnt < best_gate_cnt:
                        best_gate_cnt = new_cnt
                        best_circ = new_circ
                        wandb.log(
                            {'invoke_cnt': invoke_cnt, 'best_gate_cnt': best_gate_cnt}
                        )
                        print(
                            f"[{circ_name}] better circuit is found! best gate count: {best_gate_cnt}, candidate count: {len(candidate)}, API invoke time: {invoke_cnt}, time cost: {t - start:.3f}s",
                            flush=True,
                        )

    return best_gate_cnt, best_circ


if __name__ == '__main__':
    assert len(sys.argv) > 2

    circ_name = sys.argv[1]
    output_dir = sys.argv[2]
    wandb_mode = 'disabled'
    if len(sys.argv) > 3:
        wandb_mode = sys.argv[3]

    context = quartz.QuartzContext(
        gate_set=['x', 'cx', 'sx', 'rz', 'add'],
        filename='../ecc_set/ibm_3_2_5_ecc.json',
        no_increase=False,
        include_nop=True,
    )
    circ = quartz.PyGraph.from_qasm(
        context=context, filename=f"../circs/ibm_circs/{circ_name}.qasm"
    )

    wandb.init(
        project='quartz_ibm',
        entity='quartz',
        name=f'{circ_name}',
        mode=wandb_mode,  # 'online',
    )

    best_gate_cnt, best_circ = optimize(context, circ, circ_name)

    os.makedirs(output_dir, exist_ok=True)
    best_circ.to_qasm(filename=f'{output_dir}/{circ_name}_optimized.qasm')

    with open(f'{output_dir}/results.txt', 'a') as f:
        f.write(f"{best_gate_cnt}\t\t{circ_name}\n")
