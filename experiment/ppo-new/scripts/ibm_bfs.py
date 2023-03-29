import heapq
import os
import sys
import time

import wandb

import quartz


def ibm_add_xfer(context: quartz.QuartzContext):
    equivalent_circ_pairs = [
        (
            'OpenQASM 2.0; include "qelib1.inc"; qreg q[1]; rz(pi) q[0];',
            'OpenQASM 2.0; include "qelib1.inc"; qreg q[1]; sx q[0]; rz(pi) q[0]; sx q[0];',
        ),
        (
            'OpenQASM 2.0; include "qelib1.inc"; qreg q[1]; sx q[0]; rz(pi/2) q[0]; sx q[0];',
            'OpenQASM 2.0; include "qelib1.inc"; qreg q[1]; rz(pi/2) q[0]; sx q[0]; rz(pi/2) q[0];',
        ),
        (
            'OpenQASM 2.0; include "qelib1.inc"; qreg q[1]; sx q[0]; rz(3*pi/2) q[0]; sx q[0];',
            'OpenQASM 2.0; include "qelib1.inc"; qreg q[1]; rz(3*pi/2) q[0]; sx q[0]; rz(3*pi/2) q[0];',
        ),
    ]

    for circ_pair in equivalent_circ_pairs:
        print(circ_pair)
        context.add_xfer_from_qasm_str(src_str=circ_pair[0], dst_str=circ_pair[1])
        context.add_xfer_from_qasm_str(src_str=circ_pair[1], dst_str=circ_pair[0])

    return context


def optimize(
    context: quartz.QuartzContext,
    init_circ: quartz.PyGraph,
    circ_name: str,
    upper_limit: float = 1.05,
    print_message: bool = True,
    max_candidate_len: int = 1_000_000,
    timeout: int = 86400,
    output_dir: str = 'bfs_outputs',
) -> tuple[int, quartz.PyGraph]:
    candidate = [(init_circ.gate_count, init_circ)]
    hash_set = set([init_circ.hash()])
    best_circ: quartz.PyGraph = init_circ
    best_gate_cnt: int = init_circ.gate_count
    original_cnt = init_circ.gate_count

    t_start = time.time()
    invoke_cnt: int = 0
    popped_cnt: int = 0

    def print_and_log():
        wandb.log(
            {
                'invoke_cnt': invoke_cnt,
                'best_gate_cnt': best_gate_cnt,
                'candidate_len': len(candidate),
            }
        )
        print(
            f"[{circ_name}] best gate count: {best_gate_cnt}, {len(candidate) = }, {invoke_cnt = }, {popped_cnt = }, time cost: {time.time() - t_start:.2f} s",
            flush=True,
        )

    while candidate:
        old_len = len(candidate)
        if old_len > max_candidate_len:
            t_shr_s = time.time()
            new_candidate = []
            new_hash_set = set()
            while len(new_candidate) <= max_candidate_len // 2:
                gc, circ = heapq.heappop(candidate)
                heapq.heappush(new_candidate, (gc, circ))
                new_hash_set.add(circ.hash())
            candidate = new_candidate
            hash_set = new_hash_set
            print(
                f'queue shrinks: {old_len} -> {len(candidate)}, taking {time.time() - t_shr_s:.0f} sec, {invoke_cnt = }',
                flush=True,
            )

        _, circ = heapq.heappop(candidate)
        popped_cnt += 1
        all_nodes = circ.all_nodes()
        for node in all_nodes:
            av_xfers = circ.available_xfers_parallel(
                context=context,
                node=node,
            )
            for i_xfer in av_xfers:
                xfer = context.get_xfer_from_id(id=i_xfer)
                if time.time() - t_start > timeout:
                    print(f'[{circ_name}] timeout, exit', flush=True)
                    return best_gate_cnt, best_circ

                invoke_cnt += 1
                if print_message and invoke_cnt % 1_0_000 == 0:
                    print_and_log()

                new_circ = circ.apply_xfer(
                    xfer=xfer, node=node, eliminate_rotation=True
                )

                if new_circ is None:
                    continue

                new_hash = new_circ.hash()
                new_cnt = new_circ.gate_count
                # if new_cnt > original_cnt * upper_limit:
                if new_cnt > best_gate_cnt * upper_limit:
                    continue
                if new_hash not in hash_set:
                    hash_set.add(new_hash)
                    heapq.heappush(candidate, (new_cnt, new_circ))

                    if new_cnt < best_gate_cnt:
                        best_gate_cnt = new_cnt
                        best_circ = new_circ
                        print(f"[{circ_name}] better circuit is found!", flush=True)
                        print_and_log()
                        # save circuits
                        os.makedirs(f'{output_dir}/{circ_name}', exist_ok=True)
                        circ_file = f'{output_dir}/{circ_name}/{best_gate_cnt}_{int(time.time() - t_start)}.qasm'
                        with open(circ_file, 'w') as f:
                            f.write(best_circ.to_qasm_str())
                        print(f'[{circ_name}] wrote {circ_file} .', flush=True)

    return best_gate_cnt, best_circ


if __name__ == '__main__':
    assert len(sys.argv) > 2

    circ_name = sys.argv[1]
    output_dir = sys.argv[2]
    wandb_mode = 'disabled'
    if len(sys.argv) > 3:
        wandb_mode = sys.argv[3]

    context = quartz.QuartzContext(
        gate_set=[
            'x',
            'cx',
            'sx',
            'rz',
            'add',
            'neg',
        ],
        filename='../ecc_set/ibm_325_ecc.json',
        no_increase=False,
        include_nop=True,
    )
    context = ibm_add_xfer(context)
    circ = quartz.PyGraph.from_qasm(
        context=context, filename=f"../circs/ibm_circs/{circ_name}.qasm"
    )

    wandb.init(
        project='quartz_ibm',
        entity='quartz',
        name=f'{circ_name}',
        mode=wandb_mode,  # 'online',
    )

    best_gate_cnt, best_circ = optimize(
        context,
        circ,
        circ_name,
        max_candidate_len=2000,
        timeout=6 * 3600,
        upper_limit=1.0001,
        output_dir=output_dir,
    )

    os.makedirs(output_dir, exist_ok=True)
    best_circ.to_qasm(filename=f'{output_dir}/{circ_name}_optimized.qasm')

    with open(f'{output_dir}/results.txt', 'a') as f:
        f.write(f"{best_gate_cnt}\t\t{circ_name}\t\t{circ.gate_count}\n")
