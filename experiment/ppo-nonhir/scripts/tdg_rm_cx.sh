#!/bin/bash

set -ex

date
pwd
ls -lh

mkdir -p ftlog

export OMP_NUM_THREADS=8

# CKPT=tdg_rm_cx_iter_100.pt
# BS=3200
# CIRC=barenco_tof_3

python ppo.py c=tdg_ft c.resume=true c.ddp_port=23343 \
    c.ckpt_path=ckpts/${CKPT} c.mini_batch_size=${BS} 'c.gpus=[0,1,2,3]' \
    'c.input_graphs=[{ name: "'${CIRC}'", path: "../t_tdg_rm_circs/'${CIRC}'.qasm" }]' \
    c.cost_type=cx_count c.time_budget='24:00:00' \
    c.k_epochs=20 c.lr_scheduler=linear c.num_eps_per_iter=64 c.max_eps_len=600 \
    2>&1 | tee ftlog/tdg_rm_cx_${CIRC}.log


sleep 10
