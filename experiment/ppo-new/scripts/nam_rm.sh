#!/bin/bash

set -ex

date
pwd
ls -lh

mkdir -p ftlog

export OMP_NUM_THREADS=8

# CKPT=nam_rm_iter_100.pt
# BS=3200
# CIRC=barenco_tof_3

python ppo.py c=nam_ft c.resume=true c.ddp_port=23343 \
    c.ckpt_path=ckpts/${CKPT} c.mini_batch_size=${BS} 'c.gpus=[0,1,2,3]' \
    'c.input_graphs=[{ name: "'${CIRC}'", path: "../nam_rm_circs/'${CIRC}'.qasm" }]' \
    c.time_budget='24:00:00' \
    c.k_epochs=20 c.lr_scheduler=linear c.num_eps_per_iter=64 c.max_eps_len=600 \
    2>&1 | tee ftlog/nam_rm_${CIRC}.log


sleep 10
