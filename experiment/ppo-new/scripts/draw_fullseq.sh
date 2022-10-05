#!/bin/bash

set -ex

date
pwd
ls -lh

mkdir -p ftlog

export OMP_NUM_THREADS=8

CKPT=nam_iter_100.pt
# BS=3200
# CIRC=barenco_tof_3
# NOINC=false

python ppo.py c=nam_ft c.resume=true c.ddp_port=23343 \
    c.ckpt_path=ckpts/${CKPT} c.mini_batch_size=${BS} 'c.gpus=[0]' \
    'c.input_graphs=[{ name: "'${CIRC}'", path: "../nam_circs/'${CIRC}'.qasm" }]' \
    c.k_epochs=20 c.lr_scheduler=linear c.num_eps_per_iter=64 c.max_eps_len=600 \
    c.output_full_seq=true c.no_increase=${NOINC} \
    2>&1 | tee ftlog/draw_fullseq_nam_${CIRC}.log


sleep 10
