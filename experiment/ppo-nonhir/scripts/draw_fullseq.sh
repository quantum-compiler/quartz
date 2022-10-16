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

python ppo.py c=nam_ft c.resume=false c.ddp_port=23343 \
    c.mini_batch_size=${BS} 'c.gpus=[0]' \
    'c.input_graphs=[{ name: "'${CIRC}'", path: "../nam_circs/'${CIRC}'.qasm" }]' \
    c.k_epochs=25 c.lr_scheduler=none c.lr_gnn=3e-4 c.lr_actor=3e-4 c.lr_critic=5e-4 \
    c.num_eps_per_iter=64 c.max_eps_len=600 \
    c.output_full_seq=true c.no_increase=${NOINC} \
    2>&1 | tee ftlog/draw_fullseq_nam_${NOINC}_${CIRC}.log


sleep 10
