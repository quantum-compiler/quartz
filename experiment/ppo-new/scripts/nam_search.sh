#!/bin/bash

set -ex

date
pwd
# ls -lh

mkdir -p ftlog

export OMP_NUM_THREADS=4

CKPT=ckpts/nam_rm_iter_135_002.pt
BS=1200
CIRC=gf2^8_mult
MEM=30

python ppo.py c=nam_ft c.resume=true c.ddp_port=23343 c.ckpt_path=${CKPT} c.mini_batch_size=${BS} 'c.gpus=[0]' 'c.input_graphs=[{ name: "'${CIRC}'", path: "../circs/nam_circs/'${CIRC}'.qasm"}]' c.k_epochs=15 c.lr_scheduler=linear c.num_eps_per_iter=32 c.max_eps_len=600 c.vmem_perct_limit=${MEM} > ftlog/tuning_${CIRC}_0214_0.log 2>&1 & \
sleep 5m && python ppo.py c=nam_test c.resume=true c.ddp_port=23346 c.ckpt_path=${CKPT} 'c.gpus=[0]' 'c.input_graphs=[{ name: "'${CIRC}'", path: "../circs/nam_circs/'${CIRC}'.qasm"}]' c.num_eps_per_iter=64 c.vmem_perct_limit=${MEM} 2>&1 | tee ftlog/search_${CIRC}_0214_0.log

sleep 10
