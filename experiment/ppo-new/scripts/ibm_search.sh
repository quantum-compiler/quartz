#!/bin/bash

set -ex

date
pwd
# ls -lh

mkdir -p ftlog

export OMP_NUM_THREADS=8

# CKPT=ckpts/ibm_iter_xxx.pt
# BS=2400
# CIRC=gf2^8_mult
# MEM=30

python ppo.py c=ibm_ft c.resume=true c.ddp_port=23343 c.ckpt_path=${CKPT} c.mini_batch_size=${BS} 'c.gpus=[0]' 'c.input_graphs=[{ name: "'${CIRC}'", path: "../circs/ibm_circs/'${CIRC}'.qasm"}]' c.k_epochs=5 c.lr_scheduler=linear c.num_eps_per_iter=64 c.max_eps_len=600 c.vmem_perct_limit=${MEM} c.best_graph_output_dir=ibm_best_graphs > ftlog/tuning_ibm_${CIRC}.log 2>&1 & \
sleep 5m && \
python ppo.py c=ibm_test c.resume=true c.ddp_port=23346 c.ckpt_path=${CKPT} 'c.gpus=[0]' 'c.input_graphs=[{ name: "'${CIRC}'", path: "../circs/ibm_circs/'${CIRC}'.qasm"}]' c.num_eps_per_iter=64 c.vmem_perct_limit=${MEM} 2>&1 | tee ftlog/search_ibm_${CIRC}.log

sleep 10
