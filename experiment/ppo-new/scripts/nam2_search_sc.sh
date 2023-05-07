#!/bin/bash

set -ex

date
pwd
# ls -lh

mkdir -p ftlog

export OMP_NUM_THREADS=8

# CKPT=ckpts/nam2_rm_iter_xxx.pt
# BS=2400
# MEM=30

# CIRC_NAME=adder_8_0
# CIRC_PATH=scalability_study/adder_8/0

python ppo.py c=nam2_ft c.resume=true c.ddp_port=23343 c.ckpt_path=${CKPT} c.mini_batch_size=${BS} 'c.gpus=[0]' 'c.input_graphs=[{ name: "'${CIRC_NAME}'", path: "../circs/'${CIRC_PATH}'.qasm"}]' c.k_epochs=5 c.lr_scheduler=linear c.num_eps_per_iter=64 c.max_eps_len=600 c.vmem_perct_limit=${MEM} c.best_graph_output_dir=nam2_best_graphs_sc > ftlog/tuning_nam2_sc_${CIRC_NAME}.log 2>&1 & \
sleep 5m && \
python ppo.py c=nam2_test c.resume=true c.ddp_port=23346 c.ckpt_path=${CKPT} 'c.gpus=[0]' 'c.input_graphs=[{ name: "'${CIRC_NAME}'", path: "../circs/'${CIRC_PATH}'.qasm"}]' c.num_eps_per_iter=64 c.vmem_perct_limit=${MEM} 2>&1 | tee ftlog/search_nam2_sc_${CIRC_NAME}.log

sleep 10
