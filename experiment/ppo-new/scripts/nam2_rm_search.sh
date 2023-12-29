#!/bin/bash

set -ex

date
pwd
# ls -lh

mkdir -p ftlog

export OMP_NUM_THREADS=8

# CKPT=ckpts/nam2_rm_iter_xxx.pt
# BS=2400
MEM=${MEM:-30}
GPU=${GPU:-0}

python ppo.py c=nam2_ft c.resume=true c.ddp_port=$(bash gen_port.sh) c.ckpt_path=${CKPT} c.mini_batch_size=${BS} 'c.gpus=['${GPU}']' 'c.input_graphs=[{ name: "'${CIRC}'", path: "../circs/nam_rm_circs/'${CIRC}'.qasm"}]' c.k_epochs=5 c.lr_scheduler=linear c.num_eps_per_iter=64 c.max_eps_len=600 c.vmem_perct_limit=${MEM} c.best_graph_output_dir=nam2_rm_best_graphs > ftlog/tuning_nam2_rm_${CIRC}.log 2>&1 & \
sleep 5m && \
python ppo.py c=nam2_test c.resume=true c.ddp_port=$(bash gen_port.sh) c.ckpt_path=${CKPT} 'c.gpus=['${GPU}']' 'c.input_graphs=[{ name: "'${CIRC}'", path: "../circs/nam_rm_circs/'${CIRC}'.qasm"}]' c.num_eps_per_iter=64 c.vmem_perct_limit=${MEM} 2>&1 | tee ftlog/search_nam2_rm_${CIRC}.log

sleep 10
