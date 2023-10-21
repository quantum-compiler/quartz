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

python ppo.py c=ibm_ft c.resume=false c.ddp_port=23343 c.gnn_num_layers=${LAYER} c.mini_batch_size=${BS} 'c.gpus=[0]' 'c.input_graphs=[{ name: "'${CIRC}'", path: "../circs/ibm_circs/'${CIRC}'.qasm"}]' c.k_epochs=5 c.lr_scheduler=linear c.num_eps_per_iter=64 c.max_eps_len=600 c.vmem_perct_limit=${MEM} c.wandb_run_name_suffix=_${LAYER}l c.best_graph_output_dir=ibm_best_graphs_${LAYER}l 2>&1 | tee ftlog/tuning_ibm_${CIRC}_${LAYER}l.log

sleep 10
