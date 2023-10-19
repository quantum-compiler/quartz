#!/bin/bash
set -ex

TMP=$(CKPT=ckpts/ibm_iter_921_6l.pt BS=4800 CIRC=qgan_nativegates_ibm_tket_8 MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:17010804+1 sb.sh scripts/ibm_search_6l_wos.sh) && TMP=$(echo $TMP | awk '{print $NF}')
TMP=$(CKPT=ckpts/ibm_iter_921_6l.pt BS=4800 CIRC=portfoliovqe_nativegates_ibm_tket_8 MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:$TMP+1 sb.sh scripts/ibm_search_6l_wos.sh) && TMP=$(echo $TMP | awk '{print $NF}')
TMP=$(CKPT=ckpts/ibm_iter_921_6l.pt BS=4800 CIRC=portfolioqaoa_nativegates_ibm_tket_8 MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:$TMP+1 sb.sh scripts/ibm_search_6l_wos.sh) && TMP=$(echo $TMP | awk '{print $NF}')
TMP=$(CKPT=ckpts/ibm_iter_921_6l.pt BS=4800 CIRC=qft_nativegates_ibm_tket_8 MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:$TMP+1 sb.sh scripts/ibm_search_6l_wos.sh) && TMP=$(echo $TMP | awk '{print $NF}')
