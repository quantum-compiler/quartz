#!/bin/bash
set -ex

TMP=$(SEED=23333 CKPT=ckpts/ibm.pt BS=4800 CIRC=qgan_nativegates_ibm_tket_8 MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:17010639+1 sb.sh scripts/ibm_search_6l_gae.sh) && TMP=$(echo $TMP | awk '{print $NF}')
TMP=$(SEED=23333 CKPT=ckpts/ibm.pt BS=4800 CIRC=portfoliovqe_nativegates_ibm_tket_8 MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:$TMP+1 sb.sh scripts/ibm_search_6l_gae.sh) && TMP=$(echo $TMP | awk '{print $NF}')
TMP=$(SEED=23333 CKPT=ckpts/ibm.pt BS=4800 CIRC=portfolioqaoa_nativegates_ibm_tket_8 MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:$TMP+1 sb.sh scripts/ibm_search_6l_gae.sh) && TMP=$(echo $TMP | awk '{print $NF}')
TMP=$(SEED=23333 CKPT=ckpts/ibm.pt BS=4800 CIRC=qft_nativegates_ibm_tket_8 MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:$TMP+1 sb.sh scripts/ibm_search_6l_gae.sh) && TMP=$(echo $TMP | awk '{print $NF}')

TMP=$(SEED=42 CKPT=ckpts/ibm.pt BS=4800 CIRC=qgan_nativegates_ibm_tket_8 MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:$TMP+1 sb.sh scripts/ibm_search_6l_gae.sh) && TMP=$(echo $TMP | awk '{print $NF}')
TMP=$(SEED=42 CKPT=ckpts/ibm.pt BS=4800 CIRC=portfoliovqe_nativegates_ibm_tket_8 MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:$TMP+1 sb.sh scripts/ibm_search_6l_gae.sh) && TMP=$(echo $TMP | awk '{print $NF}')
TMP=$(SEED=42 CKPT=ckpts/ibm.pt BS=4800 CIRC=portfolioqaoa_nativegates_ibm_tket_8 MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:$TMP+1 sb.sh scripts/ibm_search_6l_gae.sh) && TMP=$(echo $TMP | awk '{print $NF}')
TMP=$(SEED=42 CKPT=ckpts/ibm.pt BS=4800 CIRC=qft_nativegates_ibm_tket_8 MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:$TMP+1 sb.sh scripts/ibm_search_6l_gae.sh) && TMP=$(echo $TMP | awk '{print $NF}')

TMP=$(SEED=98765 CKPT=ckpts/ibm.pt BS=4800 CIRC=qgan_nativegates_ibm_tket_8 MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:$TMP+1 sb.sh scripts/ibm_search_6l_gae.sh) && TMP=$(echo $TMP | awk '{print $NF}')
TMP=$(SEED=98765 CKPT=ckpts/ibm.pt BS=4800 CIRC=portfoliovqe_nativegates_ibm_tket_8 MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:$TMP+1 sb.sh scripts/ibm_search_6l_gae.sh) && TMP=$(echo $TMP | awk '{print $NF}')
TMP=$(SEED=98765 CKPT=ckpts/ibm.pt BS=4800 CIRC=portfolioqaoa_nativegates_ibm_tket_8 MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:$TMP+1 sb.sh scripts/ibm_search_6l_gae.sh) && TMP=$(echo $TMP | awk '{print $NF}')
TMP=$(SEED=98765 CKPT=ckpts/ibm.pt BS=4800 CIRC=qft_nativegates_ibm_tket_8 MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:$TMP+1 sb.sh scripts/ibm_search_6l_gae.sh) && TMP=$(echo $TMP | awk '{print $NF}')

