#!/bin/bash
set -ex

TMP=$(LAYER=2 BS=4800 CIRC=adder_8 MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:16887293+1 sb.sh scripts/ibm_search_xl.sh) && TMP=$(echo $TMP | awk '{print $NF}')
TMP=$(LAYER=2 BS=4800 CIRC=barenco_tof_10 MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:$TMP+1 sb.sh scripts/ibm_search_xl.sh) && TMP=$(echo $TMP | awk '{print $NF}')
TMP=$(LAYER=2 BS=4800 CIRC=gf2^6_mult MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:$TMP+1 sb.sh scripts/ibm_search_xl.sh) && TMP=$(echo $TMP | awk '{print $NF}')
TMP=$(LAYER=2 BS=4800 CIRC=mod_red_21 MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:$TMP+1 sb.sh scripts/ibm_search_xl.sh) && TMP=$(echo $TMP | awk '{print $NF}')

TMP=$(LAYER=6 BS=4800 CIRC=adder_8 MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:$TMP+6 sb.sh scripts/ibm_search_xl.sh) && TMP=$(echo $TMP | awk '{print $NF}')
TMP=$(LAYER=6 BS=4800 CIRC=barenco_tof_10 MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:$TMP+1 sb.sh scripts/ibm_search_xl.sh) && TMP=$(echo $TMP | awk '{print $NF}')
TMP=$(LAYER=6 BS=4800 CIRC=gf2^6_mult MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:$TMP+1 sb.sh scripts/ibm_search_xl.sh) && TMP=$(echo $TMP | awk '{print $NF}')
TMP=$(LAYER=6 BS=4800 CIRC=mod_red_21 MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:$TMP+1 sb.sh scripts/ibm_search_xl.sh) && TMP=$(echo $TMP | awk '{print $NF}')

TMP=$(LAYER=10 BS=4800 CIRC=adder_8 MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:$TMP+6 sb.sh scripts/ibm_search_xl.sh) && TMP=$(echo $TMP | awk '{print $NF}')
TMP=$(LAYER=10 BS=4800 CIRC=barenco_tof_10 MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:$TMP+1 sb.sh scripts/ibm_search_xl.sh) && TMP=$(echo $TMP | awk '{print $NF}')
TMP=$(LAYER=10 BS=4800 CIRC=gf2^6_mult MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:$TMP+1 sb.sh scripts/ibm_search_xl.sh) && TMP=$(echo $TMP | awk '{print $NF}')
TMP=$(LAYER=10 BS=4800 CIRC=mod_red_21 MEM=10 sbatch -c 128 --gpus=1 --time=6:10:00 -d after:$TMP+1 sb.sh scripts/ibm_search_xl.sh) && TMP=$(echo $TMP | awk '{print $NF}')
