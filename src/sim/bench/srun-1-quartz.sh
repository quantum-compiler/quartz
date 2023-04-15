#!/bin/bash
#SBATCH -A m4138
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node=4

rm -rf ~/qs/quartz/build/nodefile1
rm -rf ~/qs/result-srun/quartz/*_30.log
rm -rf ~/qs/result-srun/quartz/*_29.log
rm -rf ~/qs/result-srun/quartz/*_28.log
nodelist=$(scontrol show hostname $SLURM_NODELIST)
printf "%s\n" "${nodelist[@]}" > nodefile1
module use /global/common/software/m3169/perlmutter/modulefiles
module unload cray-mpich cray-libsci
module load openmpi
module load nccl
module load python
conda activate qs
export PATH=$PATH:/global/homes/z/zjia/qs/quartz/external/HiGHS/build/bin
export HYQUAS_ROOT=~/qs/HyQuas


mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit qft --n 30 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/qft_30.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit qftentangled --n 30 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/qftentangled_30.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit ghz --n 30 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/ghz_30.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit graphstate --n 30 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/graphstate_30.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit twolocalrandom --n 30 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/twolocalrandom_30.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit realamprandom --n 30 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/realamprandom_30.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit su2random --n 30 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/su2random_30.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit ae --n 30 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/aem_30.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit qpeexactm --n 30 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/qpeexact_30.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit qpeinexactm --n 30 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/qpeinexact_30.log

mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit qft --n 28 --local 28 --device 1 --use-ilp > ~/qs/result-srun/quartz/qft_28.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit qftentangled --n 28 --local 28 --device 1 --use-ilp > ~/qs/result-srun/quartz/qftentangled_28.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit ghz --n 28 --local 28 --device 1 --use-ilp > ~/qs/result-srun/quartz/ghz_28.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit graphstate --n 28 --local 28 --device 1 --use-ilp > ~/qs/result-srun/quartz/graphstate_28.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit twolocalrandom --n 28 --local 28 --device 1 --use-ilp > ~/qs/result-srun/quartz/twolocalrandom_28.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit realamprandom --n 28 --local 28 --device 1 --use-ilp > ~/qs/result-srun/quartz/realamprandom_28.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit su2random --n 28 --local 28 --device 1 --use-ilp > ~/qs/result-srun/quartz/su2random_28.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit ae --n 28 --local 28 --device 1 --use-ilp > ~/qs/result-srun/quartz/aem_28.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit qpeexactm --n 28 --local 28 --device 1 --use-ilp > ~/qs/result-srun/quartz/qpeexact_28.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit qpeinexactm --n 28 --local 28 --device 1 --use-ilp > ~/qs/result-srun/quartz/qpeinexact_28.log

mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit qft --n 29 --local 28 --device 2 --use-ilp > ~/qs/result-srun/quartz/qft_29.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit qftentangled --n 29 --local 28 --device 2 --use-ilp > ~/qs/result-srun/quartz/qftentangled_29.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit ghz --n 29 --local 28 --device 2 --use-ilp > ~/qs/result-srun/quartz/ghz_29.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit graphstate --n 29 --local 28 --device 2 --use-ilp > ~/qs/result-srun/quartz/graphstate_29.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit twolocalrandom --n 29 --local 28 --device 2 --use-ilp > ~/qs/result-srun/quartz/twolocalrandom_29.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit realamprandom --n 29 --local 28 --device 2 --use-ilp > ~/qs/result-srun/quartz/realamprandom_29.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit su2random --n 29 --local 28 --device 2 --use-ilp > ~/qs/result-srun/quartz/su2random_29.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit ae --n 29 --local 28 --device 2 --use-ilp > ~/qs/result-srun/quartz/aem_29.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit qpeexactm --n 29 --local 28 --device 2 --use-ilp > ~/qs/result-srun/quartz/qpeexact_29.log
mpirun -np 1 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit qpeinexactm --n 29 --local 28 --device 2 --use-ilp > ~/qs/result-srun/quartz/qpeinexact_29.log