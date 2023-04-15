#!/bin/bash
#SBATCH -A m4138
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 4
#SBATCH --gpus-per-node=4

rm -rf ~/qs/quartz/build/nodefile4
rm -rf ~/qs/result-srun/quartz/*_32.log
nodelist=$(scontrol show hostname $SLURM_NODELIST)
printf "%s\n" "${nodelist[@]}" > nodefile4
module use /global/common/software/m3169/perlmutter/modulefiles
module unload cray-mpich cray-libsci
module load openmpi
module load nccl
module load python
conda activate qs
export PATH=$PATH:/global/homes/z/zjia/qs/quartz/external/HiGHS/build/bin
export HYQUAS_ROOT=~/qs/HyQuas


mpirun -np 4 -npernode 1 -hostfile nodefile4 ~/qs/quartz/build/simulate --import-circuit qft --n 32 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/qft_32.log
mpirun -np 4 -npernode 1 -hostfile nodefile4 ~/qs/quartz/build/simulate --import-circuit qftentangled --n 32 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/qftentangled_32.log
mpirun -np 4 -npernode 1 -hostfile nodefile4 ~/qs/quartz/build/simulate --import-circuit ghz --n 32 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/ghz_32.log
mpirun -np 4 -npernode 1 -hostfile nodefile4 ~/qs/quartz/build/simulate --import-circuit graphstate --n 32 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/graphstate_32.log
mpirun -np 4 -npernode 1 -hostfile nodefile4 ~/qs/quartz/build/simulate --import-circuit twolocalrandom --n 32 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/twolocalrandom_32.log
mpirun -np 4 -npernode 1 -hostfile nodefile4 ~/qs/quartz/build/simulate --import-circuit realamprandom --n 32 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/realamprandom_32.log
mpirun -np 4 -npernode 1 -hostfile nodefile4 ~/qs/quartz/build/simulate --import-circuit su2random --n 32 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/su2random_32.log
mpirun -np 4 -npernode 1 -hostfile nodefile4 ~/qs/quartz/build/simulate --import-circuit ae --n 32 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/aem_32.log
mpirun -np 4 -npernode 1 -hostfile nodefile4 ~/qs/quartz/build/simulate --import-circuit qpeexactm --n 32 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/qpeexact_32.log
mpirun -np 4 -npernode 1 -hostfile nodefile4 ~/qs/quartz/build/simulate --import-circuit qpeinexactm --n 32 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/qpeinexact_32.log