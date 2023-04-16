#!/bin/bash
#SBATCH -A m4138
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 2
#SBATCH --gpus-per-node=4

rm -rf ~/qs/quartz/build/nodefile2
rm -rf ~/qs/result-srun/quartz/*_31.log
nodelist=$(scontrol show hostname $SLURM_NODELIST)
hosts=""
for HOST in $nodelist; do
    echo $HOST
    hosts="$hosts,$HOST"
done
echo $hosts
module use /global/common/software/m3169/perlmutter/modulefiles
module unload cray-mpich cray-libsci
module load openmpi
module load nccl
module load python
conda activate qs
export PATH=$PATH:/global/homes/z/zjia/qs/quartz/external/HiGHS/build/bin
export HYQUAS_ROOT=~/qs/HyQuas


mpirun -np 2 -H $hosts ~/qs/quartz/build/simulate --import-circuit qft --n 31 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/qft_31.log
mpirun -np 2 -H $hosts ~/qs/quartz/build/simulate --import-circuit qftentangled --n 31 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/qftentangled_31.log
mpirun -np 2 -H $hosts ~/qs/quartz/build/simulate --import-circuit ghz --n 31 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/ghz_31.log
mpirun -np 2 -H $hosts ~/qs/quartz/build/simulate --import-circuit graphstate --n 31 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/graphstate_31.log
mpirun -np 2 -H $hosts ~/qs/quartz/build/simulate --import-circuit twolocalrandom --n 31 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/twolocalrandom_31.log
mpirun -np 2 -H $hosts ~/qs/quartz/build/simulate --import-circuit realamprandom --n 31 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/realamprandom_31.log
mpirun -np 2 -H $hosts ~/qs/quartz/build/simulate --import-circuit su2random --n 31 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/su2random_31.log
mpirun -np 2 -H $hosts ~/qs/quartz/build/simulate --import-circuit ae --n 31 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/ae_31.log
mpirun -np 2 -H $hosts ~/qs/quartz/build/simulate --import-circuit qpeexactm --n 31 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/qpeexact_31.log
mpirun -np 2 -H $hosts ~/qs/quartz/build/simulate --import-circuit qpeinexactm --n 31 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/qpeinexact_31.log