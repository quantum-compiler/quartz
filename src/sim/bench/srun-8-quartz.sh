#!/bin/bash
#SBATCH -A m4138
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 8
#SBATCH --gpus-per-node=4

rm -rf ~/qs/quartz/build/nodefile8
rm -rf ~/qs/result-srun/quartz/*_33.log
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


mpirun -np 8 -H $hosts ~/qs/quartz/build/simulate --import-circuit qft --n 33 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/qft_33.log
mpirun -np 8 -H $hosts ~/qs/quartz/build/simulate --import-circuit qftentangled --n 33 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/qftentangled_33.log
mpirun -np 8 -H $hosts ~/qs/quartz/build/simulate --import-circuit ghz --n 33 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/ghz_33.log
mpirun -np 8 -H $hosts ~/qs/quartz/build/simulate --import-circuit graphstate --n 33 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/graphstate_33.log
mpirun -np 8 -H $hosts ~/qs/quartz/build/simulate --import-circuit twolocalrandom --n 33 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/twolocalrandom_33.log
mpirun -np 8 -H $hosts ~/qs/quartz/build/simulate --import-circuit realamprandom --n 33 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/realamprandom_33.log
mpirun -np 8 -H $hosts ~/qs/quartz/build/simulate --import-circuit su2random --n 33 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/su2random_33.log
mpirun -np 8 -H $hosts ~/qs/quartz/build/simulate --import-circuit ae --n 33 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/ae_33.log
mpirun -np 8 -H $hosts ~/qs/quartz/build/simulate --import-circuit qpeexactm --n 33 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/qpeexact_33.log
mpirun -np 8 -H $hosts ~/qs/quartz/build/simulate --import-circuit qpeinexactm --n 33 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/qpeinexact_33.log