#!/bin/bash
#SBATCH -A m4138
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 8
#SBATCH --gpus-per-node=4

cd ~/qs/nccl-tests
rm -rf ~/qs/nccl-tests/nodefile-nccl
nodelist=$(scontrol show hostname $SLURM_NODELIST)
printf "%s\n" "${nodelist[@]}" > nodefile-nccl
module use /global/common/software/m3169/perlmutter/modulefiles
module unload cray-mpich cray-libsci
module load openmpi
module load nccl
module load python
conda activate qs

NCCL_DEBUG=INFO mpirun -x LD_LIBRARY_PATH=LD_LIBRARY_PATH:/global/common/software/nersc/pm-2022q4/sw/nccl-2.15.5-ofi-r4/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/lib64:/global/common/software/m3169/perlmutter/openmpi/5.0.0rc10-ofi-cuda-22.5_11.7/gnu/lib:/opt/cray/libfabric/1.15.2.0/lib64 -np 8 -npernode 1 -hostfile nodefile-nccl ./build/alltoall_perf -b 4G -e 4G -f 2 -d double -g 4 -w 0 -n 1
