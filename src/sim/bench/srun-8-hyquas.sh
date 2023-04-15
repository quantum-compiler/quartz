#!/bin/bash
#SBATCH -A m4138
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 8
#SBATCH --gpus-per-node=4

rm -rf ~/qs/HyQuas/build/nodefile8
rm -rf ~/qs/result-srun/hyquas/*_33.log
nodelist=$(scontrol show hostname $SLURM_NODELIST)
printf "%s\n" "${nodelist[@]}" > ~/qs/HyQuas/build/nodefile8
module use /global/common/software/m3169/perlmutter/modulefiles
module unload cray-mpich cray-libsci
module load openmpi
module load nccl
module load python
conda activate qs
export PATH=$PATH:/global/homes/z/zjia/qs/quartz/external/HiGHS/build/bin
export HYQUAS_ROOT=~/qs/HyQuas

cd ~/qs/HyQuas/build
`which mpirun` -np 8 -npernode 1 -hostfile nodefile8 -x GPUPerRank=4 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_33q/qft_indep_qiskit_33.qasm > ~/qs/result-srun/hyquas/on_qft_33.log
`which mpirun` -np 8 -npernode 1 -hostfile nodefile8 -x GPUPerRank=4 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_33q/qftentangled_indep_qiskit_33.qasm > ~/qs/result-srun/hyquas/on_qftentangled_33.log
`which mpirun` -np 8 -npernode 1 -hostfile nodefile8 -x GPUPerRank=4 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_33q/ghz_indep_qiskit_33.qasm > ~/qs/result-srun/hyquas/on_ghz_33.log
`which mpirun` -np 8 -npernode 1 -hostfile nodefile8 -x GPUPerRank=4 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_33q/graphstate_indep_qiskit_33.qasm > ~/qs/result-srun/hyquas/on_graphstate_33.log
`which mpirun` -np 8 -npernode 1 -hostfile nodefile8 -x GPUPerRank=4 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_33q/twolocalrandom_indep_qiskit_33.qasm > ~/qs/result-srun/hyquas/on_twolocalrandom_33.log
`which mpirun` -np 8 -npernode 1 -hostfile nodefile8 -x GPUPerRank=4 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_33q/realamprandom_indep_qiskit_33.qasm > ~/qs/result-srun/hyquas/on_realamprandom_33.log
`which mpirun` -np 8 -npernode 1 -hostfile nodefile8 -x GPUPerRank=4 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_33q/su2random_indep_qiskit_33.qasm > ~/qs/result-srun/hyquas/on_su2random_33.log
`which mpirun` -np 8 -npernode 1 -hostfile nodefile8 -x GPUPerRank=4 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_33q/aem_indep_qiskit_33.qasm > ~/qs/result-srun/hyquas/on_ae_33.log
`which mpirun` -np 8 -npernode 1 -hostfile nodefile8 -x GPUPerRank=4 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_33q/qpeexactm_indep_qiskit_33.qasm > ~/qs/result-srun/hyquas/on_qpeexact_33.log
`which mpirun` -np 8 -npernode 1 -hostfile nodefile8 -x GPUPerRank=4 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_33q/qpeinexactm_indep_qiskit_33.qasm > ~/qs/result-srun/hyquas/on_qpeinexact_33.log

# mpirun -np 8 -npernode 1 -hostfile nodefile8 ~/qs/quartz/build/simulate --import-circuit twolocalrandom --n 33 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/twolocalrandom_33.log