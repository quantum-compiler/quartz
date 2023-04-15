#!/bin/bash
#SBATCH -A m4138
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node=4

rm -rf ~/qs/HyQuas/build/nodefile1
rm -rf ~/qs/result-srun/hyquas/*_30.log
rm -rf ~/qs/result-srun/hyquas/*_29.log
rm -rf ~/qs/result-srun/hyquas/*_28.log
nodelist=$(scontrol show hostname $SLURM_NODELIST)
printf "%s\n" "${nodelist[@]}" > ~/qs/HyQuas/build/nodefile1
module use /global/common/software/m3169/perlmutter/modulefiles
module unload cray-mpich cray-libsci
module load openmpi
module load nccl
module load python
conda activate qs
export PATH=$PATH:/global/homes/z/zjia/qs/quartz/external/HiGHS/build/bin
export HYQUAS_ROOT=~/qs/HyQuas

cd ~/qs/HyQuas/build
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=4 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_30q/qft_indep_qiskit_30.qasm > ~/qs/result-srun/hyquas/on_qft_30.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=4 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_30q/qftentangled_indep_qiskit_30.qasm > ~/qs/result-srun/hyquas/on_qftentangled_30.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=4 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_30q/ghz_indep_qiskit_30.qasm > ~/qs/result-srun/hyquas/on_ghz_30.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=4 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_30q/graphstate_indep_qiskit_30.qasm > ~/qs/result-srun/hyquas/on_graphstate_30.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=4 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_30q/twolocalrandom_indep_qiskit_30.qasm > ~/qs/result-srun/hyquas/on_twolocalrandom_30.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=4 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_30q/realamprandom_indep_qiskit_30.qasm > ~/qs/result-srun/hyquas/on_realamprandom_30.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=4 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_30q/su2random_indep_qiskit_30.qasm > ~/qs/result-srun/hyquas/on_su2random_30.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=4 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_30q/aem_indep_qiskit_30.qasm > ~/qs/result-srun/hyquas/on_ae_30.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=4 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_30q/qpeexactm_indep_qiskit_30.qasm > ~/qs/result-srun/hyquas/on_qpeexact_30.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=4 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_30q/qpeinexactm_indep_qiskit_30.qasm > ~/qs/result-srun/hyquas/on_qpeinexact_30.log

`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=2 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_29q/qft_indep_qiskit_29.qasm > ~/qs/result-srun/hyquas/on_qft_29.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=2 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_29q/qftentangled_indep_qiskit_29.qasm > ~/qs/result-srun/hyquas/on_qftentangled_29.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=2 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_29q/ghz_indep_qiskit_29.qasm > ~/qs/result-srun/hyquas/on_ghz_29.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=2 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_29q/graphstate_indep_qiskit_29.qasm > ~/qs/result-srun/hyquas/on_graphstate_29.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=2 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_29q/twolocalrandom_indep_qiskit_29.qasm > ~/qs/result-srun/hyquas/on_twolocalrandom_29.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=2 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_29q/realamprandom_indep_qiskit_29.qasm > ~/qs/result-srun/hyquas/on_realamprandom_29.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=2 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_29q/su2random_indep_qiskit_29.qasm > ~/qs/result-srun/hyquas/on_su2random_29.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=2 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_29q/aem_indep_qiskit_29.qasm > ~/qs/result-srun/hyquas/on_ae_29.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=2 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_29q/qpeexactm_indep_qiskit_29.qasm > ~/qs/result-srun/hyquas/on_qpeexact_29.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=2 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_29q/qpeinexactm_indep_qiskit_29.qasm > ~/qs/result-srun/hyquas/on_qpeinexact_29.log

`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=1 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_28q/qft_indep_qiskit_28.qasm > ~/qs/result-srun/hyquas/on_qft_28.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=1 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_28q/qftentangled_indep_qiskit_28.qasm > ~/qs/result-srun/hyquas/on_qftentangled_28.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=1 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_28q/ghz_indep_qiskit_28.qasm > ~/qs/result-srun/hyquas/on_ghz_28.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=1 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_28q/graphstate_indep_qiskit_28.qasm > ~/qs/result-srun/hyquas/on_graphstate_28.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=1 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_28q/twolocalrandom_indep_qiskit_28.qasm > ~/qs/result-srun/hyquas/on_twolocalrandom_28.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=1 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_28q/realamprandom_indep_qiskit_28.qasm > ~/qs/result-srun/hyquas/on_realamprandom_28.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=1 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_28q/su2random_indep_qiskit_28.qasm > ~/qs/result-srun/hyquas/on_su2random_28.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=1 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_28q/aem_indep_qiskit_28.qasm > ~/qs/result-srun/hyquas/on_ae_28.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=1 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_28q/qpeexactm_indep_qiskit_28.qasm > ~/qs/result-srun/hyquas/on_qpeexact_28.log
`which mpirun` -np 1 -npernode 1 -hostfile nodefile1 -x GPUPerRank=1 ../scripts/gpu-bind.sh ./main ../../quartz/circuit/MQTBench_28q/qpeinexactm_indep_qiskit_28.qasm > ~/qs/result-srun/hyquas/on_qpeinexact_28.log

# mpirun -np 8 -npernode 1 -hostfile nodefile1 ~/qs/quartz/build/simulate --import-circuit twolocalrandom --n 33 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/twolocalrandom_30.log