#!/bin/bash
#SBATCH -A m4138
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 2
#SBATCH --gpus-per-node=4

rm -rf ~/qs/HyQuas/build/nodefile2
rm -rf ~/qs/result-srun/hyquas/*_31.log
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

cd ~/qs/HyQuas/build
`which mpirun` -np 2 -H $hosts ./main ../../quartz/circuit/MQTBench_31q/qft_indep_qiskit_31.qasm > ~/qs/result-srun/hyquas/on_qft_31.log
`which mpirun` -np 2 -H $hosts ./main ../../quartz/circuit/MQTBench_31q/qftentangled_indep_qiskit_31.qasm > ~/qs/result-srun/hyquas/on_qftentangled_31.log
`which mpirun` -np 2 -H $hosts ./main ../../quartz/circuit/MQTBench_31q/ghz_indep_qiskit_31.qasm > ~/qs/result-srun/hyquas/on_ghz_31.log
`which mpirun` -np 2 -H $hosts ./main ../../quartz/circuit/MQTBench_31q/graphstate_indep_qiskit_31.qasm > ~/qs/result-srun/hyquas/on_graphstate_31.log
`which mpirun` -np 2 -H $hosts ./main ../../quartz/circuit/MQTBench_31q/twolocalrandom_indep_qiskit_31.qasm > ~/qs/result-srun/hyquas/on_twolocalrandom_31.log
`which mpirun` -np 2 -H $hosts ./main ../../quartz/circuit/MQTBench_31q/realamprandom_indep_qiskit_31.qasm > ~/qs/result-srun/hyquas/on_realamprandom_31.log
`which mpirun` -np 2 -H $hosts ./main ../../quartz/circuit/MQTBench_31q/su2random_indep_qiskit_31.qasm > ~/qs/result-srun/hyquas/on_su2random_31.log
`which mpirun` -np 2 -H $hosts ./main ../../quartz/circuit/MQTBench_31q/aem_indep_qiskit_31.qasm > ~/qs/result-srun/hyquas/on_ae_31.log
`which mpirun` -np 2 -H $hosts ./main ../../quartz/circuit/MQTBench_31q/qpeexactm_indep_qiskit_31.qasm > ~/qs/result-srun/hyquas/on_qpeexact_31.log
`which mpirun` -np 2 -H $hosts ./main ../../quartz/circuit/MQTBench_31q/qpeinexactm_indep_qiskit_31.qasm > ~/qs/result-srun/hyquas/on_qpeinexact_31.log

# mpirun -np 8 -npernode 1 -hostfile nodefile ~/qs/quartz/build/simulate --import-circuit twolocalrandom --n 33 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/twolocalrandom_31.log