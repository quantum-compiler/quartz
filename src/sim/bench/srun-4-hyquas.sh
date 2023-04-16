#!/bin/bash
#SBATCH -A m4138
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 4
#SBATCH --gpus-per-node=4

rm -rf ~/qs/HyQuas/build/nodefile4
rm -rf ~/qs/result-srun/hyquas/*_32.log
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
`which mpirun` -np 4 -H $hosts ./main ../../quartz/circuit/MQTBench_32q/qft_indep_qiskit_32.qasm > ~/qs/result-srun/hyquas/on_qft_32.log
`which mpirun` -np 4 -H $hosts ./main ../../quartz/circuit/MQTBench_32q/qftentangled_indep_qiskit_32.qasm > ~/qs/result-srun/hyquas/on_qftentangled_32.log
`which mpirun` -np 4 -H $hosts ./main ../../quartz/circuit/MQTBench_32q/ghz_indep_qiskit_32.qasm > ~/qs/result-srun/hyquas/on_ghz_32.log
`which mpirun` -np 4 -H $hosts ./main ../../quartz/circuit/MQTBench_32q/graphstate_indep_qiskit_32.qasm > ~/qs/result-srun/hyquas/on_graphstate_32.log
`which mpirun` -np 4 -H $hosts ./main ../../quartz/circuit/MQTBench_32q/twolocalrandom_indep_qiskit_32.qasm > ~/qs/result-srun/hyquas/on_twolocalrandom_32.log
`which mpirun` -np 4 -H $hosts ./main ../../quartz/circuit/MQTBench_32q/realamprandom_indep_qiskit_32.qasm > ~/qs/result-srun/hyquas/on_realamprandom_32.log
`which mpirun` -np 4 -H $hosts ./main ../../quartz/circuit/MQTBench_32q/su2random_indep_qiskit_32.qasm > ~/qs/result-srun/hyquas/on_su2random_32.log
`which mpirun` -np 4 -H $hosts ./main ../../quartz/circuit/MQTBench_32q/aem_indep_qiskit_32.qasm > ~/qs/result-srun/hyquas/on_ae_32.log
`which mpirun` -np 4 -H $hosts ./main ../../quartz/circuit/MQTBench_32q/qpeexactm_indep_qiskit_32.qasm > ~/qs/result-srun/hyquas/on_qpeexact_32.log
`which mpirun` -np 4 -H $hosts ./main ../../quartz/circuit/MQTBench_32q/qpeinexactm_indep_qiskit_32.qasm > ~/qs/result-srun/hyquas/on_qpeinexact_32.log

# mpirun -np 8 -H $hosts ~/qs/quartz/build/simulate --import-circuit twolocalrandom --n 33 --local 28 --device 4 --use-ilp > ~/qs/result-srun/quartz/twolocalrandom_32.log