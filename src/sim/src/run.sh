mpirun -np 4 -H nid003908,nid003909,nid003912,nid003913 ~/qs/quartz/build/simulate --import-circuit qftentangled --n 32 --local 28 --device 4 --use-ilp

mpirun -np 4 -H nid003908,nid003909,nid003912,nid003913 ~/qs/quartz/build/simulate --import-circuit ghz --n 32 --local 28 --device 4 --use-ilp

mpirun -np 4 -H nid003908,nid003909,nid003912,nid003913 ~/qs/quartz/build/simulate --import-circuit realamprandom --n 32 --local 28 --device 4 --use-ilp

mpirun -np 4 -H nid003908,nid003909,nid003912,nid003913 ~/qs/quartz/build/simulate --import-circuit graphstate --n 32 --local 28 --device 4 --use-ilp

mpirun -np 4 -H nid003908,nid003909,nid003912,nid003913 ~/qs/quartz/build/simulate --import-circuit twolocalrandom --n 32 --local 28 --device 4 --use-ilp

mpirun -np 4 -H nid003908,nid003909,nid003912,nid003913 ~/qs/quartz/build/simulate --import-circuit wstate --n 32 --local 28 --device 4 --use-ilp

mpirun -np 4 -H nid003908,nid003909,nid003912,nid003913 ~/qs/quartz/build/simulate --import-circuit su2random --n 32 --local 28 --device 4 --use-ilp

mpirun -np 4 -H nid003908,nid003909,nid003912,nid003913 ~/qs/quartz/build/simulate --import-circuit qft --n 32 --local 28 --device 4 --use-ilp

mpirun -np 4 -H nid003908,nid003909,nid003912,nid003913 ~/qs/quartz/build/simulate --import-circuit dj --n 32 --local 28 --device 4 --use-ilp

mpirun -np 1 -H nid001941 ~/qs/quartz/build/simulate --import-circuit qft --n 28 --local 28 --device 1 --use-ilp

mpirun -x LD_LIBRARY_PATH=LD_LIBRARY_PATH:/global/common/software/nersc/pm-2022q4/sw/nccl-2.15.5-ofi-r4/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/lib64:/global/common/software/m3169/perlmutter/openmpi/5.0.0rc10-ofi-cuda-22.5_11.7/gnu/lib:/opt/cray/libfabric/1.15.2.0/lib64 -np 4 -H nid001281,nid001316,nid001445,nid001604 ./build/alltoall_perf -b 4G -e 4G -f 2 -d double -g 4 -w 0 -n 1
mpirun -x LD_LIBRARY_PATH=LD_LIBRARY_PATH:/global/common/software/nersc/pm-2022q4/sw/nccl-2.15.5-ofi-r4/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/lib64:/global/common/software/m3169/perlmutter/openmpi/5.0.0rc10-ofi-cuda-22.5_11.7/gnu/lib:/opt/cray/libfabric/1.15.2.0/lib64 -np 1 -H nid001008 ./build/alltoall_perf -b 4G -e 4G -f 2 -d double -g 4 -w 0 -n 1

module use /global/common/software/m3169/perlmutter/modulefiles
module unload cray-mpich cray-libsci
module load openmpi
module load nccl
module load python
conda activate qs
export PATH=$PATH:/global/homes/z/zjia/qs/quartz/external/HiGHS/build/bin
export HYQUAS_ROOT=~/qs/HyQuas
