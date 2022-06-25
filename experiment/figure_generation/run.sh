export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/root/usr_local/lib:$LD_LIBRARY_PATH
python generator.py --no_increase=True --include_nop=False