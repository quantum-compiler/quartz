export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/root/usr_local/lib:$LD_LIBRARY_PATH
python gen_opt_path.py --depth=$1
