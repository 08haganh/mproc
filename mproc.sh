#!/bin/bash

anaconda_path=~/anaconda3
env=CrystalAnalyser

source ${anaconda_path}/bin/activate
conda activate ${env}

method='combinatorial'
max_workers=4
random_state=0
n_samples=10000 # only needed for method=random_sampling
output_prefix='is_even'
output_directory='output'
max_rows=10000
max_mem=30 # in MB
reduce_mem_usage=1

python /home/harry/Documents/projects/Toolbox/mproc/mproc.py \
--method=$method --max_workers=$max_workers --random_state=$random_state --n_samples=$n_samples \
--output_prefix=$output_prefix --output_directory=$output_directory --max_rows=$max_rows --max_mem=$max_mem --reduce_mem_usage=$reduce_mem_usage