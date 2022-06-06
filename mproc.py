# Script for multiprocessing a single function
# The single function should either return a dictionary or a list of dictionaries
# the dictionaries should be able to make a pandas dataframe
import os
import argparse
import numpy as np
import pandas as pd
import itertools as it
from tqdm import tqdm

#from MPROC_CONFIG import function, CONFIG
from TEST_CONFIG import function, CONFIG

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import warnings
# Ignore numpy runtime warning
warnings.filterwarnings('ignore')

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--method',type=str,default='uniform')
    parser.add_argument('--max_workers',type=int,default=1)
    parser.add_argument('--random_state',type=int,default=0)
    parser.add_argument('--n_samples',type=int,default=10000)
    parser.add_argument('--output_prefix',type=str,default='')
    parser.add_argument('--output_directory',type=str,default='output')
    parser.add_argument('--max_rows',type=int,default=10000)
    parser.add_argument('--max_mem',type=float,default=30.0)
    parser.add_argument('--reduce_mem_usage',type=int,default=1)
    args = parser.parse_args()

    assert args.method in ['combinatorial','random_sampling','uniform'] # make sure we do not try combinatorial and random sampling

    assert args.max_workers <= mp.cpu_count()-2 # Lets not kill our machine

    np.random.seed(args.random_state) # Set random state

    # Make output directory
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Define max size of dataframe for batching
    # Avoids memory issues with loading very large dataframes
    max_rows = args.max_rows

    # MPROC loop for combinatorial CONFIG file
    # Use with care as many dimensions + many parameters is bad time
    if args.method == 'combinatorial':
        prod = np.array(list(it.product(*[CONFIG[key] for key in CONFIG.keys()])))
        dict_inputs = { key : prod[:,i] for i, key in enumerate(CONFIG.keys())  }
        inputs = [dict(zip(dict_inputs,t)) for t in zip(*dict_inputs.values())]
        index = [x for x in range(len(inputs))]
        nrows = len(inputs)
        n_splits = max(int(nrows / max_rows),1)
        batches = np.array_split(inputs,n_splits)
        indices = np.array_split(index,n_splits)

    # MPROC loop for randomly sampling parameter space for each dim using numpy.choice
    elif args.method == 'random_sampling':
        # Generate random samples from parameter space
        inputs = []
        for _ in range(args.n_samples):
            temp = {}
            for param in CONFIG.keys():
                temp[param] = np.random.choice(CONFIG[param],1)[0]
            inputs.append(temp)
        index = [x for x in range(len(inputs))]
        nrows = len(inputs)
        n_splits = max(int(nrows / max_rows),1)
        batches = np.array_split(inputs,n_splits)
        indices = np.array_split(index,n_splits)  
    
    # MPROC loop for uniform CONFIG. Expects each dim to have same number of parameters
    elif args.method == 'uniform':
        inputs = [dict(zip(CONFIG,t)) for t in zip(*CONFIG.values())]
        index = [x for x in range(len(inputs))]
        nrows = len(inputs)
        n_splits = max(int(nrows / max_rows),1)
        batches = np.array_split(inputs,n_splits)
        indices = np.array_split(index,n_splits)
    else:
        print('Please enter a valid parameter construction method, options are combinatorial, random_sampling, and uniform') 

    print(f'Looping over {nrows} in {n_splits} batches of {max_rows}')
    for i, (batch, idx) in enumerate(zip(batches,indices)):
        # Multiprocess jobs
        with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            params = pd.DataFrame(data=batch.tolist(),index=idx)
            pd.DataFrame(params).to_csv(f'{args.output_directory}/batch{i}_{args.output_prefix}_params.csv')
            results = list(tqdm(ex.map(function,params.iterrows()), total=len(batch)))
        # Flatten list in case of returning list of dicts rather than single dict
        if type(results[0]) is list:
            results = [item for sublist in results for item in sublist]
        results = pd.DataFrame(data=results)
        if bool(args.reduce_mem_usage):
            results, _  = reduce_mem_usage(results)
        # Save results in chunks if memory too high
        mb = results.memory_usage().sum() / (1024**2)
        if mb > args.max_mem:
            dfs = np.array_split(results,max(int(mb / args.max_mem),1))
            for df in dfs:
                start = min(df.index)
                end = max(df.index)
                df.to_csv(f'{args.output_directory}/batch{i}_rows_{start}to{end}_{args.output_prefix}.csv',index=False)
        else:
            results.to_csv(f'{args.output_directory}/batch{i}_{args.output_prefix}.csv',index=False)    

def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in tqdm(props.columns):
        if props[col].dtype != object:  # Exclude strings
            # Print current column type
            #print("******************************")
            #print("Column: ",col)
            #print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            #print("dtype after: ",props[col].dtype)
            #print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist


if __name__ == '__main__':
    main()