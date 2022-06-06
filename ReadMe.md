# mproc
### Custom Multiprocessing scripts for running hyperparameter sweeps
 1) Create a function that takes a dataframe.iterrows() output, where the Series has all of the named variables required in the function.
 in this example we have an is_even.py function in the functions folder. The accepted outputs for this function is either a dict or a list of dicts
 2) Define a file named MPROC_CONFIG.py which contains a dictionary with all of the variables over which a parameter sweep will be completed. In this file you must import that function in step 1 as function. There are three modes in which the mproc sweep can be run. Uniform, where each parameter must have an equal number of values, combinatorial, where the cartesian product of all lists in the CONFIG dictionary will be swept through, and random sampling, which picks a value at random from each parameter to create n_samples, and sweeps through those.
 3) Run mproc.sh

Please note that the iterative functions can take as arguments other functions, such as normalisation functions, to be called within. These functions can be defined either in MPROC_CONFIG or imported from elsewhere, and entered as arguments

The script works by first splitting the parametrisations into batches based on the max rows argument. This is chosen based on the fact that the outputs from the mproc will be loaded into a pandas dataframe, the dimensions of which will be max_rows*n_features returned from function in step 1. If that product is too large, your computer will freeze during dataframe construction. Once the batches are obtained, it will run through each parametrisation in batch using max_workers. The results from the batch will be saved as a dataframe, memory reduction will be completed if reduce_mem_usage, and the dataframe will be saved. If the dataframe exceeds max_mem, the batched dataframe will be broken up further into chunks ~equal to max_mem. 

It is recommended that you always include as an output from the function the index of the input row from the param file, such that output results can be easily matched to inputs

