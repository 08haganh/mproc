# Dummy function for testing mproc.py. This function takes in a tuple from dataframe.iterrows() method and returns a dictionary of results, indexed by the index of the iterrows()
# This is designed to return dictionaries of values which are used to construct a dataframe after the multiprocessing algorithm 
import numpy as np
import pandas as pd

def is_even(kwargs):
    pd.Series(kwargs[1]).T.fillna(np.nan).replace([np.nan], [None])
    idx = kwargs[1].name
    number = kwargs[1]['number']
    if number % 2 == 0:
        data = True
    else:
        data = False

    update = {'index':idx,'is_even':data}
    return update