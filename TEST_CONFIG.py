# Import target function as function
import sys
sys.path.append('functions')
from is_even import is_even as function

# dummy_column needed as iterrows() returns single value rather than pd.Series object if only one column in dataframe
CONFIG = {
    'dummy_column':[x for x in range(1,2)],
    'number':[x for x in range(100000)]
    }