# Function that performs hdbscan clustering from a distance matrix
# kwargs is an iteration of pd.DataFrame.iterrows()
# Expects as input of a txt file of a distance matrix in condensed or square form

import numpy as np
import pandas as pd
import hdbscan
from scipy.spatial.distance import squareform

######## REMEMBER TO CHECK THAT YOU ARE USING A DISTANCE MATRIX AND NOT A SIMILARITY MATRIX ########

def hdbscan_cluster_from_distance_matrix(kwargs):

    # Load arguments
    kwargs = pd.Series(kwargs[1]).T.fillna(np.nan).replace([np.nan], [None])
    idx = str(kwargs.name)

    # Lets load the data to cluster
    cluster_data_path = kwargs['cluster_data_path']
    cluster_data = np.loadtxt(cluster_data_path,delimiter=' ',dtype=np.float64)

    # Convert to squareform if necessary
    if len(cluster_data.shape) == 1: # condensed matrix
        cluster_data = squareform(cluster_data.reshape(-1))
    elif cluster_data.shape[0] != cluster_data.shape[1]: # condensed matrix
        cluster_data = squareform(cluster_data.reshape(-1))
    else:
        cluster_data = cluster_data

    # Define HDBSCAN init args 
    hdbscan_init_args = pd.DataFrame(kwargs[['min_cluster_size','min_samples','cluster_selection_epsilon','alpha','cluster_selection_method',
                                    'allow_single_cluster','metric']]).T
    hdbscan_init_args = hdbscan_init_args.astype({
                                    'min_cluster_size': 'int32',
                                    'min_samples': 'int32',
                                    'cluster_selection_epsilon': 'float32',
                                    'alpha': 'float32',
                                    'cluster_selection_method': 'str',
                                    'allow_single_cluster': 'int32',
                                    'metric': 'str',
                                    })
    hdbscan_init_args = hdbscan_init_args.iloc[0].to_dict()
    
    # Complete clustering
    clusterer = hdbscan.HDBSCAN(**hdbscan_init_args)
    clusterer.fit(cluster_data.values)
    clusters = clusterer.labels_
    update = {'index':idx}
    update.update({ key: clusters[i] for i, key in enumerate(cluster_data.index) })
    
    return update