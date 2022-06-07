# Function that performs hdbscan clustering from a distance matrix
# kwargs is an iteration of pd.DataFrame.iterrows()
# Expects as input of a csv file with headers and an index at column=0

import numpy as np
import pandas as pd
import hdbscan

def hdbscan_cluster_from_metric_space(kwargs):

    # Load arguments
    kwargs = pd.Series(kwargs[1]).T.fillna(np.nan).replace([np.nan], [None])
    idx = str(kwargs.name)

    # Lets load the data to cluster
    cluster_data_path = kwargs['cluster_data_path']
    cluster_data = pd.read_csv(cluster_data_path,index_col=0)

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