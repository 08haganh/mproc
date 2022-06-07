import sys
sys.path.append('/home/harry/Documents/projects/Toolbox/mproc/clustering/hdbscan')
# Comment as appropriate
from hdbscan_cluster_from_distance_matrix import hdbscan_cluster_from_distance_matrix as function
#from hdbscan_cluster_from_metric_space import hdbscan_cluster_from_metric_space as function

CONFIG = {
    
    'cluster_data_path':[],
    'min_cluster_size':[],
    'min_samples':[],
    'cluster_selection_epsilon':[],
    'alpha':[],
    'cluster_selection_method':[],
    'allow_single_cluster':[],
    'metric':[]

}
