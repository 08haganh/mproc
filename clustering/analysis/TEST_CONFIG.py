# Imports
import sys
sys.path.append('/home/harry/Documents/PAHSolidStatePacking/scripts/clustering/analysis')
from cluster_analysis import cluster_analysis as function
# Only works for either distance matrices or metric spaces, not both

CONFIG = {

    'clusters_path':['/home/harry/Documents/PAHSolidStatePacking/csd15_planar_organic/descriptors/geometric_interactions/hdbscan_param_sweep1/batch0_rows_0to3233_hdbscan_param_sweep1.csv'],
    'params_path':[None],
    'assessment_space_path':['/home/harry/Documents/PAHSolidStatePacking/csd15_planar_organic/descriptors/geometric_interactions/normalised_distance_matrix.txt'],
    'true_labels_path':[None],
    'silhouette_metric':['precomputed'],
}