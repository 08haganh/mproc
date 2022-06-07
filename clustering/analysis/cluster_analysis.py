# MPROC script for analysing results from a clustering descriptor and/or parameter sweep
# If you only have 1 file this will work like a simple for loop, but this analysis is normally pretty quick so shouldn't be too bad
# Get speed improvements if used for many files with cluster assignments

# Just loops through all metrics in sklearn docs + returns number of clusters and percentage of dataset clustered

# The inputs this script expects are as follows
# .csv file containing cluster assignments. columns are structures and rows are parametrisations from hyperparameter sweep
# assessment space path  is either a distance matrix or a metric space. This is determined from the file extension, if the file extension is .txt, assumed to be a distance matrix with delimiter ' ', 
# if it is .csv, it is expected to be a metric space with headers and index at column=0
# if params path is given, it will look up a specific assessment space for each cluster from a parameter file

import numpy as np
import pandas as pd

from sklearn import metrics

from scipy.spatial.distance import squareform

def cluster_analysis(kwargs):

    # Load arguments
    kwargs = pd.Series(kwargs[1]).T.fillna(np.nan).replace([np.nan], [None])

    # Load data
    clusters = pd.read_csv(kwargs.cluster_path,index_col=0)
    if kwargs.params_path is not None:
        form = 'distance_matrix' if kwargs.assessment_space_path.split('.')[-1] == 'txt' else 'metric_space'
        if form == 'distance_matrix':
            assessment_space = np.loadtxt(kwargs.assessment_space_path,dtype=np.float64)
            # Convert to squareform if necessary
            if len(assessment_space.shape) == 1: # condensed matrix
                assessment_space = squareform(assessment_space.reshape(-1))
            elif assessment_space.shape[0] != assessment_space.shape[1]: # condensed matrix
                assessment_space = squareform(assessment_space.reshape(-1))
            else:
                assessment_space = assessment_space
        else:
            assessment_space =  pd.read_csv(kwargs.assessment_space_path,index_col=0)
    else:
        params = pd.read_csv(kwargs.params_path,index_col=0)

    # Check for true labels
    if kwargs.true_labels_path is not None:
        cluster_labels = pd.read_csv(kwargs.true_labels_path,index_col=0)
        labels_true = cluster_labels.values.reshape(-1)

    # Loop through rows
    scores = []
    for row in (clusters.iterrows()):
        index = row[1].name
        labels_pred = row[1].values.reshape(-1)
        if kwargs.metric_space_path is None:
            assessment_space = pd.read_csv(params.loc[index,'cluster_data_path'],delimiter=kwargs.delimiter,header=None).values
            # Convert to squareform if necessary
            if len(assessment_space.shape) == 1: # condensed matrix
                assessment_space = squareform(assessment_space.reshape(-1))
            elif assessment_space.shape[0] != assessment_space.shape[1]: # condensed matrix
                assessment_space = squareform(assessment_space.reshape(-1))
            else:
                assessment_space = assessment_space

        # only use clusters assigned 0 or more, -1 is assumed to be unclustered
        mask = labels_pred != -1
        if np.all(np.invert(mask)):
            update = {}
        else:
            update = {'index':index}
            update['n_clusters'] = len(np.unique(labels_pred[mask]))
            update['percentage_clustered'] = sum(mask) / len(mask)
            # Clustering without labels
            if kwargs.silhouette_metric == 'precomputed':
                # Remove mask from both axes 
                update['silhouette_score'] = metrics.silhouette_score(assessment_space[mask,:][:,mask],labels_pred[mask],metric=kwargs.silhouette_metric)
            else:
                update['silhouette_score'] = metrics.silhouette_score(assessment_space[mask],labels_pred[mask],metric=kwargs.silhouette_metric)
                update['calinski_harabasz_score'] = metrics.calinski_harabasz_score(assessment_space[mask],labels_pred[mask])
                update['davies_bouldin_score'] = metrics.davies_bouldin_score(assessment_space[mask],labels_pred[mask])
            if kwargs.true_labels_path is not None:
                # Clustering with labels
                update['rand_score'] = metrics.silhouette_score(assessment_space[mask],labels_pred[mask])
                update['mutual_info_score'] = metrics.mutual_info_score(assessment_space[mask],labels_pred[mask])
                update['adjusted_mutual_info_score'] = metrics.adjusted_mutual_info_score(assessment_space[mask],labels_pred[mask])
                update['normalized_mutual_info_score'] = metrics.normalized_mutual_info_score(assessment_space[mask],labels_pred[mask])
                update['homogeneity_score'] = metrics.homogeneity_score(labels_true, labels_pred)
                update['completeness_score'] = metrics.completeness_score(labels_true, labels_pred)
                update['fowlkes_mallows_score'] = metrics.fowlkes_mallows_score(labels_true, labels_pred)

        scores.append(update)

    return scores
