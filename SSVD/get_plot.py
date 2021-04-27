import numpy as np
import seaborn as sns

def get_plot(u, s, v, clusters, dismiss=0):
    """
    Get heat plot of clusters in rank 1 approximation.
    
    u, s, v = SSVD function output
    clusters = vector of predetermined groups
    dismiss = index discarded
    """
    
    X_star = s*u[:,None]@v[None,:]
    unique_clusters = np.unique(clusters)
    
    row_idx = np.empty(0, dtype = 'int')
    for i in range(len(unique_clusters)): #sort u by unique clusters
            idx = np.where(clusters == unique_clusters[i])
            clus_idx = idx[0][np.argsort(u[idx])]
            row_idx = np.concatenate((row_idx, clus_idx)) #update row index

    X = X_star[:,np.argsort(np.abs(v))[dismiss:]] 
    v_sort = v[np.argsort(np.abs(v))[dismiss:]] #smallest to largest absolute v (col structure of X)
    col_idx = np.argsort(v_sort) #col index for smallest to largest v
    
    sns.heatmap(X[np.ix_(row_idx, col_idx)], vmin=-1, vmax=1, cmap = 'bwr')
