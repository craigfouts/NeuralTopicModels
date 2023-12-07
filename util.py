import matplotlib.pyplot as plt
import muon as mu
import numpy as np
import os
import random
import torch
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment

def set_seed(seed=9):  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def map_labels(X_labels, Y_labels):
    scores = confusion_matrix(Y_labels, X_labels)
    row, col = linear_sum_assignment(scores, maximize=True)
    labels = np.zeros_like(X_labels)
    for i in row:
        labels[Y_labels == i] = col[i]
    return labels

def evaluate(X_labels, Y_labels):
    Y_labels = map_labels(X_labels, Y_labels)
    score = (X_labels == Y_labels).sum()/X_labels.shape[0]
    return Y_labels, score

def remove_lonely(data, labels, threshold=225., n_neighbors=12):
    locs = data[:, :2]
    knn = NearestNeighbors(n_neighbors=n_neighbors).fit(locs)
    max_dist = knn.kneighbors()[0].max(-1)
    remove_idx, = np.where(max_dist > threshold)
    data = np.delete(data, remove_idx, axis=0)
    labels = np.delete(labels, remove_idx, axis=0)
    return data, labels

def read_spine_data(filename, threshold=225., n_neighbors=12, feature_key='protein', id_key='protein:celltype'):
    mdata = mu.read(filename)
    x, y = mdata['physical'].obsm['spatial'].T
    features = mdata[feature_key].X
    data = np.concatenate([x[None].T, y[None].T, features], -1)
    ids = mdata.obs[id_key]
    labels = np.unique(ids, return_inverse=True)[1]
    if threshold is not None:
        data, labels = remove_lonely(data, labels, threshold, n_neighbors)
    return data, labels

def read_anndata(filename, id_key='leiden', spatial_key='spatial', threshold=225., n_neighbors=12):
    mdata = mu.read(filename)
    x, y = mdata.obsm[spatial_key].T
    features = mdata.X
    data = np.concatenate([x[None].T, y[None].T, features], -1)
    ids = mdata.obs[id_key]
    labels = np.unique(ids, return_inverse=True)[1]
    if threshold is not None:
        data, labels = remove_lonely(data, labels, threshold, n_neighbors)
    return data, labels

def visualize_dataset(X, X_labels, size=32, show_ax=True, filename=None, colormap='tab20'):
    fig, ax = plt.subplots()
    if not show_ax:
        ax.axis('off')
    ax.scatter(X[:, 0], X[:, 1], s=size, c=X_labels, cmap=colormap)
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')

def visualize_log(log):
    x = np.arange(len(log))
    plt.plot(x, log)
    plt.show()
