import numpy as np
import pandas as pd
from scipy.stats import skewnorm

SIZE = 30
GGBLOCKS = [[[[15, 20], [0, 5]], [[25, 30], [0, 5]], [[10, 20], [5, 10]],
            [[25, 30], [5, 10]], [[5, 15], [10, 15]], [[0, 5], [15, 20]],
            [[10, 15], [15, 20]], [[20, 25], [20, 25]], [[0, 5], [25, 30]],
            [[10, 15], [25, 30]]],
            [[[0, 15], [0, 5]], [[0, 10], [5, 10]], [[0, 5], [10, 15]]],
            [[[20, 25], [0, 10]], [[15, 30], [10, 15]]],
            [[[15, 30], [15, 20]], [[15, 20], [20, 30]], [[25, 30], [20, 30]],
            [[20, 25], [25, 30]]],
            [[[5, 10], [15, 30]], [[0, 5], [20, 25]], [[10, 15], [20, 25]]]]
CHBLOCKS = [[[[0, 10], [0, 10]], [[20, 30], [0, 10]], [[10, 20], [10, 20]],
            [[0, 10], [20, 30]], [[20, 30], [20, 30]]], 
            [[[10, 20], [0, 10]], [[0, 10], [10, 20]], [[20, 30], [10, 20]],
            [[10, 20], [20, 30]]]]

def generate_means(n_genes, n_informative, n_topics, min_=10, max_=30):
    init_means = np.random.randint(min_, max_, n_genes)
    variances = np.random.randint(1, min_/5, n_genes)
    mean_arr = np.empty((n_topics, n_genes))
    for i in range(n_topics):
        mean_arr[i, :] = init_means
    for i in range(1, n_topics):
        for j in range(n_informative):
            gene_var = variances[j]
            while True:
                if np.random.rand() > 0.5:
                    new_mean = init_means[j] + np.random.randint(2, 10) * gene_var
                else:
                    new_mean = init_means[j] - np.random.randint(2, 10) * gene_var 
                if new_mean - 5 * variances[j] > 0:
                    break
            mean_arr[i, j] = new_mean
    return mean_arr, variances

def generate_from_array(n_cells, mean_arr, variances):
    n_genes = mean_arr.shape[1]
    data = np.empty((sum(n_cells), n_genes))
    labels = np.empty(sum(n_cells), dtype=int)
    cell_count = 0
    for i in range(0, len(n_cells)):
        cells_in_class = n_cells[i]
        class_data = np.random.normal(mean_arr[i, :], variances, (cells_in_class, n_genes))
        data[cell_count:cell_count+cells_in_class, :] = class_data
        labels[cell_count:cell_count+cells_in_class] = i
        cell_count += cells_in_class
    return data, labels

def generate_data(n_genes, n_informative, n_cells=900, n_topics=2, means=None, variances=None):
    if means is None:
        means, variances = generate_means(n_genes, n_informative, n_topics)
    if isinstance(n_cells, np.ndarray):
        data, labels = generate_from_array(n_cells, means, variances)
    else:
        data, labels = generate_from_array(np.repeat(n_cells, n_topics), means, variances)
    return data, labels

def scale(X, min_, max_, mirror=False):
    X = X + (X.min() if X.min() > 0 else -X.min())
    X = X / X.max()
    if mirror:
        X = 1. - X
    X = X * (max_ - min_) + min_
    return X

def rand_locs(n_locs, x_min, x_max, y_min, y_max, dist='uniform', skew=100, mirror=False):
    if dist == 'uniform':
        x_locs = np.random.randint(x_min, x_max, size=(n_locs, 1))
    elif dist == 'skewnorm':
        x_locs = scale(skewnorm(skew).rvs((n_locs, 1)), x_min, x_max, mirror)
    else:
        raise NotImplementedError(f'Distribution "{dist}" not supported.')
    y_locs = np.random.randint(y_min, y_max, size=(n_locs, 1))
    locs = np.concatenate([x_locs, y_locs], axis=1)
    return locs

def generate_dist(n_genes, n_informative, n_cells=900, means=None, variances=None, mode='split', mixed=False, x_max=30, y_max=30):
    if means is not None and variances is not None:
        assert means.shape == (2, n_genes), "Expected argument 'means' of shape (2, n_genes)."
        assert variances.shape == (n_genes,), "Expected argument 'variances' of shape (n_genes,)"
    X, X_labels = generate_data(n_genes, n_informative, n_cells, 2, means, variances)
    class0_idx = np.where(X_labels==0)[0]
    class1_idx = np.where(X_labels==1)[0]
    n_class0 = class0_idx.shape[0]
    n_class1 = class1_idx.shape[0]
    if mixed:
        n_mixed0 = int(n_class0 / 5)
        n_mixed1 = int(n_class1 / 5)
        M_labels = X_labels.copy()
        X_labels[class0_idx[n_mixed0:]] = 0
        X_labels[class0_idx[:n_mixed0]] = 1
        X_labels[class1_idx[n_mixed1:]] = 1
        X_labels[class1_idx[:n_mixed1]] = 0
        X[class0_idx[:n_mixed0], :2] = rand_locs(n_mixed0, x_max/2, x_max, 0, y_max, 'uniform')
        X[class0_idx[n_mixed0:], :2] = rand_locs(n_class0-n_mixed0, 0, x_max/2, 0, y_max, 'uniform')
        X[class1_idx[:n_mixed1], :2] = rand_locs(n_mixed1, 0, x_max/2, 0, y_max, 'uniform')
        X[class1_idx[n_mixed1:], :2] = rand_locs(n_class1-n_mixed1, x_max/2, x_max, 0, y_max, 'uniform' )
        return X, X_labels, M_labels
    elif mode == 'split':
        X[class0_idx, :2] = rand_locs(n_class0, 0, x_max/2, 0, y_max, 'uniform')
        X[class1_idx, :2] = rand_locs(n_class1, x_max/2, x_max, 0, y_max, 'uniform')
    elif mode == 'gradient':
        X[class0_idx, :2] = rand_locs(n_class0, 0, x_max, 0, y_max, 'skewnorm')
        X[class1_idx, :2] = rand_locs(n_class1, 0, x_max, 0, y_max, 'skewnorm', mirror=True)
    else:
        raise NotImplementedError(f'Mode "{mode}" not supported.')
    return X, X_labels

def cells_per_block(blocks):
    cell_counts = []
    for block in blocks:
        cell_count = 0
        for b in block:
            cell_count = cell_count + (b[0][1] - b[0][0]) * (b[1][1] - b[1][0])
        cell_counts.append(cell_count)
    return np.array(cell_counts)

def in_block(p, block):
    return any(p[0] in range(*b[0]) and p[1] in range(*b[1]) for b in block)

def rand_blocks(n_topics, n_cells):
    blocks = [np.array([]) for _ in range(n_topics)]
    total_size = SIZE * n_cells
    for x in range(0, total_size, 5):
        for y in range(0, total_size, 5):
            c = np.random.randint(n_topics)
            blocks[c].append(np.array([[x, x + 5], [y, y + 5]]))
    return blocks

def generate_blocks(n_genes, n_informative, n_topics=2, n_cells=1, mixed=False, blocks=None, means=None, variances=None):
    total_size = SIZE * n_cells
    if blocks is None:
        blocks = rand_blocks(n_topics, n_cells)
    blocks = blocks * n_cells
    if means is not None and variances is not None:
        assert means.shape == (len(blocks), n_genes), "Expected argument 'means' of shape (n_topics, n_genes)."
        assert variances.shape == (n_genes,), "Expected argument 'variances' of shape (n_genes,)"
    cells_per_class = cells_per_block(blocks)
    X, X_labels = generate_data(n_genes, n_informative, cells_per_class, len(blocks), means, variances)
    idx = np.zeros(len(cells_per_class), dtype=np.int32)
    for i in range(1, idx.shape[0]):
        idx[i] = idx[i - 1] + cells_per_class[i - 1]
    if mixed:
        M = rand_locs(total_size * 3, 0, total_size, 0, total_size)
        M_labels = X_labels.copy()
    for p in np.mgrid[:total_size, :total_size].reshape(2, total_size * total_size).T:
        for i, block in enumerate(blocks):
            if in_block(p, block):
                X[idx[i], 0] = p[0]
                X[idx[i], 1] = p[1]
                if mixed and any(np.array_equal(m, p) for m in M):
                    M_labels[idx[i]] = (i + np.random.randint(1, len(blocks))) % len(blocks)
                idx[i] += 1
    if mixed:  
        return X, X_labels, M_labels
    return X, X_labels

def generate_dataset(n_genes, n_informative, n_cells=900, n_topics=2, blocks=None, mode=None, mixed=False, means=None, variances=None, x_max=30, y_max=30, as_df=False):
    if mode == 'blocks' or blocks is not None:
        if mixed:
            X, X_labels, M_labels = generate_blocks(n_genes, n_informative, n_topics, n_cells, mixed, blocks, means, variances)
        else:
            X, X_labels = generate_blocks(n_genes, n_informative,  n_topics, n_cells, mixed, blocks, means, variances)
    elif mode in ('split', 'gradient'):
        if mixed:
            X, X_labels, M_labels = generate_dist(n_genes, n_informative, n_cells, means, variances, mode, mixed, x_max, y_max)
        else:
            X, X_labels = generate_dist(n_genes, n_informative, n_cells, means, variances, mode, mixed, x_max, y_max)
    else:
        X, X_labels = generate_data(n_genes, n_informative, n_cells, n_topics, means, variances)
    if as_df:
        X = pd.DataFrame(X)
        columns = ['x', 'y'] + [f'Gene_{i}' for i in range(1, n_genes - 1)]
        X = X.set_axis(columns, axis=1)
    if mixed:
        return X, X_labels, M_labels
    return X, X_labels
