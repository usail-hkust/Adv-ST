from __future__ import print_function
import logging
import os
import numpy as np

from matplotlib import pyplot as plt

import networkx as nx


from mmcv import mkdir_or_exist


def n_neighbor(G, id, n_hop):
    G = nx.from_numpy_matrix(G)
    node = [id]
    node_visited = set()
    neighbors = []

    while n_hop != 0:
        neighbors = []
        for node_id in node:
            node_visited.add(node_id)
            neighbors += [id for id in G.neighbors(node_id) if id not in node_visited]
        node = neighbors
        n_hop -= 1

        if len(node) == 0:
            return neighbors

    return list(set(neighbors))






def log_test_results(model_path, list, file_name):
    'Given list, transfer list to string, and write is to csv'
    string = ','.join(str(n) for n in list)
    path = model_path + '/test_results'
    if not os.path.isdir(path):
        os.makedirs(path + '/', )

    file_path = path + "/{}.csv".format(file_name)

    '''
    Write one line of log into screen and file.
        log_file_path: Path of log file.
        string:        String to write in log file.
    '''
    with open(file_path, 'a+') as f:
        f.write(string + '\n')
        f.flush()
    print(string)


def _norm(matrix, axis=-1):
    return np.sum(matrix ** 2, axis=axis) ** 0.5







def _savefig(name='./image.jpg', dpi=300):
    plt.savefig(name, dpi=dpi)
    print('Image saved at {}'.format(name))
    plt.close()

def _softmax(logits, dim=-1):
    exp = np.exp(logits)
    sum = np.sum(exp, axis=dim, keepdims=True)
    return exp / sum

def _onehot(labels, num_classes):
    labels = np.asarray(labels, dtype=np.uint8)
    out = np.zeros((len(labels), num_classes), dtype=np.uint8)
    idx = np.arange(len(labels))
    out[idx, labels] = 1
    return out

if __name__ == '__main__':
    pass
