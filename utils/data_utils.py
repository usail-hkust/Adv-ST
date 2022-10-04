
import numpy as np
import csv
from mmcv import mkdir_or_exist
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import os
import zipfile
import numpy as np
import torch
from six.moves import urllib
import scipy.sparse as sp

def _download_url(url, save_path):  # pragma: no cover
    with urllib.request.urlopen(url) as dl_file:
        with open(save_path, "wb") as out_file:
            out_file.write(dl_file.read())


def load_PEMS_BAY_data():

    url = "https://graphmining.ai/temporal_datasets/PEMS-BAY.zip"

    # Check if zip file is in data folder from working directory, otherwise download
    if not os.path.isfile(
            os.path.join('../data/', "PEMS-BAY.zip")
    ):  # pragma: no cover
        if not os.path.exists('../data/'):
            os.makedirs('../data/')
        _download_url(url, os.path.join('../data/', "PEMS-BAY.zip"))

    if not os.path.isfile(
            os.path.join('../data/', "pems_adj_mat.npy")
    ) or not os.path.isfile(
        os.path.join('../data/', "pems_node_values.npy")
    ):  # pragma: no cover
        with zipfile.ZipFile(
                os.path.join('../data/', "PEMS-BAY.zip"), "r"
        ) as zip_fh:
            zip_fh.extractall('../data/')

    A = np.load(os.path.join('../data/', "pems_adj_mat.npy"))
    X = np.load(os.path.join('../data/', "pems_node_values.npy")).transpose(
        (1, 2, 0)
    )
    X = X.astype(np.float32)

    # Normalise as in DCRNN paper (via Z-Score Method)
    means = np.mean(X, axis=(0, 2))

    X = X #- means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X #/ stds.reshape(1, -1, 1)

    return A, X, means, stds



def load_los_data():
    los_adj = pd.read_csv(r'data/los_adj.csv', header=None)
    adj = np.mat(los_adj)
    los_tf = pd.read_csv(r'data/los_speed.csv')
    return los_tf, adj


def load_los_locations():
    los_locations = pd.read_csv(r'data/los_locations.csv', header=None)
    return np.array(los_locations)

def load_hk_locations():
    hk_locations = pd.read_csv(r'data/hk_locations.csv', header=None)
    return np.array(hk_locations)

def load_la_locations():
    la_locations = pd.read_csv(r'data/la_locations.csv', header=None)
    return np.array(la_locations)










def load_metr_la_data():
    if (not os.path.isfile("data/adj_mat.npy")
            or not os.path.isfile("data/node_values.npy")):
        with zipfile.ZipFile("data/METR-LA.zip", 'r') as zip_ref:
            zip_ref.extractall("data/")

    A = np.load("data/adj_mat.npy")
    X = np.load("data/node_values.npy").transpose((1, 2, 0))
    X = X.astype(np.float32)

    """
    for i in range(len(X)):
        print('X ', X[i,:,:].shape)
        print('X min', np.min(X[i,:,:],1).shape)
        print('X max', np.max(X[i, :, :], 1))
        print('initial:', (X[i,:,:] - np.min(X[i,:,:],1).reshape(2,1))/(np.max(X[i, :, :], 1)-np.min(X[i,:,:],1) ).reshape(2,1))
        X[i, :, :] = (X[i, :, :] - np.min(X[i, :, :], 1).reshape(2, 1)) / (np.max(X[i, :, :], 1) - np.min(X[i, :, :], 1)).reshape(2,1)
    """
    # we use max normalization method to normalize all data into [0,1]

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))

    mean = np.mean(X)
    #X = X - means.reshape(1, -1, 1)

    stds = np.std(X, axis=(0, 2))
    #X = X / stds.reshape(1, -1, 1)


    return A, X, means, stds

def z_inverse_metla(predict, target, means, stds):
    # metr la data only predict dim 0 (feature)
    predict = predict*stds[0] + means[0]
    target = target*stds[0] + means[0]
    return predict, target

def mae(pred, true):
    MAE = np.mean(np.absolute(pred - true))
    return MAE
def rmse(pred, true):

    RMSE = np.sqrt(np.mean(np.square(pred-true)))
    return RMSE
def mape(pred, true):
    y_true, y_pred = np.array(true), np.array(pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))
#Root Relative Squared Error

def rrse(pred, true):
    mean = true.mean()
    return np.divide(np.sqrt(np.sum((pred-true) ** 2)), np.sqrt(np.sum((true-mean) ** 2)))

def All_Metrics(pred, true):
    # np.array
    assert type(pred) == type(true)
    MAE = mae(pred, true)
    RMSE = rmse(pred, true)
    RRSE = rrse(pred, true)

    return MAE, RMSE, RRSE

def local_mae(pred, true):
    local_MAE = np.mean(np.absolute(pred - true))

    return local_MAE
def local_rmse(pred, true):

    RMSE = np.sqrt(np.mean(np.square(pred-true)) )
    return RMSE

def All_Local_Metrics(pred, true):
    # np.array
    assert type(pred) == type(true)
    MAE = local_mae(pred, true)
    RMSE = local_rmse(pred, true)


    return MAE, RMSE

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples

    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input])#.transpose((1,0, 2))


        target.append(X[:, 0, i + num_timesteps_input: j])
        #print('======================preparing data sets======================')
        # features.append(X[:, :, i: i + num_timesteps_input].transpose((0, 2, 1)))
        #target.append(X[:, :, i + num_timesteps_input: j].transpose((0, 2, 1)))
        #print('features shape', np.array(features).shape)
    #print('Finished!!!')
    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))

def generate_semi_dataset(X, num_timesteps):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node data divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node current target for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
        - Node future target for the future samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (3 * num_timesteps)) for i
               in range(X.shape[2] - (
                3 * num_timesteps) + 1)]

    # Save samples

    current_data, future_data, current_targets, future_targets = [], [], [], []
    for i, j in indices:
        current_data.append(
            X[:, 0, i: i + num_timesteps])#.transpose((1,0, 2))


        current_targets.append(X[:, 0, i + num_timesteps: i + 2 * num_timesteps])
        future_targets.append(X[:, 0, i + 2 * num_timesteps: j])
        #print('======================preparing data sets======================')
        # features.append(X[:, :, i: i + num_timesteps_input].transpose((0, 2, 1)))
        #target.append(X[:, :, i + num_timesteps_input: j].transpose((0, 2, 1)))
        #print('features shape', np.array(features).shape)
    #print('Finished!!!')
    return torch.from_numpy(np.array(current_data)), \
           torch.from_numpy(np.array(current_targets)), torch.from_numpy(np.array(future_targets))









def save_csv(name, save_list, root='./data_ana', msg=True, devide=True):
    mkdir_or_exist(root)
    name = os.path.join(root, name)
    one_line = []
    for save_line in save_list:
        assert isinstance(save_line, list)
        one_line.extend(save_line)
        if devide:
            one_line.append(' ')
    with open(name, 'a+', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(one_line)
    if msg:
        print("Date written to {}".format(name))



if __name__ == '__main__':
    header = ['model_dir', 'nat_acc', 'rob_acc',' ']
    class_no = np.arange(100).tolist()
    header.extend(class_no)
    header.append(' ')
    header.extend(class_no)
    save_csv('./CIFAR100_all_results.csv',  header)
