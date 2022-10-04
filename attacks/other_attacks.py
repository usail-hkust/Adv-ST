from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import  random
import networkx as nx

def attack_set_by_degree(adj, attack_nodes):
    G = nx.from_numpy_matrix(adj)
    D = G.degree()
    Degree = np.zeros(adj.shape[0])
    for i in range(adj.shape[0]):
        Degree[i] = D[i]
    # print(Degree)
    Dsort = Degree.argsort()[::-1]
    l = Dsort
    chosen_nodes = [l[i] for i in range(attack_nodes)]
    return chosen_nodes

def attack_set_by_pagerank(adj, attack_nodes):
    G = nx.from_numpy_matrix(adj)
    result = nx.pagerank(G)
    d_order = sorted(result.items(), key=lambda x: x[1], reverse=True)
    l = [x[0] for x in d_order]  # The sequence produced by pagerank algorithm
    chosen_nodes = [l[i] for i in range(attack_nodes)]
    return chosen_nodes
def attack_set_by_betweenness(adj, attack_nodes):
    G = nx.from_numpy_matrix(adj)
    result = nx.betweenness_centrality(G)
    # print(result)
    d_order = sorted(result.items(), key=lambda x: x[1], reverse=True)
    l = [x[0] for x in d_order]
    chosen_nodes = [l[i] for i in range(attack_nodes)]
    return chosen_nodes

def batch_saliency_map(input_grads):

    input_grads = input_grads.mean(dim=0)
    node_saliency_map = []
    for n in range(input_grads.shape[0]): # nth node
        node_grads = input_grads[n,:]
        node_saliency = torch.norm(F.relu(node_grads)).item()
        node_saliency_map.append(node_saliency)
    sorted_id = sorted(range(len(node_saliency_map)), key=lambda k: node_saliency_map[k], reverse=True)
    return node_saliency_map, sorted_id

def attack_set_by_saliency_map(input_grads, attack_nodes):
    node_saliency_map, sorted_id = batch_saliency_map(input_grads)
    chosen_nodes = [sorted_id[i] for i in range(attack_nodes)]
    return  chosen_nodes









def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor







def _ST_fgsm_whitebox(model,
                  X,
                  y,
                  A_wave,
                  edges,
                  edge_weights,
                  K,
                  epsilon,
                  Random,
                  find_type,
                  **kwargs):

    #X: Input data of shape (batch_size, num_nodes,
    # num_features=in_channels, num_timesteps,).

    X_fgsm = Variable(X.data, requires_grad=True)


    # first choose the nodes: randomly choose K nodes from total nodes
    batch_size_x, num_nodes, num_features, steps_length = X.size()
    ones_mat = torch.ones(1,1, num_features, steps_length).cuda() # [1,1 ,num of channel, number of time length]

    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A_wave.cpu().detach().numpy(), K)
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 1
        step_size = epsilon
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        index = attack_set_by_saliency_map(inputs_grad, K)
    else:
        raise  NameError



    chosen_attack_nodes = torch.zeros_like(X)
    chosen_attack_nodes[:,index,:,:] = ones_mat

    opt = optim.SGD([X_fgsm], lr=1e-3)
    opt.zero_grad()

    with torch.enable_grad():
        loss = nn.MSELoss()(model(X_fgsm,A_wave, edges, edge_weights), y)
    loss.backward()
    # how to clamp

    # second clamp the value according to  the neighbourhood value [min, max]
    # define the epsilon: stds: parameter free
    #X_fgsm = X_fgsm.data + epsilon * chosen_attack_nodes * X_fgsm.grad.data.sign()
    #print('X_fgsm', X_fgsm.shape)
    X_fgsm = Variable(torch.clamp(X_fgsm.data + epsilon * chosen_attack_nodes * X_fgsm.grad.data.sign(), 0.0, 1.0), requires_grad=True)

    # print('err fgsm (white-box): ', err_pgd)
    return X, X_fgsm, index

def _ST_fgsm_semibox(model,
                  X,
                  A_wave,
                  edges,
                  edge_weights,
                  K,
                  epsilon,
                  Random,
                  find_type,
                  output_len,
                  transform_ground_truth='no-linear',
                  **kwargs):

    #X: Input data of shape (batch_size, num_nodes,
    # num_features=in_channels, num_timesteps,).

    X_fgsm = Variable(X.data, requires_grad=True)
    # first choose the nodes: randomly choose K nodes from total nodes
    batch_size_x, num_nodes, num_features, steps_length = X.size()
    ones_mat = torch.ones(1,1, num_features, steps_length).cuda() # [1,1 ,num of channel, number of time length]

    if transform_ground_truth == 'linear':
        if num_features > 1:
            y = Variable(X.data[:, :,0,:], requires_grad=True)
        else:
            y = Variable(X.data, requires_grad=True).squeeze(2)

        if output_len < steps_length:

            y = Variable(y.data[:,:,:output_len], requires_grad=True)
            y = y + torch.FloatTensor(*y.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
        else:
            y = y + torch.FloatTensor(*y.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
    elif transform_ground_truth == 'no-linear':
        y_bar = model(X_fgsm,A_wave, edges, edge_weights)
        y = y_bar + torch.FloatTensor(*y_bar.shape).uniform_(-epsilon / 10 , epsilon / 10 ).cuda()
    else:
        raise  NameError


    y = Variable(torch.clamp(y, 0, 1.0), requires_grad=True)








    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A_wave.cpu().detach().numpy(), K)
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 1
        step_size = epsilon
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        index = attack_set_by_saliency_map(inputs_grad, K)
    else:
        raise  NameError



    chosen_attack_nodes = torch.zeros_like(X)
    chosen_attack_nodes[:,index,:,:] = ones_mat

    opt = optim.SGD([X_fgsm], lr=1e-3)
    opt.zero_grad()

    with torch.enable_grad():
        loss = nn.MSELoss()(model(X_fgsm,A_wave, edges, edge_weights), y)
    loss.backward()




    # second clamp the value according to  the neighbourhood value [min, max]

    eta = epsilon * chosen_attack_nodes * X_fgsm.grad.data.sign()
    X_fgsm = Variable(torch.clamp(X_fgsm.data + eta, 0.0, 1.0), requires_grad=True)


    return eta, X_fgsm, index







def _ST_pgd_whitebox(model,
                  X,
                  y,
                  A_wave,
                  edges,
                  edge_weights,
                  K,
                  epsilon,
                  num_steps,
                  Random,
                  step_size,
                  find_type,
                  **kwargs):

    X_pgd = Variable(X.data, requires_grad=True)


    # first choose the nodes: randomly choose K nodes from total nodes
    batch_size_x, num_nodes, num_features, steps_length = X.size()
    ones_mat = torch.ones(1, 1, num_features, steps_length).cuda()  # [1,1 ,num of channel, number of time length]


    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A_wave.cpu().detach().numpy(), K)
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 5
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        index = attack_set_by_saliency_map(inputs_grad, K)
    else:
        raise  NameError

    chosen_attack_nodes = torch.zeros_like(X)
    chosen_attack_nodes[:, index, :, :] = ones_mat
    if Random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon/10, epsilon/10).cuda()
        X_pgd = Variable(X_pgd.data + chosen_attack_nodes * random_noise, requires_grad=True)




    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.MSELoss()(model(X_pgd,A_wave, edges, edge_weights), y)
        loss.backward()



        # second clamp the value according to  the neighbourhood value [min, max]
        # define the epsilon: stds: parameter free
        #X_fgsm = X_fgsm.data + epsilon * chosen_attack_nodes * X_fgsm.grad.data.sign()


        eta = step_size * chosen_attack_nodes * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)



    return X, X_pgd, index




def _pgd_whitebox(model,
                  X,
                  y,
                  A_wave,
                  edges,
                  edge_weights,
                  K,
                  epsilon,
                  num_steps,
                  Random,
                  step_size,
                  find_type,
                  **kwargs):

    X_pgd = Variable(X.data, requires_grad=True)


    # first choose the nodes: randomly choose K nodes from total nodes
    batch_size_x, num_nodes, num_features, steps_length = X.size()
    ones_mat = torch.ones(1, 1, num_features, steps_length).cuda()  # [1,1 ,num of channel, number of time length]


    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A_wave.cpu().detach().numpy(), K)
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 5
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        index = attack_set_by_saliency_map(inputs_grad, K)
    else:
        raise  NameError

    chosen_attack_nodes = torch.zeros_like(X)
    chosen_attack_nodes[:, index, :, :] = ones_mat
    if Random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon/10, epsilon/10).cuda()
        X_pgd = Variable(X_pgd.data + chosen_attack_nodes * random_noise, requires_grad=True)




    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.MSELoss()(model(X_pgd,A_wave, edges, edge_weights), y)
        loss.backward()



        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon) * chosen_attack_nodes
    X_pgd = Variable(X.data + eta, requires_grad=True)
    X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    return X, X_pgd, index









def _ST_pgd_semibox(model,
                    X,
                    A_wave,
                    edges,
                    edge_weights,
                    K,
                    epsilon,
                    num_steps,
                    Random,
                    step_size,
                    find_type,
                    output_len,
                    transform_ground_truth='no-linear', **kwargs):

    X_pgd = Variable(X.data, requires_grad=True)


    # first choose the nodes: randomly choose K nodes from total nodes
    batch_size_x, num_nodes, num_features, steps_length = X.size()
    ones_mat = torch.ones(1, 1, num_features, steps_length).cuda()  # [1,1 ,num of channel, number of time length]
    if transform_ground_truth == 'linear':
        if num_features > 1:
            y = Variable(X.data[:, :,0,:], requires_grad=True)
        else:
            y = Variable(X.data, requires_grad=True).squeeze(2)

        if output_len < steps_length:

            y = Variable(y.data[:,:,:output_len], requires_grad=True)
            y = y + torch.FloatTensor(*y.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
        else:
            y = y + torch.FloatTensor(*y.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
    elif transform_ground_truth == 'no-linear':
        y_bar = model(X_pgd,A_wave, edges, edge_weights)
        y = y_bar + torch.FloatTensor(*y_bar.shape).uniform_(-epsilon / 10, epsilon / 10 ).cuda()
    else:
        raise  NameError


    y = Variable(torch.clamp(y, 0, 1.0), requires_grad=True)


    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A_wave.cpu().detach().numpy(), K)
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 1
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        index = attack_set_by_saliency_map(inputs_grad, K)
    else:
        raise  NameError

    chosen_attack_nodes = torch.zeros_like(X)
    chosen_attack_nodes[:, index, :, :] = ones_mat
    if Random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon/10, epsilon/10).cuda()
        X_pgd = Variable(X_pgd.data + chosen_attack_nodes * random_noise, requires_grad=True)




    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.MSELoss()(model(X_pgd,A_wave, edges, edge_weights), y)
        loss.backward()





        eta = step_size * chosen_attack_nodes * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)



    return eta, X_pgd, index

def _pgd_semibox(model,
                    X,
                    A_wave,
                    edges,
                    edge_weights,
                    K,
                    epsilon,
                    num_steps,
                    Random,
                    step_size,
                    find_type,
                    output_len,
                    transform_ground_truth='no-linear', **kwargs):

    X_pgd = Variable(X.data, requires_grad=True)


    # first choose the nodes: randomly choose K nodes from total nodes
    batch_size_x, num_nodes, num_features, steps_length = X.size()
    ones_mat = torch.ones(1, 1, num_features, steps_length).cuda()  # [1,1 ,num of channel, number of time length]
    if transform_ground_truth == 'linear':
        if num_features > 1:
            y = Variable(X.data[:, :,0,:], requires_grad=True)
        else:
            y = Variable(X.data, requires_grad=True).squeeze(2)

        if output_len < steps_length:

            y = Variable(y.data[:,:,:output_len], requires_grad=True)
            y = y + torch.FloatTensor(*y.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
        else:
            y = y + torch.FloatTensor(*y.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
    elif transform_ground_truth == 'no-linear':
        y_bar = model(X_pgd,A_wave, edges, edge_weights)
        y = y_bar + torch.FloatTensor(*y_bar.shape).uniform_(-epsilon / 10, epsilon / 10 ).cuda()
    else:
        raise  NameError


    y = Variable(torch.clamp(y, 0, 1.0), requires_grad=True)


    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A_wave.cpu().detach().numpy(), K)
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 1
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        index = attack_set_by_saliency_map(inputs_grad, K)
    else:
        raise  NameError

    chosen_attack_nodes = torch.zeros_like(X)
    chosen_attack_nodes[:, index, :, :] = ones_mat
    if Random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon/10, epsilon/10).cuda()
        X_pgd = Variable(X_pgd.data + chosen_attack_nodes * random_noise, requires_grad=True)




    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.MSELoss()(model(X_pgd,A_wave, edges, edge_weights), y)
        loss.backward()


        eta = step_size  * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon) * chosen_attack_nodes
    X_pgd = Variable(X.data + eta, requires_grad=True)
    X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)



    return eta, X_pgd, index




def _uniformnoise_whitebox(model,
                  X,
                  y,
                  A_wave,
                  edges,
                  edge_weights,
                  K,
                  epsilon,
                  Random,
                  step_size,
                  find_type,
                  **kwargs):

    X_randnoise = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_randnoise.shape).uniform_(-epsilon, epsilon).cuda()



    batch_size_x, num_nodes, num_features, steps_length = X.size()
    ones_mat = torch.ones(1, 1, num_features, steps_length).cuda()  # [1,1 ,num of channel, number of time length]


    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A_wave.cpu().detach().numpy(), K)
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 5
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        index = attack_set_by_saliency_map(inputs_grad, K)
    else:
        raise  NameError

    chosen_attack_nodes = torch.zeros_like(X)
    chosen_attack_nodes[:, index, :, :] = ones_mat






    eta =  chosen_attack_nodes * random_noise
    X_randnoise = Variable(X_randnoise.data + eta, requires_grad=True)
    eta = torch.clamp(X_randnoise.data - X.data, -epsilon, epsilon)
    X_randnoise = Variable(X.data + eta, requires_grad=True)
    X_randnoise = Variable(torch.clamp(X_randnoise, 0, 1.0), requires_grad=True)




    return eta, X_randnoise, index

def _normalnoise_whitebox(model,
                  X,
                  y,
                  A_wave,
                  edges,
                  edge_weights,
                  K,
                  epsilon,
                  Random,
                  step_size,
                  find_type,
                  **kwargs):

    X_randnoise = Variable(X.data, requires_grad=True)
    random_noise = epsilon * torch.normal(mean = 0, std= 1, size=X_randnoise.shape).cuda()
    #random_noise = torch.FloatTensor(*X_randnoise.shape).uniform_(-epsilon, epsilon).cuda()


    # first choose the nodes: randomly choose K nodes from total nodes
    batch_size_x, num_nodes, num_features, steps_length = X.size()
    ones_mat = torch.ones(1, 1, num_features, steps_length).cuda()  # [1,1 ,num of channel, number of time length]


    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A_wave.cpu().detach().numpy(), K)
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 5
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        index = attack_set_by_saliency_map(inputs_grad, K)
    else:
        raise  NameError

    chosen_attack_nodes = torch.zeros_like(X)
    chosen_attack_nodes[:, index, :, :] = ones_mat






    eta =  chosen_attack_nodes * random_noise
    X_randnoise = Variable(X_randnoise.data + eta, requires_grad=True)
    eta = torch.clamp(X_randnoise.data - X.data, -epsilon, epsilon)
    X_randnoise = Variable(X.data + eta, requires_grad=True)
    X_randnoise = Variable(torch.clamp(X_randnoise, 0, 1.0), requires_grad=True)



    # print('err pgd (white-box): ', err_pgd)
    return eta, X_randnoise, index


















def _ST_mim_whitebox(model,
                  X,
                  y,
                  A_wave,
                  edges,
                  edge_weights,
                  K,
                  epsilon,
                  num_steps,
                  Random,
                  step_size,
                  find_type,
                  decay_factor=1.0,
                  **kwargs):

    X_pgd = Variable(X.data, requires_grad=True)


    # first choose the nodes: randomly choose K nodes from total nodes
    batch_size_x, num_nodes, num_features, steps_length = X.size()
    ones_mat = torch.ones(1, 1, num_features, steps_length).cuda()  # [1,1 ,num of channel, number of time length]


    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A_wave.cpu().detach().numpy(), K)
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 5
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        index = attack_set_by_saliency_map(inputs_grad, K)
    else:
        raise  NameError

    chosen_attack_nodes = torch.zeros_like(X)
    chosen_attack_nodes[:, index, :, :] = ones_mat
    if Random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon/10, epsilon/10).cuda()
        X_pgd = Variable(X_pgd.data + chosen_attack_nodes * random_noise, requires_grad=True)




    previous_grad = torch.zeros_like(X.data)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.MSELoss()(model(X_pgd,A_wave, edges, edge_weights), y)
        loss.backward()
        grad = X_pgd.grad.data / torch.mean(torch.abs(X_pgd.grad.data), [1,2,3], keepdim=True)
        previous_grad = decay_factor * previous_grad + grad
        X_pgd = Variable(X_pgd.data + step_size * chosen_attack_nodes * previous_grad.sign(), requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)






    return X, X_pgd, index


def _mim_whitebox(model,
                  X,
                  y,
                  A_wave,
                  edges,
                  edge_weights,
                  K,
                  epsilon,
                  num_steps,
                  Random,
                  step_size,
                  find_type,
                  decay_factor=1.0,
                  **kwargs):

    X_pgd = Variable(X.data, requires_grad=True)


    # first choose the nodes: randomly choose K nodes from total nodes
    batch_size_x, num_nodes, num_features, steps_length = X.size()
    ones_mat = torch.ones(1, 1, num_features, steps_length).cuda()  # [1,1 ,num of channel, number of time length]


    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A_wave.cpu().detach().numpy(), K)
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 5
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        index = attack_set_by_saliency_map(inputs_grad, K)
    else:
        raise  NameError

    chosen_attack_nodes = torch.zeros_like(X)
    chosen_attack_nodes[:, index, :, :] = ones_mat
    if Random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon/10, epsilon/10).cuda()
        X_pgd = Variable(X_pgd.data + chosen_attack_nodes * random_noise, requires_grad=True)




    previous_grad = torch.zeros_like(X.data)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.MSELoss()(model(X_pgd,A_wave, edges, edge_weights), y)
        loss.backward()
        grad = X_pgd.grad.data / torch.mean(torch.abs(X_pgd.grad.data), [1,2,3], keepdim=True)
        previous_grad = decay_factor * previous_grad + grad
        X_pgd = Variable(X_pgd.data + step_size  * previous_grad.sign(), requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon) * chosen_attack_nodes
    X_pgd = Variable(X.data + eta, requires_grad=True)
    X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)



    return X, X_pgd, index






def _ST_mim_semibox(model,
                  X,
                  A_wave,
                  edges,
                  edge_weights,
                  K,
                  epsilon,
                  num_steps,
                  Random,
                  step_size,
                  find_type,
                  decay_factor=1.0,
                  output_len = 12,
                  transform_ground_truth='no-linear',
                  **kwargs):

    X_pgd = Variable(X.data, requires_grad=True)


    # first choose the nodes: randomly choose K nodes from total nodes
    batch_size_x, num_nodes, num_features, steps_length = X.size()
    ones_mat = torch.ones(1, 1, num_features, steps_length).cuda()  # [1,1 ,num of channel, number of time length]
    if transform_ground_truth == 'linear':
        if num_features > 1:
            y = Variable(X.data[:, :,0,:], requires_grad=True)
        else:
            y = Variable(X.data, requires_grad=True).squeeze(2)

        if output_len < steps_length:

            y = Variable(y.data[:,:,:output_len], requires_grad=True)
            y = y + torch.FloatTensor(*y.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
        else:
            y = y + torch.FloatTensor(*y.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
    elif transform_ground_truth == 'no-linear':
        y_bar = model(X_pgd,A_wave, edges, edge_weights)
        y = y_bar + torch.FloatTensor(*y_bar.shape).uniform_(-epsilon/ 10 , epsilon/ 10 ).cuda()
    else:
        raise  NameError






    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A_wave.cpu().detach().numpy(), K)
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 5
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward(retain_graph=True)

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        index = attack_set_by_saliency_map(inputs_grad, K)
    else:
        raise  NameError

    chosen_attack_nodes = torch.zeros_like(X)
    chosen_attack_nodes[:, index, :, :] = ones_mat
    if Random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon/10, epsilon/10).cuda()
        X_pgd = Variable(X_pgd.data + chosen_attack_nodes * random_noise, requires_grad=True)




    previous_grad = torch.zeros_like(X.data)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.MSELoss()(model(X_pgd,A_wave, edges, edge_weights), y)
        loss.backward(retain_graph=True)
        grad = X_pgd.grad.data / torch.mean(torch.abs(X_pgd.grad.data), [1,2,3], keepdim=True)
        previous_grad = decay_factor * previous_grad + grad
        X_pgd = Variable(X_pgd.data + step_size * chosen_attack_nodes * previous_grad.sign(), requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)



    return eta, X_pgd, index


def _mim_semibox(model,
                  X,
                  A_wave,
                  edges,
                  edge_weights,
                  K,
                  epsilon,
                  num_steps,
                  Random,
                  step_size,
                  find_type,
                  decay_factor=1.0,
                  output_len = 12,
                  transform_ground_truth='no-linear',
                  **kwargs):

    X_pgd = Variable(X.data, requires_grad=True)


    # first choose the nodes: randomly choose K nodes from total nodes
    batch_size_x, num_nodes, num_features, steps_length = X.size()
    ones_mat = torch.ones(1, 1, num_features, steps_length).cuda()  # [1,1 ,num of channel, number of time length]
    if transform_ground_truth == 'linear':
        if num_features > 1:
            y = Variable(X.data[:, :,0,:], requires_grad=True)
        else:
            y = Variable(X.data, requires_grad=True).squeeze(2)

        if output_len < steps_length:

            y = Variable(y.data[:,:,:output_len], requires_grad=True)
            y = y + torch.FloatTensor(*y.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
        else:
            y = y + torch.FloatTensor(*y.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
    elif transform_ground_truth == 'no-linear':
        y_bar = model(X_pgd,A_wave, edges, edge_weights)
        y = y_bar + torch.FloatTensor(*y_bar.shape).uniform_(-epsilon/ 10 , epsilon/ 10 ).cuda()
    else:
        raise  NameError






    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A_wave.cpu().detach().numpy(), K)
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 5
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward(retain_graph=True)

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        index = attack_set_by_saliency_map(inputs_grad, K)
    else:
        raise  NameError

    chosen_attack_nodes = torch.zeros_like(X)
    chosen_attack_nodes[:, index, :, :] = ones_mat
    if Random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon/10, epsilon/10).cuda()
        X_pgd = Variable(X_pgd.data + chosen_attack_nodes * random_noise, requires_grad=True)




    previous_grad = torch.zeros_like(X.data)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.MSELoss()(model(X_pgd,A_wave, edges, edge_weights), y)
        loss.backward(retain_graph=True)
        grad = X_pgd.grad.data / torch.mean(torch.abs(X_pgd.grad.data), [1,2,3], keepdim=True)
        previous_grad = decay_factor * previous_grad + grad
        X_pgd = Variable(X_pgd.data + step_size * previous_grad.sign(), requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)


    eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon) * chosen_attack_nodes
    X_pgd = Variable(X.data + eta, requires_grad=True)
    X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    # print('err pgd (white-box): ', err_pgd)
    return eta, X_pgd, index









