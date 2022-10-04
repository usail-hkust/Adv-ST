import torch



def plain_train(model, x_natural, A_wave, edges, edge_weights,y, **kwargs):

    outs = model(x_natural, A_wave, edges, edge_weights)
    loss_criterion = torch.nn.MSELoss()
    loss = loss_criterion(outs, y)

    return loss
