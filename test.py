from __future__ import print_function
import argparse
import yaml
import os
import shutil
import torch
from mmcv import Config, mkdir_or_exist
import time

from utils.env import get_root_logger, set_default_configs, init_dist, logger_info, set_random_seed


from datasets.datasets import METRLA,PeMS
from attacks.other_attacks import  _ST_fgsm_whitebox, _ST_pgd_whitebox, _uniformnoise_whitebox,  \
     _normalnoise_whitebox, _ST_mim_whitebox, _mim_whitebox, _pgd_whitebox
from torch.utils.data import DataLoader
from models.GraphWaveNet import gwnet
from models.STAWnet import  stawnet

import  numpy as np
from  datasets.datasets import  DataLoaderX
from utils.data_utils import   All_Metrics, All_Local_Metrics,load_la_locations
from utils.statistics_tools import  log_test_results
parser = argparse.ArgumentParser(description='PyTorch ST PGD Attack Evaluation')
parser.add_argument('config',
                    default='./configs/',
                    help='path to config file')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu', default=0, type=int,
                    help='which gpu to use')
parser.add_argument('--seed', type=int, default= 24, metavar='S',
                    help='random seed (default: 24)')
parser.add_argument('--rename', '-r', action='store_true', default=False,
                    help='whether allow renaing the checkpoints parameter to match')
parser.add_argument('--from_file', '-f', action='store_true', default=False,
                    help='analysis data from file')
parser.add_argument('--eval_train_data', action='store_true', default=False,
                    help='whether eval train data')
parser.add_argument('--save_features', '-s', action='store_true', default=True,
                    help='whether save features')
parser.add_argument('--individual', action='store_true', default=False,
                    help='whether to perform individual aa')
parser.add_argument('--attacker', '-a', default='ALL', # ['ALL', 'PGD']
                    help='which attack to perform')
parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
    default='none', help='job launcher')
parser.add_argument('--device_id',  '-d', default= 2, type=int,# ['TRAIN', 'TEST']
                    help='device ID')
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()
set_random_seed(args.seed)

# settings
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
if  torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')


# set configs
with open(args.config) as cf:
    cfgs = Config(yaml.safe_load(cf))
mkdir_or_exist(cfgs.model_dir)
shutil.copyfile(args.config, os.path.join(cfgs.model_dir, "config_test.yaml"))
set_default_configs(cfgs)

# setup logger
logger = get_root_logger(cfgs.log_level, cfgs.model_dir)
logger.info("Loading config file from {}".format(args.config))
logger.info("Work_dir: {}".format(cfgs.model_dir))

# init distributed env first, since logger depends on the dist info.
if args.launcher == 'none':
    distributed = False
else:
    distributed = True
    init_dist(args.launche)


torch.cuda.set_device(args.device_id)

if cfgs.dataset == 'METRLA':


    test_data = METRLA(mode='test',
                       split_train=cfgs.split_train,
                       split_val=cfgs.split_val,
                       num_timesteps_input=cfgs.num_timesteps_input,
                       num_timesteps_output=cfgs.num_timesteps_output)
    adj = test_data.A.numpy()
    locations = load_la_locations()

elif cfgs.dataset == 'PeMS':


    test_data = PeMS(mode='test',
                     split_train=cfgs.split_train,
                     split_val=cfgs.split_val,
                     num_timesteps_input=cfgs.num_timesteps_input,
                     num_timesteps_output=cfgs.num_timesteps_output)
    adj = test_data.A.numpy()
else:
    raise  NameError




def log_test_csv(val_predict , val_target , adv_val_predict, cfgs, file_name, method):
    metric_list = []
    data_set = cfgs.dataset
    metric_list.append(data_set)
    model_name = cfgs.backbone
    metric_list.append(model_name)
    method_name = method
    metric_list.append(method_name)

    clean_MAE, clean_RMSE, clean_MAPE = All_Metrics(val_predict, val_target)
    #metric_list.append(clean_MAPE)
    adv_MAE, adv_RMSE, adv_MAPE = All_Metrics(adv_val_predict, val_target)
    #metric_list.append(adv_MAPE)
    local_adv_MAE, local_adv_RMSE = All_Local_Metrics(adv_val_predict, val_predict)



    metric_list.append(clean_MAE)
    metric_list.append(adv_MAE)
    metric_list.append(local_adv_MAE)

    metric_list.append(clean_RMSE)
    metric_list.append(adv_RMSE)
    metric_list.append(local_adv_RMSE)

    log_test_results(cfgs.model_dir, metric_list, file_name)








def batch_eval(val_target, val_predict, adv_val_predict, max_speed):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average mae.
    """
    val_predict = np.vstack(val_predict)
    val_target = np.vstack(val_target)
    adv_val_predict = np.vstack(adv_val_predict)

    val_predict = val_predict * max_speed
    adv_val_predict = adv_val_predict * max_speed
    val_target = val_target * max_speed

    clean_MAE, clean_RMSE, clean_MAPE  = All_Metrics(val_predict, val_target)
    adv_MAE, adv_RMSE, adv_MAPE = All_Metrics(adv_val_predict, val_target)

    return clean_MAE, clean_RMSE, clean_MAPE, adv_MAE, adv_RMSE, adv_MAPE

def batch_eval_local(val_predict, adv_val_predict, max_speed):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average mae.
    """
    val_predict = np.vstack(val_predict)
    adv_val_predict = np.vstack(adv_val_predict)
    val_predict = val_predict * max_speed
    adv_val_predict = adv_val_predict * max_speed



    local_adv_MAE, local_adv_RMSE = All_Local_Metrics(adv_val_predict, val_predict)

    return local_adv_MAE, local_adv_RMSE





def eval_val(cfgs, val_loader, net, A_wave, edges, edge_weights, attacker, max_speed,  find_type = 'random'):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    # with torch.no_grad():
    start = time.time()
    net.eval()
    val_predict = []
    val_target = []
    adv_val_predict = []
    samples_total = len(val_loader) * cfgs.test_batch_size
    attack_nodes = int(cfgs.test_attack_nodes * len(A_wave))
    for batch_idx, (data, target) in enumerate(val_loader):
        X_batch, y_batch = data, target
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)
        out = net(X_batch, A_wave, edges, edge_weights)
        if attacker == 'ST_FGSM':
            _, X_adv, index = _ST_fgsm_whitebox(net,
                  X_batch,
                  y_batch,
                  A_wave,
                  edges,
                  edge_weights,
                  attack_nodes,
                  cfgs.test_epsilon,
                  cfgs.random,
                  find_type
                  )
        elif attacker == 'ST_PGD':
            _, X_adv, index = _ST_pgd_whitebox(net,
                             X_batch,
                             y_batch,
                             A_wave,
                             edges,
                             edge_weights,
                             attack_nodes,
                             cfgs.test_epsilon,
                             cfgs.test_num_steps,
                             cfgs.random,
                             cfgs.test_step_size,
                             find_type)
        elif attacker == 'PGD':
            _, X_adv, index = _pgd_whitebox(net,
                             X_batch,
                             y_batch,
                             A_wave,
                             edges,
                             edge_weights,
                             attack_nodes,
                             cfgs.test_epsilon,
                             cfgs.test_num_steps,
                             cfgs.random,
                             cfgs.test_step_size,
                             find_type)
        elif attacker == 'uniform':
            _, X_adv, index = _uniformnoise_whitebox(net,
                  X_batch,
                  y_batch,
                  A_wave,
                  edges,
                  edge_weights,
                  attack_nodes,
                  cfgs.test_epsilon,
                  cfgs.random,
                  cfgs.test_step_size,
                  find_type,
                  **kwargs)

        elif attacker == 'normal':
            _, X_adv, index = _normalnoise_whitebox(net,
                  X_batch,
                  y_batch,
                  A_wave,
                  edges,
                  edge_weights,
                  attack_nodes,
                  cfgs.test_epsilon,
                  cfgs.random,
                  cfgs.test_step_size,
                  find_type,
                  **kwargs)
        elif attacker == 'ST_MIM':
            _, X_adv, index = _ST_mim_whitebox(net,
                             X_batch,
                             y_batch,
                             A_wave,
                             edges,
                             edge_weights,
                             attack_nodes,
                             cfgs.test_epsilon,
                             cfgs.test_num_steps,
                             cfgs.random,
                             cfgs.test_step_size,
                             find_type,
                             decay_factor=1)

        elif attacker == 'MIM':
            _, X_adv, index = _mim_whitebox(net,
                             X_batch,
                             y_batch,
                             A_wave,
                             edges,
                             edge_weights,
                             attack_nodes,
                             cfgs.test_epsilon,
                             cfgs.test_num_steps,
                             cfgs.random,
                             cfgs.test_step_size,
                             find_type,
                             decay_factor=1)


        else:
            raise  NameError

        adv_out = net(X_adv,A_wave, edges, edge_weights)
        val_predict.append(out.cpu().detach().numpy())
        val_target.append(y_batch.cpu().detach().numpy())
        adv_val_predict.append(adv_out.cpu().detach().numpy())

        clean_MAE, clean_RMSE, clean_RRSE, adv_MAE, adv_RMSE, adv_RRSE = batch_eval(val_target, val_predict, adv_val_predict, max_speed)
        local_adv_MAE, local_adv_RMSE = batch_eval_local(val_predict, adv_val_predict, max_speed)
        if batch_idx % cfgs.log_interval == 0:
            logger_info(logger, distributed, 'Info:  [{}/{} ({:.0f}%)]\t  MAE: {:.4f} RMSE: {:.4f} RRSE: {:.4f} Glocal: Adv MAE: {:.4f} Adv RMSE: {:.4f} Adv RRSE: {:.4f} Local: Adv MAE: {:.4f} Adv RMSE: {:.4f}  time:{:.3f}'.format(
                       batch_idx * len(data), samples_total,
                       100. * batch_idx / len(val_loader),
                       clean_MAE, clean_RMSE, clean_RRSE, adv_MAE, adv_RMSE, adv_RRSE,
                       local_adv_MAE, local_adv_RMSE,
                       time.time() - start))
        #vis_attack_nodes(index, locations, adj, cfgs.dataset, attacker,find_type, cfgs.model_dir, batch_idx)

    val_predict = np.vstack(val_predict)
    val_target = np.vstack(val_target)
    adv_val_predict = np.vstack(adv_val_predict)
    clean_MAE, clean_RMSE, clean_RRSE, adv_MAE, adv_RMSE, adv_RRSE = batch_eval(val_target, val_predict, adv_val_predict, max_speed)
    local_adv_MAE, local_adv_RMSE = batch_eval_local(val_predict, adv_val_predict, max_speed)
    logger_info(logger, distributed,
               'MAE: {:.4f} RMSE: {:.4f} RRSE: {:.4f} Global: Adv MAE: {:.4f} Adv RMSE: {:.4f} Adv RRSE: {:.4f} Local: Adv MAE: {:.4f} Adv RMSE: {:.4f} '.format(
               clean_MAE, clean_RMSE, clean_RRSE, adv_MAE, adv_RMSE, adv_RRSE,
               local_adv_MAE, local_adv_RMSE))

    return val_predict * max_speed , val_target * max_speed, adv_val_predict * max_speed

def main():
    # set up data loader
    logger.info("Building test datasets {}".format(cfgs.dataset))


    test_loader = DataLoader(test_data, batch_size=cfgs.test_batch_size, shuffle=False)
    #test_loader = DataLoaderX(test_data, batch_size=cfgs.test_batch_size, shuffle=False, num_workers=6, pin_memory=True)
    A_wave = test_data.A_wave.to(device=args.device)
    edges = test_data.edges.to(device=args.device)
    edge_weights = test_data.edge_weights.to(device=args.device)
    max_speed = test_data.max_speed

    if cfgs.backbone == 'GWNET':
        dropout = 0.3
        supports = None
        gcn_bool = True
        addaptadj = True
        aptinit = None
        nhid = 32
        model = gwnet(device, num_nodes=cfgs.num_nodes, dropout=dropout, supports=supports, gcn_bool=gcn_bool,
                      addaptadj=addaptadj, aptinit=aptinit, in_dim=cfgs.num_features, out_dim=cfgs.num_timesteps_output,
                      residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8,
                      end_channels=nhid * 16).to(device=args.device)

    elif cfgs.backbone == 'STAWNET':
        nhid = 32
        num_nodes = cfgs.num_nodes
        dropout = 0.3
        gat_bool = True
        addaptadj = True
        aptonly = False
        aptinit = None
        in_dim = cfgs.num_features
        out_dim = cfgs.num_timesteps_output
        emb_length = 16
        noapt = False

        model = stawnet(device, num_nodes, dropout, gat_bool=gat_bool, addaptadj=addaptadj,
                        aptonly=aptonly, aptinit=aptinit, in_dim=in_dim, out_dim=out_dim,
                        residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8,
                        end_channels=nhid * 16, emb_length=emb_length, noapt=noapt).to(device=args.device)
    else:
        raise NameError

    load_path = cfgs.model_path
    logger.info('Loading checkpoint from %s', load_path)
    model.load_state_dict(torch.load(load_path))
    model.eval()





    if args.attacker == 'ST-TNDS':
        #  ST PGD Attack
        # white-box attack
        header = ['dataset', 'model', 'method', 'clean_MAE','adv_MAE', 'local_adv_MAE', 'clean_RMSE',
                   'adv_RMSE',  'local_adv_RMSE']

        file_name = 'data-{}_num-nodes{}_eps{}-model-{}'.format(cfgs.dataset,cfgs.test_attack_nodes,cfgs.test_epsilon,cfgs.backbone)
        log_test_results(cfgs.model_dir, header, file_name)



        logger.info('STPGD-TNDS. -Info: num attack nodes pro: {}, outputs length: {}'.format(cfgs.test_attack_nodes, cfgs.num_timesteps_output))
        val_predict , val_target , adv_val_predict = eval_val(cfgs, test_loader, model, A_wave, edges, edge_weights, args.attacker, max_speed, find_type = 'saliency')
        log_test_csv(val_predict, val_target, adv_val_predict, cfgs, file_name, 'STPGD-TNDS')


        logger_info(logger, distributed,
                    '[Remarks] {} | End of testing, saved at {}'.format(cfgs.remark, cfgs.model_dir))


    elif args.attacker == 'ALL':
        #  ALL Attack
        # white-box attack

        header = ['dataset', 'model', 'method', 'test_batch_size','clean_MAE','adv_MAE', 'local_adv_MAE', 'clean_RMSE',
                   'adv_RMSE',  'local_adv_RMSE']

        file_name = 'white_box-data-{}_num-nodes{}_eps{}-model-{}'.format(cfgs.dataset,cfgs.test_attack_nodes,cfgs.test_epsilon,cfgs.backbone)
        log_test_results(cfgs.model_dir, header, file_name)

        logger.info('STPGD-TNDS white-box attack. -Info: num attack nodes pro: {}, outputs length: {}'.format(
            cfgs.test_attack_nodes, cfgs.num_timesteps_output))
        val_predict, val_target, adv_val_predict = eval_val(cfgs, test_loader, model, A_wave, edges, edge_weights,
                                                            'ST_PGD', max_speed, find_type='saliency')
        log_test_csv(val_predict, val_target, adv_val_predict, cfgs, file_name,
                     'STPGD-TNDS white-box attack,{}'.format(cfgs.test_batch_size))

        logger.info('STMIM-TNDS white-box attack. -Info: num attack nodes pro: {}'.format(cfgs.test_attack_nodes))
        val_predict, val_target, adv_val_predict = eval_val(cfgs, test_loader, model, A_wave, edges, edge_weights,
                                                            'ST_MIM', max_speed, find_type='saliency')
        log_test_csv(val_predict, val_target, adv_val_predict, cfgs, file_name,
                     'STMIM-TNDS white-box attack,{}'.format(cfgs.test_batch_size))

        logger.info('PGD-Random  white-box attack. -Info: num attack nodes pro: {}, outputs length: {}'.format(cfgs.test_attack_nodes, cfgs.num_timesteps_output))
        val_predict , val_target , adv_val_predict = eval_val(cfgs, test_loader, model, A_wave, edges, edge_weights, 'PGD', max_speed, find_type = 'random')
        log_test_csv(val_predict, val_target, adv_val_predict, cfgs, file_name, 'PGD-Random  white-box attack,{}'.format(cfgs.test_batch_size))

        logger.info('PGD-PR white-box attack. -Info: num attack nodes pro: {}, outputs length: {}'.format(cfgs.test_attack_nodes, cfgs.num_timesteps_output))
        val_predict , val_target , adv_val_predict = eval_val(cfgs, test_loader, model, A_wave, edges, edge_weights, 'PGD', max_speed, find_type = 'pagerank')
        log_test_csv(val_predict, val_target, adv_val_predict, cfgs, file_name, 'PGD-PR white-box attack,{}'.format(cfgs.test_batch_size))


        logger.info('PGD-Centrality white-box attack. -Info: num attack nodes pro: {}, outputs length: {}'.format(cfgs.test_attack_nodes, cfgs.num_timesteps_output))
        val_predict , val_target , adv_val_predict = eval_val(cfgs, test_loader, model, A_wave, edges, edge_weights, 'PGD', max_speed, find_type = 'betweeness')
        log_test_csv(val_predict, val_target, adv_val_predict, cfgs, file_name, 'PGD-Centrality white-box attack,{}'.format(cfgs.test_batch_size))

        logger.info('PGD-Degree white-box attack. -Info: num attack nodes pro: {}, outputs length: {}'.format(cfgs.test_attack_nodes, cfgs.num_timesteps_output))
        val_predict , val_target , adv_val_predict = eval_val(cfgs, test_loader, model, A_wave, edges, edge_weights, 'PGD', max_speed, find_type = 'degree')
        log_test_csv(val_predict, val_target, adv_val_predict, cfgs, file_name, 'PGD-Degree white-box attack,{}'.format(cfgs.test_batch_size))



        logger.info('MIM-Random white-box attack. -Info: num attack nodes pro: {}'.format(cfgs.test_attack_nodes))
        val_predict , val_target , adv_val_predict = eval_val(cfgs, test_loader, model, A_wave, edges, edge_weights, 'MIM', max_speed, find_type = 'random')
        log_test_csv(val_predict, val_target, adv_val_predict, cfgs, file_name, 'MIM-Random white-box attack,{}'.format(cfgs.test_batch_size))


        logger.info(
            'MIM-PR white-box attack. -Info: num attack nodes pro: {}'.format(cfgs.test_attack_nodes))
        val_predict, val_target, adv_val_predict = eval_val(cfgs, test_loader, model, A_wave, edges, edge_weights,
                                                            'MIM', max_speed, find_type='pagerank')
        log_test_csv(val_predict, val_target, adv_val_predict, cfgs, file_name, 'MIM-PR white-box attack,{}'.format(cfgs.test_batch_size))



        logger.info(
            'MIM-Centrality white-box attack. -Info: num attack nodes pro: {}'.format(cfgs.test_attack_nodes))
        val_predict, val_target, adv_val_predict = eval_val(cfgs, test_loader, model, A_wave, edges, edge_weights,
                                                            'MIM', max_speed, find_type='betweeness')
        log_test_csv(val_predict, val_target, adv_val_predict, cfgs, file_name,
                     'MIM-Centrality white-box attack,{}'.format(cfgs.test_batch_size))


        logger.info(
            'MIM-Degree white-box attack. -Info: num attack nodes pro: {}'.format(cfgs.test_attack_nodes))
        val_predict, val_target, adv_val_predict = eval_val(cfgs, test_loader, model, A_wave, edges, edge_weights,
                                                            'MIM', max_speed, find_type='degree')
        log_test_csv(val_predict, val_target, adv_val_predict, cfgs, file_name,
                     'MIM-Degree white-box attack,{}'.format(cfgs.test_batch_size))


        logger_info(logger, distributed,
                    '[Remarks] {} | End of testing, saved at {}'.format(cfgs.remark, cfgs.model_dir))

    else:
        raise NameError




if __name__ == '__main__':
    main()
