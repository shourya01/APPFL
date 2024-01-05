import torch
import numpy as np
from mpi4py import MPI
from torch import nn as nn
import argparse
import logging
from appfl.misc.data import Dataset
from appfl.config import *
from appfl.misc.utils import set_seed
import appfl.run_mpi as rm

'''
In this script, we see whether setting clip_norm sets the norm of difference to
desired value.
'''

""" parser arguments """
parser = argparse.ArgumentParser()
parser.add_argument("--clip_update_norm", type=int, default=1, help="whether to clip or not")
parser.add_argument("--seed", type=int, default=1, help="seed for reproducability")
parser.add_argument("--batch_size", type=int, default=1, help="batch size for training")
parser.add_argument("--savedir", type=str, default='/home/L2RPN', help="dirname for saving model")
args = parser.parse_args()

def get_dataset(n_points,n_clients):
    
    """ return dataset corresponding to function y=3*x + 2 """
    
    all_x_points = np.linspace(0,1,n_points)
    split_x_points = np.array_split(all_x_points,n_clients)
    split_y_points = [3*itm+2 for itm in split_x_points]
    
    datasets = [Dataset(torch.FloatTensor(x.reshape(-1,1)),torch.FloatTensor(y.reshape(-1,1)))
                for x,y in zip(split_x_points,split_y_points)]
    retval = datasets if n_clients > 1 else datasets[0]
    
    return retval

def get_flattened_params(model):
    
    """ get numpy vector of learnable parameters of model """
    
    param_collector = []
    
    for _,val in model.named_parameters():
        param_collector.append(val.detach().cpu().numpy().reshape(-1))
        
    return np.concatenate(param_collector,axis=0)

if __name__ == "__main__":
    
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    assert comm_size == 3, "Hard coded 1 server and 2 clients for this example, comm_size is .%d"%comm_size
    
    model = nn.Sequential(nn.Linear(in_features=1,out_features=1),) # model
    loss_fn = nn.MSELoss() # loss
    metric = lambda y_true, y_pred: np.mean( np.abs(y_true-y_pred) ) # test metric
    num_clients = 2 # number of clients
    train_datasets = get_dataset(n_points = 1000, n_clients = num_clients) # get train dataset for clients
    test_datasets = None # disable test dataset
    
    cfg = OmegaConf.structured(Config)
    
    """ set cfg values to ensure experiment results are valid"""
    
    cfg.fed.args.clip_grad = True if args.clip_update_norm else False # clip grad according to input args
    cfg.fed.args.clip_value = 1 # set clip value
    cfg.fed.args.num_local_steps = 1 # set single local step
    cfg.fed.args.optim_args.lr = 1 # set client LR to 1
    cfg.fed.args.server_learning_rate = 1 # set server LR to 1
    
    cfg.train_data_batch_size = args.batch_size # set batch size according to input
    cfg.validation = False # set validation to be false
    
    cfg.fed.clientname = "ClientStepOptim" # use step-based optimizer
    cfg.num_epochs = 1 # set single global update
    cfg.save_model = True # save the model
    cfg.save_model_dirname = args.savedir # set save directory - CHANGE AS NEEDED
    
    
    set_seed(args.seed) # set random seed according to args for reproducability
    
    if comm_rank == 0:
        """ save model state before update """
        cur_param = get_flattened_params(model)
    
    if comm_rank == 0:
        """ run server """
        rm.run_server(cfg, comm, model, loss_fn, num_clients)
    else:
        """ run_client """
        rm.run_client(cfg, comm, model, loss_fn, num_clients, train_datasets[comm_rank - 1])
        
    if comm_rank == 0:
        model = torch.load(cfg.save_model_dirname+'/_Round_1.pt')
        new_param = get_flattened_params(model)
        print('old model params:')
        print(cur_param)
        print('new model params:')
        print(new_param)
        print('norm difference:')
        print(np.linalg.norm(cur_param-new_param,ord=1))
    