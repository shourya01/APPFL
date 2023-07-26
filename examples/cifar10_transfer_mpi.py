import os
import time

import numpy as np
import torch
 
import torchvision
import torchvision.transforms as transforms

from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
from models.utils import get_model

import appfl.run_serial_transfer as rs
import appfl.run_mpi_transfer as rm
# from mpi4py import MPI

import argparse

import torch.optim as optim
import logging
from torch.utils.data import DataLoader
 
 
""" read arguments """ 

parser = argparse.ArgumentParser() 

parser.add_argument('--device', type=str, default="cpu")    

## dataset and model
parser.add_argument('--dataset', type=str, default="CIFAR10")   
parser.add_argument('--num_channel', type=int, default=3)   
parser.add_argument('--num_classes', type=int, default=10)   
parser.add_argument('--num_pixel', type=int, default=32)   
parser.add_argument('--model', type=str, default="CNN")   
parser.add_argument('--train_data_batch_size', type=int, default=128)   
parser.add_argument('--test_data_batch_size', type=int, default=128)   

parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--frozen_feature_layers', type=bool, default=False)
parser.add_argument('--resnet', type=str, default='resnet152')
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--resnet_classes', type=int, nargs='+', default=[2,3,2,10])

## clients
parser.add_argument('--num_clients', type=int, default=1)    
parser.add_argument('--client_optimizer', type=str, default="Adam")    
parser.add_argument('--client_lr', type=float, default=1e-3)    
parser.add_argument('--num_local_epochs', type=int, default=1)    

## server
parser.add_argument('--server', type=str, default="ServerFedAvg")    
parser.add_argument('--num_epochs', type=int, default=20)    

parser.add_argument('--server_lr', type=float, required=False)    
parser.add_argument('--mparam_1', type=float, required=False)    
parser.add_argument('--mparam_2', type=float, required=False)    
parser.add_argument('--adapt_param', type=float, required=False)    
 
args = parser.parse_args()    

args.save_model_state_dict = False

if torch.cuda.is_available():
    print("using cuda!!!!")
    args.device="cuda"


def get_data(): 
    
    dir = os.getcwd() + "/datasets/RawData"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 
    # test data for a server
    test_data_raw = eval("torchvision.datasets." + args.dataset)(
        dir, download=True, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
    )
 

    test_data_input = []
    test_data_label = []
    for idx in range(len(test_data_raw)):
        test_data_input.append(test_data_raw[idx][0].tolist())
        test_data_label.append(test_data_raw[idx][1])

    test_dataset = Dataset(
        torch.FloatTensor(test_data_input), torch.tensor(test_data_label)
    )

    # training data for multiple clients
    train_data_raw = eval("torchvision.datasets." + args.dataset)(
        dir, download=False, train=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),                        
                        normalize,
                    ])
    )

    split_train_data_raw = np.array_split(range(len(train_data_raw)), args.num_clients)
    train_datasets = []
    for i in range(args.num_clients-1):

        train_data_input = []
        train_data_label = []
        for idx in split_train_data_raw[i]:
            train_data_input.append(train_data_raw[idx][0].tolist())
            train_data_label.append(train_data_raw[idx][1])

        train_datasets.append(
            Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )
        ) 
    
    target_train_data_input = []
    target_train_data_label = []
    for idx in split_train_data_raw[args.num_clients-1]:
        target_train_data_input.append(train_data_raw[idx][0].tolist())
        target_train_data_label.append(train_data_raw[idx][1])
    target_train_datasets = []
    target_train_datasets.append(
        Dataset(
            torch.FloatTensor(target_train_data_input),
            torch.tensor(target_train_data_label)
        )
    )
    return train_datasets, test_dataset, target_train_datasets

## Run
def main():
    
    # comm = MPI.COMM_WORLD
    # comm_rank = comm.Get_rank()
    # comm_size = comm.Get_size() 
    print("start working")
    # Reproducibility
    set_seed(1)

    """ Configuration """     
    cfg = OmegaConf.structured(Config) 

    cfg.device = args.device
    # cfg.device_server = args.device
    cfg.save_model_state_dict = args.save_model_state_dict

    ## dataset
    cfg.train_data_batch_size = args.train_data_batch_size
    cfg.test_data_batch_size = args.test_data_batch_size
    cfg.train_data_shuffle = True

    ## clients
    cfg.num_clients = args.num_clients
    cfg.fed.args.optim = args.client_optimizer
    cfg.fed.args.optim_args.lr = args.client_lr
    cfg.fed.args.num_local_epochs = args.num_local_epochs
    
    ## server
    cfg.fed.servername = args.server
    cfg.num_epochs = args.num_epochs
    
    ## outputs        
    cfg.output_dirname = "./outputs_%s_%s_%s_%s_%s_%s"%(args.dataset, args.model, args.server, args.client_optimizer, args.num_local_epochs, args.client_lr)    
    if args.server_lr != None:
        cfg.fed.args.server_learning_rate = args.server_lr
        cfg.output_dirname += "_ServerLR_%s" %(args.server_lr)
        
    if args.adapt_param != None:
        cfg.fed.args.server_adapt_param = args.adapt_param   
        cfg.output_dirname += "_AdaptParam_%s" %(args.adapt_param)             
        
    if args.mparam_1 != None:
        cfg.fed.args.server_momentum_param_1 = args.mparam_1
        cfg.output_dirname += "_MParam1_%s" %(args.mparam_1)
        
    if args.mparam_2 != None:
        cfg.fed.args.server_momentum_param_2 = args.mparam_2  
        cfg.output_dirname += "_MParam2_%s" %(args.mparam_2)        

    cfg.output_filename = "result"          
    
    
    start_time = time.time()

    """ User-defined model """    
    model = get_model(args) 
    loss_fn = torch.nn.CrossEntropyLoss()   
    
    
    ## loading models 
    cfg.load_model = False
    if cfg.load_model == True:
        cfg.load_model_dirname      = "./save_models"
        cfg.load_model_filename     = "Model"               
        model = load_model(cfg)         
    
    """ User-defined data """
    train_datasets, test_dataset, target_train_datasets = get_data()

    ## Sanity check for the user-defined data
    if cfg.data_sanity == True:
        data_sanity_check(train_datasets, test_dataset, args.num_channel, args.num_pixel)
        data_sanity_check(target_train_datasets, test_dataset, args.num_channel, args.num_pixel)        

    print(
        "-------Loading_Time=",
        time.time() - start_time,
    ) 
    
    """ saving models """
    cfg.save_model = False
    if cfg.save_model == True:
        cfg.save_model_dirname      = "./save_models"
        cfg.save_model_filename     = "Model"      
    
    # cfg.summary_file = cfg.output_dirname + "/Summary_%s.txt" %(args.dataset)
 
    
    """ Running """
    # if comm_size > 1:
    #     if comm_rank == 0:
    #         rm.run_server(cfg, comm, model, loss_fn, args.num_clients, test_dataset, args.dataset, target_train_datasets)
    #     else:
    #         rm.run_client(cfg, comm, model, loss_fn, args.num_clients-1, train_datasets, test_dataset)
    #     print("------DONE------", comm_rank)
    # else:
    #     rs.run_serial(cfg, model, loss_fn, train_datasets, test_dataset, args.dataset, target_train_datasets)
    rs.run_serial(cfg, model, loss_fn, train_datasets, test_dataset, args.dataset, target_train_datasets)
  

if __name__ == "__main__":
    main()

 
# To run CUDA-aware MPI:
# mpiexec -np 2 --mca opal_cuda_support 1 python ./cifar10_transfer_mpi.py
# To run MPI:
# mpiexec -np 2 python ./cifar10_transfer_mpi.py
# To run:
# python ./cifar10_transfer_mpi.py --num_clients 5