from cmath import nan

from collections import OrderedDict
import torch.nn as nn
from torch.optim import *
from torch.utils.data import DataLoader

import numpy as np

from omegaconf import DictConfig

import copy
import time
import logging

from .misc import *
from .algorithm import *

from mpi4py import MPI

import copy


def run_server(
    cfg: DictConfig,
    comm: MPI.Comm,
    model: nn.Module,
    loss_fn: nn.Module,
    num_clients: int,
    test_dataset: Dataset = Dataset(),
    dataset_name: str = "appfl",
    target_train_dataset: Dataset = Dataset(),
):
    """Run PPFL simulation server that aggregates and updates the global parameters of model

    Args:
        cfg (DictConfig): the configuration for this run
        comm: MPI communicator
        model (nn.Module): neural network model to train
        loss_fn (nn.Module): loss function 
        num_clients (int): the number of clients used in PPFL simulation
        test_data (Dataset): optional testing data. If given, validation will run based on this data.
        DataSet_name (str): optional dataset name
    """
    ## Start
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
    num_client_groups = np.array_split(range(num_clients-1), comm_size - 1)

    # FIXME: I think it's ok for server to use cpu only.
    device = "cpu"

    """ log for a server """
    logger = logging.getLogger(__name__)
    logger = create_custom_logger(logger, cfg)

    cfg["logginginfo"]["comm_size"] = comm_size
    cfg["logginginfo"]["DataSet_name"] = dataset_name

    ## Using tensorboard to visualize the test loss
    if cfg.use_tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(
            comment=cfg.fed.args.optim + "_clients_nums_" + str(cfg.num_clients)
        )

    "Run validation if test data is given or the configuration is enabled."
    if cfg.validation == True and len(test_dataset) > 0:
        test_dataloader = DataLoader(
            test_dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.test_data_batch_size,
            shuffle=cfg.test_data_shuffle,
        )
    else:
        cfg.validation = False

    """
    Receive the number of data from clients
    Compute "weight[client] = data[client]/total_num_data" from a server    
    Scatter "weight information" to clients        
    """
    num_data = comm.gather(0, root=0)
    total_num_data = 0
    for rank in range(1, comm_size):
        for val in num_data[rank].values():
            total_num_data += val

    weight = []
    weights = {}
    for rank in range(comm_size):
        if rank == 0:
            weight.append(0)
        else:
            temp = {}
            for key in num_data[rank].keys():
                temp[key] = num_data[rank][key] / total_num_data
                weights[key] = temp[key]
            weight.append(temp)

    weight = comm.scatter(weight, root=0)

    # TODO: do we want to use root as a client?
    # print(type(num_clients),type(num_clients-1),type(torch.tensor(num_clients-1)))
    # print(num_clients)
    # print(cfg.fed.servername)
    server = eval(cfg.fed.servername)(
        weights, copy.deepcopy(model), loss_fn, torch.tensor(num_clients-1), device, **cfg.fed.args
    )

    print('start training')
    do_continue = True
    start_time = time.time()
    test_loss = 0.0
    test_accuracy = 0.0
    best_accuracy = 0.0
    for t in range(cfg.num_epochs):
        # print(f"### training {t} th epoch")
        per_iter_start = time.time()
        do_continue = comm.bcast(do_continue, root=0)

        # We need to load the model on cpu, before communicating.
        # Otherwise, out-of-memeory error from GPU
        server.model.to("cpu")

        global_state = server.model.state_dict()

        local_update_start = time.time()
        global_state = comm.bcast(global_state, root=0)
                
        # print("waiting for local starts")
        local_states= [None for i in range(num_clients-1)]        
        for rank in range(comm_size):
            ls = ""
            if rank == 0:
                continue;
            else:
                for _, cid in enumerate(num_client_groups[rank - 1]):
                    local_states[cid] = comm.recv(source=rank, tag=cid)
                    indices = list(local_states[cid].keys())
                    # print(cid, ":", indices)
        
        cfg["logginginfo"]["LocalUpdate_time"] = time.time() - local_update_start

        # print("local states done")
        
        # print("Start Server Update")
        global_update_start = time.time()
        server.update(local_states)
        cfg["logginginfo"]["GlobalUpdate_time"] = time.time() - global_update_start
        # print("Server Update is done")

        # print("Target Server Update")
        #TODO: add another update based on the new global model and the target data ######
        target_update_start = time.time()
        batchsize = cfg.train_data_batch_size
        if cfg.batch_training == False:
            batchsize = len(target_train_dataset[0])
        output_filename = cfg.output_filename + "_client_target" 
        outfile = client_log(cfg.output_dirname, output_filename)

        # server_2 = eval(cfg.fed.servername)(
        #     [1], copy.deepcopy(server.model), loss_fn, torch.tensor(1), device, **cfg.fed.args
        # )
        # global_state = server.model.state_dict()
        # cfg.fed.args.optim_args.lr /= 5
        # client = eval(cfg.fed.clientname)(
        #     0,#num_clients,
        #     1,
        #     copy.deepcopy(model),
        #     loss_fn,
        #     DataLoader(
        #         target_train_dataset[0],
        #         num_workers=cfg.num_workers,
        #         batch_size=batchsize,
        #         shuffle=cfg.train_data_shuffle,
        #         pin_memory=True,
        #     ),
        #     cfg,
        #     outfile,
        #     test_dataloader,
        #     **cfg.fed.args,
        # )
        # client.model.load_state_dict(global_state)
        # ls = []
        # ls.append(client.update())
        # server_2.update(ls)
        # global_state = server_2.model.state_dict()
        # server.model.load_state_dict(global_state)
        # cfg.fed.args.optim_args.lr *= 5
        global_state = server.model.state_dict()
        cfg.fed.args.optim_args.lr /= 5
        client = eval(cfg.fed.clientname)(
            0,#num_clients,
            1,
            copy.deepcopy(model),
            loss_fn,
            DataLoader(
                target_train_dataset[0],
                num_workers=cfg.num_workers,
                batch_size=batchsize,
                shuffle=cfg.train_data_shuffle,
                pin_memory=True,
            ),
            cfg,
            outfile,
            test_dataloader,
            **cfg.fed.args,
        )
        client.model.load_state_dict(global_state)
        global_state = client.model.state_dict()
        server.model.load_state_dict(global_state)
        cfg.fed.args.optim_args.lr *= 5
        cfg["logginginfo"]["TargetUpdate_time"] = time.time() - target_update_start

        # validation
        validation_start = time.time()
        if cfg.validation == True:
            test_loss, test_accuracy = validation(server, test_dataloader)

            if cfg.use_tensorboard:
                # Add them to tensorboard
                writer.add_scalar("server_test_accuracy", test_accuracy, t)
                writer.add_scalar("server_test_loss", test_loss, t)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
        cfg["logginginfo"]["Validation_time"] = time.time() - validation_start
        cfg["logginginfo"]["PerIter_time"] = time.time() - per_iter_start
        cfg["logginginfo"]["Elapsed_time"] = time.time() - start_time
        cfg["logginginfo"]["test_loss"] = test_loss
        cfg["logginginfo"]["test_accuracy"] = test_accuracy
        cfg["logginginfo"]["BestAccuracy"] = best_accuracy

        server.logging_iteration(cfg, logger, t)

        """ Saving model """
        if (t + 1) % cfg.checkpoints_interval == 0 or t + 1 == cfg.num_epochs:
            if cfg.save_model == True:
                save_model_iteration(t + 1, server.model, cfg)

        if np.isnan(test_loss) == True:
            break

    """ Summary """
    server.logging_summary(cfg, logger)

    do_continue = False
    do_continue = comm.bcast(do_continue, root=0)


def run_client(
    cfg: DictConfig,
    comm: MPI.Comm,
    model: nn.Module,
    loss_fn: nn.Module,
    num_clients: int,
    train_data: Dataset,
    test_data: Dataset = Dataset(),
):
    """Run PPFL simulation clients, each of which updates its own local parameters of model

    Args:
        cfg (DictConfig): the configuration for this run
        comm: MPI communicator
        model (nn.Module): neural network model to train
        num_clients (int): the number of clients used in PPFL simulation
        train_data (Dataset): training data
        test_data (Dataset): testing data
    """

    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    ## We assume to have as many GPUs as the number of MPI processes.
    if cfg.device == "cuda":
        device = f"cuda:{comm_rank-1}"
    else:
        device = cfg.device

    num_client_groups = np.array_split(range(num_clients), comm_size - 1)

    """ log for clients"""
    outfile = {}
    for _, cid in enumerate(num_client_groups[comm_rank - 1]):
        output_filename = cfg.output_filename + "_client_%s" % (cid)
        outfile[cid] = client_log(cfg.output_dirname, output_filename)

    """
    Send the number of data to a server
    Receive "weight_info" from a server    
        (fedavg)            "weight_info" is not needed as of now.
        (iceadmm+iiadmm)    "weight_info" is needed for constructing coefficients of the loss_function         
    """
    num_data = {}
    for _, cid in enumerate(num_client_groups[comm_rank - 1]):
        num_data[cid] = len(train_data[cid])
    
    comm.gather(num_data, root=0)

    weight = None
    weight = comm.scatter(weight, root=0)

    batchsize = {}
    for _, cid in enumerate(num_client_groups[comm_rank - 1]):
        batchsize[cid] = cfg.train_data_batch_size
        if cfg.batch_training == False:
            batchsize[cid] = len(train_data[cid])

    "Run validation if test data is given or the configuration is enabled."
    if cfg.validation == True and len(test_data) > 0:
        test_dataloader = DataLoader(
            test_data,
            num_workers=cfg.num_workers,
            batch_size=cfg.test_data_batch_size,
            shuffle=cfg.test_data_shuffle,
        )
    else:
        cfg.validation = False
        test_dataloader = None

    clients = [
        eval(cfg.fed.clientname)(
            cid,
            weight[cid],
            copy.deepcopy(model),
            loss_fn,
            DataLoader(
                train_data[cid],
                num_workers=cfg.num_workers,
                batch_size=batchsize[cid],
                shuffle=cfg.train_data_shuffle,
                pin_memory=True,
            ),
            cfg,
            outfile[cid],
            test_dataloader,
            **cfg.fed.args,
        )
        for _, cid in enumerate(num_client_groups[comm_rank - 1])
    ]

    do_continue = comm.bcast(None, root=0)
    
    while do_continue:
        """Receive "global_state" """
        global_state = comm.bcast(None, root=0)

        """ Update "local_states" based on "global_state" """
        reqlist = []
        for client in clients:
            cid = client.id
            ## initial point for a client model            
            client.model.load_state_dict(global_state)

            ## client update     
            ls = client.update()                            
            req = comm.isend(ls, dest=0, tag=cid)
            reqlist.append(req)

        MPI.Request.Waitall(reqlist)
        do_continue = comm.bcast(None, root=0)

    for client in clients:
        client.outfile.close()