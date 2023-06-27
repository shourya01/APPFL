import io
import copy
import time
import logging
import numpy as np
import torch.nn as nn
from .misc import *
from mpi4py import MPI
from .algorithm import *
from torch.optim import *
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from torch.utils.data import DataLoader

def clear_model_folder(dataset_name):
    folder_path = os.getcwd()+"/output_"+ dataset_name + "_server_model"
    if not os.path.exists(folder_path):
        return
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def save_model(
    server, 
    start_time,
    loss_fn: nn.Module,
    num_clients: int,
    test_dataloader,
    cfg: DictConfig,
    dataset_name: str = "appfl"
):
    # based on the time step store the model into
    folder_path = os.getcwd()+"/output_"+ dataset_name + "_server_model"
    print(folder_path) 
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_ext = ".pt"
    file = folder_path + "/%s%s" %(str(time.time()-start_time), file_ext)

    # while os.path.exists(filename):
        # filename = dir + "/%s_%d%s" % (output_filename, uniq, file_ext)
    
    global_model = server.model.state_dict()
    # server.model.state_dict()
    model_buffer = io.BytesIO()
    torch.save(server.model, model_buffer)
    if not os.path.exists(file):
        # Create the file
        open(file, 'wb').close()

    # Save the object to the file
    with open(file, 'wb') as file_:
        file_.write(model_buffer.getvalue())

    # test_loss, test_accuracy = validation(server, test_dataloader)
    # print("store: test_accuracy",test_accuracy)
    # try:
    #     with open(file, "rb") as file:
    #         loaded_buffer = io.BytesIO(file.read())
    # except IOError:
    #     print(f"Error: Failed to open the file '{file}'")

    # try:
    #     loaded_model = torch.load(loaded_buffer)
    #     print("Model successfully loaded!")


    #     server_ = eval(cfg.fed.servername)(
    #         {}, 
    #         copy.deepcopy(loaded_model), 
    #         loss_fn, num_clients, 
    #         "cpu",
    #         **cfg.fed.args
    #     )
    #     server.model.to("cpu")
    #     test_loss, test_accuracy = validation(server_, test_dataloader)
    #     print("load: test_accuracy",test_accuracy)

    # except Exception as e:
    #     print("Error: Failed to load the model.")
    #     print(f"Exception message: {str(e)}")




def post_analysis(
    server,
    weights: {},
    model: nn.Module,
    loss_fn: nn.Module,
    num_clients: int,
    cfg: DictConfig,
    test_dataset: Dataset = Dataset(),
    dataset_name: str = "appfl"
):
    # load dataset
    if cfg.validation == True and len(test_dataset) > 0:
        test_dataloader = DataLoader(
            test_dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.test_data_batch_size,
            shuffle=cfg.test_data_shuffle,
        )
    
    
    folder_path = os.getcwd()+"/output_"+ dataset_name + "_server_model"
    test_accuracy = 0
    best_accuracy = 0
    idx = 0
    file_to_sort = []

    # iterating all the models
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.pt'):
            file_to_sort.append(float(filename[:-3]))
        else:
            continue

    file_to_sort = np.sort(np.array(file_to_sort))
    accuracy = np.zeros(len(file_to_sort))
    loss = np.zeros(len(file_to_sort))
    valid_time = np.zeros(len(file_to_sort))

    # and run validation of each model
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.pt'):
            idx = np.where(file_to_sort == float(filename[:-3]))[0]
            print(filename[:-3],filename)
        else:
            continue
        if os.path.isfile(file_path):
            # Load the saved model state_dict
            # global_state = torch.load(file_path)
            # server.model.load_state_dict(global_state)
            try:
                with open(file_path, "rb") as file:
                    loaded_buffer = io.BytesIO(file.read())
            except IOError:
                print(f"Error: Failed to open the file '{file_path}'")

            try:
                loaded_model = torch.load(loaded_buffer)
                print("Model successfully loaded!")


                server_ = eval(cfg.fed.servername)(
                    weights, 
                    copy.deepcopy(loaded_model), 
                    loss_fn, num_clients, 
                    "cpu",
                    **cfg.fed.args
                )
                server.model.to("cpu")

                # run validation
                validation_start = time.time()
                test_loss, test_accuracy = validation(server_, test_dataloader)
                # record results
                accuracy[idx] = test_accuracy
                loss[idx] = test_loss
                valid_time[idx] = time.time()-validation_start
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
            except Exception as e:
                print("Error: Failed to load the model.")
                print(f"Exception message: {str(e)}")

            # with open(file_path, 'rb') as file:
            #     fetched_buffer = io.BytesIO(file.read())
            #     # global_state = torch.load(fetched_buffer)
            #     # server.model.load_state_dict(global_state)
            #     server.model = torch.load(fetched_buffer)

    x = np.arange(len(file_to_sort))
    plt.plot(x,accuracy)
    plt.xlabel('update')
    plt.ylabel('accuracy')
    plt.savefig(folder_path+'/'+dataset_name+'_accuracy.png')
    plt.close()
    plt.plot(x,loss)
    plt.xlabel('update')
    plt.ylabel('loss')
    plt.savefig(folder_path+'/'+dataset_name+'_loss.png')
    plt.close()
    plt.plot(x,valid_time)
    plt.xlabel('update')
    plt.ylabel('validation time')
    plt.savefig(folder_path+'/'+dataset_name+'_validation_time.png')
    plt.close()

def run_server(
    cfg: DictConfig,
    comm: MPI.Comm,
    model: nn.Module,
    loss_fn: nn.Module,
    num_clients: int,
    test_dataset: Dataset = Dataset(),
    dataset_name: str = "appfl"
):
    """Run PPFL simulation server that aggregates and updates the global parameters of model in an asynchronous way

    Args:
        cfg (DictConfig): the configuration for this run
        comm: MPI communicator
        model (nn.Module): neural network model to train
        loss_fn (nn.Module): loss function 
        num_clients (int): the number of clients used in PPFL simulation
        test_dataset (Dataset): optional testing data. If given, validation will run based on this data.
        dataset_name (str): optional dataset name
    """
    ## Start
    comm_size = comm.Get_size()

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

    # Asynchronous federated learning server (aggregator)
    server = eval(cfg.fed.servername)(
        weights, 
        copy.deepcopy(model), 
        loss_fn, num_clients, 
        device,
        **cfg.fed.args
    )

    start_time = time.time()

    global_step = 0
    client_model_step = {i : 0 for i in range(0, num_clients)}
    client_local_time = {i : start_time for i in range(0, num_clients)}

    server.model.to("cpu")
    global_model = server.model.state_dict()

    # Convert the model to bytes
    gloabl_model_buffer = io.BytesIO()
    torch.save(global_model, gloabl_model_buffer)
    global_model_bytes = gloabl_model_buffer.getvalue()

    # Send (buffer size, finish flag) - INFO - to all clients in a blocking way
    for i in range(1, num_clients+1):
        comm.send((len(global_model_bytes), False), dest=i, tag=i)      # dest is the rank of the receiver, tag = dest

    # Send the buffered model - MODEL - to all clients in a NON-blocking way
    # dest is the rank of the receiver and tag = dest + comm_size 
    # we use different tags here to differentiate different types of messages (INFO v.s. MODEL)
    send_reqs = [comm.Isend(np.frombuffer(global_model_bytes, dtype=np.byte), dest=i, tag=i+comm_size) for i in range(1, num_clients+1)]

    # Wait for response (buffer size) - INFO - from clients
    recv_reqs = [comm.irecv(source=i, tag=i) for i in range(1, num_clients+1)]

    # FedAsync: main global training loop
    test_loss = 0.0
    test_accuracy = 0.0
    best_accuracy = 0.0
    acc = []
    clear_model_folder(dataset_name)
    while True:
        # Wait for response from any one client
        client_idx, local_model_size = MPI.Request.waitany(recv_reqs)

        if client_idx != MPI.UNDEFINED:
            # Record time
            local_start_time = client_local_time[client_idx]
            local_update_time = time.time() - client_local_time[client_idx]
            global_update_start = time.time()

            # Increment the global step
            global_step += 1
            logger.info(f"[Server Log] [Step #{global_step:3}] Server gets model size from client #{client_idx}")
            
            # Allocate a buffer to receive the model byte stream
            local_model_bytes = np.empty(local_model_size, dtype=np.byte)

            # Receive the model byte stream
            comm.Recv(local_model_bytes, source=client_idx+1, tag=client_idx+1+comm_size)
            logger.info(f"[Server Log] [Step #{global_step:3}] Server gets model from client #{client_idx}")

            # Load the model byte to state dict
            local_model_buffer = io.BytesIO(local_model_bytes.tobytes())
            local_model_dict = torch.load(local_model_buffer)
            
            # Perform global update
            logger.info(f"[Server Log] [Step #{global_step:3}] Server updates global model based on the model from client #{client_idx}")
            server.update(local_model_dict, client_model_step[client_idx], client_idx)
            global_update_time = time.time() - global_update_start

            # Remove the completed request from list
            recv_reqs.pop(client_idx)
            if global_step < cfg.num_epochs:
                # Convert the updated model to bytes
                global_model = server.model.state_dict()
                gloabl_model_buffer = io.BytesIO()
                torch.save(global_model, gloabl_model_buffer)
                global_model_bytes = gloabl_model_buffer.getvalue()

                # Send (buffer size, finish flag) - INFO - to the client in a blocking way
                comm.send((len(global_model_bytes), False), dest=client_idx+1, tag=client_idx+1)

                # Send the buffered model - MODEL - to the client in a blocking way
                comm.Send(np.frombuffer(global_model_bytes, dtype=np.byte), dest=client_idx+1, tag=client_idx+1+comm_size) 

                # Add new receiving request to the list
                recv_reqs.insert(client_idx, comm.irecv(source=client_idx+1, tag=client_idx+1))
                
                # Update the model step for the client
                client_model_step[client_idx] = server.global_step

                # Update the local training time of the client
                client_local_time[client_idx] = time.time()

            # Do server validation

            dir_path = os.getcwd()+"_server_model"
            save_model(server,start_time, loss_fn, num_clients,test_dataloader, cfg, dataset_name)

            validation_start = time.time()
            if cfg.validation == True:
                test_loss, test_accuracy = validation(server, test_dataloader)
                acc.append(test_accuracy)
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                if cfg.use_tensorboard:
                    # Add them to tensorboard
                    writer.add_scalar("server_test_accuracy", test_accuracy, global_step)
                    writer.add_scalar("server_test_loss", test_loss, global_step)
            cfg["logginginfo"]["Validation_time"] = time.time() - validation_start
            cfg["logginginfo"]["PerIter_time"] = time.time() - local_start_time
            cfg["logginginfo"]["Elapsed_time"] = time.time() - start_time
            cfg["logginginfo"]["test_loss"] = test_loss
            cfg["logginginfo"]["test_accuracy"] = test_accuracy
            cfg["logginginfo"]["BestAccuracy"] = best_accuracy
            cfg["logginginfo"]["LocalUpdate_time"] = local_update_time
            cfg["logginginfo"]["GlobalUpdate_time"] = global_update_time
            logger.info(f"[Server Log] [Step #{global_step:3}] Iteration Logs:")
            if global_step != 1:
                logger.info(server.log_title())
            server.logging_iteration(cfg, logger, global_step-1)

            # Break after max updates
            if global_step == cfg.num_epochs: 
                break
    x = np.arange(len(acc))
    plt.plot(x,acc)
    plt.xlabel('update')
    plt.ylabel('acc')
    folder_path = os.getcwd()+"/output_"+ dataset_name + "_server_model"
    plt.savefig(folder_path+'/'+dataset_name+'_acc.png')
    plt.close()
    post_analysis(server, weights, model, loss_fn, num_clients, cfg, test_dataset, dataset_name)
    # Cancel outstanding requests
    for recv_req in recv_reqs:
        recv_req.cancel()

    # Send a finished indicator to all clients
    send_reqs = [comm.isend((0, True), dest=i, tag=i) for i in range(1, num_clients+1)]
    MPI.Request.waitall(send_reqs)

    # server.logging_summary(cfg, logger)

def run_client(
    cfg: DictConfig,
    comm: MPI.Comm,
    model: nn.Module,
    loss_fn: nn.Module,
    num_clients: int,
    train_data: Dataset,
    test_data: Dataset = Dataset()
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

    num_client_groups = np.array_split(range(num_clients), comm_size - 1)

    """ log for clients"""
    outfile = {}
    for _, cid in enumerate(num_client_groups[comm_rank - 1]):
        output_filename = cfg.output_filename + "_client_%s" % (cid)
        outfile[cid] = client_log(cfg.output_dirname, output_filename)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    c_handler = logging.StreamHandler()
    logger.addHandler(c_handler)

    """
    Send the number of data to a server
    Receive "weight_info" from a server      
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

    cid = num_client_groups[comm_rank - 1][0]

    client = eval(cfg.fed.clientname)(
        cid,
        weight[cid],
        copy.deepcopy(model),
        loss_fn,
        DataLoader(
            train_data[cid],
            num_workers=cfg.num_workers,
            batch_size=batchsize[cid],
            shuffle=True,
            pin_memory=True,
        ),
        cfg,
        outfile[cid],
        test_dataloader,
        **cfg.fed.args,
    )

    # FedAsync: main local training loop
    while True:
        # Receive model size from the server
        global_model_size, done = comm.recv(source=0, tag=comm_rank)
        logger.info(f"[Client Log] [Client #{comm_rank-1}] Client obtains the global model size")
        outfile[cid].write(f"[Client Log] [Client #{comm_rank-1}] Client obtains the global model size\n")
        outfile[cid].flush()
        if done: 
            logger.info(f"[Client Log] [Client #{comm_rank-1}] Client receives the indicator to stop training")
            outfile[cid].write(f"[Client Log] [Client #{comm_rank-1}] Client receives the indicator to stop training\n")
            outfile[cid].flush()
            break

        # Allocate a buffer to receive the byte stream
        global_model_bytes = np.empty(global_model_size, dtype=np.byte)
        
        # Receive the byte stream
        comm.Recv(global_model_bytes, source=0, tag=comm_rank+comm_size)
        logger.info(f"[Client Log] [Client #{comm_rank-1}] Client obtains the global model")
        outfile[cid].write(f"[Client Log] [Client #{comm_rank-1}] Client obtains the global model\n")
        outfile[cid].flush()

        # Load the byte to state dict
        global_model_buffer = io.BytesIO(global_model_bytes.tobytes())
        global_model = torch.load(global_model_buffer)

        # Train the model
        client.model.load_state_dict(global_model)
        client.update()

        # Compute gradient if the algorithm is gradient-based
        if cfg.fed.args.gradient_based:
            local_model = {}
            for name in global_model:
                local_model[name] = global_model[name] - client.primal_state[name]
        else:
            local_model = copy.deepcopy(client.primal_state)

        # Convert local model to bytes
        local_model_buffer = io.BytesIO()
        torch.save(local_model, local_model_buffer)
        local_model_bytes = local_model_buffer.getvalue()

        # Send the size of local model first
        comm.send(len(local_model_bytes), dest=0, tag=comm_rank)
        
        # Send the state dict
        comm.Isend(np.frombuffer(local_model_bytes, dtype=np.byte), dest=0, tag=comm_rank+comm_size)

    client.outfile.close()


