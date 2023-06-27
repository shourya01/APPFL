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
    folder_path = os.getcwd()+"/"+ dataset_name + "_server_model"
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
    folder_path = os.getcwd()+"/"+ dataset_name + "_server_model"
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
    
    
    folder_path = os.getcwd()+"/"+ dataset_name + "_server_model"
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