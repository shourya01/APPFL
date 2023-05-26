from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
import torch

""" Federated Residual Algorithm
- This algorithm solves a convex optimization with a convex combination constraint and a bound constraint in a distributed setting.
- This algorithm has been developed for reconstructing an object from multimodal data (XRF and XRT) generated by the Advanced Photon Source (APS) facility at Argonne.
- The main idea of the proposed algorithm is to utilize "residual" for updating and customize the initial points for each client.
"""


@dataclass
class Fedres:
    type: str = "fedres"
    servername: str = "FedresServer"
    clientname: str = "FedresClient"
    args: DictConfig = OmegaConf.create(
        {
            ##
            "logginginfo": {},
            ## coefficient assigned for each client
            "coeff": {},
            ## ground truth (to compute the mean squared error for every iterations)
            "w_truth": {0: [], 1: [], 2: [], 3: []},
            ## Clients optimizer
            "optim": "SGD",
            "num_local_epochs": 1,
            "optim_args": {
                "lr": 1e3,
                "momentum": 0.9,
            },
        }
    )