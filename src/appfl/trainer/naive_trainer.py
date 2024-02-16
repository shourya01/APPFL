import copy
import time
import torch
import numpy as np
from typing import Tuple, Dict
from torch.optim import *
from .base_trainer import BaseTrainer
from appfl.privacy import laplace_mechanism_output_perturb

class NaiveTrainer(BaseTrainer):
    """
    NaiveTrainer:
        Naive trainer for FL clients, which trains the model using `torch.optim` 
        optimizers for a certain number of local epochs or local steps. 
        Users need to specify which training model to use in the configuration, 
        as well as the number of local epochs or steps.
    """  
    def train(self):
        """
        Train the model for a certain number of local epochs or steps and store the mode state
        (probably with perturbation for differential privacy) in `self.model_state`.
        """
        # Sanity check
        assert hasattr(self.train_configs, "mode"), "Training mode must be specified"
        assert self.train_configs.mode in ["epoch", "step"], "Training mode must be either 'epoch' or 'step'"
        if self.train_configs.mode == "epoch":
            assert hasattr(self.train_configs, "num_local_epochs"), "Number of local epochs must be specified"
        else:
            assert hasattr(self.train_configs, "num_local_steps"), "Number of local steps must be specified"
        
        self.model.to(self.train_configs.get("device", "cpu"))
        do_validation = self.train_configs.get("do_validation", False) and self.val_dataloader is not None
        
        # Set up logging title
        if self.round == 0:
            title = (
                ["Round", "Time", "Train Loss", "Train Accuracy"] 
                if not do_validation
                else ["Round", "Time", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"]
            )
            if self.train_configs.mode == "epoch":
                title.insert(1, "Epoch")
            self.logger.log_title(title)
        
        # # TEST
        # if do_validation:
        #     val_loss, val_accuracy = self._validate()
        # per_epoch_time = 0
        # self.logger.log_content(
        #     [self.round, 0, per_epoch_time, 0, 0] 
        #     if not do_validation
        #     else [self.round, 0, per_epoch_time, 0, 0, val_loss, val_accuracy]
        # )   
        # ## TEST END


        # Start training
        optimizer = eval(self.train_configs.optim)(self.model.parameters(), **self.train_configs.optim_args)
        if self.train_configs.mode == "epoch":
            for epoch in range(self.train_configs.num_local_epochs):
                start_time = time.time()
                train_loss, target_true, target_pred = 0, [], []
                for data, target in self.train_dataloader:
                    loss, pred, label = self._train_batch(optimizer, data, target)
                    train_loss += loss
                    target_true.append(label)
                    target_pred.append(pred)
                train_loss /= len(self.train_dataloader)
                target_true, target_pred = np.concatenate(target_true), np.concatenate(target_pred)
                train_accuracy = float(self.metric(target_true, target_pred))
                if do_validation:
                    val_loss, val_accuracy = self._validate()
                per_epoch_time = time.time() - start_time
                self.logger.log_content(
                    [self.round, epoch, per_epoch_time, train_loss, train_accuracy] 
                    if not do_validation
                    else [self.round, epoch, per_epoch_time, train_loss, train_accuracy, val_loss, val_accuracy]
                )
        else:
            start_time = time.time()
            data_iter = iter(self.train_dataloader)
            train_loss, target_true, target_pred = 0, [], []
            for _ in range(self.train_configs.num_local_steps):
                try:
                    data, target = next(data_iter)
                except:
                    data_iter = iter(self.train_dataloader)
                    data, target = next(data_iter)
                loss, pred, label = self._train_batch(optimizer, data, target)
                train_loss += loss
                target_true.append(label)
                target_pred.append(pred)
            train_loss /= len(self.train_dataloader)
            target_true, target_pred = np.concatenate(target_true), np.concatenate(target_pred)
            train_accuracy = float(self.metric(target_true, target_pred))
            if do_validation:
                val_loss, val_accuracy = self._validate()
            per_step_time = time.time() - start_time
            self.logger.log_content(
                [self.round, per_step_time, train_loss, train_accuracy] 
                if not do_validation
                else [self.round, per_step_time, train_loss, train_accuracy, val_loss, val_accuracy]
            )

        self.round += 1

        # Differential privacy
        if self.train_configs.get("use_dp", False):
            assert hasattr(self.train_configs, "clip_value"), "Gradient clipping value must be specified"
            assert hasattr(self.train_configs, "epsilon"), "Privacy budget (epsilon) must be specified"
            sensitivity = 2.0 * self.train_configs.clip_value * self.train_configs.optim_args.lr
            self.model_state = laplace_mechanism_output_perturb(
                self.model,
                sensitivity,
                self.train_configs.epsilon,
            )
        else:
            self.model_state = copy.deepcopy(self.model.state_dict())
        
        # Move to CPU for communication
        if self.train_configs.get("device", "cpu") == "cuda":
            for k in self.model_state:
                self.model_state[k] = self.model_state[k].cpu()

    def get_parameters(self) -> Dict:
        hasattr(self, "model_state"), "Please make sure the model has been trained before getting its parameters"
        return self.model_state

    def _validate(self) -> Tuple[float, float]:
        """
        Validate the model
        :return: loss, accuracy
        """
        device = self.train_configs.get("device", "cpu")
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            target_pred, target_true = [], []
            for data, target in self.val_dataloader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                val_loss += self.loss_fn(output, target).item()
                target_true.append(target.detach().cpu().numpy())
                target_pred.append(output.detach().cpu().numpy())
        val_loss /= len(self.val_dataloader)
        val_accuracy = float(self.metric(np.concatenate(target_true), np.concatenate(target_pred)))
        self.model.train()
        return val_loss, val_accuracy

    def _train_batch(self, optimizer: torch.optim.Optimizer, data, target) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Train the model for one batch of data
        :param optimizer: torch optimizer
        :param data: input data
        :param target: target label
        :return: loss, prediction, label
        """
        device = self.train_configs.get("device", "cpu")
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if self.train_configs.get("clip_grad", False) or self.train_configs.get("use_dp", False):
            assert hasattr(self.train_configs, "clip_value"), "Gradient clipping value must be specified"
            assert hasattr(self.train_configs, "clip_norm"), "Gradient clipping norm must be specified"
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.train_configs.clip_value,
                norm_type=self.train_configs.clip_norm,
            )
        return loss.item(), output.detach().cpu().numpy(), target.detach().cpu().numpy()