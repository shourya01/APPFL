from collections import OrderedDict
from .server_federated import FedServer

import torch
from torch import linalg
import copy

from ..misc.utils import validation


class ServerFedSDLBFGS(FedServer):

    def __init__(self, *args, **kwargs):
        super(ServerFedSDLBFGS, self).__init__(*args, **kwargs)

        # p - history
        # delta - lower bound on gamma_k
        self.p = kwargs["history"]
        self.delta = kwargs["delta"]

        """ Gradient history for L-BFGS """
        self.s_vectors = [] 
        self.ybar_vectors = []
        self.rho_values = []
        self.prev_params = OrderedDict()
        self.prev_grad = OrderedDict()
        self.k = 0

        # Configuration for learning rate or backwards line search
        self.max_step_size = kwargs["max_step_size"]
        self.increment = kwargs["increment"]
        self.search_control = kwargs["search_control"]



    def compute_step(self):
        super(ServerFedSDLBFGS, self).compute_pseudo_gradient()
        super(ServerFedSDLBFGS, self).update_m_vector()

        # Initial step, necessary so we can get g_{k - 1}
        if self.k == 0:
            self.make_sgd_step()
        else:
            self.make_lbfgs_step()

        self.k += 1



    def make_sgd_step(self):
        for name, _ in self.model.named_parameters():
            self.step[name] = -self.pseudo_grad[name]
            self.prev_params[name] = copy.deepcopy(self.model.state_dict()[name].reshape(-1))
            self.prev_grad[name] = copy.deepcopy(self.pseudo_grad[name].reshape(-1))


    def make_lbfgs_step(self):

        self.s_vectors.append(OrderedDict())
        self.ybar_vectors.append(OrderedDict())
        self.rho_values.append(OrderedDict())

        if self.k > self.p:
            del self.s_vectors[0]
            del self.ybar_vectors[0]
            del self.rho_values[0]

        for name, _ in self.model.named_parameters():

            shape = self.model.state_dict()[name].shape

            # Create newest s vector
            s_vector = self.model.state_dict()[name].reshape(-1) - self.prev_params[name]
            self.s_vectors[-1][name] = s_vector

            # Create newest ybar vector
            y_vector = self.pseudo_grad[name].reshape(-1) - self.prev_grad[name]
            gamma = self.compute_gamma(y_vector, s_vector)
            ybar_vector = self.compute_ybar_vector(y_vector, s_vector, gamma)
            self.ybar_vectors[-1][name] = ybar_vector

            # Create newest rho
            rho = 1.0 / (s_vector.dot(ybar_vector))
            self.rho_values[-1][name] = rho

            # Perform recursive computations and step
            v_vector = self.compute_step_approximation(name, gamma)

            try: 
                hessian = self.realize_hessian(name, gamma, shape)
                eigvals = linalg.eigvals(hessian)
                if (eigvals.real < 0.0).any():
                    __import__('pdb').set_trace()
            except RuntimeError:
                # occurs if there is not enough memory to realize the hessian
                pass
            self.step[name] = -(self.max_step_size / self.k) * v_vector.reshape(shape)

            # Store information for next step
            self.prev_params[name] = copy.deepcopy(self.model.state_dict()[name].reshape(-1))
            self.prev_grad[name] = copy.deepcopy(self.pseudo_grad[name].reshape(-1))


    
    def realize_hessian(self, name, gamma, shape):
        H = torch.eye(shape.numel(), device=self.device) * (1. / gamma)
        m = min(self.p, self.k - 1)
        r = range(m)

        for i in r:
            rs = self.rho_values[i][name] * self.s_vectors[i][name]
            I = torch.eye(shape.numel(), device=self.device) - (self.ybar_vectors[i][name].outer(rs))
            proj = rs.outer(self.s_vectors[i][name])
            H = (I.transpose(0, 1) @ H @ I) + proj

        return H
        


    def compute_gamma(self, y_vec, s_vec):
        """ Equation 3.10 """
        return max((y_vec.dot(y_vec)) / (y_vec.dot(s_vec)), self.delta)


    def compute_theta(self, y_vec, s_vec, gamma):
        """ Equation 3.8 """
        s_proj = s_vec.dot(s_vec) * gamma
        dot = s_vec.dot(y_vec)

        if dot < 0.25 * s_proj:
            return (0.75 * s_proj) / (s_proj - dot)
        return 1


    def compute_ybar_vector(self, y_vec, s_vec, gamma):
        """ Equation 3.7 """

        # H_{k, 0}^{-1} doesn't need to be computed directly since 
        # H_{k, 0}^{-1} @ vector = (gamma * I) @ vector = gamma * vector
        theta = self.compute_theta(y_vec, s_vec, gamma)
        val = (theta * y_vec) + ((1 - theta) * (gamma * s_vec))

        return val


    # def compute_step_approximation(self, name, gamma):
    #     """ Algorithm 2 """
    #     u = copy.deepcopy(self.pseudo_grad[name].reshape(-1))
    #     mu_values = []
    #     p = min(self.k - 1, self.p)
    #     r = range(p)

    #     for i in r:
    #         j = (self.k - i - 1) % p
    #         try:
    #             mu = self.rho_values[j][name] * u.dot(self.s_vectors[j][name])
    #             mu_values.append(mu)
    #             u = u - (mu * self.ybar_vectors[j][name])
    #         except IndexError:
    #             __import__('pdb').set_trace()

    #     v = (1.0 / gamma) * u
    #     for i in r:
    #         j = (self.k - p + i) % p
    #         nu = self.rho_values[j][name] * v.dot(self.ybar_vectors[j][name])
    #         v = v + ((mu_values[-(i + 1)] - nu) * self.s_vectors[j][name])
    #     
    #     return v

    def compute_step_approximation(self, name, gamma):
        """ Algorithm 3 """
        u = copy.deepcopy(self.pseudo_grad[name].reshape(-1))
        mu_values = []
        m = min(self.k - 1, self.p)
        r = range(m)

        for i in reversed(r):
            mu = self.rho_values[i][name] * u.dot(self.s_vectors[i][name])
            mu_values.append(mu)
            u = u - (mu * self.ybar_vectors[i][name])

        v = (1.0 / gamma) * u
        for i in r:
            nu = self.rho_values[i][name] * v.dot(self.ybar_vectors[i][name])
            v = v + ((mu_values[-(i + 1)] - nu) * self.s_vectors[i][name])
        
        return v


    def logging_summary(self, cfg, logger):
        super(FedServer, self).log_summary(cfg, logger)

        logger.info("client_learning_rate = %s " % (cfg.fed.args.optim_args.lr))

        if cfg.summary_file != "":
            with open(cfg.summary_file, "a") as f:

                f.write(
                    cfg.logginginfo.DataSet_name
                    + " FedSDLBFGS ClientLR "
                    + str(cfg.fed.args.optim_args.lr)
                    + " TestAccuracy "
                    + str(cfg.logginginfo.accuracy)
                    + " BestAccuracy "
                    + str(cfg.logginginfo.BestAccuracy)
                    + " Time "
                    + str(round(cfg.logginginfo.Elapsed_time, 2))
                    + "\n"
                )
