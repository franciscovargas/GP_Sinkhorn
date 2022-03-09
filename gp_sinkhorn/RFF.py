import torch
import math

import numpy as np

from copy import deepcopy
from torch.distributions.uniform import Uniform
from torch.distributions.gamma import Gamma
from torch.distributions.multivariate_normal import MultivariateNormal
from pyro.contrib.gp.kernels import Exponential, RBF, Kernel

_SUPPORTED_KERNELS = (Exponential, RBF)

class RandomFourierFeatures:

    def __init__(self, x, y, num_features, kernel=RBF, noise=1, device=None, 
                 random_seed=None, debug_rff=False, jitter=1e-6, sin_cos=False):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        self.x = x
        self.y = y
        self.num_features = num_features
        self.device = device
        self.kernel = self.init_kernel(kernel)
        self.noise = noise + jitter
        self.sin_cos = sin_cos

        self.f_kernel = self.fourier_transform(self.kernel)
        
        # API consideration: should we pass result back, or just set 
        # the member variables within the init_params method?
        self.init_params()
        self.variance = self.kernel.variance

        self.phi = self.feature_mapping(self.x)

        if debug_rff:
            self.drift = self.predict_gp
        else:
            self.ws = []
            for i in range(self.y.shape[1]):  
                
                # 3 options for how to solve: vanilla lstsq, lstsq with 
                # manual regularisation, or fully manual
                
#                 solution = torch.linalg.lstsq(self.phi, self.y[:, i]).solution
                
                solution = torch.linalg.lstsq(
                    self.phi.t().mm(self.phi) + (self.noise * 
                                                  torch.eye(self.phi.shape[1], 
                                                            device=self.device)),
                    self.phi.t().mm(self.y[:, i][:, None])).solution
                
#                 solution = self.solve_w(self.phi, self.y[:, i][:, None], 
#                                         lambda_=self.noise)
    
                self.ws.append(solution)
    
            self.drift = self.predict

    def init_kernel(self, kernel):
        """ Check whether we have an instance of the kernel, and instantiate
            one (with default params) if not.
        """
        if isinstance(kernel, _SUPPORTED_KERNELS):
            kernel = deepcopy(kernel)
        elif isinstance(kernel, Kernel):
            raise NotImplementedError("Unsupported kernel")
        else:
            kernel = kernel(input_dim=self.x.shape[1], variance=torch.tensor(1.0))
        return kernel

    def debug_kernel(self):    
        """ Return exact and approx kernels for debugging. """
        kernel_exact = self.kernel.forward(self.x)
        kernel_approx = self.phi @ self.phi.t()
        return kernel_exact, kernel_approx
                

    def fourier_transform(self, kernel):
        """ Compute the Fourier Transform of the kernel, returning a sampling
            function for the transformed density.
        """
        dim_x = self.x.shape[1]
        mean = torch.zeros(dim_x)
        if isinstance(kernel, RBF):
            sigma_new = 1 / kernel.lengthscale ** 2
            variance = sigma_new * torch.eye(dim_x)
            return MultivariateNormal(mean, variance).sample
            
        elif isinstance(kernel, Exponential):
            sigma_new = 1 / kernel.lengthscale
            variance = sigma_new * torch.eye(dim_x)
            def sample_exp(sample_shape):
                gammas = torch.tile(Gamma(0.5, 0.5).sample(sample_shape).to(self.device), 
                                    (self.num_features, 1)).T
                gaussians = MultivariateNormal(mean, variance).sample(sample_shape).to(self.device)
                return gaussians / torch.sqrt(gammas)
            return sample_exp

    def init_params(self):
        """ Randomly sample omega and b parameters of appropriate shapes. """
        self.omega = self.f_kernel([self.num_features]).double().to(self.device).t()
        self.b = Uniform(0, 2 * math.pi).sample([self.num_features]).to(self.device)

    def feature_mapping(self, x):
        """ Map input x into feature space using params b and omega. """
        scaling = math.sqrt(2 / self.num_features) * torch.sqrt(self.variance)
        basis = x.mm(self.omega) + self.b  # Use broadcasting instead of `.repeat`
        if self.sin_cos:
            num_half = int(basis.shape[1] / 2)
            sin_features = torch.sin(basis[:, :num_half])
            cos_features = torch.cos(basis[:, num_half:])
            return torch.concat([sin_features, cos_features], axis=1) * scaling
        else:
            return torch.cos(basis) * scaling
       
    def solve_w(self, phi, y, lambda_=0):
        """ Return the weights minimising MSE for basis functions phi and targets
            y, with regularisation coef lambda_.

            Same as torch.linalg.lstsq or solve, except that there's more 
            flexible regularisation.
        """
        return (phi.t().mm(phi) + 
                lambda_ * torch.eye(phi.size()[-1], device=self.device)
                ).inverse().mm(phi.t()).mm(y)


    def predict_gp(self, x_pred):
        """ Use the full gp equation to predict the value 
            for y_pred, performing the regression once per dimension.
        """
        total = torch.zeros([x_pred.shape[0], self.y.shape[1]], device=self.device)
        phi_pred = self.feature_mapping(x_pred)

        for i in range(self.y.shape[1]):

            pred = phi_pred @ self.phi.t() @ (self.phi @ self.phi.t() + 
                                              self.noise * 
                                              torch.eye(self.phi.shape[0],
                                                        device=self.device)
                                            ).inverse() @ self.y[:, i]
            total[:, i] = torch.squeeze(pred)
        return total 


    def predict(self, x_pred):
        """ Use the object's weights w and input x_pred to predict the value 
            for y_pred, performing the regression once per dimension.
        """
        total = torch.zeros([x_pred.shape[0], self.y.shape[1]], device=self.device)
        phi_pred = self.feature_mapping(x_pred)
        for i in range(self.y.shape[1]):
            w = self.ws[i]
            total[:, i] = torch.squeeze(phi_pred.matmul(w))
        return total