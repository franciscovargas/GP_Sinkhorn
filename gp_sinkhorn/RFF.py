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
                 random_seed=None):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        self.x = x
        self.y = y
        self.num_features = num_features
        self.device = device
        self.kernel = self.init_kernel(kernel)

        self.f_kernel = self.fourier_transform(self.kernel)
        
        # API consideration: should we pass result back, or just set 
        # the member variables within the init_params method?
        self.init_params()
        self.variance = self.kernel.variance

        self.ws = []
        phi = self.feature_mapping(self.x)
        for i in range(self.y.shape[1]):  
            self.ws.append(self.solve_w(phi, self.y[:, i][:, None], lambda_=noise))
            
        self.drift = lambda x: self.predict(x)

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

    def fourier_transform(self, kernel):
        """ Compute the Fourier Transform of the kernel, returning a sampling
            function for the transformed density.
        """
        dim_x = self.x.shape[1]
        sigma_new = 1 / kernel.lengthscale
        mean = torch.zeros(dim_x)
        variance = sigma_new * torch.eye(dim_x)
        if isinstance(kernel, RBF):
            return MultivariateNormal(mean, variance).sample
            
        elif isinstance(kernel, Exponential):
            def sample_exp(sample_shape):
                gammas = torch.tile(Gamma(0.5, 0.5).sample(sample_shape).to(self.device), 
                                    (self.num_features, 1)).T
                gaussians = MultivariateNormal(mean, variance).sample(sample_shape).to(self.device)
                return gaussians  / torch.sqrt(gammas)
            return sample_exp

    def init_params(self):
        """ Randomly sample omega and b parameters of appropriate shapes. """
        # Code from the paper; can refactor to simplify
        m = Uniform(torch.tensor([0.0], device=self.device), 
                    torch.tensor([2*math.pi], device=self.device))
        b = m.sample((1, self.num_features)).view(-1, self.num_features).to(self.device)
        self.omega = self.f_kernel([self.num_features]).double().to(self.device).t()
        self.b = b

    def feature_mapping(self, x):
        """ Map input x into feature space using params b and omega. """
        return (torch.cos(x.mm(self.omega) + self.b.repeat(x.shape[0], 1)) * 
                math.sqrt(2 / self.num_features)) * torch.sqrt(self.variance)

    def solve_w(self, phi, y, lambda_=0):
        """ Return the weights minimising MSE for basis functions phi and targets
            y, with regularisation coef lambda_.

            Same as torch.linalg.lstsq or solve, except that there's more 
            flexible regularisation.
        """
        return (phi.t().mm(phi) + 
                lambda_ * torch.eye(phi.size()[-1], device=self.device)
                ).inverse().mm(phi.t()).mm(y)

    def predict(self, x_pred):
        """ Use the object's weights w and input x_pred to predict the value 
            for y_pred, performing the regression once per dimension.
        """
        # assert self.y.shape[1] == x_pred.shape[1] - 1
        total = torch.zeros([x_pred.shape[0], self.y.shape[1]], device=self.device)
        phi_pred = self.feature_mapping(x_pred) * torch.sqrt(self.variance)
        for i in range(self.y.shape[1]):
            w = self.ws[i]
            total[:, i] = torch.squeeze(phi_pred.matmul(w))
        return total