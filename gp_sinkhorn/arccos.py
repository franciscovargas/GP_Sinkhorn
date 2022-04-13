
import torch

from pyro.nn.module import PyroParam
from torch.distributions import constraints
from pyro.contrib.gp.kernels import Kernel


class ArcCos(Kernel):
    
    def __init__(self, input_dim, variance_w, variance_b, variance=None, 
                 lengthscale=None, active_dims=None):
        super().__init__(input_dim, active_dims)

        self.variance_w = variance_w
        self.variance_w = PyroParam(variance_w, constraints.positive)
        
        self.variance_b = variance_b
        self.variance_b = PyroParam(variance_b, constraints.positive)

        variance = torch.tensor(1.0) if variance is None else variance
        self.variance = PyroParam(variance, constraints.positive)

        lengthscale = torch.tensor(1.0) if lengthscale is None else lengthscale
        self.lengthscale = PyroParam(lengthscale, constraints.positive)
    
    def transform(self, x):
        return self.variance_w * x + self.variance_b

    def forward(self, X, Z=None):
        """ Compute kernel matrix. """
        if Z is None:
            Z = X
        xz = self.transform(X.mm(Z.T))
        
        X_norm = torch.sqrt(self.transform(torch.linalg.norm(X, dim=1) ** 2))
        Z_norm = torch.sqrt(self.transform(torch.linalg.norm(Z, dim=1) ** 2))

        multiplier = torch.outer(X_norm, Z_norm)

        cos_theta = torch.clip(xz / multiplier, -1, 1)
        sin_theta = torch.sqrt(1 - cos_theta ** 2)

        J = sin_theta + (torch.pi - torch.acos(cos_theta)) * cos_theta

        # TODO: does it matter whether the denominator is pi or 2 * pi?
        # CNN-GP folk use 2 * pi, but pi is also believable and works.
        return self.variance * J * multiplier / (2 * torch.pi)
