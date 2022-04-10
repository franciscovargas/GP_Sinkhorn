import numpy as np

import torch
import torch.nn as nn
from torch.distributions import constraints

from pyro.nn.module import PyroParam
from pyro.contrib.gp.kernels import Kernel


class NNKernel(nn.Module):

    f32_tiny = np.finfo(np.float32).tiny

    def __init__(self, variance_b, variance_w):
        super().__init__()
        self.variance_b = variance_b
        self.variance_w = variance_w
    

    def dim_sum(self, mat, dim=2):           
        return self.variance_w * 0.5 * mat.sum(dim) + self.variance_b

    def forward(self, x, y=None):

        if y is None:
            y = x

        xy = self.variance_w * 0.5 * x.mm(y.T) + self.variance_b

        xx = self.dim_sum(x ** 2, dim=1)
        yy = self.dim_sum(y ** 2, dim=1)

        # xx = self.variance_w * 0.5 * x.dot(x) + self.variance_b
        # yy = self.variance_w * 0.5 * y.dot(y) + self.variance_b

        # xx = x.mm(x.t())

        xx_yy = torch.outer(xx, yy) + self.f32_tiny
        cos_theta = (xy * xx_yy.rsqrt()).clamp(-1, 1)
        sin_theta = torch.sqrt((xx_yy - xy ** 2).clamp(min=0))
        theta = torch.acos(cos_theta)
        return (sin_theta + (np.pi - theta) * xy) / (2 * np.pi)


class ArcCos(Kernel):
    
    def __init__(self, input_dim, variance=None, variance_b=None, 
                 variance_w=None, lengthscale=None, active_dims=None):
        super().__init__(input_dim, active_dims)

        variance = torch.tensor(1.0) if variance is None else variance
        self.variance = PyroParam(variance, constraints.positive)
        
        variance_b = torch.tensor(1.0) if variance_b is None else variance_b
        self.variance_b = PyroParam(variance_b, constraints.positive)

        variance_w = torch.tensor(1.0) if variance_w is None else variance_w
        self.variance_w = PyroParam(variance_w, constraints.positive)
        
        lengthscale = torch.tensor(1.0) if lengthscale is None else lengthscale
        self.lengthscale = PyroParam(lengthscale, constraints.positive)
        
        self.model = NNKernel(self.variance_b, self.variance_w)     
    
    def forward(self, X, Z=None):
        return self.variance * self.model(X, Z)
