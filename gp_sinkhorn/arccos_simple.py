from pyro.nn.module import PyroParam
from torch.distributions import constraints

import torch
import numpy as np
import math
from pyro.contrib.gp.kernels import Kernel


class ArcCos(Kernel):
    
    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None):
        super().__init__(input_dim, active_dims)

        variance = torch.tensor(1.0) if variance is None else variance
        self.variance = PyroParam(variance, constraints.positive)

        lengthscale = torch.tensor(1.0) if lengthscale is None else lengthscale
        self.lengthscale = PyroParam(lengthscale, constraints.positive)
    
    def forward(self, X, Z=None):
        if Z is None:
            Z = X
        t = X.mm(Z.t())
        X_norm = torch.linalg.norm(X, dim=1)
        Z_norm = torch.linalg.norm(Z, dim=1)

        # Can be more memory efficient; but that's not a problem here
        multiplier = X_norm.outer(Z_norm)

        thetas = torch.acos(torch.clip(t / multiplier, -1, 1))
            
        # Have to manually tune these for now, easier API-wise than a partial
        # function setting the init args
        sigma_b = 1
        sigma_w = 1

        J = torch.sin(thetas) + (torch.pi - thetas) * torch.cos(thetas)
        return self.variance * (sigma_w * J * multiplier / torch.pi) + sigma_b


    def forward2(self, X, Z=None):
        same = False
        if Z is None:
            same = True
            Z = X

        xy = X.mm(Z.t())
        xx = X.mm(X.t())
        yy = Z.mm(Z.t())

        f32_tiny = np.finfo(np.float32).tiny

        xx_yy = xx * yy + f32_tiny

        # Clamp these so the outputs are not NaN
        cos_theta = (xy * xx_yy.rsqrt()).clamp(-1, 1)
        sin_theta = torch.sqrt((xx_yy - xy**2).clamp(min=0))
        theta = torch.acos(cos_theta)
        xy = (sin_theta + (math.pi - theta)*xy) / (2*math.pi)

        xx = xx/2.
        if same:
            yy = xx
            
            # Make sure the diagonal agrees with `xx`
            eye = torch.eye(xy.size()[0]).unsqueeze(-1).unsqueeze(-1)
            xy = (1-eye)*xy + eye*xx
        else:
            yy = yy/2.
        return xy

