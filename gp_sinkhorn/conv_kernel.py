""" A simple convolutional kernel, where we sum over patches. 
    Inspired by https://proceedings.neurips.cc/paper/2017/file/1c54985e4f95b7819ca0357c0cb9a09f-Paper.pdf
"""

import torch
import math

import torch.nn.functional as F

from pyro.nn.module import PyroParam
from torch.distributions import constraints
from pyro.contrib.gp.kernels import Kernel
from pyro.contrib.gp.kernels import Exponential, RBF
from gp_sinkhorn.arccos import ArcCos

from copy import deepcopy


class ConvSimple(Kernel):
    
    _SUPPORTED_KERNELS = (Exponential, RBF, ArcCos)
    
    def __init__(self, input_dim, kernel_underlying, patch_size, variance=None, 
                 lengthscale=None, active_dims=None):
        super().__init__(input_dim, active_dims)
        
        self.patch_size = patch_size

        variance = torch.tensor(1.0) if variance is None else variance
        self.variance = PyroParam(variance, constraints.positive)

        lengthscale = torch.tensor(1.0) if lengthscale is None else lengthscale
        self.lengthscale = PyroParam(lengthscale, constraints.positive)
        
        self.kernel_underlying = kernel_underlying
        self.kernel_instance = None
        
        
    def init_kernel(self, kernel, input_dim):
        """ Check whether we have an instance of the kernel, and instantiate
            one (with default params) if not.
        """
        if isinstance(kernel, self._SUPPORTED_KERNELS):
            kernel = deepcopy(kernel)
        elif isinstance(kernel, Kernel):
            raise NotImplementedError("Unsupported kernel")
        else:
            kernel = kernel(input_dim=input_dim, variance=torch.tensor(1.0))
        return kernel
    
    @staticmethod
    def flatten(patches):
        return patches.flatten(start_dim=2)
    
    def extract_image_patches(self, x, stride=1, dilation=1):
        num_windows = x.shape[2] - self.patch_size + 1
        b,c,h,w = x.shape
        h2 = math.ceil(h / stride)
        w2 = math.ceil(w / stride)
        pad_row = (h2 - 1) * stride + (num_windows - 1) * dilation + 1 - h
        pad_col = (w2 - 1) * stride + (num_windows - 1) * dilation + 1 - w
        pad_row, pad_col = 0, 0
        x = F.pad(x, (pad_row//2, pad_row - pad_row//2, pad_col//2, pad_col - pad_col//2))

        # Extract patches
        patches = x.unfold(2, num_windows, stride).unfold(3, num_windows, stride)
        patches = patches.permute(0,4,5,1,2,3).contiguous()

        return patches.view(b,-1,patches.shape[-2], patches.shape[-1])
    
    
    def forward(self, X, Z=None):
        """ Compute kernel matrix. """   
        with torch.no_grad():
            if Z is None:
                Z = X
 
            if self.kernel_instance is None:
                self.kernel_instance = self.init_kernel(self.kernel_underlying, self.patch_size ** 2)

            x_point_patches = self.flatten(self.extract_image_patches(X[:,:-1].reshape(-1, 1, 28, 28)))
            z_point_patches = self.flatten(self.extract_image_patches(Z[:,:-1].reshape(-1, 1, 28, 28)))
            
            kern = torch.zeros(X.shape[0], Z.shape[0], dtype=float)
            assert x_point_patches.shape[1] == z_point_patches.shape[1]
            for i in range(x_point_patches.shape[0]):
                for j in range(z_point_patches.shape[0]):
                    k = self.kernel_instance.forward(x_point_patches[i], z_point_patches[j])
                    kern[i, j] = torch.sum(k)
            return self.variance * kern
