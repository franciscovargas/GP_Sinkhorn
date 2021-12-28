import torch
import pyro.contrib.gp as gp
from pyro.contrib.gp.util import conditional
import torch

from pyro.infer import TraceMeanField_ELBO
from pyro.infer.util import torch_backward, torch_item

import time
from pyro.contrib.gp.kernels import (Exponential, Matern32, RBF, Brownian, 
                                     Combination, Product, Sum, Kernel)
from copy import deepcopy

class GPRegression_fast(gp.models.GPRegression):

    def __init__(self, X, y, kernel, noise=None, mean_function=None, 
                 jitter=1e-6, precompute_inv=None):
        with torch.no_grad():
            if isinstance(kernel, Product): 
                kernel = deepcopy(kernel)

            self.mean_flag = mean_function
            if mean_function is not None:
                super().__init__(X, y, kernel, mean_function=mean_function,
                                 jitter=jitter, noise=noise)
            else:
                super().__init__(X, y, kernel, jitter=jitter, noise=noise)
            if precompute_inv is None:
                N = self.X.size(0)
                Kff = self.kernel(self.X).contiguous()

                # Add noise to diagonal
                Kff.view(-1)[::N + 1] += self.jitter + self.noise 
                self.Kff_inv = torch.inverse(Kff)
            else:
                self.Kff_inv = precompute_inv

    def forward(self, Xnew, full_cov=False, noiseless=True, reuse_kernel=None, 
                debug=False):
        r"""
        Computes the mean and covariance matrix (or variance) of Gaussian 
        Process posterior on a test input data :math:`X_{new}`:

        .. math:: p(f^* \mid X_{new}, X, y, k, \epsilon) = \mathcal{N}(loc, cov)

        .. note:: The noise parameter ``noise`` (:math:`\epsilon`) together with
            kernel's parameters have been learned from a training procedure 
            (MCMC or SVI).

        :param torch.Tensor Xnew: A input data for testing. Note that
            ``Xnew.shape[1:]`` must be the same as ``self.X.shape[1:]``.
        :param bool full_cov: A flag to decide if we want to predict full 
            covariance matrix or just variance.
        :param bool noiseless: A flag to decide if we want to include noise in 
            the prediction output or not.
        :returns: loc and covariance matrix (or variance) of 
            :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        self._check_Xnew_shape(Xnew)
        self.set_mode("guide")
        with torch.no_grad():

            N = self.X.size(0)

            y_residual = self.y - self.mean_function(self.X)

            # print("y_residual", y_residual.shape, self.X.shape)
            if debug:  # self.mean_flag is not None:
                import pdb; pdb.set_trace()
            if reuse_kernel == None:
                Kfs = self.kernel(self.X, Xnew)
                reuse_kernel = torch.mm(Kfs.T,self.Kff_inv)

            loc = torch.mm(reuse_kernel,y_residual.reshape((-1,1)))

    #         outer = loc + self.mean_function(Xnew)
    #         if self.mean_flag is not None:
    #             print("loc.shape", loc.shape, self.mean_function(Xnew).shape, 
    #                   outer.shape)
            mfnew = (self.mean_function(Xnew) if self.mean_flag is None else 
                     self.mean_function(Xnew).reshape(-1,1))
        return (loc + mfnew),reuse_kernel

class MultitaskGPModel():
    """
    Independent (block diagonal K) Multioutput GP model
    
    Fits a separate GP per dimension for the SDE drift estimation
    """

    def __init__(self, X, y, noise=.1, dt=1, kern=gp.kernels.RBF, 
                 gp_mean_function=None):
        """
        For our scenario d' = d+1 
        
        :param X[nxd' ndarray]: input X for GP (Flatened AR time series)
        :param y[nxd]: multioutput targets for GP
        :param dt[float]: stepsize of SDE discretisation
        :param kern[a -> gp.kernels.Base]: function that takes in parameter
                                           and returns a kernel. Can be
                                           used in conjunction to functools
                                           partial to specify params a priori
        """
        self.dim = y.shape[1]
        self.gpr_list = []
        if isinstance(kern, Kernel):
            kernel = deepcopy(kern)
        else:
            # changed from Matern32
            kernel = kern(input_dim=X.shape[1], variance=torch.tensor(1.0 / dt)) 

        # if noise is multi dimensional select the right one
        noise = [noise] * y.shape[1] if isinstance(noise, (int, float)) else noise  
        for i in range(y.shape[1]):
            gp_mean_function_i = (
                lambda xx: gp_mean_function(xx)[:, i].reshape(-1)
                if gp_mean_function 
                else None)
#                 if gp_mean_function_i is not None:
#                     import pdb; pdb.set_trace()
            if i == 0:
                gpr = GPRegression_fast(
                    X, y[:, i], kernel, noise=torch.tensor(noise[i] / dt), 
                    mean_function=gp_mean_function_i)
            else:
                gpr = GPRegression_fast(
                    X, y[:, i], kernel, noise=torch.tensor(noise[i] / dt),
                    precompute_inv=self.gpr_list[0].Kff_inv,
                    mean_function=gp_mean_function_i)
            self.gpr_list.append(gpr)

    def predict(self, X, debug=False):
        """
        Evaluates the drift on the inputs:
        
                X = cat(X_t , t) -> predict(X) = b(X_t, t)
        
        :param X[nxd' ndarray]: state + time vector inputs to evaluate
                                GPDrift on
        """
        mean_list = []
        reuse_kernel = None
        for gpr in self.gpr_list:
            # your code here
            mean,reuse_kernel = gpr(X, full_cov=True, noiseless=True,
                                    reuse_kernel=reuse_kernel, debug=debug)
            mean_list.append(mean.double().reshape((-1, 1)))
        return torch.cat(mean_list, dim=1)
    
    def fit_gp(self, num_steps=30):
        """
        Fits GP hyperparameters. Only to be potentially used 
        outside of the IPFP loop if used at all
        """
        raise("Need to fix grads before running this")
        # for gpr in self.gpr_list:
        #     optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)
        #     loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        #     losses = []
        #     for i in range(num_steps):
        #         optimizer.zero_grad()
        #         loss = loss_fn(gpr.model, gpr.guide)
        #         loss.backward(retain_graph=True)
        #         optimizer.step()
        #         losses.append(loss.item())

