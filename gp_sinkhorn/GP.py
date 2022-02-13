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
                (lambda xx: gp_mean_function(xx)[:, i].reshape(-1))
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
                                    reuse_kernel=reuse_kernel)
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

class SparseGPRegression_fast(gp.models.SparseGPRegression):

    def __init__(self, X, y, kernel, Xu, noise=None, mean_function=None, 
                 jitter=1e-6, precompute_inv=None, device=None):
        if mean_function is not None:
            super().__init__(X, y, kernel, Xu, mean_function=mean_function, 
                             jitter=jitter, noise=noise)
            self.mean_function = lambda x: torch.zeros(*x.shape[:-1]).to(device)
        else:
            super().__init__(X, y, kernel, Xu, jitter=jitter,noise=noise)
        if precompute_inv == None:
            N = self.X.size(0)
            M = self.Xu.size(0)

            Kuu = self.kernel(self.Xu).contiguous()
            Kuu.view(-1)[::M + 1] += self.jitter  # add jitter to the diagonal
            self.Luu = Kuu.cholesky()

            Kuf = self.kernel(self.Xu, self.X)

            W = Kuf.triangular_solve(self.Luu, upper=False)[0]
            D = self.noise.expand(N).to(device)

            self.W_Dinv = W / D
            K = self.W_Dinv.matmul(W.t()).contiguous()
            K.view(-1)[::M + 1] += 1  # add identity matrix to K
            self.L = K.cholesky()
        else:
            self.Luu, self.L, self.W_Dinv = precompute_inv
    
    def forward(self, Xnew, full_cov=False, noiseless=True, reuse_kernel=None):
        r"""
        Computes the mean and covariance matrix (or variance) of Gaussian Process
        posterior on a test input data :math:`X_{new}`:
        .. math:: p(f^* \mid X_{new}, X, y, k, X_u, \epsilon) = \mathcal{N}(loc, cov).
        .. note:: The noise parameter ``noise`` (:math:`\epsilon`), the inducing-point
            parameter ``Xu``, together with kernel's parameters have been learned from
            a training procedure (MCMC or SVI).
        :param torch.Tensor Xnew: A input data for testing. Note that
            ``Xnew.shape[1:]`` must be the same as ``self.X.shape[1:]``.
        :param bool full_cov: A flag to decide if we want to predict full covariance
            matrix or just variance.
        :param bool noiseless: A flag to decide if we want to include noise in the
            prediction output or not.
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        self._check_Xnew_shape(Xnew)
    
        N = self.X.size(0)
        M = self.Xu.size(0)
        
        # get y_residual and convert it into 2D tensor for packing
        mu = self.mean_function(self.X)

        y_residual = self.y.reshape(*mu.shape) - mu
        y_2D = y_residual.reshape(-1, N).t()
        W_Dinv_y = self.W_Dinv.matmul(y_2D)
        
        if reuse_kernel is None:           
            Kus = self.kernel(self.Xu, Xnew)
            Ws = Kus.triangular_solve(self.Luu, upper=False)[0]
            reuse_kernel = Ws
            
        pack = torch.cat((W_Dinv_y, reuse_kernel), dim=1)
        Linv_pack = pack.triangular_solve(self.L, upper=False)[0]
            
        # unpack
        Linv_W_Dinv_y = Linv_pack[:, :W_Dinv_y.shape[1]]
        Linv_Ws = Linv_pack[:, W_Dinv_y.shape[1]:]
        
        C = Xnew.size(0)
        loc_shape = self.y.shape[:-1] + (C,)
        loc = Linv_W_Dinv_y.t().matmul(Linv_Ws).reshape(loc_shape)

        mu_new = self.mean_function(Xnew)
        return loc.reshape(*mu_new.shape) + mu_new, reuse_kernel



class MultitaskGPModelSparse(MultitaskGPModel):
    """
    Nystr\"om approximation [Chris Williams and Matthias Seeger, 2001]
    applied to multioutput GP.
    
    Time series are subsampled randomly in a hiearachical fashion.
        - First sample a time series
        - then subsample a fixed number of timepoints within the time series
    """
    
    
    @staticmethod
    def create_inducing_points_nystrom(X, num_data_points, num_time_points):
        """
        Vectorised subsampling of inputs for Nystr\"om approximation
       
        :param X[nxd' ndarray]: input X for GP (Flatened AR time series) to subsample
        :param num_data_points[int]: Number of inducing samples from the boundary distributions
        :param num_time_points[int]: Number of inducing timesteps for the EM approximation
        """
        # First we sample the time series and then we sample time indices for each series         

        # Assuming series came in order we first find the original number of timepoints
        _max = X[:,-1].max().item()
        original_time_length = torch.where(X[:,-1] == _max)[0][0].item() + 1

        # Sample several timeseries
        perm = torch.randperm(int(X.size(0) / original_time_length))
        idx = perm[:num_data_points]
        idxs = ((idx * original_time_length).reshape(-1,1) + 
                torch.arange(original_time_length)).flatten()
        samples = X[idxs, :]

        # Sample timepoints without replacement for each timeseries
        prob_dist = torch.ones((num_data_points, original_time_length))
        inx_matrix = (torch.multinomial(prob_dist, num_time_points, 
                                        replacement=False) *
                      torch.arange(1, num_data_points+1).reshape(-1,1))
        return samples[inx_matrix.flatten(),:]
     
        
    def __init__(self, X, y, noise=.1, dt=1, num_data_points=10, 
                 num_time_points=50, nystrom_only=True, kern=gp.kernels.RBF, 
                 gp_mean_function=None, device=None):
        """
        For our scenario d' = d+1 
        
        :param X[nxd' ndarray]: input X for GP (Flatened AR time series)
        :param y[nxd]: multioutput targets for GP
        :param dt[float]: stepsize of SDE discretisation
        :param num_data_points[int]: Number of inducing samples(inducing points) from the boundary distributions
        :param num_time_points[int]: Number of inducing timesteps(inducing points) for the EM approximation
        :nystrom_only [bool]: If true disabels variational lower bound optimisation
        
        :param kern[a -> gp.krenls.Base]: function that takes in parameter
                                          and returns a kernel. Can be
                                          used in conjunction to functools
                                          partial to specify paramas a priori
        """
        self.dim = y.shape[1]
        self.gpr_list = []
        with torch.no_grad():
            self.nystrom_only = nystrom_only
            Xu = self.create_inducing_points_nystrom(X, num_data_points, 
                                                     num_time_points)

            kernel = kern(input_dim=X.shape[1]) # changed from Matern32
            for i in range(y.shape[1]):
                gp_mean_function_i = (
                    (lambda X: gp_mean_function(X)[:,i].reshape(-1,1)) 
                    if gp_mean_function 
                    else None)
                if i == 0:
                    gpr = SparseGPRegression_fast(
                        X, y[:, i], kernel, noise=torch.tensor(noise / (dt)), 
                        Xu=Xu, mean_function=gp_mean_function_i,
                        device=device
                    )
                else:
                    gpr = SparseGPRegression_fast(
                        X, y[:, i], kernel,
                        noise=torch.tensor(noise / (dt)),
                        Xu=Xu, precompute_inv=(
                            self.gpr_list[0].Luu,
                            self.gpr_list[0].L,
                            self.gpr_list[0].W_Dinv
                        ),
                        mean_function=gp_mean_function_i,
                        device=device
                    )
                self.gpr_list.append(gpr)
    
    def fit_gp(self, num_steps=30):
        """
        Fits variational approximation (inducing points) 
        as well as GP parameters
        
        VI opt should only be potentially used 
        outside of the IPFP loop if used at all, it is too
        slow inside the IPFP loop.
        """
        if self.nystrom_only: return # no var_opt if nystrom only
        
        return super().fit_gp(num_steps=num_steps)
