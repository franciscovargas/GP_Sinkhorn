import torch
import pyro.contrib.gp as gp
import math
import pyro
from pyro.contrib.gp.util import conditional
import torch

from pyro.infer import TraceMeanField_ELBO
from pyro.infer.util import torch_backward, torch_item


import time




def conditional(Xnew, X, kernel, f_loc, f_scale_tril=None, Lff=None, full_cov=False,
                whiten=False, jitter=1e-6):
    r"""
    Given :math:`X_{new}`, predicts loc and covariance matrix of the conditional
    multivariate normal distribution

    .. math:: p(f^*(X_{new}) \mid X, k, f_{loc}, f_{scale\_tril}).

    Here ``f_loc`` and ``f_scale_tril`` are variation parameters of the variational
    distribution

    .. math:: q(f \mid f_{loc}, f_{scale\_tril}) \sim p(f | X, y),

    where :math:`f` is the function value of the Gaussian Process given input :math:`X`

    .. math:: p(f(X)) \sim \mathcal{N}(0, k(X, X))

    and :math:`y` is computed from :math:`f` by some likelihood function
    :math:`p(y|f)`.

    In case ``f_scale_tril=None``, we consider :math:`f = f_{loc}` and computes

    .. math:: p(f^*(X_{new}) \mid X, k, f).

    In case ``f_scale_tril`` is not ``None``, we follow the derivation from reference
    [1]. For the case ``f_scale_tril=None``, we follow the popular reference [2].

    References:

    [1] `Sparse GPs: approximate the posterior, not the model
    <https://www.prowler.io/sparse-gps-approximate-the-posterior-not-the-model/>`_

    [2] `Gaussian Processes for Machine Learning`,
    Carl E. Rasmussen, Christopher K. I. Williams

    :param torch.Tensor Xnew: A new input data.
    :param torch.Tensor X: An input data to be conditioned on.
    :param ~pyro.contrib.gp.kernels.kernel.Kernel kernel: A Pyro kernel object.
    :param torch.Tensor f_loc: Mean of :math:`q(f)`. In case ``f_scale_tril=None``,
        :math:`f_{loc} = f`.
    :param torch.Tensor f_scale_tril: Lower triangular decomposition of covariance
        matrix of :math:`q(f)`'s .
    :param torch.Tensor Lff: Lower triangular decomposition of :math:`kernel(X, X)`
        (optional).
    :param bool full_cov: A flag to decide if we want to return full covariance
        matrix or just variance.
    :param bool whiten: A flag to tell if ``f_loc`` and ``f_scale_tril`` are
        already transformed by the inverse of ``Lff``.
    :param float jitter: A small positive term which is added into the diagonal part of
        a covariance matrix to help stablize its Cholesky decomposition.
    :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
    :rtype: tuple(torch.Tensor, torch.Tensor)
    """
    # p(f* | Xnew, X, kernel, f_loc, f_scale_tril) ~ N(f* | loc, cov)
    # Kff = Lff @ Lff.T
    # v = inv(Lff) @ f_loc  <- whitened f_loc
    # S = inv(Lff) @ f_scale_tril  <- whitened f_scale_tril
    # Denote:
    #     W = (inv(Lff) @ Kf*).T
    #     K = W @ S @ S.T @ W.T
    #     Q** = K*f @ inv(Kff) @ Kf* = W @ W.T
    # loc = K*f @ inv(Kff) @ f_loc = W @ v
    # Case 1: f_scale_tril = None
    #     cov = K** - K*f @ inv(Kff) @ Kf* = K** - Q**
    # Case 2: f_scale_tril != None
    #     cov = K** - Q** + K*f @ inv(Kff) @ f_cov @ inv(Kff) @ Kf*
    #         = K** - Q** + W @ S @ S.T @ W.T
    #         = K** - Q** + K
    start_time = time.time()

    N = X.size(0)
    M = Xnew.size(0)
    latent_shape = f_loc.shape[:-1]

    if Lff is None:
        Kff = kernel(X).contiguous()
        Kff.view(-1)[::N + 1] += jitter  # add jitter to diagonal
        Lff = Kff.cholesky()
    t = time.time()
    Kfs = kernel(X, Xnew)
    # convert f_loc_shape from latent_shape x N to N x latent_shape
    f_loc = f_loc.permute(-1, *range(len(latent_shape)))
    # convert f_loc to 2D tensor for packing
    f_loc_2D = f_loc.reshape(N, -1)
    if f_scale_tril is not None:
        # convert f_scale_tril_shape from latent_shape x N x N to N x N x latent_shape
        f_scale_tril = f_scale_tril.permute(-2, -1, *range(len(latent_shape)))
        # convert f_scale_tril to 2D tensor for packing
        f_scale_tril_2D = f_scale_tril.reshape(N, -1)

    if whiten:
        v_2D = f_loc_2D
        W = Kfs.triangular_solve(Lff, upper=False)[0].t()
        if f_scale_tril is not None:
            S_2D = f_scale_tril_2D
    else:
        pack = torch.cat((f_loc_2D, Kfs), dim=1)
        if f_scale_tril is not None:
            pack = torch.cat((pack, f_scale_tril_2D), dim=1)
        t = time.time()
        Lffinv_pack = pack.triangular_solve(Lff, upper=False)[0]
        # unpack
        v_2D = Lffinv_pack[:, :f_loc_2D.size(1)]
        W = Lffinv_pack[:, f_loc_2D.size(1):f_loc_2D.size(1) + M].t()
        if f_scale_tril is not None:
            S_2D = Lffinv_pack[:, -f_scale_tril_2D.size(1):]

    loc_shape = latent_shape + (M,)
    loc = W.matmul(v_2D).t().reshape(loc_shape)

    if full_cov:
        Kss = kernel(Xnew)
        Qss = W.matmul(W.t())
        cov = Kss - Qss
    else:
        Kssdiag = kernel(Xnew, diag=True)
        Qssdiag = W.pow(2).sum(dim=-1)
        # Theoretically, Kss - Qss is non-negative; but due to numerical
        # computation, that might not be the case in practice.
        var = (Kssdiag - Qssdiag).clamp(min=0)

    if f_scale_tril is not None:
        W_S_shape = (Xnew.size(0),) + f_scale_tril.shape[1:]
        W_S = W.matmul(S_2D).reshape(W_S_shape)
        # convert W_S_shape from M x N x latent_shape to latent_shape x M x N
        W_S = W_S.permute(list(range(2, W_S.dim())) + [0, 1])

        if full_cov:
            St_Wt = W_S.transpose(-2, -1)
            K = W_S.matmul(St_Wt)
            cov = cov + K
        else:
            Kdiag = W_S.pow(2).sum(dim=-1)
            var = var + Kdiag
    else:
        if full_cov:
            cov = cov.expand(latent_shape + (M, M))
        else:
            var = var.expand(latent_shape + (M,))
    return (loc, cov) if full_cov else (loc, var)

class GPRegression_fast(gp.models.GPRegression):

    def __init__(self, X, y, kernel, noise=None, mean_function=None, jitter=1e-6,precompute_chol=None):
        super().__init__(X, y, kernel, mean_function=mean_function, jitter=jitter,noise=noise)
        if precompute_chol == None:
            N = self.X.size(0)
            Kff = self.kernel(self.X).contiguous()
            Kff.view(-1)[::N + 1] += self.jitter + self.noise  # add noise to the diagonal
            self.Lff = Kff.cholesky()
        else:
            self.Lff = precompute_chol

    def forward(self, Xnew, full_cov=False, noiseless=True,reuse_chol=None):
        r"""
        Computes the mean and covariance matrix (or variance) of Gaussian Process
        posterior on a test input data :math:`X_{new}`:

        .. math:: p(f^* \mid X_{new}, X, y, k, \epsilon) = \mathcal{N}(loc, cov).

        .. note:: The noise parameter ``noise`` (:math:`\epsilon`) together with
            kernel's parameters have been learned from a training procedure (MCMC or
            SVI).

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
        self.set_mode("guide")
        N = self.X.size(0)

        y_residual = self.y - self.mean_function(self.X)
        loc, cov = conditional(Xnew, self.X, self.kernel, y_residual, None, self.Lff,
                               full_cov, jitter=self.jitter)

        if full_cov and not noiseless:
            M = Xnew.size(0)
            cov = cov.contiguous()
            cov.view(-1, M * M)[:, ::M + 1] += self.noise  # add noise to the diagonal
        if not full_cov and not noiseless:
            cov = cov + self.noise

        return loc + self.mean_function(Xnew), cov


class MultitaskGPModel():
    """
    Independant (block diagonal K) Multioutput GP model
    
    Fits a seperate GP per dimension for the SDE drift estimation
    """

    def __init__(self, X, y, noise=.1, dt=1, kern=gp.kernels.RBF):
        """
        For our scenario d' = d+1 
        
        :param X[nxd' ndarray]: input X for GP (Flatened AR time series)
        :param y[nxd]: multioutput targets for GP
        :param dt[float]: stepsize of SDE discretisation
        :param kern[a -> gp.krenls.Base]: function that takes in parameter
                                          and returns a kernel. Can be
                                          used in conjunction to functools
                                          partial to specify paramas a priori
        """
        self.dim = y.shape[1]
        self.gpr_list = []
        with torch.no_grad():
            kernel = kern(input_dim=X.shape[1]) # changed from Matern32
            for i in range(y.shape[1]):
                if i == 0:
                    gpr = GPRegression_fast(
                        X, y[:, i], kernel, noise=torch.tensor(noise / math.sqrt(dt))
                    )
                else:
                    gpr = GPRegression_fast(
                        X, y[:, i], kernel, noise=torch.tensor(noise / math.sqrt(dt)),precompute_chol=self.gpr_list[0].Lff
                    )
                self.gpr_list.append(gpr)

    def predict(self, X):
        """
        Evaluates the drift on the inputs:
        
                X = cat(X_t , t) -> predict(X) = b(X_t, t)
        
        :param X[nxd' ndarray]: state + time vector inputs to evaluate
                                GPDrift on
        """
        with torch.no_grad():
            mean_list = []
            Lff = None
            for gpr in self.gpr_list:
                # your code here
                mean, _ = gpr(X, full_cov=True, noiseless=True)
                mean_list.append(mean.double().reshape((-1, 1)))
            return torch.cat(mean_list, dim=1)
    
    def fit_gp(self, num_steps=30):
        """
        Fits GP hyperparameters. Only to be potentially used 
        outside of the IPFP loop if used at all
        """
        for gpr in self.gpr_list:
            optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)
            loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
            losses = []
            for i in range(num_steps):
                optimizer.zero_grad()
                loss = loss_fn(gpr.model, gpr.guide)
                loss.backward(retain_graph=True)
                optimizer.step()
                losses.append(loss.item())

        
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
        idxs = ((idx * original_time_length).reshape(-1,1) + torch.arange(original_time_length)).flatten()
        samples = X[idxs, :]

        # Sample timepoints without replacement for each timeseries
        prob_dist = torch.ones((num_data_points, original_time_length))
        inx_matrix = torch.multinomial(prob_dist, num_time_points, replacement=False) * torch.arange(1, num_data_points+1).reshape(-1,1)
        out_samps = samples[inx_matrix.flatten(),:]
     
        return out_samps
        
        
        
    def __init__(self, X, y, noise=.1, dt=1,
                 num_data_points=10, num_time_points=50,
                 nystrom_only=True, kern=gp.kernels.RBF):
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
        
        self.nystrom_only = nystrom_only
        Xu = self.create_inducing_points_nystrom(X, num_data_points, num_time_points)
        
        kernel = kern(input_dim=X.shape[1]) # changed from Matern32
        for i in range(y.shape[1]):
            gpr = gp.models.SparseGPRegression(
                X, y[:, i], kernel, noise=torch.tensor(noise / math.sqrt(dt)), Xu=Xu
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
