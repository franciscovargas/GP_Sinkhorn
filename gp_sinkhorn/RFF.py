import torch
import math

import numpy as np

from copy import deepcopy
from torch.distributions.uniform import Uniform
from torch.distributions.gamma import Gamma
from torch.distributions.multivariate_normal import MultivariateNormal
from pyro.contrib.gp.kernels import Exponential, RBF, Kernel

from gp_sinkhorn.arccos import ArcCos

_SUPPORTED_KERNELS = (Exponential, RBF, ArcCos)

class RandomFourierFeatures:
    """ Implementation for random features approach to exact kernel 
        approximation: sample some weights, and use random features to perform
        regression. 

        Note that the NN kernel is also supported, although technically its 
        random features are not derived using Fourier analysis.
    """

    def __init__(self, x, y, num_features, kernel=RBF, noise=1, device=None, 
                 random_seed=None, debug_rff=False, jitter=1e-6, sin_cos=False,
                 var_w=1, var_b=1):
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
        self.variance_w = var_w
        self.variance_b = var_b

        self.arccos = isinstance(self.kernel, ArcCos)

        self.feature_mapping_fn = (
            self.feature_mapping_nn_simple if self.arccos else
            self.feature_mapping_rff)

        if not self.arccos:
            self.f_kernel = self.fourier_transform(self.kernel)
            
        # API consideration: should we pass result back, or just set 
        # the member variables within the init_params method?
        self.init_params()
        self.variance = self.kernel.variance

        self.phi = self.feature_mapping_fn(self.x)
        self.phi_t_phi = self.phi.t().mm(self.phi)
        self.phi_t = self.phi.t()

        if debug_rff:
            self.drift = self.predict_gp
        else:
            # TODO: is this the same as a separate regression per dimension?
            # When testing with toy examples, I put in assertions to check that
            # it was equal, and it was
            with torch.no_grad():
                self.ws = torch.linalg.lstsq(
                    self.phi_t_phi + (self.noise * torch.eye(self.phi.shape[1], 
                                                            device=self.device)),
                    self.phi_t.mm(self.y)).solution
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
                                    (dim_x, 1)).T
                gaussians = MultivariateNormal(mean, variance).sample(sample_shape).to(self.device)
                return gaussians / torch.sqrt(gammas)
            return sample_exp

    def init_params(self):
        """ Randomly sample parameters of appropriate shapes to use in later
            feature mapping functions. It is crucial to use the same random 
            weights at train and test time, which is why member variables
            are set.
        """
        if self.arccos:
            n, dim_x = self.x.shape

            std_w = math.sqrt(self.variance_w  / dim_x)
            std_b = math.sqrt(self.variance_b)

            self.w0 = torch.normal(0, std_w, size=([dim_x, self.num_features])).double().to(self.device)
            self.b0 = torch.normal(0, std_b, size=([self.num_features])).to(self.device)
        else:
            self.omega = self.f_kernel([self.num_features]).double().to(self.device).t()
            self.b = Uniform(0, 2 * math.pi).sample([self.num_features]).to(self.device)

    def feature_mapping_rff(self, x):
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
        """ Use the full GP equation to predict the value 
            for y_pred, performing the regression once per dimension.
        """
        total = torch.zeros([x_pred.shape[0], self.y.shape[1]], device=self.device)
        phi_pred = self.feature_mapping_fn(x_pred)

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
        return self.feature_mapping_fn(x_pred).mm(self.ws)

    
    def feature_mapping_nn(self, x, s=1, kappa=1e-6):
        """ Random feature mapping for NN kernel, using the formula from 
            Scetbon et. al. (2020).
        
            TODO: this is not actually used (as we are still trying to 
            debug the simple version.
        """
        n, dim_x = x.shape
        variance = self.variance
        C = (variance ** (dim_x / 2)) * np.sqrt(2)
        U = MultivariateNormal(torch.zeros(dim_x), variance * torch.eye(dim_x)
            ).sample([self.num_features]).to(self.device).t().double()

        IP = x.mm(U)

        res_trans = C * (torch.maximum(IP, torch.tensor(0)) ** s)

        V = (variance - 1) / variance
        V = -(1 / 4) * V * torch.sum(U ** 2, axis=0)
        V = torch.exp(V)

        res = torch.zeros((n, self.num_features + 1), dtype=float)

        res[:, :self.num_features] = (1 / math.sqrt(self.num_features)) * res_trans * V
        res[:, -1] = kappa

        scaling = math.sqrt(2 / self.num_features) * torch.sqrt(variance)
        return res.to(self.device) * scaling
    
    def feature_mapping_nn_simple(self, x):
        """ Random feature mapping for the NN kernel, using the basic random 
            feature formula (that in Cho and Saul (2009)). Take the product 
            of w with x, add bias b, and use the ReLU nonlearity.
        """
        x1 = torch.maximum(x.mm(self.w0) + self.b0, torch.tensor(0))

        # Scaling1 is the correct one; so we don't normalise by num_features
        scaling1 = torch.sqrt(self.variance) * math.sqrt(2)
        # scaling2 = torch.sqrt(self.variance) * math.sqrt(1 / self.num_features)
        # scaling3 = torch.sqrt(self.variance) * math.sqrt(2 / self.num_features)
        # scaling4 = torch.sqrt(self.variance) * math.sqrt(4 / self.num_features)
        
        return scaling1 * x1