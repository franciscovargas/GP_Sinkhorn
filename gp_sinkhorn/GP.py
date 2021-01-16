import torch
import pyro.contrib.gp as gp
import pyro
import math

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
                gpr = gp.models.GPRegression(
                    X, y[:, i], kernel, noise=torch.tensor(noise / math.sqrt(dt))
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
            for gpr in self.gpr_list:
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
