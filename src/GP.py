import torch
import pyro.contrib.gp as gp
import pyro
import math

class MultitaskGPModel():
    """
    Independant (block diagonal K) Multioutput GP model
    
    Fits a seperate GP per dimension.
    """
    def __init__(self, X, y, noise=.1, dt=1, ):
        self.dim = y.shape[1]
        self.gpr_list = []
        for i in range(y.shape[1]):
            #kernel = gp.kernels.RBF(input_dim=X.shape[1], variance=torch.tensor(1),
            #                        lengthscale=torch.tensor(1.))
            kernel = gp.kernels.Matern32(input_dim=X.shape[1])
            gpr = gp.models.GPRegression(X, y[:, i], kernel, noise=torch.tensor(noise / math.sqrt(dt)))
            self.gpr_list.append(gpr)

    def predict(self, X):
        mean_list = []
        for gpr in self.gpr_list:
            mean, _ = gpr(X, full_cov=True, noiseless=True)

            mean_list.append(mean.double().reshape((-1, 1)))
        return torch.cat(mean_list, dim=1)
    
    def fit_gp(self,gpr, num_steps=100):
        raise("To be implemented")

        
class MultitaskGPModelSparse(MultitaskGPModel):
    """
    Nystr\"om approximation [Williams and Seeger, 2001] applied to multitask GP
    
    Time series are subsampled randomly in a hiearachical fashion.
        - First sample a time series
        - then subsample a fixed number of timepoints within the time series
    """
    
    
    @staticmethod
    def create_inducing_points_nystrom(X, num_data_points, num_time_points):
        # First we sample the time series and then we sample time indices for each series         

        # Assuming series came in order we first find the original number of timepoints
        _max = X[:,-1].max().item()
        original_time_length = torch.where(X[:,-1] == _max)[0][0].item()  + 1

        # Sample a timeseries
        perm = torch.randperm(int(X.size(0) / original_time_length))
        idx = perm[:num_data_points]
        idxs = ((idx * original_time_length).reshape(-1,1) + torch.arange(original_time_length)).flatten()
        samples = X[idxs, :]

        # In expectation approach:             
        # out_samps  = out_samps[(samples.uniform_() > 1.0 - num_time_points * 1.0 /_max)]

        # Sample timepoints for each timeseries
        prob_dist = torch.ones((num_data_points, original_time_length))
        inx_matrix = torch.multinomial(prob_dist, num_time_points, replacement=False) * torch.arange(1, num_data_points+1).reshape(-1,1)
        out_samps = samples[inx_matrix.flatten(),:]
     
        return out_samps
        
        
        
    def __init__(self, X, y, noise=.1, dt=1, num_data_points=10, num_time_points=50, nystrom_only=True):
        self.dim = y.shape[1]
        self.gpr_list = []
        
        self.nystrom_only = nystrom_only
        Xu = self.create_inducing_points_nystrom(X, num_data_points, num_time_points)
        
        for i in range(y.shape[1]):
            #kernel = gp.kernels.RBF(input_dim=X.shape[1], variance=torch.tensor(1),
            #                        lengthscale=torch.tensor(1.))
            kernel = gp.kernels.Matern32(input_dim=X.shape[1])
            gpr = gp.models.SparseGPRegression(X, y[:, i], kernel, noise=torch.tensor(noise / math.sqrt(dt)), Xu=Xu)
            self.gpr_list.append(gpr)
    
    def fit_gp(self, num_steps=30):
        if self.nystrom_only: return
        
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