import torch
import pyro.contrib.gp as gp
import math

class MultitaskGPModel():
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
        #optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)
        #loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        #losses = []
        #for i in range(num_steps):
        #    optimizer.zero_grad()
        #    loss = loss_fn(gpr.model, gpr.guide)
        #    loss.backward()
        #    optimizer.step()
        #    losses.append(loss.item())
