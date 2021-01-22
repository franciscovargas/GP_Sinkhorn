from gp_sinkhorn.gmm_torch.gmm import *
from TrajectoryNet.dataset import EBData
import torch


def get_prior_EB():
    def pred_grad(clf,x,return_pred=False):
        with torch.set_grad_enabled(True):
            x_new = x[:5].requires_grad_(True)
            pred = clf.score_samples(x_new)
            pred.backward()
            if return_pred:
                return x_new.grad.detach().numpy(),pred
            else:
                return x_new.grad.detach().numpy()
    ds = EBData('pcs', max_dim=5)
    data = torch.tensor(ds.get_data())
    clf = GaussianMixture(15,5)
    clf.fit(data)
    y = clf.predict(data)
    prior_drift = lambda X: torch.tensor([pred_grad(clf,i) for i in X])
    return prior_drift