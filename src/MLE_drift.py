
import torch
from SDE_solver import solve_sde_RK
from GP import MultitaskGPModel
from tqdm import tqdm

def fit_drift(Xts,N,dt):
    X_0 = Xts[:, 0, 0].reshape(-1, 1)  # Extract starting point
    Ys = ((Xts[:, 1:, :-1] - Xts[:, :-1, :-1]) / dt).reshape((-1, Xts.shape[2] - 1)) # Autoregressive targets y = (X_{t+e} - X_t)/dt
    Xs = Xts[:, :-1, :].reshape((-1, Xts.shape[2])) #Drop the last timepoint in each timeseries 
    gp_drift_model = MultitaskGPModel(Xs, Ys)  # Setup the GP
    # fit_gp(gp_drift_model, num_steps=5) # Fit the drift
    gp_ou_drift = lambda x: gp_drift_model.predict(x)  # Extract mean drift
    return gp_ou_drift




def MLE_IPFP(X_0,X_1,N=10,sigma=1,iteration=10,prior_drift=None):
    if prior_drift == None:
        prior_drift = lambda x: torch.tensor([0]*X_0.shape[1]).reshape((1,-1)).repeat(X_0.shape[0],1)
    dt = 1.0 / N

    # Estimating the backward drift of brownian motion
    # Start in X_0 and go forward. Then flip the series and learn a backward drift: drift_backward

    t, Xts = solve_sde_RK(b_drift=prior_drift, sigma=sigma, X0=X_0, dt=dt, N=N)
    #plot_trajectories_2(Xts, t)

    Xts[:,:,:-1] = Xts[:,:,:-1].flip(1) # Reverse the series
    drift_backward = fit_drift(Xts,N=N,dt=dt)

    result = []
    for i in tqdm(range(iteration)):
        # Estimate the forward drift
        # Start from the end X_1 and then roll until t=0
        t, Xts = solve_sde_RK(b_drift=drift_backward, sigma=sigma, X0=X_1,dt=dt, N=N)
        #plot_trajectories_2(Xts, t)

        # Reverse the series
        Xts[:,:,:-1] = Xts[:,:,:-1].flip(1)
        drift_forward = fit_drift(Xts,N=N,dt=dt)


        # Estimate backward drift
        # Start from X_0 and roll until t=1 using drift_forward
        t, Xts = solve_sde_RK(b_drift=drift_forward, sigma=sigma, X0=X_0,dt=dt, N=N)

        # Reverse the series
        Xts[:,:,:-1] = Xts[:,:,:-1].flip(1)
        drift_backward = fit_drift(Xts,N=N,dt=dt)


        T, M = solve_sde_RK(b_drift=drift_forward, sigma=sigma, X0=X_0,dt=dt, N=N)
        T2, M2 = solve_sde_RK(b_drift=drift_backward, sigma=sigma, X0=X_1,dt=dt, N=N)
        result.append([T,M,T2,M2])
    return result
