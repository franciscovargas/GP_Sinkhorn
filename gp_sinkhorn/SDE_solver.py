import torch
import math

import numpy as np

def solve_sde_RK(b_drift, sigma, X0=None, dt=1.0, N=100, t0=0.0,
                 theta=None, noise=False, forwards=True, device=None):
    """ Euler Mayurama method
    Syntax:
    ----------
    solve_sde(b_drift=None, sigma=None, X0=None, dt=None, N=100, t0=0, DW=None)
    Parameters:
    ----------
        b_drift: Time depdendent drift, the X state (with last dimension as 
                 time) defines the differential equation.
        sigma  : a  constant volatility
        X0     : Initial conditions of the SDE. Mandatory for SDEs
                 with variables > 1 (default: gaussian np.random)
        dt     : The timestep of the solution
                 (default: 1)
        N      : The number of timesteps (defines the length of the timeseries)
                 (default: 100)
        t0     : The initial time of the solution
                 (default: 0)
    """
    N = int(N) + 1
    
    n, d, *_ = X0.shape

    T = torch.tensor(dt * N).to(device)
    DWs = torch.empty((n, N - 1, d)).normal_(mean=0, std=1).to(device) * math.sqrt(dt)

    Y = torch.zeros((n, N, d + 1)).double().to(device)
    ti = torch.arange(N).double().to(device) * dt + t0

    t0rep = (t0 * torch.ones((X0.shape[0], 1)).double().to(device) if forwards
             else (T - t0) * torch.ones((X0.shape[0], 1)).double().to(device))

    if isinstance(sigma, (int, float)):
        sigmas = torch.full((N,), sigma).to(device)
    elif isinstance(sigma, tuple):
        # wedge shape: increasing from sigma_min up to sigma_max then decreasing
        sigma_min, sigma_max = sigma
        m = 2 * (sigma_max - sigma_min) / (N * dt)  # gradient
        sigmas = (sigma_max - m * torch.abs(ti - (t0 + 0.5 * N * dt))).double().to(device)

    Y = torch.cat((X0.to(device), t0rep), axis=1)[:, None, :]
    T = dt * N
    for n in range(N - 1):
        t = ti[n + 1]
        b, DW_n = b_drift(Y[:, n, :]), DWs[:, n, :]
        newY = Y[:, n, :-1] + b * dt + sigmas[n + 1] * DW_n

        trep = (t.repeat(newY.shape[0]).reshape(-1, 1) if forwards
                else T - t.repeat(newY.shape[0]).reshape(-1, 1))
        # print(trep)
        tocat = torch.cat((newY, trep), dim=1)[:, None, :]
        Y = torch.cat((Y, tocat), dim=1)
    if torch.isnan(Y).any() or torch.isinf(Y).any(): import pdb; pdb.set_trace()
    return ti, Y

