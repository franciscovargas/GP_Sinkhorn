import pickle
import torch
from gp_sinkhorn.SDE_solver import solve_sde_RK
from gp_sinkhorn.GP import MultitaskGPModel, MultitaskGPModelSparse
from gp_sinkhorn.NN import Feedforward, train_nn
from gp_sinkhorn.utils import (auxiliary_plot_routine_init, 
                               auxiliary_plot_routine_end)
import pyro.contrib.gp as gp
import math
from tqdm import tqdm
import gc
import copy
import time



def fit_drift_gp(Xts, N, dt, num_data_points=10, num_time_points=50, 
                 kernel=gp.kernels.RBF, noise=1.0, gp_mean_function=None, 
                 sparse=False, device=None):
    """
    This function transforms a set of timeseries into an autoregression problem 
    and estimates the drift function using GPs following:
    
        - Papaspiliopoulos, Omiros, Yvo Pokern, Gareth O. Roberts, and 
          Andrew M. Stuart.
          "Nonparametric estimation of diffusions: a differential equations 
           approach."
          Biometrika 99, no. 3 (2012): 511-531.
        - Ruttor, A., Batz, P., & Opper, M. (2013).
          "Approximate Gaussian process inference for the drift function in 
           stochastic differential equations."
          Advances in Neural Information Processing Systems, 26, 2040-2048.
    
    :param Xts[MxNxD ndarray]: Array containing M timeseries of length N of 
        dimension D
    :param N [int]: Number of samples in the time series
    :param dt [float]: time interval seperation between time points (sample 
        rate)
    
    :param num_data_points[int]: Number of inducing samples (inducing points) 
        from the boundary distributions
    :param num_time_points[int]: Number of inducing timesteps (inducing points) 
        for the EM approximation
    
    :return [nx(d+1) ndarray-> nxd ndarray]: returns fitted drift
    """
    # Extract starting point
    X_0 = Xts[:, 0, 0].reshape(-1, 1)  

    # Autoregressive targets y = (X_{t+e} - X_t)/dt
    Ys = ((Xts[:, 1:, :-1] - Xts[:, :-1, :-1]) / dt).reshape((-1, Xts.shape[2] - 1)) 

    # Drop the last timepoint in each timeseries
    Xs = Xts[:, :-1, :].reshape((-1, Xts.shape[2])) 

    # Set up GP

    if sparse:
        gp_drift_model = MultitaskGPModelSparse(
            Xs, Ys, dt=1, kern=kernel, noise=noise, 
            gp_mean_function=gp_mean_function, num_data_points=num_data_points, 
            num_time_points=num_time_points, device=device) 
    else:
        gp_drift_model = MultitaskGPModel(Xs, Ys, dt=1, kern=kernel, noise=noise, 
                                        gp_mean_function=gp_mean_function) 
    # fit_gp(gp_drift_model, num_steps=5) # Fit the drift
    
    def gp_ou_drift(x, debug=False):
        return gp_drift_model.predict(x, debug=debug)

#     # Extract mean drift
#     gp_ou_drift = lambda x,debug: gp_drift_model.predict(x, debug=debug)  
    return gp_ou_drift


def fit_drift_nn(Xts, N, dt, num_data_points=10, num_time_points=50, 
                 kernel=gp.kernels.RBF, noise=1.0, gp_mean_function=None, 
                 nn_model = Feedforward, device=None):
    """
    This function transforms a set of timeseries into an autoregression problem 
    and estimates the drift function using GPs following:
    
        - Papaspiliopoulos, Omiros, Yvo Pokern, Gareth O. Roberts, and 
          Andrew M. Stuart.
          "Nonparametric estimation of diffusions: a differential equations 
           approach."
          Biometrika 99, no. 3 (2012): 511-531.
        - Ruttor, A., Batz, P., & Opper, M. (2013).
          "Approximate Gaussian process inference for the drift function in 
           stochastic differential equations."
          Advances in Neural Information Processing Systems, 26, 2040-2048.
    
    :param Xts[MxNxD ndarray]: Array containing M timeseries of length N of 
        dimension D
    :param N [int]: Number of samples in the time series
    :param dt [float]: time interval seperation between time points (sample 
        rate)

    :param num_data_points[int]: Number of inducing samples(inducing points) 
        from the boundary distributions
    :param num_time_points[int]: Number of inducing timesteps(inducing points) 
        for the EM approximation
    
    :return [nx(d+1) ndarray-> nxd ndarray]: returns fitted drift
    """
    X_0 = Xts[:, 0, 0].reshape(-1, 1)  # Extract starting point
    
    # Autoregressive targets y = (X_{t+e} - X_t)/dt
    Ys = ((Xts[:, 1:, :-1] - Xts[:, :-1, :-1])  ).reshape((-1, Xts.shape[2] - 1)) 
    
    # Drop the last timepoint in each timeseries
    Xs = Xts[:, :-1, :].reshape((-1, Xts.shape[2])) 
    
    n, d = Xs.shape

    nn_drift_model = nn_model(input_size=d).double().to(device) #Setup the NN
    train_nn(nn_drift_model, Xs, Ys)

#     nn_drift_model = LinearRegression()
#     nn_drift_model.fit(Xs, Ys, method="lin")
#     nn_drift_model.eval()
    # fit_gp(gp_drift_model, num_steps=5) # Fit the drift
    
    def gp_ou_drift(x,debug=False):
        return nn_drift_model.predict(x, debug=debug) / dt
#     # Extract mean drift
#     gp_ou_drift = lambda x,debug: gp_drift_model.predict(x, debug=debug)  
    return gp_ou_drift


def MLE_IPFP(
        X_0, X_1, N=10, sigma=1, iteration=10, prior_drift=None,
        num_data_points=10, num_time_points=50, prior_X_0=None, prior_Xts=None,
        num_data_points_prior=None, num_time_points_prior=None, plot=False,
        kernel=gp.kernels.Exponential, observation_noise=1.0, decay_sigma=1, 
        refinement_iterations=5, div=1, gp_mean_prior_flag=False, log_dir=None,
        verbose=0, langevin=False, nn=False, device=None, sparse=False,
    ):
    """
    This module runs the GP drift fit variant of IPFP it takes in samples from 
    \pi_0 and \pi_1 as well as a the forward drift of the prior \P and computes 
    an estimate of the Schroedinger Bridge of \P,\pi_0,\pi_1:
    
                        \Q* = \argmin_{\Q \in D(\pi_0, \pi_1)} KL(\Q || \P)
    
    :params X_0[nxd ndarray]: Source distribution sampled from \pi_0 (initial 
        value for the bridge)
    :params X_1[mxd ndarray]: Target distribution sampled from \pi_1 (terminal 
        value for the bridge)
    
    :param N[int]: number of timesteps for Euler-Mayurama discretisations
    :param iteration[int]: number of IPFP iterations
    
    :param prior_drift[nx(d+1) ndarray-> nxd ndarray]: drift function of the 
        prior, defautls to Brownian

    :param num_data_points[int]: Number of inducing samples(inducing points) 
        from the boundary distributions
    :param num_time_points[int]: Number of inducing timesteps(inducing points) 
        for the EM approximation
    
    :param prior_X_0[mxd array]: The marginal for the prior distribution \P . 
        his is a free parameter which can be tweaked to encourage exploration 
        and improve results.

    :param prior_Xts[nxTxd array] : Prior trajectory that can be used on the 
        first iteration of IPML
    :param num_data_points_prior[int]: number of data inducing points to use for 
        the prior backwards drift estimation prior to the IPFP loop and thus can 
        afford to use more samples here than with `num_data_points`. Note 
        starting off IPFP with a very good estimate of the backwards drift of 
        the prior is very important and thus it is encouraged to be generous 
        with this parameter.
    :param num_time_points_prior[int]: number of time step inducing points to 
        use for the prior backwards drift estimation. Same comments as with 
        `num_data_points_prior`.
    :param decay_sigma[float]: Decay the noise sigma at each iteration.
    :param log_dir[str]: Directory to log the result. If None don't log.
    
    :return: At the moment returning the fitted forwards and backwards 
        timeseries for plotting. However should also return the forwards and 
        backwards drifts. 
    """

    if prior_drift is None:
        def prior_drift(x):
            return torch.tensor([0] * (x.shape[1] - 1)).reshape((1, -1)).repeat(x.shape[0], 1).to(device)
        
    fit_drift = fit_drift_nn if nn else fit_drift_gp
    # Setup for the priors backwards drift estimate
    prior_X_0 = X_0 if prior_X_0 is None else prior_X_0        
    num_data_points_prior = (num_data_points if num_data_points_prior is None 
                             else num_data_points_prior)
    num_time_points_prior = (num_time_points if num_time_points_prior is None 
                             else num_time_points_prior)
    drift_forward = None
        
    dt = 1.0 / N
    
    pow_ = int(math.floor(iteration / div))

    
    if isinstance(sigma, tuple):
        observation_noise = sigma[0] ** 2
    else:
        observation_noise = (sigma ** 2 if decay_sigma == 1.0 
                             else (sigma * (decay_sigma ** pow_)) ** 2)
    if langevin:
        d = sigma.shape[0]
        sigma[:int(d * 0.5)] = 0
    
    # Estimating the backward drift of brownian motion
    # Start in prior_X_0 and go forward. 
    # Then flip the series and learn a backward drift: drift_backward
    t, Xts = solve_sde_RK(b_drift=prior_drift, sigma=sigma, X0=prior_X_0, dt=dt, 
                          N=N, device=device)

    T_ = copy.deepcopy(t)
    M_ = copy.deepcopy(Xts)

    if prior_Xts is not None:
        Xts[:, :, :-1] = prior_Xts.flip(1) # Reverse the series
    else:
        Xts[:, :, :-1] = Xts[:, :, :-1].flip(1) # Reverse the series

    drift_backward = fit_drift(
        Xts, N=N, dt=dt, num_data_points=num_data_points_prior,
        num_time_points=num_time_points_prior, kernel=kernel, 
        noise=observation_noise, gp_mean_function=prior_drift, device=device,
        sparse=sparse
    )
    
    if plot and isinstance(sigma, (int, float)):
        auxiliary_plot_routine_init(Xts, t, prior_X_0, X_1, drift_backward, 
                                    sigma, N, dt, device)

    result = []
    
    prior_drift_backward = copy.deepcopy(drift_backward)
    
    iterations = iteration

    for i in tqdm(range(iterations)):
        # Estimate the forward drift
        # Start from the end X_1 and then roll until t=0
        if verbose:
            print("Solve drift forward ")
            t0 = time.time()
        t, Xts = solve_sde_RK(b_drift=drift_backward, sigma=sigma, X0=X_1, 
                              dt=dt, N=N, device=device)
        if verbose:
            print("Forward drift solved in ", time.time() - t0)
        del drift_forward
        gc.collect()
        # plot_trajectories_2(Xts, t)
        T2 = copy.deepcopy(t.clone().detach())
        M2 = copy.deepcopy(Xts.clone().detach())
        
        if i == 0: result.append([T_, M_, T2, M2])
        # Reverse the series
        Xts[:, :, :-1] = Xts[:, :, :-1].flip(1)

        if verbose:
            print("Fit drift")
            t0 = time.time()
        drift_forward = fit_drift(
            Xts, N=N, dt=dt, num_data_points=num_data_points,
            num_time_points=num_time_points, kernel=kernel, 
            noise=observation_noise, device=device,
            gp_mean_function=(prior_drift if gp_mean_prior_flag else None), 
            sparse=sparse,
        )
        if verbose:
            print("Fitting drift solved in ", time.time() - t0)

        # Estimate backward drift
        # Start from X_0 and roll until t=1 using drift_forward
        # HERE: HERE is where the GP prior kicks in and helps the most
        t, Xts = solve_sde_RK(b_drift=drift_forward, sigma=sigma, X0=X_0, dt=dt,
                              N=N, device=device)
        del drift_backward
        gc.collect()

        T = copy.deepcopy(t.clone().detach())
        M = copy.deepcopy(Xts.clone().detach())

        # Reverse the series
        Xts[:, :, :-1] = Xts[:, :, :-1].flip(1)

        drift_backward = fit_drift(
            Xts, N=N, dt=dt, num_data_points=num_data_points,
            num_time_points=num_time_points, kernel=kernel, 
            noise=observation_noise, device=device,
            gp_mean_function=(prior_drift if gp_mean_prior_flag else None), 
            sparse=sparse,
            # One wouuld think this should (worth rethinking this)
            # be prior drift backwards here
            # but that doesnt work as well,
            # Its kinda clear (intuitively)
            # that prior_drift backwards
            # as a fallback is not going to help
            # this prior, instead the prior of this GP
            # should be inherting the backwards drift
            # of the GP at iteration 1 sadly we dont 
            # have such an estimate thus this should be None
        )

        result.append([T, M, T2, M2])
        if i < iteration and i % div == 0:
            sigma *= decay_sigma

        gc.collect() # fixes odd memory leak
        if log_dir is not None:
            pickle.dump(result, open(f"{log_dir}/result_{i}.pkl" , "wb"))


    
    T2, M2 = solve_sde_RK(b_drift=drift_backward, sigma=sigma, X0=X_1, dt=dt, 
                          N=N, device=device)
    if iterations == 0 : return [(None, None, T2, M2)]
    
    T, M = solve_sde_RK(b_drift=drift_forward, sigma=sigma, X0=X_0, dt=dt, N=N, 
                        device=device)
    result.append([T, M, T2, M2])
    if log_dir is not None:
        pickle.dump(result, open(f"{log_dir}/result_final.pkl", "wb"))
        
    if plot and isinstance(sigma, (int, float)):
        auxiliary_plot_routine_end(Xts, t, prior_X_0, X_1, drift_backward, 
                                   sigma, N, dt, device)
        
    return result

