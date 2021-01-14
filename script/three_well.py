import numpy as np
import sys
sys.path.append('./../src/')
sys.path.append('./src/')

from SDE_solver import solve_sde_RK
from utils import plot_trajectories_2
import matplotlib.pyplot as plt
from MLE_drift import *
import torch

from celluloid import Camera
import math

feature_x = np.arange(-1.5, 1.5, 0.05)
feature_y = np.arange(-1, 2, 0.05)
x, y = np.meshgrid(feature_x, feature_y)
z = 3*np.exp(-x**2 - (y-(1.0/3)) **2) - 3*np.exp( -x**2 - (y-(5.0/3))**2) - 5 * np.exp( -(x-1)**2-y**2) - 5*np.exp(-(x+1)**2-y**2 ) + 0.2*x**4 + 0.2*(y-(1.0/3))**4

deriv_x = lambda x,y: -(0.8*x**3 + 6*x*np.exp(-x**2-(y-(5.0/3))**2) - 6*x*np.exp( -x**2 - (y-(1.0/3))**2) + 10*(x-1)*np.exp(-(x-1)**2 - y**2 ) + 10*(x+1)*np.exp(-(x+1)**2 - y**2))
deriv_y = lambda x,y: -(-6*(y-(1.0/3)) * np.exp( -x**2 - (y-(1.0/3))**2)+6*(y-(5.0/3)) * np.exp( -x**2 - (y-(5.0/3))**2) + 10*y*np.exp(-(x-1)**2-y**2)+ 10*y*np.exp(-(x+1)**2-y**2) + 0.8*(y-(1.0/3))**3)
prior_drift = lambda X: torch.tensor([[deriv_x(i[0],i[1]),deriv_y(i[0],i[1])] for i in X])
num_samples=2
sigma = 0.5
dt = 0.01
N = int(math.ceil(1.0/dt))
mu_0 = torch.tensor([1.0,0.0])
X_0 = torch.distributions.multivariate_normal.MultivariateNormal(mu_0,torch.eye(2)*0.05).sample((num_samples,1)).reshape((-1,2))
mu_1 = torch.tensor([-1.0,0.0])
X_1 = torch.distributions.multivariate_normal.MultivariateNormal(mu_1,torch.eye(2)*0.05).sample((num_samples,1)).reshape((-1,2))

result = MLE_IPFP(X_0,X_1,N=N,sigma=0.1,prior_drift=prior_drift)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
camera = Camera(fig)
M = result[-1][1]
M2 = result[-1][3]
for i in range(N):
    ax1.contourf(feature_x, feature_y, z)
    ax2.contourf(feature_x, feature_y, z)
    ax1.scatter(M[:, i, 0].detach(), M[:, i, 1].detach())
    ax2.scatter(M2[:, i, 0].detach(), M2[:, i, 1].detach())
    ax1.set_title("Forward")
    ax2.set_title("Backward")

    ax1.text(0.9, 0, r'$X_0$', fontsize=20, color='red')
    ax2.text(0.9, 0, r'$X_0$', fontsize=20, color='red')
    ax1.text(-1.1, 0, r'$X_1$', fontsize=20, color='red')
    ax2.text(-1.1, 0, r'$X_1$', fontsize=20, color='red')

    camera.snap()
animation = camera.animate()
animation.save('./assets/animation.mp4')

#HTML(animation.to_html5_video())