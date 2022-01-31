import copy
import torch
import math
import numpy as np
import matplotlib.pyplot as plt

def plot_trajectories_both_3d(Xts, t , Xts_, t_, name=None):

    fn = 14
    fig = plt.figure(figsize=(15,10))
    n = Xts.shape[0]
    
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_xlabel("$t$", fontsize=fn)
    ax.set_ylabel("$x(t)$", fontsize=fn)
    ax.set_zlabel("$y(t)$", fontsize=fn)
    for i in range(n):
        label = "$\mathbb{Q}$: Forward process" if i == 0 else None
        ti, xi = t.cpu().flatten(), Xts[i, :, :-1].detach().cpu().numpy()
        ax.plot(ti, xi[:, 0], xi[:, 1],  'b', alpha=0.3,  label=label)
    
    ax.legend(fontsize=fn)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    tt = ax.get_xticks()
    ax.set_xlabel("$t$", fontsize=fn)
    ax.set_ylabel("$x(t)$", fontsize=fn)
    ax.set_zlabel("$y(t)$", fontsize=fn)
    
    n = Xts_.shape[0]
    
    ax.set_xticks(tt.flatten() )
    ax.set_xticklabels(list(map (lambda x: '{0:.2f}'.format((x)), tt))[::-1])
    for i in range(n):
        label = "$\mathbb{P}$: Reverse process" if i == 0 else None
        ti, xi = t_.cpu().flatten(), Xts_[i,:, :-1].detach().cpu().numpy()
        ax.plot(ti, xi[:,0], xi[:,1], 'r', alpha=0.3, label=label)

    ax.legend(fontsize=fn)
    
    if name is not None:
        plt.savefig(name)
    plt.show()
    
def plot_tensor(X, color="blue"):
    plt.plot(X[:, 0].detach().cpu().numpy(), X[:, 1].detach().cpu().numpy(), ".", color=color)
    
def display_result(result, X1, X2):
    T, M, T2, M2 = result[-1]
    plot_trajectories_both_3d(M, T, M2, T2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    plt.sca(ax1)
    plot_tensor(X2, "orange")
    plot_tensor(M[:, -1, :], "blue")
    plt.sca(ax2)
    plot_tensor(X1, "orange")
    plot_tensor(M2[:, -1, :], "blue")    
    plt.show()