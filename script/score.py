import numpy as np
import matplotlib.pyplot as plt
import scprep
import pandas as pd
from TrajectoryNet.dataset import EBData
from TrajectoryNet.optimal_transport.emd import earth_mover_distance, interpolate_with_ot

from gp_sinkhorn.SDE_solver import solve_sde_RK
from gp_sinkhorn.MLE_drift import *
from gp_sinkhorn.utils import plot_trajectories_2

import torch

from celluloid import Camera
from IPython.display import HTML
import math
import sys

f = open("./../assets/result_dump/"+sys.argv[1]+"/config.txt","rb")
print(f.read())
f.close()

ds = EBData('pcs', max_dim=5)
frame_0_start, frame_0_end = np.where(ds.labels == 1)[0][0], np.where(ds.labels == 1)[0][-1]
frame_4_start, frame_4_end = np.where(ds.labels == 3)[0][0], np.where(ds.labels == 3)[0][-1]


frame_2_start, frame_2_end = np.where(ds.labels == 2)[0][0], np.where(ds.labels == 2)[0][-1]


X_mid_f = ds.get_data()[frame_2_start:frame_2_end+1]

earth_mover_distance(X_mid_f,X_mid_f)

X_0_f = ds.get_data()[frame_0_start:frame_0_end]
X_1_f = ds.get_data()[frame_4_start:frame_4_end]

many_results = pd.read_pickle('./../assets/result_dump/'+sys.argv[1]+'/result_'+sys.argv[2]+'.pkl')
for result_epoch in many_results:

    time_forward, zs_forward, time_backward, zs_backward = result_epoch

    zs_forward_ = zs_forward[:,:,:-1]
    zs_backward_ = zs_backward[:,:,:-1]
    tpi_f = int(math.floor(zs_forward_.shape[1] * 0.5))
    tpi_b = int(math.floor(zs_backward_.shape[1] * 0.5))

    emd_f = earth_mover_distance(zs_forward_[:,tpi_f,:], X_mid_f)
    emd_b = earth_mover_distance(zs_backward_[:,tpi_b,:], X_mid_f)
    print(emd_f, emd_b)