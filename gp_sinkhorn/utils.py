import matplotlib.pyplot as plt
import numpy as np
from GPy.plotting.matplot_dep.visualize import matplotlib_show
import GPy
import pods
from celluloid import Camera
from IPython.display import HTML



def plot_trajectories_2(Xts, t, remove_time=True, fig_axs=None, color='b', show=True, direction="Forward"):
    """
    Helper function that plots multple trajctories
    """

    fn = 14
    if fig_axs is None:
        fig, axs = plt.subplots(1, 1, sharey=False, figsize=(15, 10))
        axs.set_ylabel("$x(t)$", fontsize=fn)
    else:
        fig, axs = fig_axs

    n, _, _, = Xts.shape

    if remove_time:
        Xts = Xts[..., :-1]

    for i in range(n):
        label = "$\mathbb{Q}$:" + f"{direction} process" if i == 0 else None
        axs.plot(t.cpu().flatten(), Xts[i, :, :].detach().cpu().numpy().flatten(), color, alpha=0.3, label=label)

    #     plt.show()
    return (fig, axs)


class mocap_data_show(matplotlib_show):
    """Base class for visualizing motion capture data."""

    def __init__(self, vals, axes=None, connect=None, color='b'):
        if axes==None:
            fig = plt.figure()
            axes = fig.add_subplot(111, projection='3d', aspect='equal')
        super(mocap_data_show, self).__init__(vals, axes)

        self.color = color
        self.connect = connect
        self.process_values()
        self.initialize_axes()
        self.draw_vertices()
        self.finalize_axes()
        self.draw_edges()
        self.axes.figure.canvas.draw()

    def draw_vertices(self):
        self.points_handle = self.axes.scatter(self.vals[:, 0], self.vals[:, 1], self.vals[:, 2], color=self.color)

    def draw_edges(self):
        self.line_handle = []
        if self.connect is not None:
            x = []
            y = []
            z = []
            self.I, self.J = np.nonzero(self.connect)
            for i, j in zip(self.I, self.J):
                x.append(self.vals[i, 0])
                x.append(self.vals[j, 0])
                x.append(np.NaN)
                y.append(self.vals[i, 1])
                y.append(self.vals[j, 1])
                y.append(np.NaN)
                z.append(self.vals[i, 2])
                z.append(self.vals[j, 2])
                z.append(np.NaN)
            self.line_handle = self.axes.plot(np.array(x), np.array(y), np.array(z), '-', color=self.color)

    def modify(self, vals):
        self.vals = vals.copy()
        self.process_values()
        self.initialize_axes_modify()
        self.draw_vertices()
        self.initialize_axes()
        #self.finalize_axes_modify()
        self.draw_edges()
        self.axes.figure.canvas.draw()

    def process_values(self):
        raise NotImplementedError("this needs to be implemented to use the data_show class")

    def initialize_axes(self, boundary=0.05):
        """Set up the axes with the right limits and scaling."""
        bs = [(self.vals[:, i].max()-self.vals[:, i].min())*boundary for i in range(3)]
        self.x_lim = np.array([self.vals[:, 0].min()-bs[0], self.vals[:, 0].max()+bs[0]])
        self.y_lim = np.array([self.vals[:, 1].min()-bs[1], self.vals[:, 1].max()+bs[1]])
        self.z_lim = np.array([self.vals[:, 2].min()-bs[2], self.vals[:, 2].max()+bs[2]])

    def initialize_axes_modify(self):
        self.points_handle.remove()
        self.line_handle[0].remove()

    def finalize_axes(self):
#         self.axes.set_xlim(self.x_lim)
#         self.axes.set_ylim(self.y_lim)
#         self.axes.set_zlim(self.z_lim)
#         self.axes.auto_scale_xyz([-1., 1.], [-1., 1.], [-1., 1.])

        extents = np.array([getattr(self.axes, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        sz = extents[:,1] - extents[:,0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize/2
        for ctr, dim in zip(centers, 'xyz'):
            getattr(self.axes, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
        
#         self.axes.set_aspect('equal')
#         self.axes.autoscale(enable=False)

    def finalize_axes_modify(self):
        self.axes.set_xlim(self.x_lim)
        self.axes.set_ylim(self.y_lim)
        self.axes.set_zlim(self.z_lim)

class skeleton_show(mocap_data_show):
    """data_show class for visualizing motion capture data encoded as a skeleton with angles."""
    def __init__(self, vals, skel, axes=None, padding=0, color='b'):
        """data_show class for visualizing motion capture data encoded as a skeleton with angles.
        :param vals: set of modeled angles to use for printing in the axis when it's first created.
        :type vals: np.array
        :param skel: skeleton object that has the parameters of the motion capture skeleton associated with it.
        :type skel: mocap.skeleton object
        :param padding:
        :type int
        """
        self.skel = skel
        self.padding = padding
        connect = skel.connection_matrix()
        super(skeleton_show, self).__init__(vals, axes=axes, connect=connect, color=color)

    def process_values(self):
        """Takes a set of angles and converts them to the x,y,z coordinates in the internal prepresentation of the class, ready for plotting.
        :param vals: the values that are being modelled."""

        if self.padding>0:
            channels = np.zeros((self.vals.shape[0], self.vals.shape[1]+self.padding))
            channels[:, 0:self.vals.shape[0]] = self.vals
        else:
            channels = self.vals
        vals_mat = self.skel.to_xyz(channels.flatten())
        self.vals = np.zeros_like(vals_mat)
        # Flip the Y and Z axes
        self.vals[:, 0] = vals_mat[:, 0].copy()
        self.vals[:, 1] = vals_mat[:, 2].copy()
        self.vals[:, 2] = vals_mat[:, 1].copy()

    def wrap_around(self, lim, connect):
        quot = lim[1] - lim[0]
        self.vals = rem(self.vals, quot)+lim[0]
        nVals = floor(self.vals/quot)
        for i in range(connect.shape[0]):
            for j in find(connect[i, :]):
                if nVals[i] != nVals[j]:
                    connect[i, j] = False
        return connect

# main

def cmu_mocap(
    data_Y, data, subject='35', motion=['01'], in_place=True,
    optimize=True, verbose=True, plot=True,
     axes=None, time_index=1, camera=None, standardise=True
     ):
    
    if in_place:
        # Make figure move in place.
        data_Y[:, 0:3] = 0.0
  
    Y_mean = data_Y.mean(0)
    Y_std = data_Y.std(0)

    if in_place:
        Y_std[0:3] = 1.0
    
    Yn = (data_Y - Y_mean) / Y_std

    if plot:
        y = data_Y[time_index, :] if not standardise else Yn
#         import pdb; pdb.set_trace()
        data_show = skeleton_show(y[None, :], data['skel'], axes=axes)
        if camera:
            print(time_index)
            camera.snap()
        # data_show.close()

def get_subject_data(subject='35', motion=['01']):
    return pods.datasets.cmu_mocap(subject, motion)

# Skeleton 
def animate_skeleton(Y, data, notebook=True, standardise=False):
    
    fig = plt.figure()
    if notebook:
        axes = fig.add_subplot(111, projection='3d')
    else:
        axes = fig.add_subplot(111, projection='3d', aspect='auto')

    camera = Camera(fig)
    N, d = Y.shape

    for i in range(N):    
        cmu_mocap(
            Y, data, axes=axes,
            camera=camera, time_index=i, in_place=False,
            standardise=standardise
        )
    return camera
    