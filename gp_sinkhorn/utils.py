import matplotlib.pyplot as plt



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