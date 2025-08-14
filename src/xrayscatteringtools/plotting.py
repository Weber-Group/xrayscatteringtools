import matplotlib.pyplot as plt

def plot_jungfrau(x, y, f, ax=None, shading='nearest', *args, **kwargs):
    """Plot Jungfrau detector counts.

    Parameters
    ----------
    x, y : list of np.ndarray
        Coordinates for each tile of the detector.
    f : list of np.ndarray
        Data to be plotted for each tile.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.
    shading : str, optional
        Shading style for pcolormesh (default: 'nearest').

    Returns
    -------
    pcm : matplotlib.collections.QuadMesh
        The QuadMesh object created.
    """
    if ax is None:
        ax = plt.gca()
    for i in range(8):
        pcm = ax.pcolormesh(x[i], y[i], f[i], shading=shading, *args, **kwargs)
    return pcm
