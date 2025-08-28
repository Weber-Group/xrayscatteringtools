import numpy as np
import matplotlib.pyplot as plt

def plot_jungfrau(x, y, f, ax=None, shading='auto', *args, **kwargs):
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
        Shading style for pcolormesh (default: 'auto').

    Returns
    -------
    pcm : matplotlib.collections.QuadMesh
        The QuadMesh object created.
    """

    if kwargs.get('norm'):
        # If a norm is provided, use it
        norm = kwargs['norm']
    else:
        # Determine global vmin/vmax if not explicitly given
        vmin = kwargs.get('vmin', None)
        vmax = kwargs.get('vmax', None)

        if vmax is None or vmin is None:
            # Compute min and max across all tiles
            all_data = np.concatenate([fi.ravel() for fi in f])
            if vmin is None:
                vmin = np.nanmin(all_data)
            if vmax is None:
                vmax = np.nanmax(all_data)
            # Update kwargs so all tiles use the same scale
            kwargs.update(vmin=vmin, vmax=vmax)

    if ax is None:
        ax = plt.gca()
    for i in range(8):
        Xedges = compute_pixel_edges(x[i])
        Yedges = compute_pixel_edges(y[i])
        pcm = ax.pcolormesh(Xedges, Yedges, f[i], shading=shading, *args, **kwargs)
    ax.set_aspect('equal')
    return pcm

def compute_pixel_edges(coord):
    """
    Compute the array of pixel edge coordinates from pixel center coordinates.

    Parameters
    ----------
    coord : np.ndarray
        2D array (M x N) of pixel center coordinates.

    Returns
    -------
    edges : np.ndarray
        2D array ((M+1) x (N+1)) of pixel edge coordinates.
    """
    M, N = coord.shape
    edges = np.zeros((M+1, N+1))
    
    # Interior points: average of four neighboring centers
    edges[1:-1, 1:-1] = 0.25 * (
        coord[:-1, :-1] + coord[1:, :-1] +
        coord[:-1, 1:]  + coord[1:, 1:]
    )
    
    # Edges along rows (top and bottom)
    edges[0, 1:-1]  = coord[0, :-1]  - 0.5*(coord[1, :-1] - coord[0, :-1])
    edges[-1, 1:-1] = coord[-1, :-1] + 0.5*(coord[-1, :-1] - coord[-2, :-1])
    
    # Edges along columns (left and right)
    edges[1:-1, 0]  = coord[:-1, 0]  - 0.5*(coord[:-1, 1] - coord[:-1, 0])
    edges[1:-1, -1] = coord[:-1, -1] + 0.5*(coord[:-1, -1] - coord[:-1, -2])
    
    # Corners
    edges[0,0]   = coord[0,0] - 0.5*(coord[1,0] - coord[0,0]) - 0.5*(coord[0,1] - coord[0,0])
    edges[0,-1]  = coord[0,-1] - 0.5*(coord[1,-1] - coord[0,-1]) + 0.5*(coord[0,-1] - coord[0,-2])
    edges[-1,0]  = coord[-1,0] + 0.5*(coord[-1,0] - coord[-2,0]) - 0.5*(coord[-1,1] - coord[-1,0])
    edges[-1,-1] = coord[-1,-1] + 0.5*(coord[-1,-1] - coord[-2,-1]) + 0.5*(coord[-1,-1] - coord[-1,-2])
    
    return edges
