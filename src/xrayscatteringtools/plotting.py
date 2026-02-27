import numpy as np
import matplotlib.pyplot as plt
from .utils import J4M

def plot_j4m(f, x=None, y=None, ax=None, shading='auto', *args, **kwargs):
    """Plot Jungfrau 4M detector counts with saved detector data.

    Parameters
    ----------
    f : np.ndarray
        2D array (8 x 512 x 1024) of detector counts for each tile.
    x, y : np.ndarray, optional
        Coordinates for each tile of the detector. If None, saved coordinates are used and plotted with plot_jungfrau.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.
    shading : str, optional
        Shading style for pcolormesh (default: 'auto').

    Returns
    -------
    pcm : matplotlib.collections.QuadMesh
        The QuadMesh object created.
    """
    if x is None or y is None:
        x = J4M.x
        y = J4M.y
    return plot_jungfrau(y, -x, f, ax=ax, shading=shading, *args, **kwargs)

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

def edges_from_centers(centers):
    """
    Compute the edges of bins from their center positions.

    Parameters
    ----------
    centers : array_like
        1D array of bin center positions. Must contain at least two elements.

    Returns
    -------
    edges : ndarray
        1D array of bin edge positions, with length `len(centers) + 1`.

    Raises
    ------
    ValueError
        If `centers` contains fewer than two elements.

    Notes
    -----
    The edges are calculated as follows:
    - Inner edges are the midpoints between adjacent centers.
    - The first and last edges are extrapolated based on the adjacent bin width.

    Examples
    --------
    >>> centers = [1, 2, 3]
    >>> edges_from_centers(centers)
    array([0.5, 1.5, 2.5, 3.5])
    """
    centers = np.asarray(centers)
    if centers.size < 2:
        raise ValueError("Array must have at least 2 elements to calculate edges.")

    # 1. Calculate midpoints for inner edges
    inner_edges = (centers[:-1] + centers[1:]) / 2.0

    # 2. Extrapolate the first and last edges based on the adjacent bin width
    first_edge = centers[0] - (centers[1] - centers[0]) / 2.0
    last_edge = centers[-1] + (centers[-1] - centers[-2]) / 2.0

    # 3. Combine into a single array
    return np.concatenate(([first_edge], inner_edges, [last_edge]))
