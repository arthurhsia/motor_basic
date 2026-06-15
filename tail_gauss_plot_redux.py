import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent


def tail_gauss_plot_redux(electrodes, weights, ax=None):
    """
    Project electrode weights onto the brain surface using a Gaussian kernel.

    Parameters
    ----------
    electrodes : ndarray, shape (n_elec, 3)
        Electrode MNI coordinates (x, y, z).
    weights : ndarray, shape (n_elec,)
        Signed r^2 values to color the brain surface.
    ax : Axes3D or None
        Existing 3-D axes to draw into; a new figure is created if None.

    Returns
    -------
    ax : Axes3D
    """
    halfbrains = sio.loadmat(str(_SCRIPT_DIR / 'halfbrains.mat'))
    cm_data = sio.loadmat(str(_SCRIPT_DIR / 'dg_colormap.mat'))
    cmap = ListedColormap(cm_data['cm'])

    if electrodes[:, 0].mean() < 0:
        cortex = halfbrains['leftbrain']
        view_azim = 270
        light_sign = -1
    else:
        cortex = halfbrains['rightbrain']
        view_azim = 90
        light_sign = 1

    verts = cortex['vert'][0, 0]          # (N_verts, 3)
    tris = cortex['tri'][0, 0] - 1        # convert 1-indexed → 0-indexed

    # Gaussian spreading (gsp=50 matches MATLAB default)
    gsp = 50.0
    c = np.zeros(len(verts))
    for k in range(len(electrodes)):
        d2 = np.sum((verts - electrodes[k]) ** 2, axis=1)
        c += weights[k] * np.exp(-d2 / gsp)

    # Per-face color = mean of three vertex values
    face_vals = c[tris].mean(axis=1)
    max_abs = np.max(np.abs(face_vals))
    if max_abs == 0:
        max_abs = 1.0
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    face_colors = cmap(norm(face_vals))

    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_trisurf(
        verts[:, 0], verts[:, 1], verts[:, 2],
        triangles=tris, shade=True, alpha=1.0,
    )
    surf.set_facecolor(face_colors)

    # Electrode dots
    ax.scatter(
        electrodes[:, 0] * 1.01, electrodes[:, 1], electrodes[:, 2],
        s=4, c='white', depthshade=False, zorder=5,
    )

    ax.view_init(elev=0, azim=view_azim)
    ax.axis('off')
    ax.set_box_aspect([1, 1, 1])
    return ax
