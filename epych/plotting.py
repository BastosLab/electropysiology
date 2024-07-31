#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np

from . import colormaps

def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta/2, f[-1] + delta/2]

def imagesc(ax, cs, alpha=None, cmap=None, **kwargs):
    if cmap is None:
        cmap = colormaps.parula
    if kwargs['vmin'] is None and kwargs['vmax'] is None:
        std_dev = cs.std()
        kwargs['vmin'] = 2 * -std_dev
        kwargs['vmax'] = 2 * std_dev
    x, y = np.linspace(0, cs.shape[1]), np.linspace(0, cs.shape[0])
    return ax.imshow(cs, alpha=alpha, aspect='auto', interpolation='none',
                     extent=extents(x) + extents(y), cmap=cmap, **kwargs)

def heatmap(fig, ax, data, alpha=None, title=None, cbar=True, vmin=-1e-4,
            vmax=1e-4, cmap=None):
    cbar = cbar or vmin is None or vmax is None
    img = imagesc(ax, data, alpha=alpha, vmin=vmin, vmax=vmax, origin='lower',
                  cmap=cmap)
    if cbar:
        fig.colorbar(img, ax=ax)
    if title is not None:
        ax.set_title(title)
