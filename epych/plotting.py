#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np

def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta/2, f[-1] + delta/2]

def imagesc(ax, cs, **kwargs):
    if kwargs['vmin'] is None and kwargs['vmax'] is None:
        m = max(np.abs(cs.min()), np.abs(cs.max()))
        kwargs['vmin'] = -m
        kwargs['vmax'] = m
    x, y = np.linspace(0, cs.shape[1]), np.linspace(0, cs.shape[0])
    return ax.imshow(cs, aspect='auto', interpolation='none',
                     extent=extents(x) + extents(y), **kwargs)

def heatmap(fig, ax, data, title=None, cbar=True, vmin=-1e-4, vmax=1e-4):
    img = imagesc(ax, data, vmin=vmin, vmax=vmax, origin='lower')
    if cbar:
        fig.colorbar(img, ax=ax)
    if title is not None:
        ax.set_title(title)
