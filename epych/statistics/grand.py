#!/usr/bin/python3

import copy
import matplotlib.pyplot as plt
import numpy as np

from .. import plotting, signal, statistic

class GrandAverage(statistic.Statistic[signal.EpochedSignal]):
    def __init__(self, iid_shape, data=None):
        super().__init__(iid_shape, data=data)
        if data is None:
            self._data = {"n": 0, "sum": np.zeros((*self.iid_shape, 1))}

    def apply(self, element: signal.EpochedSignal):
        assert element.data.shape[0] == self.iid_shape[0]
        data = element.data
        if self.iid_shape[1] < element.data.shape[1]:
            data = data[:, :self.iid_shape[1], :]
        running = copy.deepcopy(self.data)
        running["sum"] += data.sum(axis=-1, keepdims=True)
        running["n"] += element.num_trials
        return running

    def result(self):
        return self.data["sum"] / self.data["n"]

    def heatmap(self, ax=None, fig=None, title=None, vmin=None, vmax=None,
                origin="lower"):
        if ax is None:
            ax = plt.gca()
        if fig is None:
            fig = plt.gcf()

        data = self.result().squeeze()
        plotting.heatmap(fig, ax, data, cbar=False, title=title, vmin=vmin,
                         vmax=vmax)

        num_xticks = len(ax.get_xticks())
        xtick_locs = np.linspace(0, data.shape[1], num_xticks)
        xticks = np.linspace(0, data.shape[-1], num_xticks)
        xticks = ["%0.2f" % t for t in xticks]
        ax.set_xticks(xtick_locs, xticks)
