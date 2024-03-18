#!/usr/bin/python3

import copy
import matplotlib.pyplot as plt
import numpy as np
from typing import TypeVar

from .. import plotting, signal, statistic

from . import alignment

T = TypeVar('T', bound=signal.EpochedSignal)

class GrandAverage(statistic.Statistic[T]):
    def __init__(self, alignment: alignment.LaminarAlignment, data=None):
        super().__init__((alignment.num_channels, alignment.num_times),
                         data=data)
        self._alignment = alignment
        self._channels = None
        self._dt = None
        if data is None:
            self._data = {"n": 0, "sum": np.zeros((*self.iid_shape, 1)),
                          "timestamps": np.zeros(self.iid_shape[1])}
        self._signal_class = None

    def apply(self, element: T):
        assert element.data.shape[0] == self.num_channels
        data = element.data
        if self.num_times < element.data.shape[1]:
            data = data[:, :self.num_times, :]
        running = copy.deepcopy(self.data)

        if self._channels is None:
            low, _, high = self._alignment.result()[0]
            channels_range = [c in range(low, high) for c
                              in element.channels.channel.values]
            self._channels = element.channels.loc[channels_range]
            self._dt = element.dt
            self._signal_class = element.__class__
        running["n"] += element.num_trials
        running["sum"] += data.sum(axis=-1, keepdims=True)
        running["timestamps"] +=\
            element.times[:self.num_times] * element.num_trials
        return running

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

    @property
    def num_channels(self):
        return self.iid_shape[0]

    @property
    def num_times(self):
        return self.iid_shape[1]

    def plot(self, **kwargs):
        return self.result().plot(**kwargs)

    def result(self):
        data = self.data["sum"] / self.data["n"]
        times = self.data["timestamps"] / self.data["n"]
        return self._signal_class(self._channels, data, self._dt,
                                  times).evoked()
