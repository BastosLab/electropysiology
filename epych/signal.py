#!/usr/bin/python3

import collections.abc
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class Signal(collections.abc.Sequence):
    def __init__(self, channels, data, dt, timestamps):
        assert len(data.shape) >= 2
        assert len(timestamps) == data.shape[1]

        self._channels = channels
        self._data = data
        self._dt = dt
        self._timestamps = timestamps

    @property
    def channel_info(self):
        return self._channels

    @property
    def data(self):
        return self._data

    @property
    def df(self):
        return 1. / self.T

    @property
    def dt(self):
        return self._dt

    @property
    def f0(self):
        return 1. / self.dt

    def fmap(self, f):
        return self.__class__(self.channel_info, f(self.data), self.dt,
                              self.times)

    @property
    def fNQ(self):
        return self.f0 / 2.

    def heatmap(self, ax=None, xlims=None, ylims=None):
        if ax is None:
            ax = plt.gca()

        sns.heatmap(self.data.squeeze(), ax=ax, linewidth=0, cmap='viridis',
                    cbar=False, robust=True)
        if xlims is not None:
            ax.set_xlim(*xlims)
        if ylims is not None:
            ax.set_ylim(*ylims)

    def mask_epoch(self, onset, offset):
        self._data = np.nan_to_num(self._data, copy=False)
        first, last = self.sample_at(onset), self.sample_at(offset)
        self._data[:, :first] *= 0
        self._data[:, last:] *= 0

    @property
    def num_channels(self):
        return self.data.shape[0]

    def __len__(self):
        return len(self._timestamps)

    def sample_at(self, t):
        return np.nanargmin((self._timestamps - t) ** 2)

    def select_channels(self, k, v):
        groups = self.channel_info.groupby(k).groups
        rows = [self.channel_info.index.get_loc(c) for c in groups[v]]
        return self.__class__(self.channel_info.take(rows),
                              self.data[rows, :], self.dt, self.times)

    def sort_channels(self, key):
        indices = self.channel_info.sort_values(key, ascending=False).index
        return [self.channel_info.index.get_loc(i) for i in indices]

    @property
    def T(self):
        return self.dt * len(self)

    @property
    def times(self):
        return self._timestamps

    def time_to_samples(self, t):
        return math.ceil(t * self.f0)

    def __getitem__(self, key):
        if isinstance(key, int):
            key = slice(key, key+1, None)
        if key.step is None:
            key = slice(key.start, key.stop, 1)

        duration = key.stop - key.start
        key = slice(self.sample_at(key.start), self.sample_at(key.stop),
                    key.step)
        return self.__class__(self.channel_info, self.data[:, key], self.dt,
                              self.times[key])

class EpochedSignal(Signal):
    def __init__(self, channels, data, dt, timestamps):
        assert len(data.shape) == 3
        super().__init__(channels, data, dt, timestamps)

    def erp(self):
        return self.fmap(lambda xs: xs.mean(-1, keepdims=True))

    def mask_epochs(self, onsets, offsets):
        assert len(onsets) == len(offsets)

        self._data = np.nan_to_num(self._data, copy=False)
        for trial in range(len(onsets)):
            first = self.sample_at(onsets[trial])
            last = self.sample_at(offsets[trial])
            self._data[:, :first, trial] *= 0
            self._data[:, last:, trial] *= 0

    @property
    def num_trials(self):
        return self.data.shape[2]

    def select_trials(self, trials):
        return self.__class__(self.channel_info, self.data[:, :, trials],
                              self.dt, self.times)
