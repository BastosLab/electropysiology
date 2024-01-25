#!/usr/bin/python3

import collections.abc
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class Signal(collections.abc.Sequence):
    def __init__(self, channels, data, dt, timestamps):
        assert len(data.shape) == 3
        assert len(timestamps) == data.shape[1]

        self._channels = channels
        self._data = data
        self._dt = dt
        self._timestamps = timestamps

    def baseline_correct(self, start, stop):
        start, stop = self.sample_at(start), self.sample_at(stop) - 1
        f = lambda data: data - data[:, start:stop].mean(axis=1, keepdims=True)
        return self.fmap(f)

    @property
    def channels(self):
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

    def erp(self, baseline_time=None):
        if baseline_time is not None:
            data = self.baseline_correct(0, baseline_time).data
        else:
            data = self.data
        return ContinuousSignal(self.channels, data.mean(-1, keepdims=True),
                                self.dt, self.times)

    @property
    def f0(self):
        return 1. / self.dt

    def fmap(self, f):
        return self.__class__(self.channels, f(self.data), self.dt,
                              self.times)

    @property
    def fNQ(self):
        return self.f0 / 2.

    def __len__(self):
        return len(self._timestamps)

    def mask_epochs(self, onsets, offsets):
        assert len(onsets) == len(offsets)

        self._data = np.nan_to_num(self._data, copy=False)
        for trial in range(len(onsets)):
            first = self.sample_at(onsets[trial])
            last = self.sample_at(offsets[trial])
            self._data[:, :first, trial] *= 0
            self._data[:, last:, trial] *= 0

    @property
    def num_channels(self):
        return self.data.shape[0]

    @property
    def num_trials(self):
        return self.data.shape[2]

    def sample_at(self, t):
        return np.nanargmin(np.abs(self._timestamps - t))

    def select_channels(self, k, v):
        groups = self.channels.groupby(k).groups
        rows = [self.channels.index.get_loc(c) for c in groups[v]]
        return self.__class__(self.channels.take(rows),
                              self.data[rows, :], self.dt, self.times)

    def select_trials(self, trials):
        return self.__class__(self.channels, self.data[:, :, trials],
                              self.dt, self.times)

    def sort_channels(self, key):
        indices = self.channels.sort_values(key, ascending=False).index
        return [self.channels.index.get_loc(i) for i in indices]

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
        return self.__class__(self.channels, self.data[:, key], self.dt,
                              self.times[key])

class ContinuousSignal(Signal):
    iid_signal = Signal

    def __init__(self, channels, data, dt, timestamps):
        assert data.shape[2] == 1
        super().__init__(channels, data, dt, timestamps)

    def epoch(self, intervals, time_shift=0.):
        assert intervals.shape[1] == 2 and intervals.shape[0] >= 1

        trials_data = []
        for trial, (start, end) in enumerate(intervals):
            start, end = self.sample_at(start), self.sample_at(end)
            trials_data.append(self._data[:, start:end, :])
        trials_samples = min(data.shape[1] for data in trials_data)
        trials_data = [data[:, :trials_samples] for data in trials_data]
        trials_data = np.concatenate(trials_data, axis=-1)
        timestamps = np.arange(0., trials_data.shape[1] * self.dt, self.dt)
        return self.iid_signal(self.channels, trials_data, self.dt,
                               timestamps + time_shift)

    def line_plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.times, self.data.T.squeeze(), **kwargs)

    def heatmap(self, ax=None, fig=None, title=None, vmin=None, vmax=None,
                origin="lower"):
        if ax is None:
            ax = plt.gca()
        if fig is None:
            fig = plt.gcf()

        data = self.data.squeeze()
        sns.heatmap(self.data.squeeze(), ax=ax, linewidth=0, cmap='viridis',
                    cbar=True, vmin=vmin, vmax=vmax)
        if origin == "lower":
            ax.invert_yaxis()

        # img = imagesc(ax, data, vmin=vmin, vmax=vmax, origin='lower')
        # fig.colorbar(img, ax=ax)
        if title is not None:
            ax.set_title(title)

        xtick_locs = np.linspace(0, data.shape[1], 20)
        xticks = np.linspace(self.times[0], self.times[-1], 20)
        xticks = ["%0.2f" % t for t in xticks]
        ax.set_xticks(xtick_locs, xticks)

    def plot(self, *args, **kwargs):
        return self.line_plot(*args, **kwargs)
