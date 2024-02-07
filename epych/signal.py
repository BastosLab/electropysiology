#!/usr/bin/python3

import collections.abc
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy

class Signal(collections.abc.Sequence):
    def __init__(self, channels: pd.DataFrame, data, dt, timestamps):
        assert len(data.shape) >= 2

        self._channels = channels
        self._data = data
        self._dt = dt
        self._timestamps = timestamps

    @property
    def channels(self):
        return self._channels

    @property
    def df(self):
        return 1. / self.T

    @property
    def dt(self):
        return self._dt

    def _data_slices(self, channels, times, trials):
        if channels is None:
            channels = slice(0, self.num_channels, 1)
        else:
            if isinstance(channels, int):
                channels = slice(channels, channels+1, None)
            if channels.step is None:
                channels = slice(channels.start, channels.stop, 1)
        if times is None:
            times = slice(self.times[0], self.times[-1], None)
        else:
            if isinstance(times, int):
                times = slice(times, times+1, None)
            if times.step is None:
                times = slice(times.start, times.stop, 1)
        if trials is None:
            trials = slice(None, None, None)
        else:
            if isinstance(trials, int):
                trials = slice(trials, trials+1, None)
            if trials.step is None:
                trials = slice(trials.start, trials.stop, 1)
        return channels, times, trials

    @property
    def f0(self):
        return 1. / self.dt

    def fmap(self, f):
        return self.__class__(self.channels, f(self.data), self.dt,
                              self.times)

    @property
    def fNQ(self):
        return self.f0 / 2.

    def get_data(self, channels=None, times=None, trials=None):
        assert isinstance(channels, slice)
        assert isinstance(times, slice)
        assert isinstance(trials, slice)
        raise NotImplementedError

    def __len__(self):
        return len(self._timestamps)

    @property
    def num_channels(self):
        return len(self._channels)

    @property
    def num_trials(self):
        raise NotImplementedError

    def sample_at(self, t):
        return np.nanargmin(np.abs(self._timestamps - t))

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

class EpochedSignal(Signal):
    def __init__(self, channels: pd.DataFrame, data, dt, timestamps):
        assert len(data.shape) == 3
        assert len(channels) == data.shape[0]
        assert len(timestamps) == data.shape[1]

        super().__init__(channels, data, dt, timestamps)

    def __add__(self, sig):
        assert self.__class__ == sig.__class__
        assert (self.channels == sig.channels).all().all()
        assert self.dt == sig.dt
        assert len(self) == len(sig)
        num_samples = min(self.data.shape[1], sig.data.shape[1])
        timestamps = np.arange(num_samples) * self.dt
        data = self.data[:num_samples] + sig.data[:num_samples]
        return self.__class__(self.channels, data, self.dt, timestamps)

    def baseline_correct(self, start, stop):
        start, stop = self.sample_at(start), self.sample_at(stop) - 1
        f = lambda data: data - data[:, start:stop].mean(axis=1, keepdims=True)
        return self.fmap(f)

    @property
    def data(self):
        return self._data

    def downsample(self, n):
        channels = self.channels.loc[0::n]
        data = self.data[0::n, :, :]
        return self.__class__(channels, data, self.dt, self.times)

    def evoked(self):
        data = self.data.mean(-1, keepdims=True)
        return EvokedSignal(self.channels, data, self.dt, self.times)

    def get_data(self, channels, times, trials):
        channels, times, trials = self._data_slices(channels, times, trials)

        times = slice(self.sample_at(times.start),
                      self.sample_at(times.stop) + 1, times.step)
        return self._data[channels, times, trials]

    def __getitem__(self, key):
        if isinstance(key, int):
            key = slice(key, key+1, None)
        if key.step is None:
            key = slice(key.start, key.stop, 1)

        key = slice(self.sample_at(key.start), self.sample_at(key.stop),
                    key.step)
        return self.__class__(self.channels, self.data[:, key], self.dt,
                              self.times[key])

    def mask_epochs(self, onsets, offsets):
        assert len(onsets) == len(offsets)

        self._data = np.nan_to_num(self._data, copy=False)
        for trial in range(len(onsets)):
            first = self.sample_at(onsets[trial])
            last = self.sample_at(offsets[trial])
            self._data[:, :first, trial] *= 0
            self._data[:, last:, trial] *= 0

    def median_filter(self, cs=3):
        return self.fmap(
            lambda data: scipy.ndimage.median_filter(data, size=(cs, 1, 1))
        )

    @property
    def num_channels(self):
        return self.data.shape[0]

    @property
    def num_trials(self):
        return self.data.shape[2]

    def select_channels(self, k, v):
        groups = self.channels.groupby(k).groups
        rows = [self.channels.index.get_loc(c) for c in groups[v]]
        return self.__class__(self.channels.take(rows),
                              self.data[rows, :], self.dt, self.times)

    def select_trials(self, trials):
        return self.__class__(self.channels, self.data[:, :, trials],
                              self.dt, self.times)

    def __sub__(self, sig):
        assert self.__class__ == sig.__class__
        assert (self.channels == sig.channels).all().all()
        assert self.dt == sig.dt
        assert len(self) == len(sig)
        num_samples = min(self.data.shape[1], sig.data.shape[1])
        timestamps = np.arange(num_samples) * self.dt
        data = self.data[:num_samples] - sig.data[:num_samples]
        return self.__class__(self.channels, data, self.dt, timestamps)

class EvokedSignal(EpochedSignal):
    def __init__(self, channels, data, dt, timestamps):
        assert data.shape[2] == 1
        super().__init__(channels, data, dt, timestamps)

    def annotate_channels(self, ax, key):
        prev_channel = ""
        for c, channel in enumerate(self.channels[key].values):
            if channel == prev_channel:
                continue
            prev_channel = channel
            line = ax.axhline(c, linestyle="--", color="black")
            ax.annotate(channel.decode(), line.get_xydata()[0, :])

    def line_plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.times, self.data.T.squeeze(), **kwargs)

    def heatmap(self, ax=None, fig=None, title=None, vmin=None, vmax=None,
                origin="lower", channel_annotations="location"):
        if ax is None:
            ax = plt.gca()
        if fig is None:
            fig = plt.gcf()

        data = self.data.squeeze()
        sns.heatmap(self.data.squeeze(), ax=ax, linewidth=0, cmap='viridis',
                    cbar=True, vmin=vmin, vmax=vmax)
        if origin == "lower":
            ax.invert_yaxis()

        if title is not None:
            ax.set_title(title)

        num_xticks = len(ax.get_xticks())
        xtick_locs = np.linspace(0, data.shape[1], num_xticks)
        xticks = np.linspace(self.times[0], self.times[-1], num_xticks)
        xticks = ["%0.2f" % t for t in xticks]
        ax.set_xticks(xtick_locs, xticks)

        if channel_annotations is not None and\
           channel_annotations in self.channels.columns:
            self.annotate_channels(ax, channel_annotations)

    def plot(self, *args, **kwargs):
        return self.line_plot(*args, **kwargs)

class RawSignal(Signal):
    epoched_signal = EpochedSignal

    def __init__(self, channels: pd.DataFrame, data, dt, timestamps,
                 channels_dim=0, time_dim=1):
        assert len(data.shape) == 2
        assert len(channels) == data.shape[channels_dim]
        assert len(timestamps) == data.shape[time_dim]

        self._channels_dim = channels_dim
        self._time_dim = time_dim
        super().__init__(channels, data, dt, timestamps)

    def evoked(self, start=None, stop=None, step=None):
        data, timestamps = self.get_data(None, slice(start, stop, step), None)
        return self.epoched_signal(self.channels, data[:, :, np.newaxis],
                                   self.dt, timestamps)

    def epoch(self, intervals, time_shift=0.):
        assert intervals.shape[1] == 2 and intervals.shape[0] >= 1

        trials_data = []
        for trial, (start, end) in enumerate(intervals):
            trials_data.append(self[start:end])
        trials_samples = min(data.shape[1] for data in trials_data)
        trials_data = [data[:, :trials_samples] for data in trials_data]
        trials_data = np.concatenate(trials_data, axis=-1)
        timestamps = np.arange(trials_data.shape[1]) * self.dt + time_shift
        return self.epoched_signal(self.channels, trials_data, self.dt,
                                   timestamps)

    def get_data(self, channels, times, trials):
        channels, times, trials = self._data_slices(channels, times, trials)
        assert trials == slice(None, None, None)

        times = slice(self.sample_at(times.start), self.sample_at(times.stop),
                      times.step)

        keys = [None, None]
        keys[self._channels_dim] = channels
        keys[self._time_dim] = times
        data = self._data[keys[0], keys[1]]
        data = data.swapaxes(self._channels_dim, 0)
        return data[:, :, np.newaxis]

    def __getitem__(self, key):
        return self.get_data(None, key, None)

    @property
    def num_trials(self):
        return 1
