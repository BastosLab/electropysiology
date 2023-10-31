#!/usr/bin/python3

import collections.abc
import math
import numbers
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import seaborn as sns

from . import preprocess

class Signal(collections.abc.Sequence):
    def __init__(self, channels, data, dt, sampling_times):
        assert len(data.shape) == 3
        assert len(sampling_times) == data.shape[1]

        self._channels = channels
        self._data = data
        self._dt = dt
        self._sampling_times = sampling_times

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

    def mask_trial(self, tr, onset, offset):
        first, last = self.sample_at(onset), self.sample_at(offset)
        S = last - first
        self._data[:, :first, tr] = np.zeros([self.num_channels, first])
        self._data[:, S:, tr] = np.zeros([self.num_channels, len(self) - S])

    @property
    def num_channels(self):
        return self.data.shape[0]

    def __len__(self):
        return len(self._sampling_times)

    @property
    def num_trials(self):
        return self.data.shape[2]

    def sample_at(self, t):
        return np.nanargmin((self._sampling_times - t) ** 2)

    def select_channels(self, k, v):
        groups = self.channel_info.groupby(k).groups
        rows = [self.channel_info.index.get_loc(c) for c in groups[v]]
        return self.__class__(self.channel_info.take(rows),
                              self.data[rows, :, :], self.dt, self.times)

    def sort_channels(self, key):
        indices = self.channel_info.sort_values(key, ascending=False).index
        return [self.channel_info.index.get_loc(i) for i in indices]

    @property
    def T(self):
        return self.dt * len(self)

    @property
    def times(self):
        return self._sampling_times

    def time_to_samples(self, t):
        return math.ceil(t * self.f0)

    def __getitem__(self, key):
        if isinstance(key, int):
            key = slice(key, key+1, None)
        if key.step is None:
            key = slice(key.start, key.stop, 1)

        duration = key.stop - key.start
        times = np.linspace(key.start, key.stop, self.time_to_samples(duration))
        key = slice(self.sample_at(key.start), self.sample_at(key.stop),
                    key.step)
        if len(times) > key.stop - key.start:
            times = times[:key.stop - key.start]
        return self.__class__(self.channel_info, self.data[:, key, :], self.dt,
                              times)

class Spectrum:
    def __init__(self, df, pows):
        self._df = df
        self._freqs = np.arange(0, pows.shape[1] / df, self.df)[np.newaxis, :]
        self._pows = pows

    def closest_freq(self, f):
        return np.nanargmin((self.freqs - f) ** 2)

    @property
    def df(self):
        return self._df

    @property
    def freqs(self):
        return self._freqs

    def heatmap(self, fbottom=0, ftop=None, ax=None):
        if ax is None:
            ax = plt.gca()
        if ftop is None:
            ftop = self.freqs[-1]

        sns.heatmap(self.pows, ax=ax, linewidth=0, cmap='viridis', cbar=False,
                    robust=True)
        ax.set_xlim(left=fbottom, right=ftop)

    def decibels(self):
        return Spectrum(self.df, 10 * np.log10(self.pows))

    @property
    def pows(self):
        return self._pows

    def relative(self):
        max_pow = self.pows.max(axis=0, keepdims=True)
        return Spectrum(self.df, self.pows / max_pow)

    def trial_mean(self, axis=-1):
        return Spectrum(self.df, self.pows.mean(axis=axis))

class LocalFieldPotential(Signal):
    @property
    def erp(self):
        return self.fmap(lambda xs: xs.mean(-1, keepdims=True))

    def plot(self, **kwargs):
        plt.plot(self.times, self.data.T.squeeze(), **kwargs)

    def csd(self, sigma, s):
        channel_csds = []
        for i in range(2, self.num_channels - 2):
            vi = (self.data[i-2] - 2 * self.data[i] + self.data[i+1])
            channel_csds.append(-sigma * vi / (2 * s ** 2))
        channel_csds = np.stack(channel_csds, axis=0)
        return self.__class__(self.channel_info[2:-2], channel_csds, self.dt,
                              self.times)

    def power_spectrum(self, dBs=True, relative=False, taper=None):
        xs = self.data
        if taper is not None:
            xs = taper(xs.shape[1])[np.newaxis, :, np.newaxis] * xs
        xf = fft.rfft(xs - xs.mean(axis=1, keepdims=True), axis=1)
        pows = (2 * self.dt ** 2 / self.T) * (xf * xf.conj())
        pows = pows[:, 0:xs.shape[1] // 2].real

        spectrum = Spectrum(self.df, pows)
        if relative:
            spectrum = spectrum.relative()
        if dBs:
            spectrum = spectrum.decibels()
        return spectrum.trial_mean()

class ConditionTrials:
    def __init__(self, events, lfp=None, mua=None, spikes=None,
                 zscore_mua=True):
        assert lfp is not None or mua is not None or spikes is not None
        self._events = events
        self._lfp, self._mua, self._spikes = lfp, mua, spikes
        if zscore_mua and self._mua is not None:
            self._mua = self._mua.fmap(preprocess.zscore_trials)

        self._num_trials = None
        for thing in (lfp, mua, spikes):
            if thing is not None:
                assert len(thing.data.shape) == 3 # Channels x Times x Trials
                if self._num_trials:
                    assert thing.num_trials == self._num_trials
                else:
                    self._num_trials = thing.num_trials

    @property
    def num_trials(self):
        return self._num_trials

    @property
    def events(self):
        return self._events

    def _event_bounds(self, event):
        event_keys = list(self.events.keys())
        successor = event_keys[event_keys.index(event) + 1]
        return self.events[event], self.events[successor]

    def time_lock(self, event, duration=True, before=0., after=0.):
        onsets, offsets = self._event_bounds(event)
        if not isinstance(duration, bool):
            offsets = onsets + duration
        onsets = onsets - before
        offsets = offsets + after
        first, last = onsets.min(), offsets.max()

        lfp, mua, spikes = None, None, None
        if self._lfp is not None:
            lfp = self._lfp[first:last]
        if self._mua is not None:
            mua = self._mua[first:last]
        if self._spikes is not None:
            spike = self._spikes[first:last]

        for tr in range(self.num_trials):
            if lfp is not None:
                lfp.mask_trial(tr, onsets[tr], offsets[tr])
            if mua is not None:
                mua.mask_trial(tr, onsets[tr], offsets[tr])
            if spikes is not None:
                spikes.mask_trial(tr, onsets[tr], offsets[tr])

        # TODO: store and fetch the analog signals that provide ground-truth for
        # time indexing.

        return TimeLockedSeries(lfp, mua, spikes)

class TimeLockedSeries:
    def __init__(self, lfp=None, mua=None, spikes=None):
        assert lfp is not None or mua is not None or spikes is not None

        self._shape = None
        self._lfp, self._mua, self._spikes = lfp, mua, spikes
        for thing in (lfp, mua, spikes):
            if thing is not None:
                assert len(thing.data.shape) == 3 # Channels x Times x Trials
                if self._shape:
                    assert thing.data.shape == self._shape
                else:
                    self._shape = thing.data.shape

    @property
    def lfp(self):
        return self._lfp

    @property
    def mua(self):
        return self._mua

    @property
    def spikes(self):
        return self._spikes
