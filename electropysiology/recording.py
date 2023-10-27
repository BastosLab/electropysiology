#!/usr/bin/python3

import math
import numbers
import numpy as np
import matplotlib.pyplot as plt

from . import preprocess

class ConditionTrials:
    def __init__(self, dt, events, sample_times, lfp=None, mua=None,
                 spikes=None, zscore_mua=True):
        assert lfp is not None or mua is not None or spikes is not None
        self._dt = dt
        self._events = events
        self._sample_times = sample_times
        self._lfp, self._mua, self._spikes = lfp, mua, spikes
        if zscore_mua and self._mua:
            self._mua = preprocess.zscore_trials(self._mua)

        self._shape = None
        for thing in (lfp, mua, spikes):
            if thing is not None:
                assert len(thing.shape) == 3 # Channels x Times x Trials
                if self._shape:
                    assert thing.shape == self._shape
                else:
                    self._shape = thing.shape

    @property
    def num_channels(self):
        return self._shape[0]

    @property
    def num_times(self):
        return self._shape[1]

    @property
    def num_trials(self):
        return self._shape[2]

    @property
    def dt(self):
        return self._dt

    @property
    def f0(self):
        return 1. / self.dt

    @property
    def fNQ(self):
        return self.f0 / 2.

    @property
    def T(self):
        return self.dt * self._ntimes

    @property
    def events(self):
        return self._events

    def _event_bounds(self, event):
        event_keys = list(self.events.keys())
        successor = event_keys[event_keys.index(event) + 1]
        return self.events[event], self.events[successor]

    def sample_at(self, t):
        return np.nanargmin((self._sample_times - t) ** 2)

    def time_to_samples(self, t):
        return math.ceil(t * self.f0)

    def time_lock(self, event, duration=True, before=0., after=0.):
        onsets, offsets = self._event_bounds(event)
        if not isinstance(duration, bool):
            offsets = onsets + duration
        onsets = onsets - before
        offsets = offsets + after

        first, last = onsets.min(), offsets.max()
        max_samples = self.time_to_samples(last - first)
        times = np.linspace(first, last, max_samples)

        lfps, muas, spikes = None, None, None
        if self._lfp is not None:
            lfps = np.zeros([self.num_channels, max_samples, self.num_trials])
        if self._mua is not None:
            muas = np.zeros([self.num_channels, max_samples, self.num_trials])
        if self._spikes is not None:
            spikes = np.zeros([self.num_channels, max_samples, self.num_trials])

        for tr in range(self.num_trials):
            onset = self.sample_at(onsets[tr])
            offset = self.sample_at(offsets[tr])
            S = offset - onset
            if self._lfp is not None:
                lfps[:, :S, tr] = self._lfp[:, onset:offset, tr]
            if self._mua is not None:
                muas[:, :S, tr] = self._mua[:, onset:offset, tr]
            if self._spikes is not None:
                spikes[:, :S, tr] = self._spikes[:, onset:offset, tr]

        # TODO: store and fetch the analog signals that provide ground-truth for
        # time indexing.

        return TimeLockedSeries(times, lfps, muas, spikes)

class TimeLockedSeries:
    def __init__(self, times, lfp=None, mua=None, spikes=None):
        assert times is not None
        assert lfp is not None or mua is not None or spikes is not None

        self._times = times
        self._shape = None
        self._lfp, self._mua, self._spikes = lfp, mua, spikes
        for thing in (lfp, mua, spikes):
            if thing is not None:
                assert len(thing.shape) == 3 # Channels x Times x Trials
                if self._shape:
                    assert thing.shape == self._shape
                else:
                    self._shape = thing.shape
                assert self._shape[1] == times.shape[0]

    @property
    def times(self):
        return self._times

    @property
    def lfp(self):
        return self._lfp

    @property
    def mua(self):
        return self._mua

    @property
    def spikes(self):
        return self._spikes

    @property
    def erp(self):
        return self.lfp.mean(-1)

    def plot_erp(self):
        plt.plot(self.times, self.erp.T)
