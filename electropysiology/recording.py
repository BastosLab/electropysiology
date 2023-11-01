#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from . import preprocess

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
