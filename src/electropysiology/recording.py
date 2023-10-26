#!/usr/bin/python3

import numbers
import numpy as np

from . import preprocess

class ConditionTrials:
    def __init__(self, dt, event_codes, event_times, lfp=None, mua=None,
                 spikes=None, zscore_mua=True):
        assert lfp or mua or spikes
        self._dt = dt
        self._event_codes = event_codes
        self._lfp, self._mua, self._spikes = lfp, mua, spikes
        if zscore_mua:
            self._mua = preprocess.zscore(self._mua)

        self._shape = None
        for thing in (lfp, mua, spikes):
            if thing:
                assert len(thing.shape) == 3 # Channels x Times x Trials
                if ntimes:
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
    def event_codes(self):
        return self._event_codes

    def _event_indices(self, event):
        return np.nonzero(self._event_codes == self.event_codes[event])[0]

    def _event_bounds(self, event):
        events = self._event_indices(event)
        onsets = event_times[trials, events].astype(np.int64)
        offsets = event_times[trials, events+1].astype(np.int64)
        return onsets, offsets

    def time_lock(self, event, duration=True, before=0., after=0.):
        onsets, offsets = self._event_bounds(event)
        onsets = onsets - before
        if isinstance(duration, numbers.Number):
            offsets = onsets + duration
        offsets = offsets + after
        max_T = int(offsets.max() - onsets.min())
        times = np.arange(0, int(max_T))

        lfps, muas, spikes = None, None, None
        if self._lfp:
            lfps = np.zeros([self.num_channels, max_T, self.num_trials])
        if self._mua:
            muas = np.zeros([self.num_channels, max_T, self.num_trials])
        if self._spikes:
            spikes = np.zeros([self.num_channels, max_T, self.num_trials])

        for tr in range(self.num_trials):
            onset, offset = onsets[t], offsets[t]
            T = int((offset - onset) * dt)
            if self._lfp:
                lfps[:, :T, tr] = self._lfp[:, onset:offset, tr]
            if self._mua:
                muas[:, :T, tr] = self._mua[:, onset:offset, tr]
            if self._spikes:
                spikes[:, :T, tr] = self._spikes[:, onset:offset, tr]

        # TODO: store and fetch the analog signals that provide ground-truth for
        # time indexing.

        return times, lfps, muas, spikes
