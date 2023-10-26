#!/usr/bin/python3

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
