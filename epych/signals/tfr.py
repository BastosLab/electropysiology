#!/usr/bin/python3

import dask.array
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantities as pq

from .. import plotting, signal
from ..statistics.spectrum import PowerSpectrum

class TimeFrequencyRepr(signal.Signal):
    def __init__(self, channels: pd.DataFrame, data, dt, freqs, timestamps):
        self._freqs = freqs
        super().__init__(channels, data, dt, timestamps)

    def baseline(self, start, end):
        first = np.abs(self.times - start).argmin()
        last = np.abs(self.times - end).argmin()
        base_mean = self.data[:, first:last, :].magnitude.mean(axis=1,
                                                               keepdims=True)
        base_mean = base_mean * self.data.units
        tfrs = (self.data - base_mean) / base_mean * 100 * pq.percent
        return self.__class__(self.channels, tfrs, self.dt, self.freqs,
                              self.times)

    def channel_depths(self, column=None):
        if column is not None and column in self.channels:
            return self.channels[column].values
        return np.arange(len(self.channels))

    def channel_mean(self):
        middle_channel = len(self.channels) // 2
        channels = self.channels[middle_channel:(middle_channel + 1)]
        data = self.data.magnitude.mean(axis=0, keepdims=True) * self.data.units
        return self.__class__(channels, data, self.dt, self.freqs, self.times)

    def decibels(self):
        return self.fmap(lambda data: 10 * np.log10(data))

    @property
    def freqs(self):
        return self._freqs

    def fmap(self, f):
        return self.__class__(self.channels, f(self.data), self.dt, self.freqs,
                              self.times)

    @property
    def fmax(self):
        return self._freqs[-1]

class EpochedTfr(TimeFrequencyRepr, signal.EpochedSignal):
    def __init__(self, channels: pd.DataFrame, data, dt, freqs, timestamps):
        assert len(data.shape) == 4
        assert len(channels) == data.shape[0]
        assert len(timestamps) == data.shape[1]
        assert timestamps.units == dt.units

        super(EpochedTfr, self).__init__(channels, data, dt, freqs, timestamps)

    def evoked(self):
        erp = super().evoked()
        return EvokedTfr(erp.channels, erp.data, erp.dt, self.freqs, erp.times)

class EvokedTfr(TimeFrequencyRepr, signal.EvokedSignal):
    def __init__(self, channels: pd.DataFrame, data, dt, freqs, timestamps):
        assert data.shape[-1] == 1
        super(EvokedTfr, self).__init__(channels, data, dt, freqs, timestamps)

    def evoked(self):
        erp = super().evoked()
        return EvokedTfr(erp.channels, erp.data, erp.dt, self.freqs, erp.times)

    def plot(self, *args, **kwargs):
        if "events" in kwargs:
            events = kwargs.pop("events")
            def callback(self, ax):
                for (event, (time, color)) in events.items():
                    ymin, ymax = ax.get_ybound()
                    xtime = self.sample_at(time)
                    ax.vlines(xtime, ymin, ymax, colors=color,
                              linestyles='dashed', label=event)
                    ax.annotate(event, (xtime + 0.005, ymax))
            kwargs["callback"] = callback
        return self.heatmap(*args, **kwargs)
