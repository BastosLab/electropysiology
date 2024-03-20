#!/usr/bin/python3

from collections.abc import Iterable
import functools
import numpy as np
import os
import pandas as pd
import pickle
import re

from .. import signal, statistic

def subcortical_median(channels, locations):
    return round(np.median(channels))

def cortical_l4(channels, locations):
    l4 = os.path.commonprefix([l.decode() for l in locations]) + "4"
    l4_mask = [l4 in loc.decode() for loc in locations]
    return round(np.median(channels[l4_mask]))

class LaminarAlignment(statistic.Statistic[signal.EpochedSignal]):
    def __init__(self, center_loc=cortical_l4, column="location", data=None):
        self._center_loc = center_loc
        self._column = column
        self._num_times = None
        super().__init__((1,), data=data)

    def align(self, i: int, sig: signal.EpochedSignal) -> signal.EpochedSignal:
        low, l4, high = self.result()[i]
        alignment_mask = [c in range(int(low), int(high)) for c
                          in range(len(sig.channels))]
        result = sig.select_channels(alignment_mask)
        return result.__class__(result.channels,
                                result.data[:, :self.num_times], result.dt,
                                result.times[:self.num_times])

    def apply(self, element: signal.EpochedSignal):
        channels_index = element.channels.channel\
                         if "channel" in element.channels.columns\
                         else element.channels.index
        l4_center = self._center_loc(channels_index, element.channels.location)

        sample = np.array((channels_index.values[0], l4_center,
                           channels_index.values[-1]))[np.newaxis, :]
        if self.num_times is None or len(element) < self.num_times:
            self._num_times = len(element)
        if self.data is None:
            return sample
        return np.concatenate((self.data, sample), axis=0)

    def fmap(self, f):
        return self.__class__(self._area, self._column, f(self.data))

    @property
    def num_channels(self):
        low, _, high = self.result()[0]
        return high - low

    @property
    def num_times(self):
        return self._num_times

    def result(self):
        l4_channels = self._data[:, 1]
        low_distance = (l4_channels - self._data[:, 0]).min()
        high_distance = (self._data[:, 2] - l4_channels).min()
        return np.array([l4_channels - low_distance, l4_channels,
                         l4_channels + high_distance]).T.round()

def laminar_alignment(name, sig):
    return LaminarAlignment()

def hippocampal_alignment(name, sig):
    return LaminarAlignment(center_loc=subcortical_median)

def location_prefix(probe, sig: signal.Signal):
    return os.path.commonprefix([
        loc.decode() for loc in sig.channels.location.values
    ])

def location_set(probe, sig: signal.Signal):
    locations = set([loc.decode() for loc in sig.channels.location.values])
    return functools.reduce(lambda x, y: x + y, locations)

class AlignmentSummary(statistic.Summary):
    def __init__(self, signal_key=location_prefix, alignment=laminar_alignment):
        super().__init__(signal_key, alignment)

    @classmethod
    def unpickle(cls, path):
        assert os.path.isdir(path)

        with open(path + "/summary.pickle", mode="rb") as f:
            self = pickle.load(f)
        self._stats = {}
        ls = [entry for (entry, _, _) in os.walk(path) if os.path.isdir(entry)]
        for entry in sorted(ls[1:]):
            entry = os.path.relpath(entry, start=path)
            self._stats[entry] = LaminarAlignment.unpickle(path + "/" + entry)
        self._statistic = LaminarAlignment
        return self
