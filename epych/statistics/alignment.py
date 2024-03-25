#!/usr/bin/python3

from abc import abstractmethod
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

class ChannelAlignment(statistic.Statistic[signal.EpochedSignal]):
    def __init__(self, column="location", data=None):
        self._column = column
        self._num_times = None
        super().__init__((1,), data=data)

    def align(self, i: int, sig: signal.EpochedSignal) -> signal.EpochedSignal:
        low, center, high = self.result()[i]
        alignment = [c in range(int(low), int(high)) for c in
                     range(len(sig.channels))]
        result = sig.select_channels(alignment)
        return result.__class__(result.channels,
                                result.data[:, :self.num_times], result.dt,
                                result.times[:self.num_times])

    def apply(self, element: signal.EpochedSignal):
        channels_index = element.channels.channel if "channel"\
                         in element.channels.columns else element.channels.index
        descriptors = element.channels.loc[channels_index.index][self._column]
        center = round(np.median(np.where(self.center_filter(descriptors))[0]))
        center = channels_index.iloc[center]

        sample_columns = element.channels.loc[:, [self._column, "channel"]]
        sample = np.array((channels_index.values[0], center,
                           channels_index.values[-1]))[np.newaxis, :]
        if self.num_times is None or len(element) < self.num_times:
            self._num_times = len(element)
        if self.data is None:
            return {self._column: [sample_columns], "sample": sample}
        return {
            self._column: self.data[self._column] + [sample_columns],
            "sample": np.concatenate((self.data["sample"], sample), axis=0)
        }

    @abstractmethod
    def center_filter(self, descriptors):
        raise NotImplementedError

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
        center_channels = self.data["sample"][:, 1]
        low_distance = (center_channels - self.data["sample"][:, 0]).min()
        high_distance = (self.data["sample"][:, 2] - center_channels).min()
        return np.array([center_channels - low_distance, center_channels,
                         center_channels + high_distance]).T.round()

def laminar_alignment(name, sig):
    return LaminarAlignment()

def subcortical_alignment(name, sig):
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
