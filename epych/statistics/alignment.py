#!/usr/bin/python3

from abc import abstractmethod
from collections import Counter
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
    l4 = os.path.commonprefix(locations) + "4"
    l4_mask = [l4 in loc for loc in locations]
    return round(np.median(channels[l4_mask]))

def add_dicts(left, right, monoid=None):
    if monoid is None:
        monoid = lambda x, y: x + y
    return {k: monoid(left[k], right[k]) for k in left.keys() & right.keys()}

class ChannelAlignment(statistic.Statistic[signal.EpochedSignal]):
    def __init__(self, column="location", data=None):
        self._column = column
        super().__init__((1,), data=data)

    def align(self, i: int, sig: signal.EpochedSignal) -> signal.EpochedSignal:
        alignment, locations = self.result()
        num_times = self.num_times
        low, center, high = alignment[i]

        channels = sig.channels.channel if "channel" in sig.channels.columns\
                   else pd.Series(list(range(len(sig.channels))))
        result = sig.select_channels(channels.isin(range(low, high)).values)
        result.channels.location = locations
        return result.__class__(result.channels.reset_index(drop=True),
                                result.data[:, :num_times], result.dt,
                                result.times[:num_times])

    def apply(self, element: signal.EpochedSignal):
        channels_index = element.channels.channel if "channel"\
                         in element.channels.columns else element.channels.index
        descriptors = element.channels.loc[channels_index.index][self._column]
        center_channels = np.where(self.center_filter(descriptors))[0]
        center = channels_index.iloc[round(np.median(center_channels))]

        sample = {
            "center": [center],
            "channels": [element.channels],
            "num_times": [len(element)],
        }
        if self.data is None:
            return sample
        return add_dicts(self.data, sample)

    @abstractmethod
    def center_filter(self, descriptors):
        raise NotImplementedError

    def fmap(self, f):
        return self.__class__(self._area, self._column, f(self.data))

    @property
    def num_channels(self):
        low, _, high = self.result()[0][0]
        return high - low

    @property
    def num_times(self):
        return min(self.data["num_times"])

    @functools.cache
    def result(self):
        centers = np.array(self.data["center"])
        distances = np.array([
            [channels[0] - centers[i], channels[-1] - centers[i]]
            for i, channels in enumerate(chans.channel.values for chans in
                                         self.data["channels"])
        ])
        chans_down, chans_up = -distances[:, 0].max(), distances[:, 1].min()
        alignment = np.stack((centers - chans_down, centers,
                              centers + chans_up), axis=-1)
        locations = []
        for i, chans in enumerate(self.data["channels"]):
            indices = range(alignment[i, 0], alignment[i, 2])
            chans = chans.loc[chans.channel.isin(indices)]
            locations.append(chans.location.values)
        locations = [Counter(locs).most_common(1)[0][0] for locs
                     in np.stack(locations, axis=-1)]
        return alignment, locations

class LaminarAlignment(ChannelAlignment):
    def __init__(self, area="VIS", data=None):
        self._area = area
        super().__init__(column="location", data=data)

    @property
    def area(self):
        return self._area

    def center_filter(self, descriptors):
        l4 = os.path.commonprefix(list(descriptors.values)) + "4"
        return [l4 in loc for loc in descriptors]

    def column_filter(self, column):
        return [self.area in loc for loc in column.values]

def laminar_alignment(name, sig):
    area = os.path.commonprefix(list(sig.channels.location.values))
    return LaminarAlignment(area=area)

def subcortical_alignment(name, sig):
    return LaminarAlignment(center_loc=subcortical_median)

def location_prefix(probe, sig: signal.Signal):
    return os.path.commonprefix(list(sig.channels.location.values))

def location_set(probe, sig: signal.Signal):
    locations = set([loc for loc in sig.channels.location.values])
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
