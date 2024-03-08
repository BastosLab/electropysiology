#!/usr/bin/python3

from collections.abc import Iterable
import numpy as np
import os
import pandas as pd
import re

from .. import signal, statistic

class LaminarAlignment(statistic.Statistic[signal.EpochedSignal]):
    def __init__(self, column="location", data=None):
        self._column = column
        super().__init__((1,), data=data)

    def align(self, i: int, sig: signal.EpochedSignal) -> signal.EpochedSignal:
        low, l4, high = self.data[i]
        alignment_mask = [c in range(low, high+1) for c
                          in range(len(sig.channels))]
        return sig.select_channels(alignment_mask)

    def apply(self, element: signal.Signal):
        area_l4 = os.path.commonprefix([l.decode() for l
                                        in element.channels.location]) + "4"
        l4_mask = [area_l4 in loc.decode() for loc in element.channels.location]

        channels_index = element.channels.channel\
                         if "channel" in element.channels.columns\
                         else element.channels.index
        l4_center = round(np.median(channels_index[l4_mask]))

        sample = np.array((channels_index.values[0], l4_center,
                           channels_index.values[-1]))[np.newaxis, :]
        if self.data is None:
            return sample
        return np.concatenate((self.data, sample), axis=0)

    def fmap(self, f):
        return self.__class__(self._area, self._column, f(self.data))

    def result(self):
        l4_channels = self._data[:, 1]
        low_distance = (l4_channels - self._data[:, 0]).mean()
        high_distance = (self._data[:, 2] - l4_channels).mean()
        return np.array([l4_channels - low_distance, l4_channels,
                         l4_channels + high_distance]).T.round()

class AlignmentSummary(statistic.Summary):
    def __init__(self):
        super().__init__(LaminarAlignment)

    def signal_key(self, sig: signal.Signal):
        return os.path.commonprefix([
            loc.decode() for loc in sig.channels.location.values
        ])

    @classmethod
    def unpickle(cls, path):
        return statistic.Summary.unpickle(path, LaminarAlignment)
