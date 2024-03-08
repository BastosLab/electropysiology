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

    def calculate(self, elements: Iterable[signal.Signal]):
        for element in elements:
            self._data = self.apply(element)
        l4_channels = self._data[:, 1]
        superficial_distance = (l4_channels - self._data[:, 0]).mean()
        deep_distance = (self._data[:, 2] - l4_channels).mean()
        return np.array([l4_channels - superficial_distance, l4_channels,
                         l4_channels + deep_distance]).T.round()

    def fmap(self, f):
        return self.__class__(self._area, self._column, f(self.data))
