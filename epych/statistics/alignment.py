#!/usr/bin/python3

from collections.abc import Iterable
import numpy as np
import os
import pandas as pd
import re

from .. import signal, statistic

class LaminarAlignment(statistic.Statistic[signal.EpochedSignal]):
    def __init__(self, area="VIS", column="location", data=None):
        self._area = area
        self._column = column
        super().__init__((1,), data=data)

    def apply(self, element: signal.Signal):
        area_mask = [self._area in loc.decode() for loc in
                     element.channels[self._column].values]
        area_channels = element.channels.loc[area_mask]
        area_l4 = os.path.commonprefix([l.decode() for l
                                        in area_channels.location]) + "4"
        l4_mask = [area_l4 in loc.decode() for loc in element.channels.location]
        l4_center = round(np.median(element.channels.loc[l4_mask].index))

        sample = np.array((area_channels.index[0], l4_center,
                           area_channels.index[-1]))[np.newaxis, :]
        if self.data is None:
            return sample
        return np.concatenate((self.data, sample), axis=0)

    def calculate(self, elements: Iterable[signal.Signal]):
        for element in elements:
            self._data = self.apply(element)
        l4_channel = np.median(self._data[:, 1])
        superficial_distance = (self._data[:, 0] - l4_channel).mean()
        deep_distance = (self._data[:, 2] - l4_channel).mean()

        superficial_channel = max(round(l4_channel + superficial_distance), 0)
        deep_channel = min(round(l4_channel + deep_distance),
                           max(self._data[:, 2]))
        return np.array([superficial_channel, l4_channel, deep_channel])

    def fmap(self, f):
        return self.__class__(self._area, self._column, f(self.data))
