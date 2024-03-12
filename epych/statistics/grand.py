#!/usr/bin/python3

import numpy as np

from .. import signal, statistic

class GrandAverage(statistic.ChannelwiseStatistic[signal.EpochedSignal]):
    def apply(self, element: signal.EpochedSignal):
        assert element.data.shape[:-1] == self.iid_shape
        running = self._data if self._data is not None else {
            "n": 0, "sum": np.zeros(*self.iid_shape),
        }
        running["sum"] += element.data.sum(axis=-1)
        running["n"] += element.num_trials
        return running

    def result(self):
        return self.data["sum"] / self.data["n"]
