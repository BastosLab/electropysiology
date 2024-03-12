#!/usr/bin/python3

import numpy as np

from .. import signal, statistic

class GrandAverage(statistic.Statistic[signal.EpochedSignal]):
    def __init__(self, iid_shape, data=None):
        if data is None:
            data = {"n": 0, "sum": np.zeros(*self.iid_shape)}
        super().__init__(iid_shape, data=data)

    def apply(self, element: signal.EpochedSignal):
        assert element.data.shape[:-1] == self.iid_shape
        running = self._data.copy()
        running["sum"] += element.data.sum(axis=-1)
        running["n"] += element.num_trials
        return running

    def result(self):
        return self.data["sum"] / self.data["n"]
