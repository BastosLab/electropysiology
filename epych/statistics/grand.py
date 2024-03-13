#!/usr/bin/python3

import copy
import numpy as np

from .. import signal, statistic

class GrandAverage(statistic.Statistic[signal.EpochedSignal]):
    def __init__(self, iid_shape, data=None):
        super().__init__(iid_shape, data=data)
        if data is None:
            self._data = {"n": 0, "sum": np.zeros((*self.iid_shape, 1))}

    def apply(self, element: signal.EpochedSignal):
        assert element.data.shape[0] == self.iid_shape[0]
        data = element.data
        if self.iid_shape[1] < element.data.shape[1]:
            data = data[:, :self.iid_shape[1], :]
        running = copy.deepcopy(self.data)
        running["sum"] += data.sum(axis=-1, keepdims=True)
        running["n"] += element.num_trials
        return running

    def result(self):
        return self.data["sum"] / self.data["n"]
