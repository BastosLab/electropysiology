#!/usr/bin/python3

import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy
from typing import TypeVar

from .. import plotting, signal, statistic

from . import alignment

T = TypeVar('T', bound=signal.EpochedSignal)

class GrandConcatenation(statistic.Statistic[T]):
    def __init__(self, alignment: alignment.LaminarAlignment, data=None):
        super().__init__((alignment.num_channels, alignment.num_times),
                         data=data)
        self._alignment = alignment
        self._dt = None
        if data is None:
            self._data = {"channels": None, "k": 0, "cat": None,
                          "timestamps": np.zeros(self.iid_shape[1])}
        self._signal_class = None

    @property
    def alignment(self):
        return self._alignment

    def apply(self, element: T):
        element = self.alignment.align(self.data["k"], element)
        assert len(element.channels) == self.num_channels
        assert element.data.shape[0] == self.num_channels
        data = element.data
        if self.num_times < element.data.shape[1]:
            data = data[:, :self.num_times, :]
        running = copy.deepcopy(self.data)

        channels = element.channels.reset_index(drop=True)
        if running["channels"] is None:
            running["channels"] = channels
            self._dt = element._dt
            self._signal_class = element.__class__
        else:
            for column in running["channels"].columns:
                if running["channels"][column].values.dtype == np.int64:
                    running["channels"][column] += channels[column]

        running["k"] += 1
        running["cat"] = data if running["cat"] is None else\
                         np.cat((running["cat"], data), axis=-1)
        running["timestamps"] += element.times[:self.num_times]
        return running

    @property
    def num_channels(self):
        return self.iid_shape[0]

    @property
    def num_times(self):
        return self.iid_shape[1]

    def result(self):
        times = self.data["timestamps"] / self.data["k"]
        channels = self.data["channels"].copy()
        for column in channels.columns:
            if channels[column].values.dtype == np.int64:
                channels[column] //= self.data["k"]
        return self._signal_class(channels, self.data["cat"], self._dt, times)

class GrandAverage(statistic.Statistic[T]):
    def __init__(self, alignment: alignment.LaminarAlignment, data=None):
        super().__init__((alignment.num_channels, alignment.num_times),
                         data=data)
        self._alignment = alignment
        self._dt = None
        if data is None:
            self._data = {"channels": None, "k": 0, "n": 0,
                          "sum": np.zeros((*self.iid_shape, 1)),
                          "timestamps": np.zeros(self.iid_shape[1])}
        self._signal_class = None

    @property
    def alignment(self):
        return self._alignment

    def apply(self, element: T):
        element = self.alignment.align(self.data["k"], element)
        assert len(element.channels) == self.num_channels
        assert element.data.shape[0] == self.num_channels
        data = element.data
        if self.num_times < element.data.shape[1]:
            data = data[:, :self.num_times, :]
        running = copy.deepcopy(self.data)

        channels = element.channels.reset_index(drop=True)
        if running["channels"] is None:
            running["channels"] = channels
            self._dt = element._dt
            self._signal_class = element.__class__
        else:
            for column in running["channels"].columns:
                if running["channels"][column].values.dtype == np.int64:
                    running["channels"][column] += channels[column]

        running["k"] += 1
        running["n"] += element.num_trials
        running["sum"] += data.sum(axis=-1, keepdims=True)
        running["timestamps"] += element.times[:self.num_times]
        return running

    def heatmap(self, ax=None, fig=None, title=None, vmin=None, vmax=None,
                origin="lower"):
        if ax is None:
            ax = plt.gca()
        if fig is None:
            fig = plt.gcf()

        data = self.result().squeeze()
        plotting.heatmap(fig, ax, data, cbar=False, title=title, vmin=vmin,
                         vmax=vmax)

        num_xticks = len(ax.get_xticks())
        xtick_locs = np.linspace(0, data.shape[1], num_xticks)
        xticks = np.linspace(0, data.shape[-1], num_xticks)
        xticks = ["%0.2f" % t for t in xticks]
        ax.set_xticks(xtick_locs, xticks)

    @property
    def num_channels(self):
        return self.iid_shape[0]

    @property
    def num_times(self):
        return self.iid_shape[1]

    def plot(self, **kwargs):
        return self.result().plot(**kwargs)

    def result(self):
        data = self.data["sum"] / self.data["n"]
        times = self.data["timestamps"] / self.data["k"]
        channels = self.data["channels"].copy()
        for column in channels.columns:
            if channels[column].values.dtype == np.int64:
                channels[column] //= self.data["k"]
        return self._signal_class(channels, data, self._dt, times).evoked()

class GrandVariance(statistic.Statistic[T]):
    def __init__(self, alignment: alignment.ChannelAlignment,
                 mean: signal.EvokedSignal, data=None):
        super().__init__((alignment.num_channels, alignment.num_times),
                         data=data)
        self._alignment = alignment
        self._mean = mean
        if data is None:
            self._data = {"diffs": np.zeros((*self.iid_shape, 1)), "k": 0,
                          "n": 0}

    @property
    def alignment(self):
        return self._alignment

    def apply(self, element: T):
        element = self.alignment.align(self.data["k"], element)
        assert len(element.channels) == self.alignment.num_channels
        assert element.data.shape[0] == self.alignment.num_channels
        data = element.data
        if self.alignment.num_times < element.data.shape[1]:
            data = data[:, :self.alignment.num_times, :]
        running = copy.deepcopy(self.data)

        running["diffs"] += ((data - self.mean.data) ** 2).sum(axis=-1,
                                                               keepdims=True)
        running["k"] += 1
        running["n"] += element.num_trials
        return running

    @property
    def mean(self):
        return self._mean

    def result(self):
        variance = self.data["diffs"] / (self.data["n"] - 1)
        return self.mean.__class__(self.mean.channels, variance, self.mean._dt,
                                   self.mean.times)

def t_test(left: GrandVariance, right: GrandVariance):
    assert left.iid_shape == right.iid_shape
    k = n = 0

    std_err2l = left.result().data.magnitude / left.data["n"]
    std_err2r = right.result().data.magnitude / right.data["n"]
    ts = (left.mean.data - right.mean.data).magnitude
    ts /= np.sqrt(std_err2l + std_err2r)
    dfs = (std_err2l + std_err2r) ** 2
    dfs /= std_err2l ** 2 / (left.data["n"] - 1) +\
           std_err2r ** 2 / (right.data["n"] - 1)
    pvals = scipy.special.stdtr(dfs, -np.abs(ts)) * 2
    return ts, pvals

def summary_t_test(left: statistic.Summary, right: statistic.Summary):
    results = {}
    for key in (left.stats.keys() & right.stats.keys()):
        results[key] = t_test(left.stats[key], right.stats[key])
    return results
