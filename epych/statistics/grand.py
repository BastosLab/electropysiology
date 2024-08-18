#!/usr/bin/python3

import copy
import matplotlib.pyplot as plt
import numpy as np
import opencv2 as cv
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
                         np.concatenate((running["cat"], data), axis=-1)
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

class GrandNonparametricClusterTest(Statistic[T]):
    def __init__(self, alignment: alignment.LaminarAlignment, alpha=0.05,
                 data=None, partitions=1000):
        super().__init__((alignment.num_channels, alignment.num_times),
                         data=data)
        self._alignment = alignment
        self._alpha = alpha
        if data is None:
            self._data = {"left": None, "right": None}
        self._partitions = partitions

    @property
    def alignment(self):
        return self._alignment

    @property
    def alpha(self):
        return self._alpha

    def apply(self, element: tuple[T, T]):
        assert self.data["left"] is None and self.data["right"] is None
        assert element[0].dt == element[1].dt
        assert element[0].channels == element[1].channels

        return {"left": element[0], "right": element[0]}

    @property
    def partitions(self):
        return self._partitions

    def result(self):
        ldata, rdata = self.data["left"].data, self.data["right"].data
        lmean, rmean = ldata.mean(axis=-1), rdata.mean(axis=-1)
        lvar = np.var(ldata, axis=-1, ddof=1)
        rvar = np.var(rdata, axis=-1, ddof=1)
        ts, pvals = t_stats(ldata.shape[-1], lmean, lvar, rdata.shape[-1],
                            rmean, rvar)

        combined_data = np.concatenate((ldata, rdata), axis=-1)
        cluster_sizes = []
        for k in range(self.partitions):
            rng = np.random.default_rng()
            trials = rng.permutation(combined_data.shape[-1])
            ltrials = trials[:ldata.shape[-1]]
            rtrials = trials[ldata.shape[-1]:]

            l_pseudo = combined_data[:, :, ltrials]
            r_pseudo = combined_data[:, :, rtrials]
            l_pseudomean = l_pseudo.mean(axis=-1)
            r_pseudomean = r_pseudo.mean(axis=-1)
            l_pseudovar = np.var(l_pseudo, axis=-1, ddof=1)
            r_pseudovar = np.var(r_pseudo, axis=-1, ddof=1)
            ts, _ = t_stats(len(ltrials), l_pseudomean, l_pseudovar,
                            len(rtrials), r_pseudomean, r_pseudovar)
            N, labels = cv.connectedComponents(ts, connectivity=8)
            cluster_sizes.append(max([labels == n for n in range(1, N)]))

        critical_bin = round(self.alpha * self.partitions)
        bins = sorted(cluster_sizes, reverse=True)
        critical_val = bins[critical_bin]

        N, labels = cv.connectedComponents(ts, connectivity=8)
        for cluster in range(1, N):
            if sum(labels == cluster) < critical_val:
                ts[labels == cluster] = 0

        return self.data["left"].__class__(self.data["left"].channels, ts,
                                           self.data["left"].dt,
                                           self.data["left"].times)

def t_stats(ln, lmean, lvar, rn, rmean, rvar):
    l_stderr, r_stderr = lvar / ln, rvar / rn
    ts = (lmean - rmean) / np.sqrt(l_stderr + r_stderr)
    dfs = (l_stderr + r_stderr) ** 2
    dfs /= l_stderr ** 2 / (ln - 1) + r_stderr ** 2 / (rn - 1)
    pvals = scipy.special.stdtr(dfs, -np.abs(ts)) * 2
    return ts, pvals

def t_test(left: GrandVariance, right: GrandVariance):
    assert left.iid_shape == right.iid_shape
    k = n = 0

    return t_stats(left.data["n"], left.mean.data.magnitude,
                   left.result().data.magnitude, right.data["n"],
                   right.mean.data.magnitude, right.result().data.magnitude)

def summary_t_test(left: statistic.Summary, right: statistic.Summary):
    results = {}
    for key in (left.stats.keys() & right.stats.keys()):
        results[key] = t_test(left.stats[key], right.stats[key])
    return results
