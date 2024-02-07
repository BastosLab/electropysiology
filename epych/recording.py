#!/usr/bin/python3

from collections import Counter
import collections.abc as abc
import functools
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import quantities as pq
import typing

from . import preprocess, signal

def empty_intervals():
    return pd.DataFrame(columns=["trial", "type", "start", "end"])

class Sampling(abc.Sequence):
    def __init__(self, intervals: pd.DataFrame, trials: pd.DataFrame,
                 units: dict[str, pq.UnitQuantity], **signals):
        for column in units:
            assert column in intervals.columns or column in trials.columns
        assert set(intervals.columns) >= {"type", "start", "end"}
        assert trials.index.name == "trial"
        for signal in signals.values():
            assert len(trials) in (0, signal.num_trials)
        assert isinstance(units["start"], pq.UnitTime) and\
               isinstance(units["end"], pq.UnitTime)
        self._intervals = intervals
        self._signals = signals
        self._trials = trials
        self._units = units

    def __add__(self, other):
        assert self.signals.keys() == other.signals.keys()
        assert (self.trials == other.trials).all().all()
        assert self.units == other.units
        intervals = empty_intervals()
        signals = {k: self.signals[k] + other.signals[k] for k in self.signals}
        return self.__class__(intervals, self.trials, self.units, **signals)

    def baseline_correct(self, start, stop):
        return self.smap(lambda v: v.baseline_correct(start, stop))

    def erp(self):
        intervals = []
        for epoch_type in self.intervals["type"].unique():
            epochs = self.intervals.loc[self.intervals["type"] == epoch_type]
            epochs = epochs.mean(0, numeric_only=True)
            epochs = pd.DataFrame(data=epochs.values[np.newaxis, :],
                                  columns=epochs.index)
            intervals.append(epochs.assign(type=[epoch_type]))
        if intervals:
            intervals = pd.concat(intervals)
        else:
            intervals = empty_intervals()

        trials = self.trials.mean(axis=0, numeric_only=True)
        trials = pd.DataFrame(data=trials.values[np.newaxis, :],
                              columns=trials.index.values)
        trials = trials.assign(trial=[0]).set_index("trial")

        signals = {k: v.evoked() for k, v in self.signals.items()}
        return Recording(intervals, trials, self.units, **signals)

    def __getitem__(self, key):
        if isinstance(key, int):
            key = slice(key, key+1, None)
        if key.step is None:
            key = slice(key.start, key.stop, 1)

        onset, offset = key.start, key.stop
        inner_intervals = self.intervals["start"].values >= onset &\
                          self.intervals["end"].values <= offset
        inner_intervals = self.intervals.loc[inner_intervals]
        return self.__class__(inner_intervals, self.trials, self.units,
                              **{k: v[key] for k, v in self.signals.items()})

    @property
    def intervals(self):
        return self._intervals

    def __len__(self):
        return math.min(len(signal) for signal in self.signals.values())

    def select_trials(self, f, *columns):
        trial_entries = (list(self.trials[col].values) for col in columns)
        selections = np.array([f(*entry) for entry in zip(*trial_entries)])

        trials = self.trials.loc[selections]
        signals = {k: s.select_trials(selections) for k, s
                   in self.signals.items()}
        return Sampling(self.intervals, trials, self.units, **signals)

    @property
    def signals(self):
        return self._signals

    def smap(self, f):
        return self.__class__(self.intervals, self.trials, self.units, **{
            k: f(v) for k, v in self.signals.items()
        })

    def __sub__(self, other):
        assert self.signals.keys() == other.signals.keys()
        assert (self.trials == other.trials).all().all()
        assert self.units == other.units
        intervals = empty_intervals()
        signals = {k: self.signals[k] - other.signals[k] for k in self.signals}
        return self.__class__(intervals, self.trials, self.units, **signals)

    def time_lock(self, time, before=0., after=0.):
        key = slice(time - before, time + after, None)
        return self[key]

    @property
    def trials(self):
        return self._trials

    @property
    def units(self):
        return self._units

class Recording(Sampling):
    def __init__(self, intervals: pd.DataFrame, trials: pd.DataFrame,
                 units: dict[str, pq.UnitQuantity], **signals):
        assert len(trials) <= 1
        super().__init__(intervals, trials, units, **signals)

    def epoch(self, inner_epochs, outer_epochs=None, before=0., after=0.):
        targets = self.intervals.loc[inner_epochs]
        if outer_epochs is not None:
            parent = self.intervals.loc[outer_epochs]

            mask = []
            target_times = zip(targets["start"], targets["end"])
            for t, (start, end) in enumerate(target_times):
                idx = np.where((parent["start"] < start) & (end < parent["end"]))[0]
                if len(idx):
                    mask.append((targets.index[t], idx[0]))
            mask = np.array(mask)
            targets, epochs = targets.loc[mask[:, 0]], parent.loc[mask[:, 1]]
        else:
            epochs = targets
        assert len(targets) == len(epochs)
        befores = (targets["start"].values - epochs["start"].values).mean() + before
        afters = (epochs["end"].values - targets["end"].values).mean() + after
        onsets, offsets = targets["start"] - befores, targets["end"] + afters

        epoch_intervals = np.stack((onsets.values, offsets.values), axis=-1)
        trials = []
        for t, (onset, offset) in enumerate(epoch_intervals):
            inners = (self.intervals["start"] > onset) &\
                     (self.intervals["end"] < offset)
            inners = self.intervals.loc[inners]
            inners = inners.assign(trial=[t] * len(inners))
            inners.loc[:, "start":"end"] -= onset
            for inner in inners.itertuples():
                remainder = {
                    k: v for k, v in inner._asdict().items() if k not in
                    {"trial", "type", "start", "end", "Index"}
                }
                trials.append({
                    "trial": t,
                    inner.type + "_start": inner.start - befores,
                    inner.type + "_end": inner.end - befores,
                    **remainder
                })
        trial_columns = [set(trial.keys()) for trial in trials]
        trial_columns = trial_columns[0].union(*trial_columns[1:])
        trials = {
            k: [trial[k] if (k in trial) else pd.NA for trial in trials]
            for k in trial_columns
        }
        trials = pd.DataFrame(data=trials,
                              columns=list(trial_columns)).set_index("trial")
        trials = trials.groupby("trial").sum()
        signals = {k: s.epoch(epoch_intervals, -befores) for k, s in
                   self.signals.items()}
        return Sampling(empty_intervals(), trials, self.units, **signals)

    def plot(self, vmin=None, vmax=None, figure=None, figargs={}, **events):
        timespan = np.array([sig.times[-1] - sig.times[0] for sig in
                             self.signals.values()]).sum() * 4
        fig, axes = plt.subplot_mosaic([[sig for sig in self.signals]],
                                       figsize=(timespan, 3), dpi=100)

        for sig, ax in axes.items():
            self.signals[sig].plot(ax=ax, fig=fig, title=sig, vmin=vmin,
                                   vmax=vmax)
            for (event, time) in events.items():
                ymin, ymax = ax.get_ybound()
                xtime = self.signals[sig].sample_at(time)
                ax.vlines(xtime, ymin, ymax, colors='lightgreen',
                          linestyles='dashed', label=event)
                ax.annotate(event, (xtime + 0.005, ymax))

        fig.tight_layout()
        plt.show()
        if figure is not None:
            fig.savefig(figure, **figargs)
        plt.close(fig)
