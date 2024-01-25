#!/usr/bin/python3

import collections.abc as abc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import quantities as pq
import seaborn as sns
import typing

from . import preprocess, signal

def epochs_from_records(intervals):
    return pd.DataFrame.from_records(intervals,
                                     columns=["type", "start", "end"])

def events_from_records(events):
    return pd.DataFrame.from_records(events, columns=["type", "time"])

class Sampling:
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
            intervals = pd.DataFrame(columns=["trial", "type", "start", "end"])

        trials = self.trials.mean(axis=0, numeric_only=True)
        trials = pd.DataFrame(data=trials.values[np.newaxis, :],
                              columns=trials.index.values)
        trials = trials.assign(trial=[0]).set_index("trial")

        signals = {k: v.erp() for k, v in self.signals.items()}
        return Recording(intervals, trials, self.units, **signals)

    def time_lock(self, time, before=0., after=0.):
        onset, offset = time - before, time + after

        signals = {k: s[onset:offset] for k, s in self.signals.items()}

        inner_intervals = self.intervals["start"].values >= onset &\
                          self.intervals["end"].values <= offset
        inner_intervals = self.intervals.loc[inner_intervals]
        if len(inner_intervals):
            inner_intervals[:, "start":"end"] -= onset
        return Sampling(inner_intervals, self.trials, self.units, **signals)

    @property
    def intervals(self):
        return self._intervals

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

    @property
    def trials(self):
        return self._trials

    @property
    def units(self):
        return self._units

class Recording(Sampling):
    def __init__(self, intervals: Intervals, **signals):
        self._intervals = intervals
        uniques = intervals.uniques().table
        unique_types = [ty in intervals.events for ty in uniques["type"].values]
        uniques = uniques.loc[unique_types]
        unique_events = uniques["type"].values
        unique_array = uniques["start"].values
        index = None
        if "trial" in uniques.columns:
            unique_array = (uniques["trial"].values,) + unique_array
            index = "trial"
        else:
            index = uniques.index
        unique_data = {k: v for (k, v) in zip(unique_events, unique_array)}
        trials = TrialInfo(pd.DataFrame(unique_data, index=index),
                                        {e: self._intervals.units["start"] for e
                                         in unique_events})
        super().__init__(trials, **signals)

    @property
    def epochs(self):
        for epoch in self.intervals.epochs:
            table = self.intervals.table
            rows = table.loc[table["type"] == epoch]
            start, end = rows["start"].item(), rows["end"].item()
            yield (epoch, start, end)

    @property
    def events(self):
        for event in self.intervals.events:
            table = self.intervals.table
            time = table.loc[table["type"] == event]["start"].item()
            yield (event, time)

    def epoch(self, epoch_type, before=0., after=0.):
        epochs = self.epochs[self.epochs["type"] == epoch_type]
        onsets, offsets = epochs["start"], epochs["end"]
        onsets, offsets = onsets - before, offsets + after
        first, last = onsets.min(), offsets.max()
        signals = {k: s[first:last] for k, s in self.signals.items()}
        for sig in signals.values():
            sig.mask_epochs(onsets, offsets)

        epoch_events = (self.events["time"] >= first) &\
                       (self.events["time"] <= last)
        epoch_events = self.events.loc[epoch_events]
        epoch_events = pd.DataFrame(epoch_events["time"].values,
                                    columns=list(epoch_events["type"].values))
        return Sampling(epoch_events, self.units, **signals)

    @property
    def intervals(self):
        return self._intervals

    def plot(self):
        fig, axes = plt.subplot_mosaic([[sig] for sig in self.signals],
                                       layout='constrained', sharex=True)
        for sig, ax in axes.items():
            ax.set_title(sig)
            self.signals[sig].plot(ax=ax)

        for (event, time) in self.events:
            for ax in axes.values():
                ymin, ymax = ax.get_ybound()
                ax.vlines(time, ymin, ymax, colors='black', linestyles='dashed',
                          label=event)
                ax.annotate(event, (time + 0.005, ymax))
