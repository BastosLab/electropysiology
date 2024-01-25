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
    def __init__(self, trials: TrialInfo, **signals):
        for signal in signals.values():
            assert len(trials) in (0, signal.num_trials)
        self._signals = signals
        self._trials = trials

    def erp(self):
        event_types = list(self.trials.events)
        event_times = []
        for event in self.trials.events:
            times = self.trials[event].values * self.trials.unit(event)
            event_times.append(times.rescale(pq.second).mean())
        event_times = np.array(event_times) * pq.second
        intervals = Intervals(pd.DataFrame(data={"type": event_types,
                                                 "start": event_times,
                                                 "end": event_times}),
                              {"end": pq.second, "start": pq.second})
        signals = {k: v.erp() for k, v in self.signals.items()}
        return Recording(intervals, **signals)

    def event_lock(self, event, before=0., after=0.):
        assert self.trials.is_event(event)

        event_times = self.trials[event]
        onsets = (event_times - before).to_numpy()
        offsets = (event_times + after).to_numpy()
        first, last = onsets.min(), offsets.max()

        signals = {k: s[first:last] for k, s in self.signals.items()}
        for sig in signals.values():
            sig.mask_epochs(onsets, offsets)

        columns = []
        for column in self.trials.columns:
            if self.trials.is_event(column):
                v = self.trials[column]
                if ((v >= first) & (v <= last)).all():
                    columns.append(column)
            else:
                columns.append(column)
        trials = self.trials.filter(items=columns, axis="columns")
        return Sampling(trials, **signals)

    def select_trials(self, f, *columns):
        trial_entries = (list(self.trials[col].values) for col in columns)
        selections = np.array([f(*entry) for entry in zip(*trial_entries)])

        trials = self.trials.select(selections)
        signals = {k: s.select_trials(selections) for k, s
                   in self.signals.items()}
        return Sampling(trials, **signals)

    @property
    def signals(self):
        return self._signals

    @property
    def trials(self):
        return self._trials

class Intervals:
    def __init__(self, table: pd.DataFrame, units: dict[str, pq.UnitQuantity]):
        for column in units:
            assert column in table.columns
        assert "type" in table.columns
        assert "start" in table.columns and "end" in table.columns
        assert isinstance(units["start"], pq.UnitTime) and\
               isinstance(units["end"], pq.UnitTime)
        self._table = table
        self._units = units

    @property
    def epochs(self):
        for ty in self.table["type"].unique():
            if self.is_epoch(ty):
                yield ty

    @property
    def events(self):
        for ty in self.table["type"].unique():
            if self.is_event(ty):
                yield ty

    def filter(self, *args, **kwargs):
        table = self._table.filter(*args, **kwargs)
        units = {k: v for k, v in self._units.items() if k in table.columns}
        return Intervals(table, units)

    def is_epoch(self, key: str) -> bool:
        rows = self.table.loc[self.table["type"] == key]
        return all((rows["end"] - rows["start"]) > 0.)

    def is_event(self, key: str) -> bool:
        rows = self.table.loc[self.table["type"] == key]
        return all((rows["end"] - rows["start"]) == 0.)

    def query(self, key: str):
        return self.table.loc[self.table["type"] == key]

    @property
    def table(self):
        return self._table

    def uniques(self):
        uniques = ~self.table.duplicated("type", keep=False)
        return Intervals(self.table.loc[uniques], self.units)

    def unit(self, col: str):
        return self._units.get(col, None)

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
