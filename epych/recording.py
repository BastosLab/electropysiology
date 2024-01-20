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
    return pd.DataFrame.from_records(events, index="name",
                                     columns=["name", "time"])

class Trials:
    def __init__(self, table: pd.DataFrame, units: dict[str, pq.UnitQuantity]):
        self._table = table
        for column in units:
            assert column in self._table.columns
        self._units = units

    @property
    def columns(self):
        return self.table.columns

    @property
    def events(self):
        for column in self.columns:
            if self.is_event(column):
                yield column

    def filter(self, *args, **kwargs):
        table = self._table.filter(*args, **kwargs)
        units = {k: v for k, v in self._units.items() if k in table.columns}
        return Trials(table, units)

    def __getitem__(self, key):
        return self.table.__getitem__(key)

    def is_event(self, key: str) -> bool:
        return isinstance(self.unit(key), pq.UnitTime)

    def __iter__(self):
        return self.table.__iter__()

    def __len__(self):
        return len(self.table)

    def select(self, key):
        table = self.table.loc[key]
        return Trials(table, self._units)

    @property
    def table(self):
        return self._table

    def unit(self, col: str):
        return self._units.get(col, None)

    @property
    def units(self):
        return self._units

class Sampling:
    def __init__(self, trials: Trials, **signals):
        for signal in signals.values():
            assert signal.num_trials == len(trials)
        self._signals = signals
        self._trials = trials

    def erp(self):
        events = [(event, self.trials[event].mean()) for event in self.trials.events]
        units = self.trials.units
        signals = {k: v.erp() for k, v in self.signals.items()}
        return Recording([], events, units, **signals)

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

class ContinuousRecording:
    def __init__(self, intervals, events, **signals):
        self._epochs = epochs_from_records(intervals)
        for (name, start, end) in self._epochs.iterttuples(False, None):
            events.append((name, start))
            events.append((name, end))
        self._events = events_from_records(events)
        self._signals = signals

    @property
    def epochs(self):
        return self._epochs

    @property
    def events(self):
        return self._events

    def epoch(self, epoch_type, before=0., after=0.):
        epochs = self.epochs[self.epochs["type"] == epoch_type]
        onsets, offsets = epochs["start"], epochs["end"]
        onsets, offsets = onsets - before, offsets + after
        first, last = onsets.min(), offsets.max()
        signals = {k: s[first:last] for k, s in self.signals.items()}
        for sig in signals.values():
            sig.mask_epochs(onsets, offsets)

        events = {k: v for k, v in self.events.items()
                  if ((v >= first) & (v <= last)).all()}
        return EpochedSeries(events, **signals)

class EpochedSeries:
    def __init__(self, trial_info, units, **signals):
        for sig in signals.values():
            assert isinstance(sig, signal.EpochedSignal)
            assert sig.num_trials == len(trial_info)
        self._signals = signals
        self._trial_info = trial_info
        self._units = units

    @property
    def events(self):
        for column in self.trial_info.columns:
            if isinstance(self.units[column], pq.UnitTime):
                yield column

    def plot(self, trial=None):
        fig, axes = plt.subplot_mosaic([[sig] for sig in self.signals],
                                       layout='constrained', sharex=True)
        for sig, ax in axes.items():
            ax.set_title(sig)
            if trial is not None:
                signal = self.signals[sig].select_trials([trial])
            else:
                signal = self.signals[sig].erp()
            signal.plot(ax=ax)

        for event in self.events:
            if trial is not None:
                event_time = self.trial_info[event][trial]
            else:
                event_time = self.trial_info[event].mean()
            for ax in axes.values():
                ymin, ymax = ax.get_ybound()
                ax.vlines(event_time, ymin, ymax, colors='black',
                          linestyles='dashed', label=event)
                ax.annotate(event, (event_time + 0.005, ymax))

    def select_trials(self, f, *columns):
        trial_entries = (list(self.trial_info[col].values) for col in columns)
        selections = np.array([f(*entry) for entry in zip(*trial_entries)])

        trial_info = self.trial_info.loc[selections]
        signals = {k: s.select_trials(selections) for k, s
                   in self.signals.items()}
        return EpochedSeries(trial_info, self.units, **signals)

    @property
    def signals(self):
        return self._signals

    def time_lock(self, event, before=0., after=0.):
        event_times = self.trial_info[event]
        onsets = (event_times - before).to_numpy()
        offsets = (event_times + after).to_numpy()
        first, last = onsets.min(), offsets.max()

        signals = {k: s[first:last] for k, s in self.signals.items()}
        for sig in signals.values():
            sig.mask_epochs(onsets, offsets)

        columns = []
        for column in self.trial_info.columns:
            if isinstance(self.units[column], pq.UnitTime):
                v = self.trial_info[column]
                if ((v >= first) & (v <= last)).all():
                    columns.append(column)
            else:
                columns.append(column)
        trial_info = self.trial_info.filter(items=columns, axis="columns")
        units = {k: v for k, v in self.units.items() if k in columns}
        return EpochedSeries(trial_info, units, **signals)

    @property
    def trial_info(self):
        return self._trial_info

    @property
    def units(self):
        return self._units
