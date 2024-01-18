#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import quantities as pq
import seaborn as sns

from . import preprocess, signal

def epochs_from_records(intervals):
    return pd.DataFrame.from_records(intervals,
                                     columns=["type", "start", "end"])

def events_from_records(events):
    return pd.DataFrame.from_records(events, index="name",
                                     columns=["name", "time"])

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
