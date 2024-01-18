#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from . import preprocess

class Recording:
    def __init__(self, events, trials, **signals):
        self._events = events
        self._signals = signals
        self._trials = trials
        for signal in self.signals:
            assert self.signals[signal].data.shape[-1] == len(self.trials)

    @property
    def events(self):
        return self._events

    def _event_bounds(self, event):
        event_keys = list(self.events.keys())
        successor = event_keys[event_keys.index(event) + 1]
        return self.events[event].values, self.events[successor].values

    def select_trials(self, f, *columns):
        trial_entries = (list(self.trials[col].values) for col in columns)
        selections = np.array([f(*entry) for entry in zip(*trial_entries)])

        events = self.events.loc[selections]
        trials = self.trials.loc[selections]
        signals = {k: s.select_trials(selections)
                   for k, s in self.signals.items()}
        return Recording(events, trials, **signals)

    @property
    def signals(self):
        return self._signals

    @property
    def trials(self):
        return self._trials

    def time_lock(self, event, duration=True, before=0., after=0.):
        onsets, offsets = self._event_bounds(event)
        if not isinstance(duration, bool):
            offsets = onsets + duration
        onsets = onsets - before
        offsets = offsets + after
        first, last = onsets.min(), offsets.max()

        signals = {k: s[first:last] for k, s in self.signals.items()}
        for sig in signals.values():
            sig.mask_events(onsets, offsets)

        events = {k: v for k, v in self.events.items()
                  if ((v >= first) & (v <= last)).all()}
        return TimeLockedSeries(events, **signals)

class TimeLockedSeries:
    def __init__(self, events, **signals):
        self._events = events
        self._signals = signals
        self._shape = None
        for signal in self.signals.values():
            if self._shape is None:
                self._shape = signal.data.shape
            else:
                assert signal.data.shape == self.shape
        assert len(self.shape) == 3 # Channels x Times x Trials

    @property
    def events(self):
        return self._events

    def plot_trial(self, t=None):
        fig, axes = plt.subplot_mosaic([[sig] for sig in self.signals],
                                       layout='constrained', sharex=True)
        for sig, ax in axes.items():
            ax.set_title(sig)
            if t is not None:
                signal = self.signals[sig].select_trials([t])
                signal.plot(ax=ax)
            else:
                signal = self.signals[sig].erp()
                signal.plot(ax=ax)

        for event in self.events:
            if t is not None:
                event_time = self.events[event][t]
            else:
                event_time = self.events[event].mean()
            for ax in axes.values():
                ymin, ymax = ax.get_ybound()
                ax.vlines(event_time, ymin, ymax, colors='black',
                          linestyles='dashed', label=event)
                ax.annotate(event, (event_time + 0.005, ymax))

    @property
    def shape(self):
        return self._shape

    @property
    def signals(self):
        return self._signals
