#!/usr/bin/python3

import dask.array
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantities as pq

from .. import plotting, signal
from ..statistics import spectrum

class TimeFrequencyRepr(signal.Signal):
    def __init__(self, channels: pd.DataFrame, data, dt, freqs, timestamps):
        self._freqs = freqs
        super().__init__(channels, data, dt, timestamps)

    def baseline(self, start, end, decibels=False):
        first = np.abs(self.times - start).argmin()
        last = np.abs(self.times - end).argmin()
        base_mean = self.data[:, first:last, :].magnitude.mean(axis=1,
                                                               keepdims=True)
        base_mean = base_mean * self.data.units
        if decibels:
            tfrs = 10 * np.log10(self.data / base_mean) * spectrum.decibel
        else:
            tfrs = (self.data - base_mean) / base_mean * 100 * pq.percent
        return self.__class__(self.channels, tfrs, self.dt, self.freqs,
                              self.times)

    def channel_depths(self, column=None):
        if column is not None and column in self.channels:
            return self.channels[column].values
        return np.arange(len(self.channels))

    def channel_mean(self):
        middle_channel = len(self.channels) // 2
        channels = self.channels[middle_channel:(middle_channel + 1)]
        data = self.data.magnitude.mean(axis=0, keepdims=True) * self.data.units
        return self.__class__(channels, data, self.dt, self.freqs, self.times)

    def decibels(self):
        return self.fmap(lambda data: 10 * np.log10(data))

    @property
    def freqs(self):
        return self._freqs

    def fmap(self, f):
        return self.__class__(self.channels, f(self.data), self.dt, self.freqs,
                              self.times)

    @property
    def fmax(self):
        return self._freqs[-1]

    def __replace__(self, /, **changes):
        parameters = {field: changes.get(field, getattr(self, field)) for field
                      in ["channels", "data", "dt", "freqs", "times"]}
        return self.__class__(*parameters.values())

    def select_freqs(self, low, high):
        low_idx = np.argmin(np.abs(self.freqs - low))
        high_idx = np.argmin(np.abs(self.freqs - high))
        return self.__replace__(data=self.data[:, :, low_idx:high_idx, :],
                                freqs=self.freqs[low_idx:high_idx])

class EpochedTfr(TimeFrequencyRepr, signal.EpochedSignal):
    def __init__(self, channels: pd.DataFrame, data, dt, freqs, timestamps):
        assert len(data.shape) == 4
        assert len(channels) == data.shape[0]
        assert len(timestamps) == data.shape[1]
        assert timestamps.units == dt.units

        super(EpochedTfr, self).__init__(channels, data, dt, freqs, timestamps)

    def evoked(self):
        erp = super().evoked()
        return EvokedTfr(erp.channels, erp.data, erp.dt, self.freqs, erp.times)

class EvokedTfr(TimeFrequencyRepr, signal.EvokedSignal):
    def __init__(self, channels: pd.DataFrame, data, dt, freqs, timestamps):
        assert data.shape[-1] == 1
        super(EvokedTfr, self).__init__(channels, data, dt, freqs, timestamps)

    def evoked(self):
        erp = super().evoked()
        return EvokedTfr(erp.channels, erp.data, erp.dt, self.freqs, erp.times)

    def heatmap(self, alpha=None, ax=None, cmap=None, fbottom=0, fig=None,
                filename=None, ftop=None, title=None, vlim=None, vmin=None,
                vmax=None, baseline=None, cbar_ends=None, **events):
        lone = fig is None
        if fig is None:
            fig = plt.figure(figsize=(self.plot_width * 4, 3))
        if alpha is not None:
            alpha = alpha.squeeze()
        if ax is None:
            ax = fig.add_subplot()
        if ftop is None:
            ftop = self.fmax.item()
        vlim = 2 * self.data.std() if vlim is None else vlim
        if vmax is None:
            vmax = vlim
        if vmin is None:
            vmin = -vlim

        freqs = self.freqs
        times = self.times
        tfrs = self.data.squeeze()
        title = "Spectrogram" if title is None else title
        if tfrs.units.dimensionality.string == "%":
            title += " (% change from baseline)"
        plotting.heatmap(fig, ax, tfrs.T, alpha=alpha, cmap=cmap, title=title,
                         vmin=vmin, vmax=vmax, cbar_ends=cbar_ends)

        ax.set_xlim(0, len(times))
        xticks = [int(xtick) for xtick in ax.get_xticks()]
        zero_tick = self.sample_at(0.)
        zerotick_loc = (np.abs(np.array(xticks) - zero_tick)).argmin()
        xticks.insert(zerotick_loc, zero_tick)
        xticks[-1] = min(xticks[-1], len(times) - 1)
        xtick_times = times[xticks].round(decimals=2)
        xtick_times[zerotick_loc] = 0. * xtick_times.units
        ax.set_xticks(xticks, xtick_times)
        ax.set_xlabel("Time (seconds)")

        ax.set_ylim(0, tfrs.shape[-1])
        yticks = [int(ytick) for ytick in ax.get_yticks()]
        yticks[-1] = min(yticks[-1], tfrs.shape[-1] - 1)
        ax.set_yticks(yticks, ['{0:,.2f}'.format(f) for f in freqs[yticks]])
        ymin, ymax = ax.get_ybound()
        ax.set_ylabel("Frequency (Hz)")

        if baseline is not None:
            bxmin = self.sample_at(baseline[0])
            bxmax = self.sample_at(baseline[1])
            ax.axvspan(bxmin, bxmax, alpha=0.1, color='k')
            ax.annotate("Baseline", (bxmin + 0.5, ymax - 10))

        for (event, (time, color)) in events.items():
            xtime = self.sample_at(time)
            ax.vlines(xtime, *ax.get_ybound(), colors=color,
                      linestyles='dashed', label=event)
            ax.annotate(event, (xtime + 0.5, ymax - 10), color=color)

        band_bounds = np.unique(list(spectrum.THETA_BAND) +\
                                list(spectrum.ALPHA_BETA_BAND) +\
                                list(spectrum.GAMMA_BAND))
        yfreqs = [np.nanargmin(np.abs(freqs - bound)) for bound in band_bounds]
        ax.hlines(yfreqs, *ax.get_xbound(), colors='gray', linestyles='dotted')

        if lone:
            fig.set_layout_engine("tight")
        if filename is not None:
            fig.savefig(filename, dpi=100)
        if lone:
            plt.show()
            plt.close(fig)

    def plot(self, *args, **kwargs):
        return self.heatmap(*args, **kwargs)

    @property
    def plot_width(self):
        width = (self.times[-1] - self.times[0])
        if hasattr(width, "units"):
            width = width.magnitude
        return width

class BandPower(signal.Signal):
    def __init__(self, bands: pd.DataFrame, data, dt, timestamps):
        super().__init__(bands, data, dt, timestamps)

    def __replace__(self, /, **changes):
        parameters = {field: changes.get(field, getattr(self, field)) for field
                      in ["channels", "data", "dt", "freqs", "times"]}
        return self.__class__(*parameters.values())

class EpochedBandPower(BandPower, signal.EpochedSignal):
    def __init__(self, bands: pd.DataFrame, data, dt, timestamps):
        assert len(data.shape) == 4
        assert len(bands) == data.shape[0]
        assert len(timestamps) == data.shape[1]
        assert timestamps.units == dt.units

        super(EpochedBandPower, self).__init__(bands, data, dt, timestamps)

    def evoked(self):
        erp = super().evoked()
        return EvokedBandPower(erp.channels, erp.data, erp.dt, erp.times)

class EvokedBandPower(BandPower, signal.EvokedSignal):
    def __init__(self, bands: pd.DataFrame, data, dt, timestamps):
        assert data.shape[-1] == 1
        super(EvokedBandPower, self).__init__(bands, data, dt, timestamps)

    def evoked(self):
        erp = super().evoked()
        return EvokedBandPower(erp.channels, erp.data, erp.dt, erp.times)

    def plot(self, *args, **kwargs):
        return self.line_plot(*args, **kwargs)

    @property
    def plot_width(self):
        width = (self.times[-1] - self.times[0])
        if hasattr(width, "units"):
            width = width.magnitude
        return width
