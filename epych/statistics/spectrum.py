#!/usr/bin/python3

import matplotlib.pyplot as plt
import mne
import numpy as np
import quantities as pq
import scipy.fft as fft
import syncopy as spy

from .. import plotting, signal, statistic

THETA_BAND = (1. * pq.Hz, 4. * pq.Hz)
ALPHA_BETA_BAND = (8. * pq.Hz, 30. * pq.Hz)
GAMMA_BAND = (50. * pq.Hz, 150. * pq.Hz)

class PowerSpectrum(statistic.ChannelwiseStatistic[signal.EpochedSignal]):
    def __init__(self, df, channels, f0, fmax=150, taper=None, data=None):
        if not hasattr(fmax, "units"):
            fmax = np.array(fmax) * pq.Hz
        self._df = df.rescale("Hz")
        self._f0 = f0.rescale("Hz")
        self._freqs = np.arange(0, fmax.item(), df.item())
        self._freqs = (self._freqs + df.item()) * df.units
        self._taper = taper
        super().__init__(channels, (int((fmax / df).item()),), data=data)

    def apply(self, element: signal.EpochedSignal):
        assert (element.channels == self.channels).all().all()
        assert element.df == self.df
        assert element.f0 >= self.f0

        channels = [str(ch) for ch in list(self.channels.index.values)]
        xs = element.data.magnitude - element.data.magnitude.mean(axis=-1,
                                                                  keepdims=True)
        xs = mne.EpochsArray(np.moveaxis(xs, -1, 0),
                             mne.create_info(channels, self.f0.item()),
                             tmin=element.times[0].item())
        data = spy.mne_epochs_to_tldata(xs)
        cfg = spy.get_defaults(spy.freqanalysis)
        cfg.foi = self.freqs.magnitude.squeeze()
        cfg.ft_compat = True
        cfg.keeptrials = 'yes'
        cfg.method = 'mtmfft'
        cfg.output = 'pow'
        cfg.parallel = True
        cfg.polyremoval = 0
        cfg.t_ftimwin = 0.4
        cfg.taper = self._taper
        cfg.tapsmofrq = 4
        cfg.toi = "all"
        psd = np.stack(spy.freqanalysis(cfg, data).show(), axis=-1)

        if self.data is None:
            return np.moveaxis(psd, 0, 1)
        return np.concatenate((self.data, psd), axis=-1)

    def band_power(self, fbottom, ftop):
        ibot = np.nanargmin((self.freqs - fbottom) ** 2)
        itop = np.nanargmin((self.freqs - ftop) ** 2)
        return self.data[:, ibot:itop+1].mean(axis=1)

    def closest_freq(self, f):
        return np.nanargmin((self.freqs - f) ** 2)

    def decibels(self):
        return self.fmap(lambda vals: 10 * np.log10(vals))

    @property
    def df(self):
        return self._df

    @property
    def dt(self):
        return (1. / self.f0).rescale('s')

    def evoked(self):
        return self.fmap(lambda vals: vals.mean(axis=-1))

    def fmap(self, f):
        return self.__class__(self.df, self.channels, self.f0, fmax=self.fmax,
                              data=f(self.data))

    @property
    def f0(self):
        return self._f0

    @property
    def fmax(self):
        return self._freqs[-1]

    @property
    def freqs(self):
        return self._freqs

    def heatmap(self, fbottom=0, ftop=None, ax=None, fig=None):
        if ax is None:
            ax = plt.gca()
        if fig is None:
            fig = plt.gcf()
        if ftop is None:
            ftop = self.freqs[0, -1]

        plotting.heatmap(fig, ax, self.data, title="Power Spectral Density",
                         vmin=0., vmax=self.data.max())
        ax.set_xlim(left=fbottom, right=ftop)

    def plot_channels(self, stat, ax=None, xlims=None):
        if ax is None:
            ax = plt.gca()

        channels = np.arange(0, self.data.shape[0])
        ax.plot(stat, channels)
        if xlims is not None:
            ax.set_xlim(*xlims)
        ax.invert_yaxis()

    def relative(self):
        max_pow = self.data.max(axis=0, keepdims=True)
        return self.fmap(lambda vals: vals / max_pow)

    def result(self):
        return self.data.mean(axis=-1)

class Spectrogram(statistic.ChannelwiseStatistic[signal.EpochedSignal]):
    def __init__(self, df, channels, f0, fmax=150, taper=None, data=None):
        if not hasattr(fmax, "units"):
            fmax = np.array(fmax) * pq.Hz
        self._df = df.rescale("Hz")
        self._f0 = f0.rescale("Hz")
        self._freqs = np.arange(0, fmax.item(), df.item())
        self._freqs = (self._freqs + df.item()) * df.units
        self._taper = taper
        super().__init__(channels, (int((fmax / df).item()),), data=data)

    def apply(self, element: signal.EpochedSignal):
        assert (element.channels == self.channels).all().all()
        assert element.df == self.df
        assert element.f0 >= self.f0

        channels = [str(ch) for ch in list(self.channels.index.values)]
        xs = element.data.magnitude - element.data.magnitude.mean(axis=-1,
                                                                  keepdims=True)
        xs = mne.EpochsArray(np.moveaxis(xs, -1, 0),
                             mne.create_info(channels, self.f0.item()),
                             proj=False, tmin=element.times[0].item())
        data = spy.mne_epochs_to_tldata(xs)
        cfg = spy.get_defaults(spy.freqanalysis)
        cfg.foi = self.freqs.magnitude.squeeze()
        cfg.ft_compat = True
        cfg.keeptrials = 'yes'
        cfg.method = 'mtmconvol'
        cfg.output = 'pow'
        cfg.parallel = True
        cfg.polyremoval = 0
        cfg.t_ftimwin = 0.4
        cfg.taper = self._taper
        cfg.tapsmofrq = 4
        cfg.toi = "all"
        tfr = spy.freqanalysis(cfg, data)

        element_data = [(tfr, element.times)]
        return (self.data if self.data is not None else []) + element_data

    def closest_freq(self, f):
        return np.nanargmin((self.freqs - f) ** 2)

    @property
    def df(self):
        return self._df

    @property
    def dt(self):
        return (1. / self.f0).rescale('s')

    def fmap(self, f):
        return self.__class__(self.df, self.channels, self.f0, fmax=self.fmax,
                              data=f(self.data))

    @property
    def f0(self):
        return self._f0

    @property
    def fmax(self):
        return self._freqs[-1]

    @property
    def freqs(self):
        return self._freqs

    def heatmap(self, ax=None, baseline=None, fbottom=0, fig=None, ftop=None):
        if ax is None:
            ax = plt.gca()
        if fig is None:
            fig = plt.gcf()
        if ftop is None:
            ftop = self.fmax.item()
        freqs = self.data[0][0].freq
        time = self.times
        tfrs = self.result(baseline=baseline, channel_mean=True,
                           trial_mean=True) * 100
        boundary = max(abs(tfrs.min()), abs(tfrs.max()))
        plotting.heatmap(fig, ax, tfrs.T, title="Time Frequency Representation",
                         vmin=-boundary, vmax=boundary)

        ax.set_xlim(0, len(time))
        xticks = [int(xtick) for xtick in ax.get_xticks()]
        xticks[-1] = min(xticks[-1], len(time) - 1)
        ax.set_xticks(xticks, time[xticks].round(decimals=2))

        ax.set_ylim(0, tfrs.shape[-1])
        yticks = [int(ytick) for ytick in ax.get_yticks()]
        yticks[-1] = min(yticks[-1], tfrs.shape[-1] - 1)
        ax.set_yticks(yticks, freqs[yticks])

    def result(self, baseline=None, channel_mean=True, decibels=False,
               trial_mean=True):
        tfrs = np.concatenate([np.stack(tfr.show(), axis=0)
                               for (tfr, _) in self.data], axis=0)
        tfrs = tfrs.swapaxes(0, -1)
        if baseline is not None:
            first = np.abs(self.times - baseline[0]).argmin()
            last = np.abs(self.times - baseline[1]).argmin()
            base_mean = tfrs[:, first:last, :, :].mean(axis=1, keepdims=True)
            tfrs = (tfrs - base_mean) / base_mean
        if decibels:
            tfrs = 10 * np.log10(tfrs)
        if channel_mean:
            tfrs = tfrs.mean(axis=0)
        if trial_mean:
            tfrs = tfrs.mean(axis=-1)
        return tfrs

    @property
    def times(self):
        times = np.stack([times.magnitude for _, times in self.data],axis=-1)
        return times.mean(-1)
