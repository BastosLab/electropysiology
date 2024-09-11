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
        xs = mne.EpochsArray(np.moveaxis(element.data.magnitude, -1, 0),
                             mne.create_info(channels, self.f0.item()),
                             tmin=element.times[0].item())
        data = spy.mne_epochs_to_tldata(xs)
        cfg = spy.get_defaults(spy.freqanalysis)
        cfg.foi = self.freqs.magnitude.squeeze()
        cfg.ft_compat = True
        cfg.keeptrials = 'no'
        cfg.method = 'mtmfft'
        cfg.output = 'pow'
        cfg.t_ftimwin = 0.25
        cfg.taper = self._taper
        cfg.tapsmofrq = 4
        cfg.toi = "all"
        psd = spy.freqanalysis(cfg, data).show()

        if self.data is None:
            return np.moveaxis(psd, 0, 1)[:, :, np.newaxis]
        return np.concatenate((self.data, tfrs), axis=-1)

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
