#!/usr/bin/python3

import dask.array
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import quantities as pq
import scipy.fft as fft
import syncopy as spy
from tqdm import tqdm

from .. import plotting, signal, signals, statistic

mne.set_log_level("CRITICAL")

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
    def __init__(self, df, channels, f0, chunk_trials=4, fmax=150, taper=None,
                 data=None, path=None):
        if not hasattr(fmax, "units"):
            fmax = np.array(fmax) * pq.Hz
        self._chunk_trials = chunk_trials
        self._df = df.rescale("Hz")
        self._f0 = f0.rescale("Hz")
        self._freqs = np.arange(0, fmax.item(), df.item())
        self._freqs = (self._freqs + df.item()) * df.units
        self._k = 0
        self._taper = taper
        self._path = path
        super().__init__(channels, (int((fmax / df).item()),), data=data)

    def apply(self, element: signal.EpochedSignal):
        assert (element.channels == self.channels).all().all()
        assert element.df == self.df
        assert element.f0 >= self.f0

        element_data = []
        channels = [str(ch) for ch in list(self.channels.index.values)]
        xs = element.data.magnitude - element.data.magnitude.mean(axis=-1,
                                                                  keepdims=True)
        tois = []
        for c in tqdm(range(0, element.num_trials, self._chunk_trials)):
            trials = slice(c, c + self._chunk_trials)
            trial_xs = mne.EpochsArray(
                np.moveaxis(xs[:, :, trials], -1, 0),
                mne.create_info(channels, int(self.f0.item())), proj=False,
            )

            data = spy.mne_epochs_to_tldata(trial_xs)
            cfg = spy.get_defaults(spy.freqanalysis)
            cfg.foi = self.freqs.magnitude.squeeze()
            cfg.ft_compat = True
            cfg.keeptrials = 'yes'
            cfg.method = 'mtmconvol'
            cfg.output = 'pow'
            cfg.polyremoval = 0
            cfg.t_ftimwin = 0.4
            cfg.taper = self._taper
            # Temporal resolution of 80ms.
            cfg.toi = np.arange(
                0, element.times[-1].magnitude - element.times[0].magnitude,
                0.08
            )
            # cfg.toi = "all"
            tfr = spy.freqanalysis(cfg, data)
            tois.append(tfr.time[0])
            path, ext = os.path.splitext(tfr.filename)
            if self.path:
                path = self.path
            tfr.save(filename=path + "/tfr_" + str(c) + ext)
            tfr._close()

            element_data.append(tfr.filename)

            del data
            del trial_xs
            spy.cleanup(interactive=False)

        toi = np.array(tois).mean(axis=0) * element.times[0].units +\
              element.times[0] + 0.2 * pq.second
        self._k += 1
        if self.data is None:
            return (element_data, toi)
        else:
            return (self.data[0] + element_data, self.data[1] + toi)

    def closest_freq(self, f):
        return np.nanargmin(np.abs(self.freqs - f))

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
        if self.data:
            return spy.load(self.data[0][0]).freq
        return self._freqs

    def result(self):
        elements = [spy.load(element) for element in self.data[0]]
        times = elements[0].sampleinfo[:, 1] - elements[0].sampleinfo[:, 0]
        channels = elements[0].channel
        shape = [len(elements[0].channel), int(times.mean()),
                 len(elements[0].freq)]
        tfrs = []
        ntrials = 0
        for element in elements:
            element_tfrs = element.show()
            if isinstance(element_tfrs, list):
                element_tfrs = np.stack(element_tfrs, axis=-1)
            else:
                element_tfrs = element_tfrs[:, :, :, np.newaxis]
            element_tfrs = np.moveaxis(element_tfrs, 2, 0)
            assert len(element_tfrs.shape) == 4
            tfrs.append(dask.array.from_array(element_tfrs))
            ntrials += element_tfrs.shape[-1]
            del element_tfrs
        del elements
        tfrs = dask.array.concatenate(tfrs, axis=-1)

        return signals.tfr.EpochedTfr(channels, tfrs.compute() * pq.Hz,
                                      np.diff(self.times).mean(), self.freqs,
                                      self.times)

    @property
    def path(self):
        return self._path

    @property
    def times(self):
        return self.data[1] / self._k
