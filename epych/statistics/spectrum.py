#!/usr/bin/python3

import copy
import hdf5storage as mat
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import scipy.fft as fft

from .. import plotting, signal, statistic

THETA_BAND = (1., 4.)
ALPHA_BETA_BAND = (8., 30.)
GAMMA_BAND = (50., 150.)

class PowerSpectrum(statistic.Statistic):
    def __init__(self, df, channels, f0, values=None):
        self._df = df
        self._freqs = np.arange(0, f0, df)[np.newaxis, :]
        super().__init__(channels, (f0,), values=values)

    def band_power(self, fbottom, ftop):
        ibot = np.nanargmin((self.freqs - fbottom) ** 2)
        itop = np.nanargmin((self.freqs - ftop) ** 2)
        return self.values[:, ibot:itop+1].mean(axis=1)

    def closest_freq(self, f):
        return np.nanargmin((self.freqs - f) ** 2)

    def decibels(self):
        return self.fmap(lambda vals: 10 * np.log10(vals))

    @property
    def df(self):
        return self._df

    def evoked(self):
        return self.fmap(lambda vals: vals.mean(axis=-1))

    def fmap(self, f):
        return self.__class__(self.df, self.channels, self.f0,
                              values=f(self.values))

    @property
    def f0(self):
        return self._freqs[:, -1] + self.df

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

        plotting.heatmap(fig, ax, self.values, "Power Spectral Density",
                         vmin=0., vmax=self.values.max())
        ax.set_xlim(left=fbottom, right=ftop)

    def plot_channels(self, stat, ax=None, xlims=None):
        if ax is None:
            ax = plt.gca()

        channels = np.arange(0, self.values.shape[0])
        ax.plot(stat, channels)
        if xlims is not None:
            ax.set_xlim(*xlims)
        ax.invert_yaxis()

    def relative(self):
        max_pow = self.values.max(axis=0, keepdims=True)
        return self.fmap(lambda vals: vals / max_pow)

    def update(self, s: signal.EpochedSignal, taper=None):
        assert (s.channels == self.channels).all().all()
        assert s.df == self.df

        xs = s.data
        if taper is not None:
            xs = taper(xs.shape[1])[np.newaxis, :, np.newaxis] * xs
        xf = fft.rfft(xs - xs.mean(axis=1, keepdims=True), axis=1)
        pows = (2 * s.dt ** 2 / s.T) * (xf * xf.conj())
        pows = pows[:, 0:xs.shape[1] // 2].real
        assert pows.shape[1] == self.f0

        if self.values is None:
            self._vals = pows
        else:
            self._vals = np.concatenate((self.values, pows), axis=-1)

        return self.values
