#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class Spectrum:
    def __init__(self, df, pows):
        self._df = df
        self._freqs = np.arange(0, pows.shape[1] / df, self.df)[np.newaxis, :]
        self._pows = pows

    def band_power(self, fbottom, ftop):
        ibot = np.nanargmin((self.freqs - fbottom) ** 2)
        itop = np.nanargmin((self.freqs - ftop) ** 2)
        return self.pows[:, ibot:itop+1].mean(axis=-1)

    def closest_freq(self, f):
        return np.nanargmin((self.freqs - f) ** 2)

    @property
    def df(self):
        return self._df

    @property
    def freqs(self):
        return self._freqs

    def heatmap(self, fbottom=0, ftop=None, ax=None):
        if ax is None:
            ax = plt.gca()
        if ftop is None:
            ftop = self.freqs[-1]

        sns.heatmap(self.pows, ax=ax, linewidth=0, cmap='viridis', cbar=False,
                    robust=True)
        ax.set_xlim(left=fbottom, right=ftop)

    def decibels(self):
        return Spectrum(self.df, 10 * np.log10(self.pows))

    def plot_channels(self, stat, ax=None, xlims=None):
        if ax is None:
            ax = plt.gca()

        channels = np.arange(0, self.pows.shape[0])
        ax.plot(stat, channels)
        if xlims is not None:
            ax.set_xlim(*xlims)
        ax.invert_yaxis()

    @property
    def pows(self):
        return self._pows

    def relative(self):
        max_pow = self.pows.max(axis=0, keepdims=True)
        return Spectrum(self.df, self.pows / max_pow)

    def trial_mean(self, axis=-1):
        return Spectrum(self.df, self.pows.mean(axis=axis))
