#!/usr/bin/python3

import copy
import hdf5storage as mat
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns

THETA_BAND = (1., 4.)
ALPHA_BETA_BAND = (8., 30.)
GAMMA_BAND = (50., 150.)

class Spectrum:
    def __init__(self, df, pows, channels):
        assert isinstance(channels, pd.DataFrame)
        assert len(channels) == pows.shape[0]

        self._channels = channels
        self._df = df
        self._freqs = np.arange(0, pows.shape[1] / df, self.df)[np.newaxis, :]
        self._pows = pows

    def band_power(self, fbottom, ftop):
        ibot = np.nanargmin((self.freqs - fbottom) ** 2)
        itop = np.nanargmin((self.freqs - ftop) ** 2)
        return self.pows[:, ibot:itop+1].mean(axis=-1)

    @property
    def channels(self):
        return self._channels

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
            ftop = self.freqs[0, -1]

        sns.heatmap(self.pows, ax=ax, linewidth=0, cmap='viridis', cbar=False,
                    robust=True)
        ax.set_xlim(left=fbottom, right=ftop)

    def decibels(self):
        return Spectrum(self.df, 10 * np.log10(self.pows), self.channels)

    def pickle(self, path):
        assert os.path.isdir(path) or not os.path.exists(path)
        os.makedirs(path, exist_ok=True)

        self.channels.to_csv(path + '/channels.csv')
        mat.savemat(path + '/power_spectrum.mat', {"freqs": self.freqs,
                                                   "pows": self.pows})
        other = copy.copy(self)
        other._channels = other._freqs = other._pows = None
        with open(path + "/power_spectrum.pickle", mode="wb") as f:
            pickle.dump(other, f)

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
        return Spectrum(self.df, self.pows / max_pow, self.channels)

    def select_channels(self, mask):
        channels = self.channels.loc[mask] if self.channels is not None else None
        pows = self.pows[mask, :]
        return Spectrum(self.df, pows, channels)

    def trial_mean(self, axis=-1):
        return Spectrum(self.df, self.pows.mean(axis=axis), self.channels)

    @classmethod
    def unpickle(cls, path):
        assert os.path.isdir(path)

        with open(path + "/power_spectrum.pickle", mode="rb") as f:
            self = pickle.load(f)

        arrays = mat.loadmat(path + "/power_spectrum.mat")
        self._freqs, self._pows = arrays["freqs"], arrays["pows"]
        self._channels = pd.read_csv(path + '/channels.csv', index_col=0)
        self._channels["location"] = self._channels["location"].apply(eval)
        return self
