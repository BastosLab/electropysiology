#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft

from .. import signal
from ..spectrum import Spectrum

class LocalFieldPotential(signal.Signal):
    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.times, self.data.T.squeeze(), **kwargs)

    def csd(self, sigma, s):
        channel_csds = []
        for i in range(2, self.num_channels - 2):
            vi = (self.data[i-2] - 2 * self.data[i] + self.data[i+1])
            channel_csds.append(-sigma * vi / (2 * s ** 2))
        channel_csds = np.stack(channel_csds, axis=0)
        return self.__class__(self.channel_info[2:-2], channel_csds, self.dt,
                              self.times)

    def power_spectrum(self, dBs=True, relative=False, taper=None):
        xs = self.data
        if taper is not None:
            xs = taper(xs.shape[1])[np.newaxis, :, np.newaxis] * xs
        xf = fft.rfft(xs - xs.mean(axis=1, keepdims=True), axis=1)
        pows = (2 * self.dt ** 2 / self.T) * (xf * xf.conj())
        pows = pows[:, 0:xs.shape[1] // 2].real

        spectrum = Spectrum(self.df, pows)
        if relative:
            spectrum = spectrum.relative()
        if dBs:
            spectrum = spectrum.decibels()
        return spectrum.trial_mean()
