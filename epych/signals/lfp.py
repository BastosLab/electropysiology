#!/usr/bin/python3

from elephant.current_source_density import estimate_csd
import matplotlib.pyplot as plt
from neo import AnalogSignal
import numpy as np
import quantities as pq
import scipy.fft as fft

from .. import signal
from ..spectrum import Spectrum

class LocalFieldPotential(signal.Signal):
    def current_source_density(self, method="StandardCSD"):
        csd_trials = []
        for trial in range(self.num_trials):
            neo_lfp = AnalogSignal(self.data[:, :, trial].transpose(),
                                   units="V", sampling_rate = self.f0 * pq.Hz)
            channel_depths = self.channels["vertical"].values[:, np.newaxis]
            neo_lfp.annotate(coordinates=channel_depths * pq.mm)
            csd_trials.append(np.array(
                estimate_csd(neo_lfp, method=method).transpose()
            ))
        return self.__class__(self.channels, np.stack(csd_trials, axis=-1),
                              self.dt, self.times)

    def erp(self):
        erp = super().erp()
        return ContinuousLfp(erp.channels, erp.data, erp.dt, erp.times)

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
        return spectrum

class ContinuousLfp(LocalFieldPotential, signal.ContinuousSignal):
    iid_signal = LocalFieldPotential

    def plot(self, *args, **kwargs):
        return self.heatmap(*args, **kwargs)
