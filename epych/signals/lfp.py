#!/usr/bin/python3

from elephant.parallel import ProcessPoolExecutor
from elephant.current_source_density import estimate_csd
import matplotlib.pyplot as plt
from neo import AnalogSignal
import numpy as np
import quantities as pq
import scipy.fft as fft

from .. import signal
from ..spectrum import Spectrum

class LocalFieldPotential(signal.Signal):
    def channel_depths(self, column=None):
        if column is not None and column in self.channels:
            return self.channels[column].values
        return np.arange(len(self.channels))

    def current_source_density(self, depth_column=None, method="StandardCSD"):
        data = self.get_data(None, None, None)
        if method is None:
            csd_channels = []
            for i in range(2, self.num_channels - 2):
                vi = (data[i-2] - 2 * data[i] + data[i+1])
                csd_channels.append(-0.4 * vi / (2 * 0.2 ** 2))
            csd_trials = np.stack(csd_channels, axis=0)
            channels = self.channels[2:-2]
        else:
            csd_trials = []
            depths = self.channel_depths(depth_column)[:, np.newaxis] * pq.mm
            neo_lfps = []
            for trial in range(self.num_trials):
                neo_lfp = AnalogSignal(data[:, :, trial].transpose(),
                                       units="V",
                                       sampling_rate = self.f0 * pq.Hz)
                neo_lfp.annotate(coordinates=depths)
                neo_lfps.append(neo_lfp)
            csd_trials = ProcessPoolExecutor().execute(estimate_csd, neo_lfps,
                                                       method=method)
            csd_trials = np.stack([np.array(t.transpose()) for t in csd_trials],
                                  axis=-1)
            channels = self.channels
        return self.__class__(channels, csd_trials, self.dt, self.times)

    def power_spectrum(self, dBs=True, relative=False, taper=None):
        xs = self.get_data(None, None, None)
        if taper is not None:
            xs = taper(xs.shape[1])[np.newaxis, :, np.newaxis] * xs
        xf = fft.rfft(xs - xs.mean(axis=1, keepdims=True), axis=1)
        pows = (2 * self.dt ** 2 / self.T) * (xf * xf.conj())
        pows = pows[:, 0:xs.shape[1] // 2].real

        spectrum = Spectrum(self.df, pows, self.channels)
        if relative:
            spectrum = spectrum.relative()
        if dBs:
            spectrum = spectrum.decibels()
        return spectrum

class EpochedLfp(LocalFieldPotential, signal.EpochedSignal):
    def __init__(self, channels, data, dt, timestamps):
        assert len(data.shape) == 3
        assert len(channels) == data.shape[0]
        assert len(timestamps) == data.shape[1]

        super(EpochedLfp, self).__init__(channels, data, dt, timestamps)

    def evoked(self):
        erp = super().evoked()
        return EvokedLfp(erp.channels, erp.data, erp.dt, erp.times)

class EvokedLfp(LocalFieldPotential, signal.EvokedSignal):
    def __init__(self, channels, data, dt, timestamps):
        assert data.shape[2] == 1
        super(EvokedLfp, self).__init__(channels, data, dt, timestamps)

    def evoked(self):
        erp = super().evoked()
        return EvokedLfp(erp.channels, erp.data, erp.dt, erp.times)

    def plot(self, *args, **kwargs):
        return self.heatmap(*args, **kwargs)

class RawLfp(LocalFieldPotential, signal.RawSignal):
    epoched_signal = EpochedLfp

    def __init__(self, channels, data, dt, timestamps, channels_dim=0,
                 time_dim=1):
        assert len(data.shape) == 2
        assert len(channels) == data.shape[channels_dim]
        assert len(timestamps) == data.shape[time_dim]

        self._channels_dim = channels_dim
        self._time_dim = time_dim
        super(RawLfp, self).__init__(channels, data, dt, timestamps,
                                     channels_dim, time_dim)
