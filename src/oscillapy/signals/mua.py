#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft

from .. import preprocess
from .. import signal

class MultiUnitActivity(signal.Signal):
    def __init__(self, channels, data, dt, sampling_times, zscore=True):
        if zscore:
            data = preprocess.zscore_trials(data)
        super().__init__(channels, data, dt, sampling_times)

    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.times, self.data.T.squeeze(), **kwargs)
