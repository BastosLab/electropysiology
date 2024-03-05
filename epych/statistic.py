#!/usr/bin/python3

from collections.abc import Iterable
import copy
import hdf5storage as mat
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import typing
from typing import Generic, Optional, TypeVar

from . import signal

T = TypeVar('T')

class Statistic(Generic[T]):
    def __init__(self, iid_shape, data: Optional[np.ndarray]=None):
        assert data is None or data.shape[:len(iid_shape)] == iid_shape
        self._iid_shape = iid_shape
        self._data = data

    def apply(self, element: tuple[T, ...]) -> np.ndarray:
        raise NotImplementedError

    def calculate(self, elements: Iterable[tuple[T, ...]]):
        for element in elements:
            self._data = self.apply(element)
        return self._data

    @property
    def data(self):
        return self._data

    def fmap(self, f):
        return self.__class__(self.iid_shape, f(self.values))

    @property
    def iid_shape(self):
        return self._iid_shape

    def pickle(self, path):
        assert os.path.isdir(path) or not os.path.exists(path)
        os.makedirs(path, exist_ok=True)

        arrays = {k: v for k, v in self.__dict__.items()
                  if isinstance(v, np.ndarray)}
        mat.savemat(path + ("/%s.mat" % self.__class__.name), arrays)
        other = copy.copy(self)
        for k in arrays:
            setattr(other, k, None)
        with open(path + ("/%s.pickle" % self.__class__.name), mode="wb") as f:
            pickle.dump(other, f)

    @classmethod
    def unpickle(cls, path):
        assert os.path.isdir(path)

        with open(path + ("/%s.pickle" % self.__class__.name), mode="rb") as f:
            self = pickle.load(f)

        arrays = mat.loadmat(path + ("/%s.mat" % self.__class__.name))
        for k in arrays:
            setattr(self, k, arrays[k])
        return self
