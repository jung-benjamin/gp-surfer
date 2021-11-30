#! /usr/bin/env python3

"""Transform the Data for GPR Training

Defines a base class that is used to transform data
for training, store the transformation parameters,
and untransform the data.
An example would be mapping the input data to values
between 0 and 1.

Mapping to 0 and 1 and transforming to a normal
distribution are implemented.
"""

import numpy as np
from abc import ABC, abstractmethod

class Transformation(ABC):
    """Abstract base class for transforming data"""

    def __init__(self, parameters=None):
        """Initialize the class, optionally with transformation parameters"""
        self.transformation_parameters = parameters

    @abstractmethod
    def transform(self, x):
        """Transform the data array"""
        pass

    @abstractmethod
    def untransform(self, x):
        """Revert the transformation"""
        pass

    @property
    def transformation_parameters(self):
        """Parameters used to transform the data"""
        return self._transformation_parameters

    @transformation_parameters.setter
    def transformation_parameters(self, p):
        """Set new transformtion parameters"""
        self._transformation_parameters = p


class Normalize(Transformation):
    """Normalize data to [0, 1]"""

    def transform(self, x):
        """Divide the data by its maximum

        Data (x) is a numpy array with two dimensions.
        Dim 0 iterates over the set of points.
        Dim 1 iterates over the different variables.
        """
        p = self.transformation_parameters
        if p is None:
            p = x.max(axis=0)
            self.transformation_parameters = p
        return x / p

    def untransform(self, x):
        """Multiply the data by the transformation parameter"""
        p = self.transformation_parameters
        if p is None:
            print("No transformation parameter exists.")
        return x * p


class StandardNormalize(Transformation):
    """Map data to a standard normal distribution"""

    def transform(self, x):
        """Transform x to standard normal distribution

        Subtract the mean and divide by the standard deviation.
        x is a numpy array with two dimensions.
        Dim 0 iterates over the set of points.
        Dim 1 iterates over the different variables.
        """
        p = self.transformation_parameters
        if p is None:
            p = (x.mean(axis=0), x.std(axis=0, ddof=1))
            self.transformation_parameters = p
        return (x - p[0]) / p[1]

    def untransform(self, x):
        """Revert the standard normal transformation"""
        p = self.transformation_parameters
        if p is None:
            print("No transformation parameter exists.")
        return x * p[1] + p[0]
