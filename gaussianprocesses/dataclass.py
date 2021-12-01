#! /usr/bin/env python3

"""Classes for storing the data of the GPR model"""

import numpy as np

class Data():
    """Store x and y data for the GPR model"""

    def __init__(self, xdata=None, ydata=None):
        """Set x and y data of the class

        It is assumed that the lengths of x and y match.

        Parameters
        ----------
        xdata : np.ndarray, floats, (N x D)
            Input data for a gpr model. N is the number of 
            points in the data, and D is the length of each
            data point.
        ydata : np.ndarray, floats, (N)
            Output data for the gpr model, N is the number of 
            data points.
        """
        self.x = xdata
        self.y = ydata

    @property
    def x(self):
        """Return the x data"""
        return self._x

    @x.setter
    def x(self, x):
        """Set the x data"""
        self._x = x

    @property
    def y(self):
        """Return the y data"""
        return self._y

    @y.setter
    def y(self, y):
        """Set the y data"""
        self._y = y


class ModelData()
    """Store training, validation and testing data"""

    def __init__(self, xtrain, xtest, xvalidate, ytrain, ytest, yvalidate):
        """Set the data for the class"""
        self.train = Data(xtrain, ytrain)
        self.test = Data(xtest, ytest)
        self.validate = Data(xvalidate, yvalidate)

    @property
    def train(self):
        """Return the training data"""
        return self._train

    @train.setter
    def train(self, d):
        """Set the training data"""
        self._train = d

    @property
    def test(self):
        """Return the testing data"""
        return self._test

    @test.setter
    def test(self, d):
        """Set the testing data"""
        self._test = d

    @property
    def validate(self):
        """Return the validation data"""
        return self._validate

    @validate.setter
    def validate(self, d):
        """Set the validation data"""
        self._validate = d

