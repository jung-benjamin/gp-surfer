#! /usr/bin/env python3

"""Classes for storing the data of the GPR model"""

import numpy as np
import matplotlib.pyplot as plt

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

    def plot(self, show=True, **pltkwargs):
        """Create scatterplots of the data

        Automatically adjusts the number of subplots
        according to the dimensions of the x-data.
        Tries to keep the plot close to a square shape.

        Parameters
        ----------
        show : bool, optional (default is True)
            If True, calls plt.show(). Otherwise, plt.close()
            is called.
        pltkwargs
            Keyword arguments for matplotlib subplots.
        """
        dim = self.x.shape[1]
        num_ax = (int(np.round(np.sqrt(dim))), int(np.ceil(np.sqrt(dim))))
        fig, axes = plt.subplots(*num_ax, **pltkwargs)
        for i, ax in zip(range(dim), axes.flatten()):
            ax.scatter(self.x[:,i], self.y)
        if show:
            plt.show()
        else:
            plt.close()


class ModelData():
    """Store training, validation and testing data"""

    def __init__(self, **kwargs):
        """Set the data for the class"""
        arguments = {'x_train' : None, 'y_train' : None,
                     'x_test' : None, 'y_test' : None,
                     'x_validate' : None, 'y_validate' : None
                     }
        arguments.update(kwargs)
        self.train = arguments
        self.test = arguments
        self.validate = arguments

    @property
    def train(self):
        """Return the training data"""
        return self._train

    @train.setter
    def train(self, d):
        """Set the training data"""
        self._train = Data(d['x_train'], d['y_train'])

    @property
    def test(self):
        """Return the testing data"""
        return self._test

    @test.setter
    def test(self, d):
        """Set the testing data"""
        self._test = Data(d['x_test'], d['y_test'])

    @property
    def validate(self):
        """Return the validation data"""
        return self._validate

    @validate.setter
    def validate(self, d):
        """Set the validation data"""
        self._validate = Data(d['x_validate'], d['y_validate'])

    def concatenate(self):
        """Concatenate the three categories into one Data object"""
        conc = Data()
        categories = [self.train, self.test, self.validate]
        conc.x = np.concatenate([n.x for n in categories if n.x is not None], axis=0)
        conc.y = np.concatenate([n.y for n in categories if n.y is not None], axis=0)
        if conc.x.shape[0] != conc.y.shape[0]:
            raise ValueError('Lengths of x and y do not match')
        return conc

    def plot(self, dataset, show=True):
        """Create scatterplots a dataset

        Plots either the training, validation or test data.

        Parameters
        ----------
        dataset : str
            Determine which dataset to plot. Either 'train',
            'test' or 'validate'.
        """
        data = getattr(self, dataset)
        print(data)
        data.plot(show)
