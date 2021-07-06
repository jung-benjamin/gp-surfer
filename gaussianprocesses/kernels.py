#/ /usr/bin/env python3

"""Kernel objects for gaussian process regression

A base class for the kernels is defined with which new
kernels can be defined by implementing the required
methods.
"""

import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist


class Kernel(ABC):
    """Abstract base class for a GPR kernel."""

    def __init__(self, parameters):
        """Initialize the kernel with parameter values

        The list entry of parameters should be the noise
        that is added to the kernel function.

        Parameters
        ----------
        parameters
            array like, floats
        """
        self.parameters = parameters

    def __call__(self, x1, x2, grad=False):
        """Construct the covariance matrix

        Apply the kernel function to x1 and x1.
        Optionally compute the gradient as well.
        """
        if grad:
            return self.kernel_function(x1, x2), self.kernel_gradient(x1, x2)
        else:
            return self.kernel_function(x1, x2)

    @abstractmethod
    def kernel_function(self, x1, x2):
        """Construct the covariance matrix"""
        pass

    @abstractmethod
    def kernel_gradient(self, x1, x2):
        """Compute the gradient of the covariance matrix"""
        pass


class SquaredExponential(Kernel):
    """A squared exponential kernel"""

    def kernel_function(self, x1, x2):
        """Describe the squared exponential kernel here"""
        sqdist = (np.sum(x1**2, axis = 1).reshape(-1, 1)
                  + np.sum(x2**2, 1)
                  - 2 * np.dot(x1, x2.T)
                  )
        cov = self.parameters[0]**2 * np.exp(-0.5 * sqdist / params[1]**2)
        noise = params[-1]**2 * np.eye(x1.shape[0])
        return cov + noise

    def kernel_gradient(self, x1, x2):
        """Compute the gradient of the kernel function"""
        sqdist = (np.sum(x1**2, axis = 1).reshape(-1, 1)
                  + np.sum(x2**2, 1)
                  - 2 * np.dot(x1, x2.T)
                  )
        k = self.kernel_function(x1, x2)
        gradients = [2 * params[0] * k,
                     np.multiply(sqdist / params[1]**3, k),
                     2 * params[-1] * np.eys(x1.shape[0]
                     ]
        return gradients


class AnisotropicSquaredExponential(Kernel):
    """An anisotropic squared exponential kernel"""

    def kernel_function(self, x1, x2):
        """Compute the covariance matrix"""
        lam = np.eye(len(x1[0]))
        length_scales = 1 / np.array(parameters[1:-1])
        np.fill_diagonal(lam, length_scales)
        x1 = np.dot(x1, lam)
        x2 = np.dot(x2, lam)
        sqdist = cdist(x1, x2, metric = 'sqeuclidean').T
        cov = params[0]**2 * np.exp(-0.5 * sqdist)
        noise = params[-1]**2 * np.eye(x1.shape[0])
        return cov + noise

    def kernel_gradient(self, x1, x2):
        """Compute the gradient of the covariance matrix"""
        k = self.kernel_function(x1, x2)
        g = [cdist(np.expand_dims(x1[:,i], -1), #np.newaxis?
                   np.expand_dims(x2[:,i], -1), 
                   metric = 'sqeuclidian'
                   ) / params[i+1]
             for i in range(x1.shape[1])
             ]
        gradients = ([2 * params[0] * k]
                     + [np.multiply(g[i], k) for i in range(x1.shape[1])]
                     + [2 * params[-1] * np.eye(x1.shape[0])]
                     )
        return gradients
        
