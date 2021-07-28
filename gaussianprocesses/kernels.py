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

        The last entry of parameters should be the noise
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

    def __add__(self, other):
        return Sum([self, other])


class Combination(Kernel):
    """Combine a list of kernels"""

    def __init__(self, kernels):
        self.kernel_list = kernels

    @abstractmethod
    def _combine(self):
        """Method to combine the kernel functions"""
        pass

    def kernel_function(self, x1, x2):
        return self._combine([k.kernel_function(x1, x2) for k in self.kernel_list])

    def kernel_gradient(self, x1, x2):
        return self._combine([k.kernel_gradient(x1, x2) for k in self.kernel_list])


class Sum(Combination):
    """Calculate the sum of kernels"""

    def _combine(self, l):
        s = l[0]
        for elem in l[1:]:
            s = s + elem
        return s


class SquaredExponential(Kernel):
    """A squared exponential kernel"""

    def kernel_function(self, x1, x2):
        """Describe the squared exponential kernel here"""
        sqdist = (np.sum(x1**2, axis = 1).reshape(-1, 1)
                  + np.sum(x2**2, 1)
                  - 2 * np.dot(x1, x2.T)
                  )
        cov = self.parameters[0]**2 * np.exp(-0.5 * sqdist / self.parameters[1]**2)
        noise = self.parameters[-1]**2 * np.eye(x1.shape[0])
        return cov + noise

    def kernel_gradient(self, x1, x2):
        """Compute the gradient of the kernel function"""
        sqdist = (np.sum(x1**2, axis = 1).reshape(-1, 1)
                  + np.sum(x2**2, 1)
                  - 2 * np.dot(x1, x2.T)
                  )
        k = self.kernel_function(x1, x2)
        gradients = [2 * self.parameters[0] * k,
                     np.multiply(sqdist / self.parameters[1]**3, k),
                     2 * self.parameters[-1] * np.eye(x1.shape[0])
                     ]
        return gradients


class AnisotropicSquaredExponential(Kernel):
    """An anisotropic squared exponential kernel"""

    def kernel_function(self, x1, x2):
        """Compute the covariance matrix"""
        lam = np.eye(len(x1[0]))
        length_scales = 1 / np.array(self.parameters[1:-1])
        np.fill_diagonal(lam, length_scales)
        x1 = np.dot(x1, lam)
        x2 = np.dot(x2, lam)
        sqdist = cdist(x1, x2, metric = 'sqeuclidean').T
        cov = self.parameters[0]**2 * np.exp(-0.5 * sqdist)
        noise = self.parameters[-1]**2 * np.eye(x1.shape[0])
        return cov + noise

    def kernel_gradient(self, x1, x2):
        """Compute the gradient of the covariance matrix"""
        k = self.kernel_function(x1, x2)
        g = [cdist(np.expand_dims(x1[:,i], -1), #np.newaxis?
                   np.expand_dims(x2[:,i], -1), 
                   metric = 'sqeuclidean'
                   ) / self.parameters[i+1]
             for i in range(x1.shape[1])
             ]
        gradients = ([2 * self.parameters[0] * k]
                     + [np.multiply(g[i], k) for i in range(x1.shape[1])]
                     + [2 * self.parameters[-1] * np.eye(x1.shape[0])]
                     )
        return gradients


if __name__ == '__main__':
    p = [1., 1., 1., 1.]
    rng = np.random.default_rng()
    x_1 = rng.standard_normal((10,2))
    x_2 = rng.standard_normal((10,2))

    sqe = SquaredExponential(p)
    k, g = sqe(x_1, x_2, grad = True)
    
    asqe = AnisotropicSquaredExponential(p)
    ak, ag = asqe(x_1, x_2, grad = True)

    test = sqe + asqe
    tk, tg = test(x_1, x_2, grad = True)
    print(tk, tg)
