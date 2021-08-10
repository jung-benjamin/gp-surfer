#! /usr/bin/env python3

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

        Apply the kernel function to x1 and x2.
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

    def __mul__(self, other):
        return Product([self, other])


class Combination(Kernel):
    """Combine a list of kernels"""

    def __init__(self, kernels):
        self.kernel_list = kernels

    @abstractmethod
    def _combine_function(self):
        """Method to combine the kernel functions"""
        pass

    @abstractmethod
    def _combine_gradient(self):
        """Method to combine the kernel gradients"""
        pass

    def kernel_function(self, x1, x2):
        return self._combine_function([k.kernel_function(x1, x2) for k in self.kernel_list])

    def kernel_gradient(self, x1, x2):
        return self._combine_gradient([k.kernel_function(x1, x2) for k in self.kernel_list],
                                      [k.kernel_gradient(x1, x2) for k in self.kernel_list]
                                      )


class Sum(Combination):
    """Calculate the sum of kernels"""

    def _combine_function(self, l):
        s = l[0]
        for elem in l[1:]:
            s = s + elem
        return s

    def _combine_gradient(self, l, g):
        s = g[0]
        for elem in g[1:]:
            s = s + elem
        return s


class Product(Combination):
    """Calculate the product of kernels"""

    def _combine_function(self, l):
        p = l[0]
        for elem in l[1:]:
            p = p * elem
        return p

    def _combine_gradient(self, l, g):
        p = []
        for i, grad in enumerate(g):
            k = l.copy()
            k.pop(i)
            p_kernels = k[0]
            for elem in k[1:]:
                p_kernels = p_kernels * elem
            p_grad = [p_kernels * grad_elem for grad_elem in grad]
            p += p_grad
        return p


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
    
    def __init__(self, parameters):
        """Initialize the kernel with parameters"""
        Kernel.__init__(self, parameters)
        self.lambda_ = None
        
    def kernel_function(self, x1, x2):
        """Compute the covariance matrix"""
#         lam = np.eye(len(x1[0]))
#         length_scales = 1 / np.array(self.parameters[1:-1])
#         np.fill_diagonal(lam, length_scales)
        if self.lambda_ is None:
            lam = self.create_lambda(x1)
        else:
            lam = self.lambda_
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
    
    def create_lambda(self, x1):
        """Calculate the matrix with lengthscales for the kernel"""
        lam = np.eye(len(x1[0]))
        length_scales = 1 / np.array(self.parameters[1:-1])
        np.fill_diagonal(lam, length_scales)
        self.lambda_ = lam
        return lam
        

