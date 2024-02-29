#! /usr/bin/env python3
"""Test script for the kernels module"""

import numpy as np
from gaussianprocesses import kernels


def create_kernel_input(pars=4, shape=(10, 2)):
    """Create test input for a kernel"""
    p = list(np.ones(pars))
    rng = np.random.default_rng()
    x_1 = rng.standard_normal(shape)
    x_2 = rng.standard_normal(shape)
    return p, x_1, x_2


def kernel_shape(kernel, x1, x2):
    k = kernel(x1, x2, grad=False)
    assert k.shape == (x1.shape[0], x2.shape[0])


def kernel_gradient_length(kernel, x1, x2):
    k, g = kernel(x1, x2, grad=True)
    assert len(g) == len(kernel.parameters)


def test_sqe_kernel_shape():
    p, x1, x2 = create_kernel_input()
    return kernel_shape(kernels.SquaredExponential(p), x1, x2)


def test_asqe_kernel_shape():
    p, x1, x2 = create_kernel_input()
    return kernel_shape(kernels.AnisotropicSquaredExponential(p), x1, x2)


def test_linear_kernel_shape():
    p, x1, x2 = create_kernel_input()
    return kernel_shape(kernels.Linear(p), x1, x2)


def test_sqe_gradient_length():
    p, x1, x2 = create_kernel_input(pars=3)
    return kernel_gradient_length(kernels.SquaredExponential(p), x1, x2)


def test_asqe_gradient_length():
    p, x1, x2 = create_kernel_input()
    asqe = kernels.AnisotropicSquaredExponential(p)
    k, g = asqe(x1, x2, grad=True)
    assert len(g) == len(asqe.parameters)


def test_linear_gradient_length():
    p, x1, x2 = create_kernel_input(pars=5)
    return kernel_gradient_length(kernels.Linear(p), x1, x2)


def test_sqe_asqe_kernel_sum_shape():
    p, x1, x2 = create_kernel_input()
    sqe = kernels.SquaredExponential(p)
    asqe = kernels.AnisotropicSquaredExponential(p)
    sum_k = sqe + asqe
    k = sum_k(x1, x2, grad=False)
    assert k.shape == (10, 10)


def test_sqe_asqe_kernel_sum_value():
    p, x1, x2 = create_kernel_input()
    sqe = kernels.SquaredExponential(p)
    asqe = kernels.AnisotropicSquaredExponential(p)
    sum_k = sqe + asqe
    k = sum_k(x1, x2, grad=False)
    k1 = sqe(x1, x2, grad=False)
    k2 = asqe(x1, x2, grad=False)
    assert np.all(k == k1 + k2)


def test_sqe_asqe_kernel_product_shape():
    p, x1, x2 = create_kernel_input()
    sqe = kernels.SquaredExponential(p)
    asqe = kernels.AnisotropicSquaredExponential(p)
    sum_k = sqe * asqe
    k = sum_k(x1, x2, grad=False)
    assert k.shape == (10, 10)


def test_sqe_asqe_kernel_product_value():
    p, x1, x2 = create_kernel_input()
    sqe = kernels.SquaredExponential(p)
    asqe = kernels.AnisotropicSquaredExponential(p)
    sum_k = sqe * asqe
    k = sum_k(x1, x2, grad=False)
    k1 = sqe(x1, x2, grad=False)
    k2 = asqe(x1, x2, grad=False)
    assert np.all(k == k1 * k2)


def test_sqe_asqe_gradient_sum_length():
    p, x1, x2 = create_kernel_input()
    sqe = kernels.SquaredExponential(p)
    asqe = kernels.AnisotropicSquaredExponential(p)
    sum_k = sqe + asqe
    k, g = sum_k(x1, x2, grad=True)
    k1, g1 = sqe(x1, x2, grad=True)
    k2, g2 = asqe(x1, x2, grad=True)
    assert len(g) == (len(g1) + len(g2))


def test_sqe_asqe_gradient_product_length():
    p, x1, x2 = create_kernel_input()
    sqe = kernels.SquaredExponential(p)
    asqe = kernels.AnisotropicSquaredExponential(p)
    prod_k = sqe * asqe
    k, g = prod_k(x1, x2, grad=True)
    k1, g1 = sqe(x1, x2, grad=True)
    k2, g2 = asqe(x1, x2, grad=True)
    assert len(g) == (len(g1) + len(g2))
