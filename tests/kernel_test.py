#! /usr/bin/env python3

"""Test script for the kernels module"""

import numpy as np
from gaussianprocesses import kernels

def create_kernel_input():
    """Create test input for a kernel"""
    p = [1., 1., 1., 1.]
    rng = np.random.default_rng()
    x_1 = rng.standard_normal((10,2))
    x_2 = rng.standard_normal((10,2))
    return p, x_1, x_2

def test_sqe_kernel_shape():
    p, x1, x2 = create_kernel_input()
    sqe = kernels.SquaredExponential(p)
    k = sqe(x1, x2, grad = False)
    assert k.shape == (10, 10)

def test_asqe_kernel_shape():
    p, x1, x2 = create_kernel_input()
    asqe = kernels.AnisotropicSquaredExponential(p)
    k = asqe(x1, x2, grad = False)
    assert k.shape == (10, 10)

def test_sqe_asqe_kernel_sum():
    p, x1, x2 = create_kernel_input()
    sqe = kernels.SquaredExponential(p)
    asqe = kernels.AnisotropicSquaredExponential(p)
    sum_k = sqe + asqe
    k = sum_k(x1, x2, grad = False)
    assert k.shape == (10, 10)

# sqe = kernels.SquaredExponential(p)
# k, g = sqe(x_1, x_2, grad = True)
# print('SQE kernel shape ', k.shape)
# print('SQE gradient shape ', len(g))
# for n in g:
#     print(n.shape)
# 
# asqe = kernels.AnisotropicSquaredExponential(p)
# ak, ag = asqe(x_1, x_2, grad = True)
# print('ASQE kernel shape ', ak.shape)
# print('ASQE gradient shape ', len(ag))
# for n in ag:
#     print(n.shape)
# 
# test = sqe + asqe
# tk, tg = test(x_1, x_2, grad = True)
# print('Sum kernel shape ', tk.shape)
# print('Sum gradient shape ', len(tg))
# for n in tg:
#     print(n.shape)

#print(tk, tg)
