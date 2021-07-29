#! /usr/bin/env python3

"""Test script for the kernels module"""

import numpy as np
from gaussianprocesses import kernels

p = [1., 1., 1., 1.]
rng = np.random.default_rng()
x_1 = rng.standard_normal((10,2))
x_2 = rng.standard_normal((10,2))

sqe = kernels.SquaredExponential(p)
k, g = sqe(x_1, x_2, grad = True)

asqe = kernels.AnisotropicSquaredExponential(p)
ak, ag = asqe(x_1, x_2, grad = True)

test = sqe + asqe
tk, tg = test(x_1, x_2, grad = True)
print(tk, tg)
