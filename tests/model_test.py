#! /usr/bin/env python3

"""Tests for the model module"""

import numpy as np
from gaussianprocesses import models, kernels

p = np.array([1., 1., 1., 1.])
rng = np.random.default_rng()
x_1 = rng.standard_normal((10,2))
y_1 = rng.standard_normal((10,1))
x_2 = rng.standard_normal((20,2))


asqe = kernels.AnisotropicSquaredExponential(p)
gpr = models.GaussianProcessRegression(x_data = x_1,
                                       y_data = y_1,
                                       kernel = asqe
                                       )

gpr.optimize(n_steps = 10)
