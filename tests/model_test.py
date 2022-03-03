#! /usr/bin/env python3

"""Tests for the model module"""

import numpy as np
from gaussianprocesses import models, kernels

def create_data():
    """Create data arrays for tests"""
    p = np.array([1., 1., 1., 1.])
    rng = np.random.default_rng(12345)
    x_1 = rng.standard_normal((10,2))
    y_1 = rng.standard_normal((10,1))
    return p, x_1, y_1

def create_model():
    """Create a GPR model for tests"""
    p, x_1, y_1 = create_data()
    asqe = kernels.AnisotropicSquaredExponential(p)
    trafo = {'x_trafo' : 'Normalize', 'y_trafo' : 'StandardNormalize'}
    gpr = models.GaussianProcessRegression(x_train=x_1,
                                           y_train=y_1,
                                           kernel=asqe,
                                           transformation=trafo,
                                           )
    return gpr

def test_model_optimize():
    """Test optimize method of the GPR class"""
    m = create_model()
    m.optimize(n_steps=10)
    assert True

def test_store_json(tmp_path):
    """Test creation of json files with GPR class"""
    m = create_model()
    m.optimize(n_steps=10)
    p = tmp_path / 'test.json'
    m.store_kernel(p, how='json')
    assert True

def test_load_json(tmp_path):
    """Teast loading of json files with GPR class"""
    m = create_model()
    p = tmp_path / 'test.json'
    m.store_kernel(p, how='json')
    n = models.GaussianProcessRegression.from_json(p)
    assert True
