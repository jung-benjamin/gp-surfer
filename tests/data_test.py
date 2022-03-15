#! /usr/bin/env python3

"""Tests for the dataclass module"""

import numpy as np
from gaussianprocesses import dataclass as dc

def create_data():
    """Create data arrays for tests"""
    rng = np.random.default_rng(12345)
    x_1 = rng.standard_normal((10,2))
    y_1 = rng.standard_normal((10,1))
    return x_1, y_1

def test_data_plot():
    """Test the for Data.plot method"""
    data = dc.Data(*create_data())
    data.plot(show=False)
    assert True

def test_modeldata_plot():
    """Test for the ModelData.plot method"""
    d = create_data()
    dm = dc.ModelData(x_train=d[0], y_train=d[1])
    dm.plot(dataset='train', show=False)
