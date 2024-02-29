#! /usr/bin/env python3
"""Tests for the transformation module"""

import numpy as np
from gaussianprocesses import transformations

np.random.seed(100)
rng = np.random.default_rng()
r_vals = rng.random(size=(10, 2)) * 100


def test_normalize_forward():
    trafo = transformations.Normalize()
    t = trafo.transform(r_vals)
    assert (t.min() >= 0 and t.max() <= 1)


def test_normalize_backwards():
    trafo = transformations.Normalize()
    t = trafo.transform(r_vals)
    u = trafo.untransform(t)
    assert np.all(np.isclose(r_vals, u, atol=np.finfo(float).eps))


def test_standardnormalize_forward():
    trafo = transformations.StandardNormalize()
    t = trafo.transform(r_vals)
    p = trafo.transformation_parameters
    m = np.isclose(p[0], r_vals.mean(axis=0), atol=np.finfo(float).eps)
    s = np.isclose(p[1], r_vals.std(axis=0, ddof=1), atol=np.finfo(float).eps)
    assert (np.all(m) and np.all(s))


def test_standardnormalize_backwards():
    trafo = transformations.StandardNormalize()
    t = trafo.transform(r_vals)
    u = trafo.untransform(t)
    assert np.all(np.isclose(r_vals, u, atol=np.finfo(float).eps))
