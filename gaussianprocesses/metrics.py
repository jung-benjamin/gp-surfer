#! /usr/bin/env python3
"""Metrics to evaluate quality of the GP models"""

import numpy as np


def total_sum_squares(data, axis=None):
    '''Calculate the total sum of squares of errors

    The squared sum of the difference between each data point
    to the mean of the data is the total variance of the data
    (not divided by the number of data points).

    Parameters
    ----------
    data
        (nx1) array, data points

    Output
    ------
    ss_tot
        float, total variance of array.

    '''
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    mean = data.mean()
    sq_diff = (data - mean)**2
    ss_tot = sq_diff.sum()
    return ss_tot


def sum_squares_residuals(pred, data):
    '''Calculate the sum of squares of the residual errors

    The squared sum of the residuals (difference between model
    prediction and data) represents the unexplained variance
    of the model.

    Parameters
    ----------
    pred
        (nx1) array, model predictions
    data
        (nx1) array, data points


    Returns
    -------
    ss_res
        float
    '''
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    res = (pred - data)**2
    ss_res = res.sum()
    return ss_res


def r_squared(pred, data):
    '''Calculate the R squared metric for predictions

    R squared is the coefficient of determination of the
    model predictions. It is calculated by dividing the
    sum of squares of the residuals by the total sum of
    squares and subtracting from 1.

    Parameters
    ----------
    pred
        (nx1) array, model predictions
    data
        (nx1) array, data points

    Returns
    -------
    Rsquare
        float, R squared value
    '''
    ss_res = sum_squares_residuals(pred, data)
    ss_tot = total_sum_squares(data)
    r_square = 1 - (ss_res / ss_tot)
    return r_square


def rmse(pred, data):
    """Calculate the root mean squared error

    Parameters
    ----------
    pred
        (nx1) array, model predictions
    data
        (nx1) array, data points

    Returns
    -------
    rmse
        float, root mean squared error
    """
    ss_res = sum_squares_residuals(pred, data)
    mse = ss_res / len(data)
    return np.sqrt(mse)


def range_normalized_rmse(pred, data):
    """Calculate the range normalized root mean squared error

    The range normalized root mean squared error (RNRMSE) is
    calculated by dividing the RMSE by the range of the data.

    Parameters
    ----------
    pred
        (nx1) array, model predictions
    data
        (nx1) array, data points

    Returns
    -------
    rrmse
        float, range normalized root mean squared error
    """
    rmse_val = rmse(pred, data)
    return rmse_val / (data.max() - data.min())


def mean_normalized_rmse(pred, data):
    """Calculate the mean normalized root mean squared error

    The mean normalized root mean squared error (MNRMSE) is
    calculated by dividing the RMSE by the mean of the data.

    Parameters
    ----------
    pred
        (nx1) array, model predictions
    data
        (nx1) array, data points

    Returns
    -------
    mnrms
        float, mean normalized root mean squared error
    """
    rmse_val = rmse(pred, data)
    return rmse_val / data.mean()


def iq_normalized_rmse(pred, data):
    """Calculate the interquartile range normalized root mean squared error

    The interquartile range normalized root mean squared error (IQRRMSE) is
    calculated by dividing the RMSE by the interquartile range of the data.

    Parameters
    ----------
    pred
        (nx1) array, model predictions
    data
        (nx1) array, data points

    Returns
    -------
    iqr_rms
        float, interquartile range normalized root mean squared error
    """
    rmse_val = rmse(pred, data)
    return rmse_val / (np.percentile(data, 75) - np.percentile(data, 25))


def mape(pred, data):
    """Calculate the mean absolute percentage error

    The mean absolute percentage error (MAPE) is calculated
    by summing over the absolute differences between actual
    and predicted value divided by the actual value. The
    sum is divided by the number of data points. The mape is
    given in %.

    Parameters
    ----------
    pred
        (nx1) array, model predictions
    data
        (nx1) array, data points

    Returns
    -------
    mape
        float, mean absolute percentage error
    """
    return np.abs(((data - pred) / data)).mean() * 100


class MetricFactory:
    """Factory class to create metric functions

    The factory class creates a dictionary of metric functions
    based on the input arguments. The metric functions are
    stored in the dictionary and can be accessed by their name.

    Parameters
    ----------
    metrics
        list of strings, names of the metrics to be created

    Returns
    -------
    dict
        dictionary of metric functions
    """

    metric_dict = {
        'r_squared': r_squared,
        'rmse': rmse,
        'range_normalized_rmse': range_normalized_rmse,
        'rnmse': range_normalized_rmse,
        'mean_normalized_rmse': mean_normalized_rmse,
        'mnrms': mean_normalized_rmse,
        'iq_normalized_rmse': iq_normalized_rmse,
        'iqrrmse': iq_normalized_rmse,
        'mape': mape,
    }

    def get_metric(self, metric):
        """Get the metric function by name"""
        if metric not in self.metric_dict:
            raise ValueError(f"Metric {metric} not found")
        return self.metric_dict[metric]
