#/ /usr/bin/env python3

"""Functions to prepare the data for the gpr

This module contains functions to normalize the 
input and the output data and to revert the normalizations
"""


def map_to_zero_one(arr, axis=0):
    """Normalize the data to between 0 and 1
    
    All entries along an axis of the array
    are divided by the maximum along that axis.
    """
    arr_max = arr.max(axis = axis)
    normed_arr = arr / arr_max
    return normed_arr, arr_max

def revert_zero_one_map(arr, factor):
    """Revert the normalization of the array"""
    return arr * factor

def map_to_standard_normal(arr, axis=0):
    """Map an array to a standard normal distribution"""
    arr_mean = arr.mean(axis = axis)
    arr_std = arr.std(axis = axis)
    normed_arr = (arr - arr_mean) / arr_std
    return normed_arr, arr_mean, arr_std

def revert_standard_normal_map(arr, mean, std):
    """Revert the normalization of an array
    
    Undo the normalization to a standard normal distribution
    by multiplying with the standard deviation and adding
    the mean value.
    """
    return arr * std + mean