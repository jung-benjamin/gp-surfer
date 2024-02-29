#! /usr/bin/env python3
"""Calculate R-squared for a set of GP models."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import gaussianprocesses as gp


def argparser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    testdata_x = 'Csv file with input values of test data'
    parser.add_argument('-x', '--test-x', help=testdata_x, type=Path)
    testdata_y = 'Csv file with output values of test data'
    parser.add_argument('-y', '--test-y', help=testdata_y, type=Path)
    model_file = 'Json file with GP model filepaths'
    parser.add_argument('-m', '--model-file', help=model_file, type=Path)
    model_dir = 'Parent directory of surrogate models.'
    parser.add_argument('-d', '--model-dir', help=model_dir, type=Path)
    num_points = 'Number of data points to include.'
    parser.add_argument('-n',
                        '--num-points',
                        help=num_points,
                        default=100,
                        type=int)
    output_file = 'Name of output file'
    parser.add_argument('-o', '--out-file', help=output_file, type=Path)
    return parser.parse_args()


def read_csv_data(datafile, index_col=None):
    """Read the data from the csv file into a dataframe."""
    data = pd.read_csv(datafile, index_col=index_col, header=[0])
    try:
        data.drop('Unnamed: 0', axis=0, inplace=True)
    except KeyError:
        pass
    return data


def load_model_dict(fp, base):
    """Load GP models from the filepaths file.
    
    The filepaths file stores the path to the model file
    relative to a base directory.
    """
    with fp.open('r') as f:
        path_dict = json.load(f)
    models = {}
    for n, it in path_dict.items():
        models[n] = gp.models.GaussianProcessRegression.from_json(
            Path(base, *it))
    return models


def load_test_data(fp_x, fp_y, num=100):
    """Load the x and y values of the test data."""
    x_test = read_csv_data(fp_x).iloc[-num:, :]
    y_test = read_csv_data(fp_y, index_col=0).iloc[:, -num:]
    return x_test, y_test


def calc_rsquared(model_dict, test_x, test_y):
    """Calculate r-squared for each model."""
    r_sq = {}
    for n, model in model_dict.items():
        r_sq[n] = model.evaluate_predictions(test_x.values,
                                             test_y.loc[n].values)
    return r_sq


def load_models(args):
    """Load models based on input arguments"""
    return load_model_dict(args.model_file, args.model_dir)


def load_data(args):
    """Load test data based on input arguments"""
    return load_test_data(args.test_x, args.test_y, args.num_points)


def store_results(results, fp):
    """Store results dictionary to a json file"""
    with fp.with_suffix('.json').open('w') as f:
        json.dump(results, f, indent=True)


if __name__ == '__main__':
    args = argparser()
    models = load_models(args)
    x_test, y_test = load_data(args)
    results = calc_rsquared(model_dict=models, test_x=x_test, test_y=y_test)
    store_results(results, args.out_file)
