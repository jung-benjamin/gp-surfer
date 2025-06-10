#! /usr/bin/env python3
"""Calculate R-squared for a set of GP models."""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gaussianprocesses as gp

METRICS = ["rmse", "r_squared", "mape", "rnrmse", "mnrmse", "iqrrmse"]


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
    parser.add_argument('-o', '--out-dir', help=output_file, type=Path)

    subset_group = parser.add_mutually_exclusive_group()
    subset_group.add_argument('--subset-list',
                              nargs='+',
                              help='List of model names to include as subset.',
                              type=str)
    subset_group.add_argument(
        '--subset-file',
        help='JSON file containing list of model names to include as subset.',
        type=Path)
    return parser.parse_args()


def read_csv_data(datafile, index_col=None):
    """Read the data from the csv file into a dataframe."""
    data = pd.read_csv(datafile, index_col=index_col, header=[0])
    try:
        data.drop('Unnamed: 0', axis=0, inplace=True)
    except KeyError:
        pass
    return data


def load_model_dict(fp, base, subset=None):
    """Load GP models from the filepaths file.
    
    The filepaths file stores the path to the model file
    relative to a base directory.
    """
    with fp.open('r') as f:
        path_dict = json.load(f)
    models = {}

    names = subset if subset else path_dict.keys()
    for n in names:
        models[n] = gp.models.GaussianProcessRegression.from_json(
            Path(base, *path_dict[n]))
    return models


def load_test_data(fp_x, fp_y, num=100):
    """Load the x and y values of the test data."""
    x_test = read_csv_data(fp_x).iloc[-num:, :]
    y_test = read_csv_data(fp_y, index_col=0).iloc[:, -num:]
    return x_test, y_test


def plot_predictions(model_dict, test_x, test_y, output_dir=Path(".")):
    """Calculate the given metric for each model."""
    for n, model in model_dict.items():
        fig, axes = model.plot_predictions(
            x=test_x.values,
            y=test_y.loc[n].values,
            save=output_dir / f"{n}_predictions.png",
        )
        plt.close(fig)


def plot_models():
    """Evaluate the GP models."""
    args = argparser()
    if args.subset_file:
        with args.subset_file.open('r') as f:
            subset = json.load(f)
    elif args.subset_list:
        subset = args.subset_list
    else:
        subset = None
    models = load_model_dict(args.model_file, args.model_dir, subset=subset)
    x_test, y_test = load_test_data(args.test_x, args.test_y, args.num_points)
    plot_predictions(models, x_test, y_test, args.out_dir)


if __name__ == '__main__':
    plot_models()
