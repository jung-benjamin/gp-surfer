#! /usr/bin/env python3

"""Train Gaussian processes as surrogate models"""

import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import gaussianprocesses as gp


def argparser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    name = 'Name to identify the set of gp-models.'
    parser.add_argument('name', help=name)
    x_data = 'Csv file containg the input variable data set.'
    parser.add_argument('-x', '--x-data', help=x_data)
    y_data = 'Csv file containing the output value data set.'
    parser.add_argument('-y', '--y-data', help=y_data)
    run = 'Identify the iteration of training.'
    parser.add_argument('-r', '--run', help=run, type=int, default=1)
    ntrain = 'Number of points in the training data.'
    parser.add_argument('-n', '--num-train', help=ntrain, type=int, default=200)
    target = 'Target R-squared value for the GP models.'
    parser.add_argument('-t', '--target', help=target, type=float, default=0.99)
    max_iter = 'Maximum number of iterations for optimizing r-squared.'
    parser.add_argument('--max-iter', help=max_iter, default=1, type=int)
    seed = 'Seed for the random number generator.'
    parser.add_argument('-s', '--seed', help=seed, type=int, default=2021)
    loglevel = 'Set the logging level.'
    parser.add_argument('-l', '--loglevel', help=loglevel, default='INFO')
    logloc = 'Set directory location for the log file.'
    parser.add_argument('--log-loc', help=logloc, default='.', type=Path)
    model_loc = 'Set directory where GP models are stored.'
    parser.add_argument('--model-loc', help=model_loc, default='.', type=Path)
    group = parser.add_mutually_exclusive_group()
    id_file = 'Json file with a list of nuclide ids.'
    group.add_argument('-i', '--id-file', help=id_file, type=Path)
    remove = 'Specify isotopes to ignore while training via a json file.'
    group.add_argument('--remove', help=remove, type=Path)
    return parser.parse_args()


def set_logger(args):
    """Configure the logging file."""
    level = getattr(logging, args.loglevel.upper())
    log = args.log_loc / f'training_{args.name}_{args.run}.log'
    msg = '%(levelname)s:%(message)s'
    logging.basicConfig(filename=log, format=msg, level=level)


def read_csv_data(datafile, index_col=None):
    """Read the data from the csv file into a dataframe."""
    data = pd.read_csv(datafile, index_col=index_col, header=[0])
    try:
        data.drop('Unnamed: 0', axis=0, inplace=True)
    except KeyError:
        pass
    return data


def get_model_ids(args, y_data):
    """Create a list of ids for the surrogate models."""
    if args.id_file:
        with args.id_file.open('r') as f:
            id_list = sorted(json.load(f))
    elif args.remove:
        with args.remove.open('r') as f:
            rem = json.load(f)
            id_list = sorted(set(y_data.index) - set(rem))
    else:
        id_list = sorted(y_data.index)
    return id_list


def train(name, x, y, outpath, ntrain=200, maxiter=1, seed=2021, target=0.99):
    """Train a gp-model"""
    train_x = x.values
    n_params = train_x.shape[1]
    train_y = y.loc[name,:].values
    trafo = {'x_trafo' : 'Normalize', 'y_trafo' : 'StandardNormalize'}
    kernel = gp.kernels.AnisotropicSquaredExponential([1]*(n_params+1) + [10])
    model = gp.models.GaussianProcessRegression(
        x_train=train_x, y_train=train_y, kernel=kernel, transformation=trafo
    )
    num_val = int((len(x) - ntrain) / 2 + ntrain)
    model.split_data(idx_train=(0, ntrain),idx_val=(num_val, None))
    try:
        metric, params, info = model.optimize_metric(
            target=target, seed=seed, maxiter=maxiter, full_output=True
        )
        model.store_kernel(outpath, how='json')
        logging.info(f'Model {name} validation: r-squared={metric}')
        test_metric = model.evaluate_predictions()
        logging.info(f'Model {name} test: r-squared={test_metric}')
        logging.info(f'Model {name} info: {info}')
        logging.info(f'Model {name} parameters: {params}')
        logging.info(f'Model {name} path: {outpath}')
    except np.linalg.LinAlgError:
        logging.exception(f'Training {name} failed.')
        return
    if test_metric < target:
        return 0
    else:
        return 1


def get_set_id(args):
    """Create the id for the set of GP models."""
    return f'{args.name}_{args.run}'


def make_model_dir(args):
    """Create directory where GP-models are stored."""
    base = args.model_loc
    set_id = get_set_id(args)
    if not base.name == set_id:
        model_dir = base / set_id
    else:
        model_dir = base
    try:
        model_dir.mkdir(parents=True)
    except FileExistsError:
        pass
    return model_dir


def train_gp_surrogates():
    """Train Gaussian processes as surrogate models"""
    args = argparser()
    set_logger(args)
    x_data = read_csv_data(args.x_data)
    y_data = read_csv_data(args.y_data, index_col=0)
    data_len = min([x_data.shape[0], y_data.shape[1]])
    x_data = x_data.iloc[:data_len]
    y_data = y_data.iloc[:,:data_len]
    logging.debug(f'Input variable data shape: {x_data.shape}')
    logging.debug(f'Output value data shape: {y_data.shape}')
    id_list = get_model_ids(args, y_data)
    logging.info(f'Training GP models for: {id_list}')
    model_dir = make_model_dir(args)
    logging.info(f'Location of the GP-models: {model_dir}')
    crashed, failed, success = [], [], []
    file_paths = {}
    for iso in id_list:
        file_name = f'{args.name}_{iso}_{args.run}.json'
        training = train(
            name=iso,
            x=x_data,
            y=y_data,
            outpath=model_dir / file_name,
            maxiter=args.max_iter,
            ntrain=args.num_train,
            seed=args.seed,
            target=args.target
        )
        if training:
            success.append(iso)
            file_paths[iso] = [get_set_id(args), file_name]
        elif training is None:
            crashed.append(iso)
        else:
            failed.append(iso)
    path_file = args.model_loc / f'{args.name}_filepaths.json'
    logging.info(f'Writing path file to {path_file}')
    try:
        with path_file.open('r') as f:
            existing_file_paths = json.load(f)
    except FileNotFoundError:
        pass
    else:
        existing_file_paths.update(file_paths)
        file_paths = existing_file_paths
    with path_file.open('w') as f:
        json.dump(file_paths, f, indent=True)
    logging.info(f'Training successful: {success}')
    logging.info(f'Training unsuccessful: {failed}')
    logging.info(f'Training crashed: {crashed}')

if __name__ == '__main__':
    train_gp_surrogates()