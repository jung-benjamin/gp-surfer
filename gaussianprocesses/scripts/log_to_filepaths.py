#! /usr/bin/env python3
"""Create a filepaths file from training log data."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

PATH_REGEX = re.compile(
    r"""
    ([a-zA-z]{1,2}-\d{1,3}m?|[a-zA-z]{1,2}-\d{1,3}\*?)
    \s
    path
    :\s
    ([a-zA-Z0-9/\.\*_\-]*)
    """, re.X)

RSQ_REGEX = re.compile(
    r"""
    ([a-zA-z]{1,2}-\d{1,3}m?|[a-zA-z]{1,2}-\d{1,3}\*?)
    \s
    (test|validation)
    :\sr-squared=
    (-?\d*\.\d*)
    """, re.X)


def argparser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    infile = 'Input file with the log of the training run.'
    parser.add_argument('infile', help=infile, nargs='*', type=Path)
    threshold = 'Threshold R-squared for selecting surrogate models.'
    parser.add_argument('-t',
                        '--threshold',
                        help=threshold,
                        type=float,
                        default=0.99)
    drop_parents = 'Parents of the paths to drop before writing.'
    parser.add_argument('-d', '--drop-parents', help=drop_parents, type=Path)
    rsq_type = 'Select which R-squared value to use.'
    parser.add_argument('-r',
                        '--rsq-type',
                        help=rsq_type,
                        choices=['test', 'validation'],
                        default='test')
    outfile = 'Filepath (.json) for storing the filepaths.'
    parser.add_argument('-o', '--output', help=outfile, type=Path)
    update = 'Set flag to prohibit updating of existing output file.'
    parser.add_argument('-u', '--update', help=update, action='store_false')
    return parser.parse_args()


def read_rsquared(args):
    """Read r-squared values from the input file."""
    rsq_list = []
    for infile in args.infile:
        with open(infile, 'r') as f:
            log = f.read()
        rsq = RSQ_REGEX.findall(log)
        rsq_df = pd.DataFrame(rsq, columns=['Nuclide', 'Case', 'R-sq'])
        rsq_df.set_index('Nuclide', inplace=True)
        rsq_df['R-sq'] = rsq_df['R-sq'].astype('float64')
        rsq_list.append(rsq_df.pivot(columns='Case').droplevel(0, axis=1))
    if len(args.infile) > 1:
        keys = [Path(n).stem for n in args.infile]
        return pd.concat(rsq_list, axis=1, keys=keys)
    return pd.concat(rsq_list, axis=1)


def read_modelpaths(args):
    """Read paths to stored GP-models from the input file."""
    path_list = []
    for infile in args.infile:
        with open(infile, 'r') as f:
            log = f.read()
        path = PATH_REGEX.findall(log)
        if args.drop_parents:
            path = [(n, Path(p).relative_to(args.drop_parents))
                    for n, p in path]
        else:
            path = [(n, Path(p)) for n, p in path]
        path_df = pd.DataFrame(path, columns=['Nuclide', 'Path'])
        path_df.set_index('Nuclide', inplace=True)
        path_df['Path'] = path_df['Path'].map(Path)
        path_list.append(path_df)
    if len(args.infile) > 1:
        keys = [Path(n).stem for n in args.infile]
        return pd.concat(path_list, axis=1, keys=keys)
    return pd.concat(path_list, axis=1)


def query_outpath():
    """Request path for output file."""
    print('Output path exists and update is False.')
    o = Path(input('File path for the model file:')).with_suffix('.json')
    return o


def write_modelfile(args, paths):
    """Write the path to the best GP-models to a file"""
    if paths.columns.nlevels > 1:
        raise NotImplementedError
    else:
        d = {}
        for n, row in paths.iterrows():
            d[n] = row['Path'].parts
    if args.output:
        op = args.output
    else:
        op = Path(f'{args.infile[0].stem}_filepaths').with_suffix('.json')
    if op.exists() and args.update:
        with open(op, 'r') as f:
            ya = json.load(f)
        ya.update(d)
        d = ya
    elif op.exists():
        count = 0
        while op.exists():
            if count >= 2:
                override = input('Should the file be updated (y/n)? ')
                if override == 'y':
                    with open(op, 'r') as f:
                        ya = json.load(f)
                    ya.update(d)
                    d = ya
                    break
                else:
                    pass
            op = query_outpath()
            count += 1
    with open(op, 'w') as f:
        json.dump(d, f, indent=True)


def select_models(args):
    """Select surrogate models by R-squared value in log-file."""
    rsq = read_rsquared(args)
    paths = read_modelpaths(args)
    select = rsq[rsq[args.rsq_type] >= args.threshold]
    use_paths = paths.loc[select.index]
    write_modelfile(args, use_paths)


def log_to_filepaths():
    args = argparser()
    select_models(args)


if __name__ == '__main__':
    log_to_filepaths()
