#! /usr/bin/env python3
"""Analyze the scaling runs"""

import re
import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def argparser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    name = 'Name to identify the scaling run.'
    parser.add_argument('name', help=name, nargs='+')
    return parser.parse_args()


output_regex = re.compile(r'output_(\d+)_(\d+)')
logfile_regex = re.compile(r'training_(.+)-(\d+)-(\d+)_')


def get_scaling_config(fp):
    """Read number of cores from the filename."""
    if fp.suffix == '.log':
        group = logfile_regex.search(fp.name)
    elif fp.suffix == '.txt':
        group = output_regex.search(fp.name)
    return {
        'ncores': int(group.group(2)),
        'number': int(group.group(3)),
        'run': group.group(1)
    }


def read_log_runtimes(f):
    """Read runtimes of gp training from the logfile."""
    step_regex = re.compile(r'([a-zA-Z]+-*\d+) optimizer runtime: (\d+\.\d+)')
    loop_regex = re.compile(r'loop runtime: (\d+\.\d+)')
    script_regex = re.compile(r'Total runtime: (\d+\.\d+)')
    with f.open('r') as f:
        log = f.read()
        loop_runtime = loop_regex.search(log)
        script_runtime = script_regex.search(log)
        step_runtimes = step_regex.findall(log)
    return {
        'Total': float(script_runtime.group(1)),
        'Loop': float(loop_runtime.group(1)),
        'Steps': [(x, float(y)) for (x, y) in step_runtimes]
    }


def calc_speedup(df):
    """Calculate speedup in a dataframe of runtimes"""
    return df.loc[pd.IndexSlice[:, 1, :]] / df


def get_runtimes_from_logfile(fp):
    """Create dataframe with runtimes given a logfile."""
    cfg = get_scaling_config(fp)
    runtimes = read_log_runtimes(fp)
    runtime_df = pd.DataFrame(runtimes['Steps'], columns=['Type', 'Runtime'])
    runtime_df['Label'] = ['Step'] * len(runtime_df)
    loop_s = pd.Series({
        'Type': 'Loop',
        'Runtime': runtimes['Loop'],
        'Label': 'Loop'
    })
    script_s = pd.Series({
        'Type': 'Total',
        'Runtime': runtimes['Total'],
        'Label': 'Total'
    })
    df = pd.concat([runtime_df.T, loop_s, script_s], axis=1,
                   ignore_index=True).T
    df['Ncores'] = cfg['ncores']
    df['Number'] = cfg['number']
    df['Run'] = cfg['run']
    return df


def get_runtimes_from_dir(d):
    """Read runtimes from the logfiles in a directory"""
    return pd.concat(map(get_runtimes_from_logfile, d.glob('*.log')),
                     ignore_index=True)


def get_logfile_data(args):
    """Read runtime data from all directories"""
    sdirs = map(lambda x: Path('output', x), args.name)
    return pd.concat(map(get_runtimes_from_dir, sdirs))


def make_point_plots(log_data):
    """Create seaborn point plots of the runtimes."""
    gb = log_data.groupby('Run')
    fig, axes = plt.subplots(1, gb.ngroups, sharey='row')
    if gb.ngroups > 1:
        for ax, (g, group) in zip(axes, gb):
            sns.pointplot(group, x='Ncores', y='Runtime', hue='Label', ax=ax)
            ax.set_title(g)
    else:
        sns.pointplot(log_data, x='Ncores', y='Runtime', hue='Label', ax=axes)
    plt.tight_layout()
    plt.savefig('pointplot.png')


def plot_speedup_runtime(log_data):
    """Create seaborn relplots of runtimes and speedup."""
    mean_time = log_df.groupby(['Type', 'Ncores', 'Label',
                                'Run'])['Runtime'].mean()
    speedup = mean_time.groupby(level=0,
                                axis=0).apply(calc_speedup).droplevel(0,
                                                                      axis=0)
    rt = mean_time.copy()
    rt.name = 'Value'
    rt = pd.DataFrame(rt)
    rt['Quantity'] = 'Runtime'
    sp = speedup.swaplevel(1, 3).swaplevel(2, 3)
    sp.name = 'Value'
    sp = pd.DataFrame(sp)
    sp['Quantity'] = 'Speedup'
    data = pd.concat([rt, sp])
    fig, ax = plt.subplots(1, 1)
    sns.relplot(data,
                x='Ncores',
                y='Value',
                hue='Label',
                col='Run',
                row='Quantity',
                facet_kws={'sharey': 'row'})
    plt.savefig('relplot.png')


def plot_averaged_speedup_runtime(log_data):
    """Plot speedup and runtime averaged over the training steps."""
    mean_time = log_df.groupby(['Type', 'Ncores', 'Label',
                                'Run'])['Runtime'].mean()
    speedup = mean_time.groupby(level=0,
                                axis=0).apply(calc_speedup).droplevel(0,
                                                                      axis=0)
    rt = mean_time.copy()
    rt.name = 'Value'
    rt = pd.DataFrame(rt)
    rt['Quantity'] = 'Runtime'
    sp = speedup.swaplevel(1, 3).swaplevel(2, 3)
    sp.name = 'Value'
    sp = pd.DataFrame(sp)
    sp['Quantity'] = 'Speedup'
    data = pd.concat([rt, sp])
    averaged = data.groupby(['Label', 'Ncores', 'Run', 'Quantity']).mean()
    sns.relplot(averaged,
                x='Ncores',
                y='Value',
                row='Quantity',
                col='Run',
                hue='Label',
                facet_kws={'sharey': 'row'})
    plt.savefig('speedup.png')


def store_scaling(log_data, args):
    """Store scaling data to a csv file."""
    mean_time = log_df.groupby(['Type', 'Ncores', 'Label',
                                'Run'])['Runtime'].mean()
    speedup = mean_time.groupby(level=0,
                                axis=0).apply(calc_speedup).droplevel(0,
                                                                      axis=0)
    speedup = speedup.swaplevel(1, 3).swaplevel(2, 3)
    data = pd.concat([mean_time, speedup], keys=['Time', 'Speedup'], axis=1)
    data.to_csv(f'scaling_{"_".join(args.name)}.csv')
    print(data)

if __name__ == '__main__':
    args = argparser()
    log_df = get_logfile_data(args)
    make_point_plots(log_df)
    plot_speedup_runtime(log_df)
    plot_averaged_speedup_runtime(log_df)
    store_scaling(log_df, args)
