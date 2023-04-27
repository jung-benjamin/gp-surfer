#! /usr/bin/env python3
"""A script to create submission files and filepaths for scaling runs

Things this script needs to do:
- Create an overarching directory for the files of one scaling run
- Create an individual directory for the output of each simulation
- Create input files for each individual simulation
- Write one (or more) SLURM job array scripts with the correct variables

Variables
---------
n_cores : int
    Number of cores to use.
"""

import os
import json
import argparse
from pathlib import Path

from ropsam import slurm_utils

BASE = Path(os.environ['HOME']) / 'code' / 'gaussianprocesses' / 'scaling'


def argparser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    name = 'Name for the scaling run'
    parser.add_argument('name', help=name)
    ncores = 'Number of cores to use.'
    parser.add_argument('-n',
                        '--ncores',
                        help=ncores,
                        type=int,
                        nargs='*',
                        default=1)
    account = 'SLURM account to use.'
    parser.add_argument('-a', '--account', help=account)
    mail = 'Email address for SLURM notifications.'
    parser.add_argument('-m', '--mail', help=mail)
    submit = 'Submit the SLURM job directly.'
    parser.add_argument('-s', '--submit', action='store_true')
    conda_env = 'Select the conda environment.'
    parser.add_argument('--conda-env', help=conda_env, default='gp-env')
    more_env = 'Add more environment variables.'
    parser.add_argument('--more-env', help=more_env)
    non_exclusive = 'Reserve only the required cores on a node, instead of 48.'
    parser.add_argument('--non-excl', help=non_exclusive, action='store_true')
    set_scipy_blas = 'Set BLAS environment variable for scipy'
    parser.add_argument('--set-blas', help=set_scipy_blas, action='store_true')
    set_scipy_mkl = 'Set MKL environment variable for scipy'
    parser.add_argument('--set-mkl', help=set_scipy_mkl, action='store_true')
    return parser.parse_args()


def write_jobarray(ncores, njobs, args):
    """Write a SLURM job file for submitting an array job."""
    settings = slurm_utils.SimulationSettings(
        simfile_path='',
        simfile_name=('-x data/x_data.csv ' + 
            '-y data/y_data.csv -i isolist.json ' +
            f'--log-loc output/{args.name} ' +
            f'--model-loc output/{args.name} ' +
            f'{args.name}-{ncores}-' + '${(l:2::0:)SLURM_ARRAY_TASK_ID}'
            ),
        output_path=BASE / 'output' / args.name,
        output_name=f'output_{ncores}_%a.txt',
        slurm_job_name=f'gp-scaling-{args.name}-{ncores}',
        pwd=BASE,
        slurm_cpus_per_task={'True': ncores, 'False': 48}[str(args.non_excl)],
        slurm_time='48:00:00',
        slurm_array=f'1-{njobs}',
        slurm_mem_per_cpu='2G',
        conda_profile='. ~/anaconda/etc/profile.d/conda.sh',
        conda_env=args.conda_env,
        env_OMP_NUM_THREADS=ncores,
        echo_1='echo OMP Threads $OMP_NUM_THREADS',
        echo_2='echo MKL Threads $MKL_NUM_THREADS',
        echo_3='echo OpenBLAS Threads $OPENBLAS_NUM_THREADS',
    )
    settings.use('train_gp_surrogates')
    if args.account:
        settings.add_settings(slurm_account=args.account)
    if args.mail:
        settings.add_settings(slurm_mail_type='END,FAIL',
                              slurm_mail_user=args.mail)
    if args.set_blas:
        settings.add_settings(env_OPENBLAS_NUM_THREADS=ncores)
    if args.set_mkl:
        settings.add_settings(env_MKL_NUM_THREADS=ncores)
    if args.more_env:
        s1 = args.more_env.split(',')
        s2 = [x.split('=') for x in s1]
        d2 = dict(s2)
        settings.add_settings(**d2)
    cmd_list = settings.make_command_list()
    header = [f'Gaussianprocesses scaling run on {ncores} cores.']
    fw = slurm_utils.BatchFileWriter(cmd_list, header, **settings.slurm_vars)
    fw.write_batchfile(BASE / 'output' / args.name /
                       f'{args.name}_{ncores}_jobarray.sh')
    return BASE / 'output' / args.name / f'{args.name}_{ncores}_jobarray.sh'


if __name__ == '__main__':
    args = argparser()
    print(args.name)
    try:
        (BASE / 'output' / args.name).mkdir()
    except FileExistsError:
        pass
    for c in args.ncores:
        bf = write_jobarray(c, 10, args)
        slurm_utils.submit_slurm(bf, test_print=not args.submit)
