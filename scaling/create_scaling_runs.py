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
# CODE = Path(os.environ['HOME']) / 'differentiating-reactors'
# CORES = [1, 2, 4, 8, 16, 32]
# INPUT_DICT = {
#     "name": "test",
#     "ratiofile": str(CODE / 'data' / 'ratiolists.json'),
#     "key": "osborn",
#     "ev_file": str(CODE / 'data' / 'synthetic_evidence.csv'),
#     "ev_id": "c64p_0704",
#     "mixing_type": "normalize",
#     "model_files": {
#         "A": str(CODE / 'surrogates' / 'c64p_filepaths.json'),
#         "B": str(CODE / 'surrogates' / 'yb4p_filepaths.json')
#     },
#     "limits": {
#         "alphaA": {
#             "lower": 0,
#             "upper": 2
#         },
#         "burnupA": {
#             "lower": 0.1,
#             "upper": 1
#         },
#         "powerA": {
#             "lower": 0.03,
#             "upper": 0.15
#         },
#         "coolingA": {
#             "lower": 0,
#             "upper": 10000
#         },
#         "enrichmentA": {
#             "lower": 0.711,
#             "upper": 1.5
#         },
#         "alphaB": {
#             "lower": 0,
#             "upper": 2
#         },
#         "burnupB": {
#             "lower": 0.1,
#             "upper": 1
#         },
#         "powerB": {
#             "lower": 0.03,
#             "upper": 0.15
#         },
#         "coolingB": {
#             "lower": 0,
#             "upper": 10000
#         },
#         "enrichmentB": {
#             "lower": 0.711,
#             "upper": 1.5
#         }
#     },
#     "test_run": "False",
#     "surrogate_path": str(CODE / 'surrogates'),
# }
# ENV_VARS = {
#     'pymc5-env': {
#         'env_PYTENSOR_FLAGS':
#         ('exception_verbosity=high,' +
#          'base_compiledir=${TMP}/theano_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}'
#          )
#     },
#     'pymc5-linux': {
#         'env_PYTENSOR_FLAGS':
#         ('exception_verbosity=high,' +
#          'base_compiledir=${TMP}/theano_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}'
#          )
#     },
#     'pymc-linux': {
#         'env_THEANO_FLAGS':
#         ('exception_verbosity=high,' +
#          'base_compiledir=${TMP}/pytensor_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}'
#          )
#     }
# }


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
    no_cores = 'Do not set number of cors in PyMC sampling function.'
    parser.add_argument('--no-cores', help=no_cores, action='store_true')
    num_chains = 'Number of chains in the inference.'
    parser.add_argument('--num-chains', help=num_chains, type=int, default=1)
    conda_env = 'Select the conda environment.'
    parser.add_argument('--conda-env',
                        help=conda_env,
                        default='gp-env')
    draw = 'Number of draw steps in each Markov chain'
    parser.add_argument('--draw', type=int, default=1000, help=draw)
    more_env = 'Add more environment variables.'
    parser.add_argument('--more-env', help=more_env)
    non_exclusive = 'Reserve only the required cores on a node, instead of 48.'
    parser.add_argument('--non-excl', help=non_exclusive, action='store_true')
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
        pwd=BASE / args.name,
        slurm_cpus_per_task={'True': ncores, 'False': 48}[str(args.non_excl)],
        slurm_time='48:00:00',
        slurm_array=f'1-{njobs}',
        slurm_mem_per_cpu='2G',
        conda_profile='. ~/anaconda/etc/profile.d/conda.sh',
        conda_env=args.conda_env,
        # env_OMP_NUM_THREADS=ncores,
        # env_PYTENSOR_FLAGS=
        # ('exception_verbosity=high,' +
        #  'base_compiledir=${TMP}/pytensor_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}'
        #  ),
        echo_1='echo $OMP_NUM_THREADS',
        echo_2='echo $MKL_NUM_THREADS',
        echo_3='echo $OPENBLAS_NUM_THREADS',)
        # **ENV_VARS[args.conda_env])
    settings.use('train_gp_surrogates')
    if args.account:
        settings.add_settings(slurm_account=args.account)
    if args.mail:
        settings.add_settings(slurm_mail_type='END,FAIL',
                              slurm_mail_user=args.mail)
    if args.more_env:
        s1 = args.more_env.split(',')
        s2 = [x.split('=') for x in s1]
        d2 = dict(s2)
        settings.add_settings(**d2)
    cmd_list = settings.make_command_list()
    header = [f'Gaussianprocesses scaling run on {ncores} cores.']
    fw = slurm_utils.BatchFileWriter(cmd_list, header, **settings.slurm_vars)
    fw.write_batchfile(BASE / 'output' / args.name / f'{args.name}_{ncores}_jobarray.sh')
    return BASE / 'output' / args.name / f'{args.name}_{ncores}_jobarray.sh'


if __name__ == '__main__':
    args = argparser()
    print(args.name)
    try:
        (BASE / 'output' / args.name).mkdir()
    except FileExistsError:
        pass
    if args.ncores == 'all':
        for c in CORES:
            bf = write_jobarray(c, 10, args)
            slurm_utils.submit_slurm(bf, test_print=not args.submit)
    else:
        for c in args.ncores:
            bf = write_jobarray(c, 10, args)
            slurm_utils.submit_slurm(bf, test_print=not args.submit)
