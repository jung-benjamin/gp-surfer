#! /usr/bin/env python3
"""Write job array scripts for RASOR runs."""

import argparse
import json
import os
from pathlib import Path

from simple_slurm import Slurm


def update_command(template, keys):
    """Update the job array script."""
    with open(template, "r") as f:
        job = f.read()
    for key, value in keys.items():
        job = job.replace(f"@{key.upper()}@", str(value))
    return job


def argparser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    job_template = "Template file for the job array script."
    parser.add_argument("job_template", type=Path, help=job_template)
    workdir = "Working directory for the jobs."
    parser.add_argument("-w", "--wdir", type=str, help=workdir, required=True)
    x_train = "Path to training data inputs."
    parser.add_argument("-x",
                        "--xtrain",
                        type=Path,
                        help=x_train,
                        required=True)
    y_train = "Path to training data outputs."
    parser.add_argument("-y",
                        "--ytrain",
                        type=Path,
                        help=y_train,
                        required=True)
    name = "Name of the model."
    parser.add_argument("-n", "--name", type=str, help=name, required=True)
    log_location = "Location to save log files."
    parser.add_argument(
        "-l",
        "--logloc",
        type=Path,
        help=log_location,
        default=Path(os.environ["HPC_SCRATCH"]) / "logs",
    )
    modelloc = "Location to save model files."
    parser.add_argument(
        "-m",
        "--modelloc",
        type=Path,
        help=modelloc,
        required=True,
    )
    iter = "Number of iterations to run."
    parser.add_argument("-i", "--iter", type=int, help=iter, default=100)
    nuclides = "File with nuclide names."
    parser.add_argument("-f",
                        "--nuclides",
                        type=Path,
                        help=nuclides,
                        required=True)
    run = "Repeat number of the job."
    parser.add_argument("--run", type=int, help=run, default=1)
    num_train = "Number of training points."
    parser.add_argument("--numtrain", type=int, help=num_train, default=200)
    seed = "Random seed for the job."
    parser.add_argument("-s", "--seed", type=int, help=seed, default=2021)
    job_file = "Path to save job file."
    parser.add_argument("-j",
                        "--job-file",
                        type=Path,
                        help=job_file,
                        default=Path.cwd() / "jobfile.sh")

    parser.add_argument(
        "--slurm-args",
        help=("Slurm cookies to pass to the job script.\n" +
              "Will override any values in a config file." +
              "  Format: <slurm_cookie>=<value>\n" +
              "  Use _ instead of -. Leading -- will be added."),
        nargs="*")
    parser.add_argument(
        "--slurm-config",
        type=Path,
        help=(
            "Path to a slurm configuration file. Expects a JSON file.\n" +
            "If none is provided, ./slurm_config.json is used if available.\n"
            + "Otherwise looks for $HOME/slurm_config.json"))
    parser.add_argument("--code-env",
                        help="Command(s) to load the code environment.",
                        default="")
    return parser.parse_args()


if __name__ == '__main__':
    args = argparser()
    vars = vars(args)

    slurm_args = vars.pop("slurm_args")
    slurm_config = vars.pop("slurm_config")
    if slurm_config is None:
        f1 = Path.cwd() / "slurm_config.json"
        f2 = Path.home() / "slurm_config.json"
        if f1.exists():
            slurm_config = f1
        elif f2.exists():
            slurm_config = f2
    if slurm_config is not None:
        with open(slurm_config, "r") as f:
            slurm_cfg = json.load(f)
    else:
        slurm_cfg = {}
    if slurm_args is not None:
        slurm_arg_dict = {
            n: v
            for n, v in (arg.split("=") for arg in slurm_args)
        }
        slurm_cfg.update(slurm_arg_dict)

    slurm = Slurm(**slurm_cfg)
    slurm.add_cmd("# Load environment...")
    slurm.add_cmd(vars.pop("code_env", ""))
    slurm.add_cmd(
        "echo \"Environment loaded\"\n\n# Set up the working directory...")
    slurm.add_cmd(f"WDIR={vars.pop('wdir')}")
    slurm.add_cmd(f"mkdir -p $WDIR")
    slurm.add_cmd(f"cd $WDIR")
    slurm.add_cmd(
        "echo \"Working directory set to $WDIR\"\n\n# Run commands...")

    template = vars.pop("job_template")
    jobfile = vars.pop("job_file")
    job = update_command(template, vars)
    slurm.add_cmd(job)

    with open(jobfile, "w") as f:
        f.write(str(slurm).replace("\$", r"$"))
