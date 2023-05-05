from setuptools import find_packages, setup

setup(
    name='gaussianprocesses',
    version='0.5.1',
    packages=find_packages(
        include=['gaussianprocesses', 'gaussianprocesses.*']),
    description='A package for gaussian process regression.',
    entry_points={
        'console_scripts': [
            'train_gp_surrogates = gaussianprocesses.scripts:train_gp_surrogates',
            'log_to_filepaths = gaussianprocesses.scripts:log_to_filepaths'
        ]
    },
)
