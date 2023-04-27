from setuptools import find_packages, setup

setup(
    name='gaussianprocesses',
    version='0.4.0',
    packages=find_packages(
        include=['gaussianprocesses', 'gaussianprocesses.*']),
    description='A package for gaussian process regression.',
    entry_points={
        'console_scripts': [
            'train_gp_surrogates = scripts:train_gp_surrogates',
            'log_to_filepaths = scripts:log_to_filepaths'
        ]
    },
)
