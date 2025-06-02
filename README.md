[![DOI](https://zenodo.org/badge/994780944.svg)](https://doi.org/10.5281/zenodo.15576628)

## Gaussianprocesses
A package for gaussian process regression in nuclear archaeology

The package ~~uses the Theano backend~~ uses numpy and scipy.
A `theano` backend may be added later to be more easily compatible with pymc3 to facilitate bayesian inference methods.

Kernel instances can be added and multiplied to create a new kernel instance.
When doing this, the kernel functions and the kernel gradients are modified and the hyperparameter lists are concatenated.

Currently only three kernels are implemented: the _squared exponential_, the _anisotropic squared exponential_ and an _"anisotropic" linear_ kernel.

To implement your own kernels simply create a subclass of the kernel base class. 
Then define the methods:
```python
def kernel_function(self, x1, x2)
```
and
```python
def kernel_gradient(self, x1, x2)
```

The `GaussianProcessRegression` class handles the hyperparameter optimization and can load and store prevously trained kernels.

## Installing
Go into the directory containing this source code and run `pip install .`

## Example
The example can be executed with 
```bash
train_gp_surrogates example -x input_parameters_1000.csv -y output_values_1000.csv -i id_file_example.json
```

