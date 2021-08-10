A package for gaussian process regression in nuclear archaeology

The package uses the Theano backend, with the intent to be easily
compatible with pymc3 to facilitate bayesian inference methods.

Kernel instances can be added and multiplied to create a new kernel
instance. When doing this, the kernel functions and the kernel gradients 
are modified and the hyperparameter lists are concatenated.

Currently only three kernels are implemented: the squared exponential,
the anisotropic squared exponential and an "anisotropic" linear kernel.

To implement your own kernels simply create a subclass of the kernel
base class. Then define the methods
```
def kernel_function(self, x1, x2)
```
and
```
def kernel_gradient(self, x1, x2)
```
