#! /usr/bin/env python3

"""Gaussian process regression model classes

Classes for building a gpr model and optimizing the
hyperparameters.
"""

import json
import numpy as np
import warnings
import matplotlib.pyplot as plt

from json import JSONEncoder, JSONDecoder
from scipy import linalg
from scipy.optimize import fmin_l_bfgs_b

import gaussianprocesses.metrics as metrics
import gaussianprocesses.transformations as tr
import gaussianprocesses.kernels as kernels
from gaussianprocesses.dataclass import ModelData


class NumpyArrayEncoder(JSONEncoder):
    """JSONEncoder that supports numpy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class GaussianProcessRegression():
    """A gaussian process regression model

    The model contains methods for computing the
    log likelihood and the posterior predictive function
    as well as for optimizing the hyperparameters.
    """

    def __init__(self, kernel, **kwargs):
        """Initialize the regression model with a kernel

        Parameters
        ----------
        kernel
            kernel object
        kwargs
            Keyword arguments for specifying the training,
            testing and validation data, as well as the type
            of transformation. The keyword 'transformation'
            takes precedent over 'x_trafo' and 'y_trafo'.
        """
        arguments = {'x_train' : None,
                     'y_train' : None,
                     'x_test' : None,
                     'y_test' : None,
                     'x_validate' : None,
                     'y_validate' : None,
                     'transformation' : None,
                     'x_trafo' : None,
                     'y_trafo' : None,
                     }
        arguments.update(kwargs)
        self.kernel = kernel
        self.data = ModelData(**arguments)
        if arguments['transformation'] is not None:
            self.transformation = arguments['transformation']
        else:
            self.transformation = arguments

    ## This only works for kernels that are implemented as an object
    ## Kernels that are combinations of kernels cannot be loaded
    @classmethod
    def from_dict(cls, d):
        """Create instance of the model by loading a dictionary

        Parameters
        ----------
        d : dict
            Dictionary containing parameters to initialize the kernel.

        Returns
        -------
        model : GaussianProcessRegression object
            Instance of the GaussianProcessRegression class.
        """
        model = GaussianProcessRegression(**d)
        model.kernel.parameters = d['Params']
        model.kernel_inv = d['K_inv']
        model.alpha_ = d['alpha_']
        model.LAMBDA = d['LAMBDA']
        return model

    @classmethod
    def from_json(cls, j):
        """Create instance of the model by loading a json file

        Parameters
        ----------
        j : str, path-like
            File path of the json file.

        Returns
        -------
        model : GaussianProcessRegression object
            Instance of the GaussianProcessRegression class.
        """
        with open(j, 'r') as f:
            jdict = json.load(f)
        d = {}
        for n, it in jdict.items():
            if n == 'x_trafo' or n == 'y_trafo':
                d[n] = (it[0], np.array(it[1]))
            elif isinstance(it, list):
                d[n] = np.array(it)
            else:
                d[n] = it
        model = cls.from_dict(d)
        return model

    @property
    def kernel(self):
        """Return the kernel instance"""
        return self._kernel

    @kernel.setter
    def kernel(self, k):
        """Set a kernel for the object"""
        if isinstance(k, kernels.Kernel):
            self._kernel = k
        elif isinstance(k, str):
            self._kernel = getattr(kernels, k)()
        else:
            raise Exception('Kernel must be string or kernel object!')

    @property
    def transformation(self):
        """Return the transformation classes"""
        trafo_dict = {'x_trafo' : self._x_transformation,
                      'y_trafo' : self._y_transformation,
                      }
        return trafo_dict

    @transformation.setter
    def transformation(self, t):
        """Set the transformation classes"""
        if isinstance(t, dict):
            self.x_transformation = t['x_trafo']
            self.y_transformation = t['y_trafo']
        elif t is None:
            self._x_transformation = None
            self._y_transformation = None
        else:
            raise Exception('Transformation must be a dict or None!')

    @property
    def x_transformation(self):
        """Return only the transformation for x data"""
        return self._x_transformation

    @x_transformation.setter
    def x_transformation(self, xt):
        """Set only the x data transformation"""
        if isinstance(xt, tr.Transformation):
            self._x_transformation = xt
        elif isinstance(xt, str):
            self._x_transformation = getattr(tr, xt)()
        elif isinstance(xt, tuple) or isinstance(xt, list):
            self._x_transformation = getattr(tr, xt[0])()
            self._x_transformation.transformation_parameters = np.array(xt[1])
        else:
            self._x_transformation = None

    @property
    def y_transformation(self):
        """Return only the transformation for y data"""
        return self._y_transformation

    @y_transformation.setter
    def y_transformation(self, yt):
        """Set only the y data transformation"""
        if isinstance(yt, tr.Transformation):
            self._y_transformation = yt
        elif isinstance(yt, str):
            self._y_transformation = getattr(tr, yt)()
        elif isinstance(yt, tuple) or isinstance(yt, list):
            self._y_transformation = getattr(tr, yt[0])()
            self._y_transformation.transformation_parameters = np.array(yt[1])
        else:
            self._y_transformation = None

    def transform_x(self, x):
        """Apply the transformation to x data"""
        trafo = self._x_transformation
        if trafo is None:
            return x
        else:
            return trafo.transform(x)

    def transform_y(self, y):
        """Apply the transformation to the y data"""
        trafo = self._y_transformation
        if trafo is None:
            return y
        else:
            return trafo.transform(y)

    def untransform_x(self, x):
        """Revert the transformation of the x data"""
        trafo = self._x_transformation
        if trafo is None:
            return x
        else:
            return trafo.untransform(x)

    def untransform_y(self, y):
        """Revert the transformation of the y data"""
        trafo = self._y_transformation
        if trafo is None:
            return y
        else:
            return trafo.untransform(y)

    def split_data(self, idx_train=None, idx_val=None):
        """Split the data into different sets

        Concatenates all three data categories into one
        array and redistributes the data to the categories
        according to the indices.
        If idx_train and idx_val are None, the data is split
        into three equally long parts. The indices specified
        for slicing should not cover overlapping intervals.

        Parameters
        ----------
        idx_train
            tuple int, start and stop indices to indicate the
            training set
        """
        ## different options to split data could be added
        ## also consider using some sort of validation during
        ## optimization?
        ## or kfold cross-validation
        conc = self.data.concatenate()
        if idx_train is None and idx_val is None:
            split_x = np.array_split(conc.x, 3)
            split_y = np.array_split(conc.y, 3)
        elif idx_val is None:
            idx1 = np.arange(*idx_train)
            split_x = [conc.x[idx1,:]]
            split_y = [conc.y[idx1]]
            split_x += [np.delete(conc.x, idx1, axis=0)]
            split_y += [np.delete(conc.y, idx1, axis=0)]
            split_x += [None]
            split_y += [None]
        elif idx_train is None:
            idx2 = np.arange(*idx_val)
            split_x = [None]
            split_y = [None]
            split_x += [np.delete(conc.x, idx2, axis=0)]
            split_y += [np.delete(conc.y, idx2, axis=0)]
            split_x += [conc.x[idx2,:]]
            split_y += [conc.y[idx2]]
        else:
            idx1 = np.arange(*idx_train)
            idx2 = np.arange(*idx_val)
            idx3 = np.concatenate([idx1, idx2])
            split_x = [conc.x[idx1,:]]
            split_y = [conc.y[idx1]]
            split_x += [np.delete(conc.x, idx3, axis=0)]
            split_y += [np.delete(conc.y, idx3, axis=0)]
            split_x += [conc.x[idx2,:]]
            split_y += [conc.y[idx2]]
        self.data.train = {'x_train' : split_x[0], 'y_train' : split_y[0]}
        self.data.test = {'x_test' : split_x[1], 'y_test' : split_y[1]}
        self.data.validate = {'x_validate' : split_x[2], 'y_validate' : split_y[2]}

    def posterior_predictive(self, x_test, cov=False):
        """Compute statistics of the posterior predictive distribution

        Compute the conditional distribution of a subset of elements,
        conditional on the training data.

        Parameters
        ----------
        x_test
            array like, test data for which the distribution is
            evaluated

        Output
        ------
        mean
            posterior mean vector (n x d)
        cov
            covariance matrix (n x n).
        """
        xt = self.transform_x(self.data.train.x)
        xtest = self.transform_x(x_test)
        yt = self.transform_y(self.data.train.y)
        K = self.kernel(xt, xt, grad=False)
        K_s =  self.kernel(xt, xtest, grad=False)
        L_ = linalg.cholesky(K, lower=True)
        ## This try an except may cause errors to go unnoticed.
        try:
            alpha_ = linalg.cho_solve((L_,True), yt)
        except ValueError:
            if cov:
                return 0, 0
            return 0
        mu_s = np.dot(K_s, alpha_)

        """Needs to be implemented better"""
        if cov:
            # This part can be commented out if a calculation of the posterior variance
            # is not desired or needed. It doubles the calculation time
            #K_ss = kernel(X_s, X_s, Type,params[:-1]) + params[-1] * np.ones(len(X_s))
            #L_inv = linalg.solve_triangular(L_.T,np.eye(L_.shape[0]))
            #K_inv = L_inv.dot(L_inv.T)
            #cov_s = K_ss - K_s.dot(K_inv).dot(K_s)
            raise NotImplementedError
            #cov_s = 0
            #return mu_s[0], cov_s
        else:
            return self.untransform_y(mu_s[0])

    def log_marginal_likelihood(self, theta,):
        """Compute the negative log marginal likelihood

        The negative log marginal likelihood is computed for
        the training data with respect to the hyperparameters.

        Parameters
        ----------
        theta
            array like (1 x n_params), hyperparameters
        split
            tuple of int, split data into training and
            test set

        Output
        ------
        nll:
            float, negative loglikelihood of model.
        dnll:
            array of floats (1xn_params),
            gradient of the negative loglikehood w.r.t hyperparameters
        """
        # Numerical Implementation of Eq. (7) as described
        # in http://www.gaussianprocess.org/gpm/chapters/RW2.pdf, Section
        # 2.2, Algorithm 2.1.
        self.kernel.parameters = theta
        x_train = self.transform_x(self.data.train.x)
        y_train = self.transform_y(self.data.train.y)
        K, dK = self.kernel(x_train, x_train, grad = True)
        try:
            L = linalg.cholesky(K, lower=True)
            L_inv = linalg.solve_triangular(L.T,np.eye(L.shape[0]))
            K_inv = L_inv.dot(L_inv.T)

            alpha = linalg.cho_solve((L, True), y_train)

            nll = (np.sum(np.log(np.diagonal(L)))
                   + 0.5 * np.dot(y_train.T, alpha)
                   + 0.5 * len(x_train) * np.log(2*np.pi)
                  )

            Tr_arg = alpha.dot(alpha.T) - K_inv

            dnll = [-0.5 * np.trace(Tr_arg.dot(dK[i]))
                    for i in range(theta.shape[0])
                   ]
            dnll = np.array(dnll)

            return nll, dnll

        except (np.linalg.LinAlgError, ValueError):
            # In case K is not positive semidefinite
            return np.inf,np.array([np.inf for i in range(theta.shape[0])])

    def optimize(self, n_steps=1, first_params=None, seed=2021, verbose=False):
        """Optimize the hyperparameters of the kernel

        Optimize the kernel hyperparameters and optionally
        restart the optimization with random starting
        values and return the best result. For optimal
        performance the data should probably be normalized
        somehow.

        Parameters
        ----------
        n_steps
            int, number of restarts of the optimizer
        split
            tuple of int, split data into training and
            test set

        Output
        ------
        optimized_params
            list, optimized hyperparameters
        """
        n_params = len(self.kernel.parameters)

        def obj_func(theta):
            """Objective function for the optimizer"""
            return self.log_marginal_likelihood(theta)

        opt_position = np.zeros((n_steps, n_params))
        opt_value = np.zeros(n_steps)

        rng = np.random.default_rng(seed)
        start_params = rng.random(size=(n_steps, n_params))
        if first_params is not None:
            start_params[0,:] = first_params

        for i, p in enumerate(start_params):
            res = fmin_l_bfgs_b(obj_func,
                                p,
                                fprime = None,
                                bounds = self.kernel.bounds,
                               )
            opt_position[i] = res[0]
            opt_value[i] = res[1]

        min_idx = opt_value.argmin()
        if verbose:
            print('Loglikelihood = {} \n'.format(np.exp(opt_value[min_idx])))
            print('Hyperparameters = \n {} \n'.format(opt_position[min_idx]))

        optimized_params = opt_position[min_idx]
        self.kernel.parameters = optimized_params

    def optimize_metric(self, target, metric='r_squared', n_steps=1000,
                        seed=2021, maxiter=100, full_output=False):
        """Repeat hyperparameter optimization to improve metric

        Repeatedly optimizes the hyperparameters and computes the
        evaluation metric for the validation data. Selects the
        optimizer result with the best evaluation.

        Parameters
        ----------
        target : float
            If the metric is larger than target, the repetition stops.
        metric : string
            Identifies the evaluation metric from the metrics module.
        n_steps : int
            Number of steps for the optimizer method.
        seed : int
            Starting seed of the optimizer method. Is increased by
            one for each repetition of the optimizer in this method.
        maxiter : int
            Maximum number of repetitions if the target is not met.
        full_output : bool
            If True, returns more information about the optimization
            process.

        Returns
        -------
        metric : float
            Value of the metric for the best result of the optimizer.
        parameters : float
            Model hyperparameters that produced the metric with the
            validation data.
        info : dict (optional)
            Information dictionary.
        """
        n_params = len(self.kernel.parameters)
        parameters = np.zeros((maxiter, n_params))
        metrics = np.zeros(maxiter)
        for i in range(maxiter):
            self.optimize(n_steps=n_steps, seed=seed)
            parameters[i] = self.kernel.parameters
            m = self.evaluate_predictions('validate', metric=metric)
            metrics[i] = m
            if m > target:
                break
            seed += 1
        best = metrics.argmax()
        self.kernel.parameters = parameters[best]
        if full_output:
            info = {'niter' : i+1}
            return metrics[best], parameters[best], info
        return metrics[best], parameters[best]

    def compute_alpha(self):
        """Compute the alpha of the kernel with the training data"""
        xt = self.transform_x(self.data.train.x)
        yt = self.transform_y(self.data.train.y)
        K = self.kernel(xt, xt, grad=False)
        L_ = linalg.cholesky(K, lower=True)
        try:
            alpha_ = linalg.cho_solve((L_,True), yt)
        except ValueError:
            return 0
        return alpha_

    def compute_kernel_inverse(self):
        """Compute inverse of the kernel matrix with the training data"""
        xt = self.transform_x(self.data.train.x)
        K = self.kernel(xt, xt, grad=False)
        L_ = linalg.cholesky(K, lower=True)
        L_inv = linalg.solve_triangular(L_.T, np.eye(L_.shape[0]))
        K_inv = L_inv.dot(L_inv.T)
        return K_inv

    def predictions(self, x='test', cov=False):
        """Calculate model predictions for a set of points

        Evaluates the posterior predictive for each point
        in the given data set. Optionally, the testing or
        validation data can be used.
        Each datapoint is considered individually to evaluate
        the y value given the training data. Considering all
        datapoints only makes sense if the correlation between
        the new datapoints is considered as well, which is
        not implemented here.

        Parameters
        ----------
        x : str or np.ndarray(float), default = 'test'
            If x is a string, it specifies either the testing
            ('test') or validation ('validate') data set of
            the model. Otherwise it needs to be an array of
            dimension (n_points, n_params).
        cov : bool
            If true, the posterior variance is also calculated.

        Returns
        -------
        predictions : np.ndarray(float)
            The model prediction for each datapoint.
        variance : np.ndarray(float), if cov is True
            The posterior variance of each prediction.
        """
        if isinstance(x, str):
            data = getattr(self.data, x).x
        else:
            data = x
        data = np.expand_dims(data, axis=1)
        predictions = np.zeros(data.shape[0])
        if cov:
            raise NotImplementedError
        else:
            for i, d in enumerate(data):
                predictions[i] = self.posterior_predictive(d)
            return predictions

    def evaluate_predictions(self, x='test', y=None, metric='r_squared'):
        """Compare model predictions to the data

        Use some metric to evaluate the deviation of
        the model predictions to a set of datapoints.

        Parameters
        ----------
        x : str or np.ndarray(float), default = 'test'
            If x is a string, it specifies either the testing
            ('test') or validation ('validate') data set of
            the model. Otherwise it needs to be an array of
            dimension (n_points, n_params).
        y : str or np.ndarray(float), default = None
            Data, against which the model predictions are compared.
            If y is None, the y data in the Data object specified by x
            is used. If y is a string, it specifies either the testing or
            validation data. Otherwise, the points in y should correspond
            to the points in x.
        metric : str
            Specifies a metric from the metrics module.

        Returns
        -------
        performance : float
            The value of the performance metric.
        """
        prediction = self.predictions(x=x)
        if y is None:
            compare = getattr(self.data, x).y
        elif isinstance(y, str):
            compare = getattr(self.data, y).y
            if y != x:
                warnings.warn('X and Y data do not match.')
        if isinstance(metric, list):
            performance = {}
            for m in metric:
                try:
                    test_func = getattr(metrics, m)
                except AttributeError:
                    continue
                performance[m] = test_func(prediction, compare)
        else:
            test_func = getattr(metrics, metric)
            performance = test_func(prediction, compare)
        return performance

    def plot_predictions(self, x='test', y=None, plot_compare=True, **pltkwargs):
        """Plot the model predictions"""
        prediction = self.predictions(x=x)
        if y is None:
            compare = getattr(self.data, x).y
        elif isinstance(y, str):
            compare = getattr(self.data, y).y
            if y != x:
                warnings.warn('X and Y data do not match.')
        if isinstance(x, str):
            xdata = getattr(self.data, x).x
        else:
            xdata = x

        fig, axes = plt.subplots(1, xdata.shape[1], **pltkwargs)
        for i, ax in enumerate(axes):
            ax.scatter(xdata[:,i], prediction, label='Prediction')
            if plot_compare:
                ax.scatter(xdata[:,i], compare, label='True')
        plt.show()

    def store_kernel(self, path, how='npy', exact_copy=False):
        """Store kernel with precomputed matrices

        Stores precomputed matrices for the training data
        and the optimized hyperparameters of the kernel.
        This allows the kernel to be loaded with low
        computational effort.

        Parameters
        ----------
        path : str
            Intended file location.
        how : str
            Determines the dataformat for the stored information.
        exact_copy : bool (optional, default is False)
            If set to True, the test and validation data is also
            stored.
        """
        params = self.kernel.parameters
        try:
            lam = self.kernel.create_lambda(self.data.train.x)
        except AttributeError:
            warnings.warn("Kernel has no lambda.")
            lam = None
        alpha = self.compute_alpha()
        inv = self.compute_kernel_inverse()
        if how == 'npy':
            d = {'Params': params,
                 'LAMBDA': lam,
                 'alpha_': alpha,
                 'K_inv': inv,
                 'x_train': self.data.train.x,
                 'y_train': self.data.train.y,
                 'y_trafo': (self.y_transformation.__class__.__name__,
                             self.y_transformation.transformation_parameters),
                 'x_trafo': (self.x_transformation.__class__.__name__,
                             self.x_transformation.transformation_parameters),
                 'kernel': self.kernel.__class__.__name__,
                 }
            if exact_copy:
                d['x_test'] = self.data.test.x
                d['y_test'] = self.data.test.y
                d['x_validate'] = self.data.validate.x
                d['y_validate'] = self.data.validate.y
            np.save(path, d)
        elif how == 'json':
            d = {'Params': params,
                 'LAMBDA': lam,
                 'alpha_': alpha,
                 'K_inv': inv,
                 'x_train': self.data.train.x,
                 'y_train': self.data.train.y,
                 'y_trafo': (self.y_transformation.__class__.__name__,
                             self.y_transformation.transformation_parameters),
                 'x_trafo': (self.x_transformation.__class__.__name__,
                             self.x_transformation.transformation_parameters),
                 'kernel': self.kernel.__class__.__name__,
                 }
            if exact_copy:
                d['x_test'] = self.data.test.x
                d['y_test'] = self.data.test.y
                d['x_validate'] = self.data.validate.x
                d['y_validate'] = self.data.validate.y
            with open(path, 'w') as sp:
                json.dump(d, sp, cls=NumpyArrayEncoder)


class GPRQuotient():
    """Calculate the quotient of two GPR model predictions

    This composite class of two GaussianProcessRegression classes
    evaluates the predictions of two trained models an calcualtes
    the quotient.
    """

    def __init__(self, mi, mj):
        """Set the composite classes

        Parameters
        ----------
        mi, mj : GaussianProcessRegression
            GPR models that comprise the quotient.
        """
        self.mi = mi
        self.mj = mj

    @classmethod
    def from_json(cls, fi, fj):
        """Instantiate the class from json files

        Parameters
        ----------
        f1, f2 : str, path-like
            Filepaths to files that contain all parameters of
            the two comprising classes.

        Returns
        -------
        quotient : GPRQuotient
            The composite class.
        """
        mi = GaussianProcessRegression.from_json(fi)
        mj = GaussianProcessRegression.from_json(fj)
        quotient = cls(mi, mj)
        return quotient

    def predictions(self, x, inv=False):
        """Calculate the quotient of the predictions

        Parameters
        ----------
        x : array of floats
            Data for which the predictions are calculated.
            Both comprising models are evaluated at the same
            data points.
        inf : bool, optional (default is False)
            If bool is True, then the inverse of the quotien is
            returned, i.e. mj / mi, instead of mi / mj.

        Returns
        pred : array of floats
            Quotient of the predictions of the comprising classes.
        """
        pi = self.mi.predictions(x=x)
        pj = self.mj.predictions(x=x)
        quotient = pi / pj
        if inv:
            return 1 / quotient
        return quotient
