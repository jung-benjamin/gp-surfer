#/ /usr/bin/env python3

"""Gaussian process regression model classes

Classes for building a gpr model and optimizing the
hyperparameters.
"""

import numpy as np
from scipy import linalg
from scipy.optimize import fmin_l_bfgs_b

class GaussianProcessRegression():
    """A gaussian process regression model
    
    The model contains methods for computing the
    log likelihood and the posterior predictive function
    as well as for optimizing the hyperparameters.
    """
    
    def __init__(self, x_data, y_data, kernel):
        """Initialize the regression model with a kernel
        
        Parameters
        ----------
        x_data
            array (n x d), training data input values
        y_data
            array (n x 1), training data output values
        kernel
            kernel object
        """
        self.kernel = kernel
        self.x_train = np.array(x_data)
        self.y_train = np.array(y_data)
        self.x_test = None
        self.y_test = None
        
    def split_data(self, idx_train):
        """Split the data into training and test set
        
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
        x_train = self.x_train[idx_train[0]:idx_train[1], :]
        y_train = self.y_train[idx_train[0]:idx_train[1]]
        x_test = np.concatenate([self.x_train[:idx_train[0],:], self.x_train[idx_train[1]:,:]], axis = 0)
        y_test = np.concatenate([self.y_train[:idx_train[0]], self.y_train[idx_train[1]:]], axis = 0)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
    def posterior_predictive(self, x_test):
        '''Compute statistics of the posterior predictive distribution

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
        '''
        K = self.kernel(self.x_train, self.x_train, grad=False)
        K_s =  self.kernel(self.x_train, x_test, grad=False)

        L_ = linalg.cholesky(K, lower=True)
        try:
            alpha_ = linalg.cho_solve((L_,True), self.y_train)
        except ValueError:
            return 0, 0
        mu_s = np.dot(K_s, alpha_)

        """Needs to be implemented better"""
        # This part can be commented out if a calculation of the posterior variance
        # is not desired or needed. It doubles the calculation time

        #K_ss = kernel(X_s, X_s, Type,params[:-1]) + params[-1] * np.ones(len(X_s))
        #L_inv = linalg.solve_triangular(L_.T,np.eye(L_.shape[0]))
        #K_inv = L_inv.dot(L_inv.T)
        #cov_s = K_ss - K_s.dot(K_inv).dot(K_s)

        cov_s = 0
        return mu_s[0], cov_s

    def log_marginal_likelihood(self, theta, split=(None, None)):
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
        x_train = self.x_train#[split[0]:split[1], :]
        y_train = self.y_train#[split[0]:split[1]]
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
        
    def optimize(self, n_steps=1):
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
        ????
        """
        n_params = len(self.kernel.parameters)
        
        def obj_func(theta):
            """Objective function for the optimizer"""
            return self.log_marginal_likelihood(theta)
        
        opt_results = []
        for i in range(n_steps):
            res = fmin_l_bfgs_b(obj_func,
                                np.random.random(n_params),
                                fprime = None,
                                bounds = self.kernel.bounds,
                               )
            opt_results.append([res[0],res[1]])
        
        opt_results = np.array(opt_results)
        min_idx = np.argmin(opt_results[:, 1])

        print('Loglikelihood = {} \n'.format(np.exp(opt_results[min_idx, 1])))
        print('Hyperparameters = \n {} \n'.format(opt_results[min_idx, 0]))

        optimized_params = opt_results[min_idx, 0]
        self.kernel.parameters = optimized_params
        
    def compute_alpha(self):
        """Compute the alpha of the kernel with the training data"""
        K = self.kernel(self.x_train, self.x_train, grad = False)
        L_ = linalg.cholesky(K, lower=True)
        try:
            alpha_ = linalg.cho_solve((L_,True), self.y_train)
        except ValueError:
            return 0
        return alpha_

    def compute_kernel_inverse(self):
        """Compute inverse of the kernel matrix with the training data"""
        K = self.kernel(self.x_train, self.x_train, grad = False)
        L_ = linalg.cholesky(K, lower=True)
        L_inv = linalg.solve_triangular(L_.T, np.eye(L_.shape[0]))
        K_inv = L_inv.dot(L_inv.T)
        return K_inv
