#/ /usr/bin/env python3

"""Gaussian process regression model classes

Classes for building a gpr model and optimizing the
hyperparameters.
"""

class GaussianProcessRegression():
    """A gaussian process regression model
    
    The model contains methods for computing the
    log likelihood and the posterior predictive function
    as well as for optimizing the hyperparameters.
    """
    
    def __init__(kernel):
        """Initialize the regression model with a kernel"""
        self.kernel = kernel
        
    def posterior_predictive(x_test, x_train, y_train):
    '''  
    Compute statistics of the posterior predictive distribution
    
    Compute the conditional distribution of a subset of elements,
    conditional on the training data.
    
    Parameters
    ----------
    x_test
        array like, test data for which the distribution is 
        evaluated
    x_train
        array like (m x d), training locations
    y_train
        array like (m x 1), training values

    Output
    ------
        mean
            posterior mean vector (n x d)
        cov
            covariance matrix (n x n).
    '''
    k = self.kernel(x_train, x_train, grad=False)
    k_test =  self.kernel(x_train, x_test, grad=False)
    

    L_ = linalg.cholesky(K, lower=True)
    try:
        alpha_ = linalg.cho_solve((L_,True), y_train)
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

    def log_marginal_likelihood(x_train, y_train, theta):
        """Compute the negative log marginal likelihood
        
        The negative log marginal likelihood is computed for 
        the training data with respect to the hyperparameters.
        
        Parameters
        ----------
        x_train
            array like (m x d), training locations
        y_train
            array like (m x 1), values at training locations
        theta
            array like (1 x n_params), hyperparameters
            
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
        K, dK = kernel(X_train, X_train, Type,theta,gradient=True) 
        try:
            L = linalg.cholesky(K,lower=True)
            L_inv = linalg.solve_triangular(L.T,np.eye(L.shape[0]))
            K_inv = L_inv.dot(L_inv.T)
            
            alpha = linalg.cho_solve((L, True), Y_train)
            
            nll = np.sum(np.log(np.diagonal(L))) +\
                0.5 * np.dot(Y_train.T,alpha) +\
                    0.5 * len(X_train) * np.log(2*np.pi)
            
            Tr_arg = alpha.dot(alpha.T) - K_inv
            
            dnll = [-0.5* np.trace(Tr_arg.dot(dK[i])) \
                             for i in range(theta.shape[0])]
            dnll = np.array(dnll)

            return nll, dnll
        
        except (np.linalg.LinAlgError, ValueError):
            # In case K is not positive semidefinite
            return np.inf,np.array([np.inf for i in range(theta.shape[0])])