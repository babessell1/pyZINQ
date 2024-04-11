
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
from numba import jit, njit
from scipy.stats import chi2, norm


@njit
def firth_likelihood(loglike_val, H):
    return -(loglike_val + .5*np.log(np.linalg.det(-H)))


@njit
def fit_firth(betas, hessian, y, X):
    pi = 1 / (  1 + np.exp( -(X @ betas) )  )
    W = np.diag(pi * (1 - pi))
    cov = np.linalg.pinv(-hessian)

    # build hat matrix
    rootW = np.sqrt(W)
    H = X.T @ rootW.T
    H = cov @ H
    H = (rootW @ X) @ H

    # penalised score
                        # h_i
    U = X.T @ (y - pi + np.diag(H) * (.5 - pi))
    new_betas = betas + cov @ U
        
    return new_betas


def firth_logistic_regression(y : np.array,
                              X : np.ndarray,
                              max_iter : int = 1000,
                              tol : float = 1e-6,
                              test : str = 'lrt',
                              ) -> tuple[np.array, np.array, float]:
    """
    Perform Firth logistic regression.

    beta_hat = argmin | sum(y_i  - pi_i + h_i(1/2 - pi_i))x_{ir} |
        i -> N, r -> p = len(beta)

    Where: 
        h_i = ith diagonal element of:
            W^(1/2)X(X'WX)^(-1)X'W^(1/2)

        |I(beta)|^(1/2) = (X'WX)^(1/2) = Fisher information matrix

        pi_i = 1 / (1 + exp(-x_i*beta)) = probability of success
        W = diag(pi_i(1-pi_i)) = weight matrix
        X = design matrix (n x p) 
    """
    X = sm.add_constant(X)
    logit_model = sm.Logit(y, X)

    # fit null model
    null_model = sm.Logit(y, np.ones((len(y), 1)))
    null_result = null_model.fit(disp=0)
    start_vec = np.zeros(X.shape[1])
    start_vec[0] = null_result.params[0]
    betas = start_vec

    H = logit_model.hessian(betas)
    ll = logit_model.loglike(betas)

    conv = False
    #TODO: figure out how to calculate the hessian so I can throw this in a jit
    for i in range(max_iter):
        new_betas = fit_firth(betas, H, y, X)
        ll_next = logit_model.loglike(betas)
        H_next = logit_model.hessian(betas)

        while firth_likelihood(ll_next, H_next) < firth_likelihood(ll, H):
            new_betas = betas + 0.5 * (new_betas - betas)
            ll, H = ll_next, H_next
            ll_next = logit_model.loglike(new_betas)
            H_next = logit_model.hessian(new_betas)
            conv = (np.linalg.norm(new_betas - betas) < tol)
            if conv: break
        
        betas = new_betas
        if conv: break

    if new_betas is None:
        sys.stderr.write('Firth regression failed\n')
        return None

    # Calculate stats
    fitll = ll_next

    # add small value to the hessian to ensure it is invertible
    H_next += np.eye(H_next.shape[0]) * 1e-6
    bse = np.sqrt(np.diag(np.linalg.pinv(-H_next)))

    if test == 'null_model':
        return None, None, fitll, None, None
    
    elif test == 'wald':
        # Wald test for each coefficient
        stats = abs(betas / bse)
        pvals = 2 * ( 1 - norm.cdf(stats) )

    elif test == 'lrt':
        stats, pvals = [], [] # store chi2 stats and pvals
        # Likelihood ratio test
        for i, bse in enumerate(bse):
            null_X = np.delete(X, i, axis=1)

            _, _, null_fitll, _, _ = \
                firth_logistic_regression(
                    y, null_X, max_iter, tol, test='null_model')
            
            lr = -2 * (null_fitll - fitll)
            stats.append(lr)     
            lr_pval = 1 if lr < 0 else chi2.sf(lr, 1)
            pvals.append(lr_pval)


    return betas, bse, fitll, stats, pvals