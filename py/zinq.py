import sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, pearsonr, sf, norm
from scipy import optimize
from numba import jit, njit
# quantile regression
import statsmodels.api as sm
from statsmodels.formula.api import smf
from firth import firth_logistic_regression
from helpers import *

class ZINQ:
    """
    Zero-inflated quantile test with ensembling for generic data.

    Parameters:
    -----------
    data_matrix : tuple[pd.DataFrame]
        tuple of n x 2 dataframes (n samples x sample name + feature name)
        corresponding to different algorithms or methods producing similar 
        for the same samples. Missing values for features are allowed. (np.nan)
    
    metadata : pd.DataFrame
        n x c dataframe (n samples x sample name + covariate name)
        with covariates and the variable to test. Missing data 
        here is NOT allowed.

    data_names : list[str]
        names of the data sources for key-access to results.

    test_variable : str
        variable to test in the metadata dataframe.

    covariates2include : list[str]
        list of covariates to correct for. Default is ["all"].
        Can also include "libsize" to use library size as a covariate

    quantile_levels : list[float]
        quantile levels to regress on. Default is [0.1, 0.25, 0.5, 0.75, 0.9]
    
    method : str
        method to combine p-values. Default is "MinP".
        Options are "MinP" and "Cauchy".

    seed : int
        seed for random number generation. Default is 37.

        
    Properties:
    -----------
    dnames : list[str]
        names of the data sources for key-access to results.

    data : dict[str, pd.DataFrame]
        dictionary of dataframes corresponding to different algorithms
    
    meta : dict[str, pd.DataFrame]
        dictionary of metadata dataframes corresponding to different algorithms

    test_var : str
        variable to test in the metadata dataframe.

    quantiles : list[float]
        quantile levels to regress on.

    method : str
        method to combine p-values.

    seed : int

    binary : bool
        True if the test variable is binary, False otherwise.
    
    warning_codes : dict[str, list[int]]
        warning codes for each feature.
        1: When library size is a confounder.
        2: When all read counts are zero.
        3: When there are limited non-zero read counts (<30 or <15).
        4: When there is a perfect separation w.r.t. the variable(s) of interest.
    
    z_pvalues : dict[str, dict[str, float]]
        p-values for firth logistic regression.

    q_pvalues : dict[str, dict[str, float]]
        p-values from quantile regression.
    
    combined_pvalues : dict[str, dict[str, float]]
        combined p-values.

    Public Methods:
    ---------------
    run_zinq()
        Run Entire ZINQ pipeline and return dataframes with p-values
        ...
        returns: resulting p values for each data source in the same
                 it was provided in the constructor
        -> tuple(pd.DataFrame(columns=[
            "Firth_Logistic", "Quantile", "Combined"], index=data_names))

    run_sanity_check()
        Run sanity check before applying ZINQ.
        ...
        returns: warning_codes 
        -> dict[str, list[int]]

    run_marginal_tests()
        Run marginal tests for the Firth logistic and quantile regression
        components for each data source.
        ...
        returns: z_pvalues, q_pvalues
        -> tuple(dict[dict[str, float]], dict[dict[str, float]])
    
    run_test_combination()
        Combine the marginal p-values for each data source.
        ...
        returns: combined_pvalues
        -> dict[dict[str, float]]
    
    Private Methods:
    ---------------
    _check(dname: str) -> list[int]
        Sanity check a single data source.

    ZINQ
    """
    def __init__(self, 
                 data_matrix : tuple[pd.DataFrame], # list of dataframes correpsonding to different algorithms prodcucing similar data [n x 2, n x 2, ...] 1 is response, 2 is variable of interest
                 metadata : pd.DataFrame, # list of dataframes correpsonding to different algorithms prodcucing similar data [n x p, n x p, ...]
                 data_names : list[str], # names of the data sources
                 response : str, # response variable
                 covariates2include : list = ["all"], # list of covariates to correct for
                 quantile_levels : list = [0.1, 0.25, 0.5, 0.75, 0.9], # quantile levels to regress on
                 method : str = "MinP", # method to combine p-values
                 count_data : bool = False, # count data need to perform dithering
                 seed : int = 2020): # seed for ditheirng
    
        self.covars = metadata.columns if covariates2include == ["all"] else covariates2include
        self.dnames = data_names
        self.quantiles = quantile_levels
        self.method = method
        self.seed = seed
        self.binary = True if len(np.unique(self.meta[self.test_var])) == 2 else False

        self.Z = metadata[self.covars].to_numpy() # covariates
        self.C = metadata[response].to_numpy() # response variable
        
        # add ensemble key for ensembled data sources
        # in combination test
        data_names.append("ensemble" if len(data_names) > 1 else None)

        self.warning_codes = self.z_pvalues = self.q_pvalues = self.quant = self.Y = self.weights = {}
        for dname in self.dnames:
            #self.X[dname] = np.hstack((self.data[dname], _ZC)) # design matrix
            self.warning_codes[dname] = [] 
            self.z_pvalues[dname] = -1 # firth logistic regression p-values
            self.q_pvalues[dname] = -1 # quantile regression p-values
            self.Y[dname] = data_matrix[0].iloc[:, 1].to_numpy() # count data
            self.weights[dname] = 1 # TODO: implement weights


    def run_zinq(self) -> tuple[pd.DataFrame]:
        """
        Run Entire ZINQ pipeline and return dataframes with p-values
        """
        self.run_sanity_check()
        self.run_marginal_tests()
        return self.run_test_combination()


    def _check_all(self) -> dict[int: list[int]]:
        """
        Sanity check a single data source
        """
        return {dname: self._check(dname) for dname in self.dnames}

    def _check(self, dname) -> list[int]:
        """
        Sanity check a single data source
        """
        return [
            i for i,chk in enumerate([
                self._check_lib_confound(dname),
                self._check_all_zero(dname),
                self._check_limited_non_zero(dname),
                self._check_perfect_separation(dname)
            ]) if chk]


    def _check_lib_confound(self, dname) -> list[list[int]]:
        """
        Check if library size is a confounder.
        """
        lib_size = self.data[dname].sum(axis=1)
        test_vars = self.meta[self.test_var]
        if self.binary:
            responses = test_vars.unique()
            _, pval = ttest_ind(
                lib_size[test_vars.eq(responses[0])],
                lib_size[test_vars.eq(responses[1])]
            )
        else: # quantitative
            _, pval = pearsonr(lib_size, self.meta[self.test_var])
        
        return True if pval < 0.05 else False
    

    def _check_all_zero(self, dname) -> list[int]:
        """
        Check if all read counts are zero.
        """
        return True if self.data[dname].sum(axis=1).eq(0).all() else False
    

    def _check_limited_non_zero(self, dname, thresh=30) -> list[int]:
        """
        Check if there are limited non-zero read counts.
        """
        return True if self.data[dname].sum(axis=1).lt(thresh).all() else False
    

    def _check_perfect_separation(self, dname) -> list[int]:
        """
        Check if there is perfect separation w.r.t. the variable(s) of interest.
        """
        test_var = self.meta[self.test_var]
        uniq = test_var.unique() # unique values of the test variable
        t1idx = np.which(test_var.eq(uniq[0])) # index of the first unique value
        t2idx = np.which(test_var.eq(uniq[1])) # index of the second unique value
        z_idx = np.which(self.data[dname].eq(0)) # index of zero values
        zeros_in_t1 = np.intersect1d(t1idx, z_idx) # zero values in the first unique value
        zeros_in_t2 = np.intersect1d(t2idx, z_idx)
        len_1, len_2 = len(zeros_in_t1), len(zeros_in_t2)

        return True if (len_1 == 0 or len_2 == 0) and (len_1 + len_2) > 0 else False


    def run_sanity_check(self) -> dict[str, list[int]]:
        """
        Run sanity check before applying ZINQ.
        """
        codes = self._check_all()
        code_map = {
            1: "Library size is a confounder.",
            2: "All read counts are zero.",
            3: "Limited non-zero read counts (<30 or <15).",
            4: "Perfect separation w.r.t. the variable(s) of interest."
        }
        for dname in self.dnames: codes[dname] = self._check(dname)
        print(f"""{'\n\n'.join([f'''                  
        Source {dname}, has the following warnings: \n
        {'\n\t'.join([f"{codes[dname].count(cnt)} samples where {code_map[cnt]}" 
        for cnt in codes.keys()])}''' for dname in self.dnames])}""")

        self.warning_codes = codes
        return codes
    
    
    @staticmethod
    def _firth_regress(C : np.array, x : np.ndarray) -> tuple[np.array[float]]: # x 5 
        # betas, bse, fitll, stats, pvals 
        return firth_logistic_regression(C, x)

    
    def run_firth_regression(self, dname):
        """
        Perform Firth logistic regression on a single data source.
        For a single 
        """
        # betas, bse, fitll, stats, pvals 
        return self._firth_regress(self.C, self.Z[dname])

    
    def _get_quantile(self, dname) -> np.ndarray:
        """
        Build quantile matrix for quantile regression.
        """
        # get non-zero indices
        idx_nonzero = np.where(self.Y[dname].ne(0))
        zero_rate = len(idx_nonzero) / len(self.Y[dname])
        yq = dither(self.Y[dname][idx_nonzero], type="right", value=1)

        return yq, zero_rate
        
       
    @staticmethod
    def _rank_score_test(c_star, betas, quantiles, m, width):
        """
        Rank score test for quantile regression.
        """
        # this is ripped straight from the R code
        rs = [np.sum((tau - betas[k]) * c_star) / np.sqrt(m) for k, tau in enumerate(quantiles)]
        if width == 1:
            cov_rs = quantiles * (1 - quantiles)
        else:
            cov_rs = np.zeros((width, width))
            for k in range(width - 1):
                for l in range(k + 1, width):
                    cov_rs[k, l] = min(quantiles[k], quantiles[l]) - quantiles[k] * quantiles[l]
            cov_rs = cov_rs + cov_rs.T + np.diag(quantiles * (1 - quantiles))
        sigma_hat = cov_rs * np.sum(c_star ** 2) / m
        if width == 1:
            sigma_hat = np.sqrt(sigma_hat)
        else:
            sigma_hat = np.sqrt(np.diag(sigma_hat))
        
        # marginal p-value in quantile regression
        pval_quantile = 2 * (1 - norm.cdf(np.abs(rs / sigma_hat)))

        return pval_quantile


    @staticmethod
    def _fit_quant_model(model, q):
        """
        Fit quantile model.
        """
        res = model.fit(q=q)
        qpred0 = res.predict()

        return qpred0
        

    @staticmethod
    def _quant_regress(self, C : np.array, y : np.array, yq : np.array, zr : float) -> tuple[np.array[float]]:
        """
        Perform quantile regression on a single data source.
        """
        z = self.Z
        # project out the covariates
        C_star = C - z @ np.linalg.solve(z.T @ z) @ z.T @ C

        # rq equivilent for python is smf.quantreg
        # need as df
        data = pd.DataFrame(np.hstack((C, y)), columns=["Case"].extend(self.covars))
        model = smf.quantreg(model, data)
        qpred0 = [self._fit_quant_model(model, tau) for tau in self.quantiles]
        m = len(y) 
        width = len(self.quantiles)
        pvals = self._rank_score_test(C_star, qpred0, self.quantiles, m, width)

        return pvals


    def run_quantile_regression(self, dname):
        # set X_quant
        yq, zr = self._get_quantile(dname)
        pvals = self._quant_regress(self.C, self.Y[dname], yq, zr)

        return pvals


    def _marginal_test(self, dname): # ?
        _, _, _, _, firth_pvals = self.run_firth_regression(dname)
        quant_pvals = self.run_quantile_regression(dname)
        return firth_pvals, quant_pvals
    

    def run_marginal_tests(self) -> tuple[dict[str, dict[str, float]]]:
        """
        Run marginal tests for the Firth logistic and quantile regression
        components for each data source.
        """
        for dname in self.dnames:
            self.z_pvalues[dname], self.q_pvalues[dname] = self._marginal_test(dname)


    @staticmethod
    def _combine_cauchy(z_pvals, q_pvals, meta_weights, taus, zero_rate):
        """
        Combine p-values using Cauchy combination
        """
        # weights based on zero inflation rate
        weights = taus*(taus <= .5) + (1-taus)*(taus > .5)
        weights = weights / np.sum(weights) * (1-zero_rate)

        # meta weights based on user input for each data source
        norm_mweights = meta_weights / np.sum(meta_weights)

        # for each data source, add weighted transformed p-values
        cauchy_transform = 0
        for i in range(len(meta_weights)):
            w = weights[i]
            z = z_pvals[i]
            q = q_pvals[i]
            zr = zero_rate[i]
            nw = norm_mweights[i]
            cauchy_transform += nw * (
                zr * np.tan( (.5-z)*np.pi ) + sum( w*np.tan( (.5-q)*np.pi ) )
            )

            pval = 1 - sf(cauchy_transform)

        return pval


    def run_test_combination(self) -> float:
        qpvals = [pval for pval in self.q_pvalues.values()]
        zpvals = [pval for pval in self.z_pvalues.values()]
        weights = [weight for weight in self.weights.values()]
        taus = [tau for tau in self.quantiles]
        zero_rate = [zr for zr in self.zero_rate.values()]

        if self.test_combination == "cauchy" or True: #TODO: implement MinP too
            self.p_combined = self._combine_cauchy(qpvals, zpvals, weights, taus, zero_rate)

        return self.p_combined


    def run_zinq(self):
        """
        Run Entire ZINQ pipeline and return dataframes with p-values
        """
        self.run_sanity_check()
        self.run_marginal_tests()

        return self.run_test_combination()