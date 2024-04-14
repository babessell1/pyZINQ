import sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, pearsonr, norm, cauchy
from scipy import optimize
from numba import jit, njit
# quantile regression
import statsmodels.formula.api as smf
from py.firth import firth_logistic_regression
from py.helpers import *

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
                 data_matrix : tuple[np.ndarray], # list of dataframes correpsonding to different algorithms prodcucing similar data (n x 1, n x 1, ...)
                 metadata : pd.DataFrame, # list of dataframes correpsonding to different algorithms prodcucing similar data (n x p, n x p, ...)
                 data_names : list[str], # names of the data sources
                 response : str, # response variable
                 covariates2include : list = ["all"], # list of covariates to correct for
                 quantile_levels : list = [0.1, 0.25, 0.5, 0.75, 0.9], # quantile levels to regress on
                 method : str = "cauchy", # method to combine p-values
                 count_data : bool = False, # count data need to perform dithering
                 seed : int = 2024): # seed for ditheirng
    
        self.covars = metadata.columns if covariates2include == ["all"] else covariates2include
        if response in self.covars: self.covars.remove(response)
        self.response = response
        self.dnames = data_names
        self.quantiles = quantile_levels
        self.method = method
        self.seed = seed
        self.binary = True if len(np.unique(metadata[response])) == 2 else False
        self.count_data = count_data

        self.Z = metadata[self.covars].to_numpy() # covariates
        self.C = metadata[response].to_numpy() # response variable
        
        self.warning_codes = {}
        self.z_pvalues = {} # zero inflation p value
        self.q_pvalues = {} # quantile regression p value
        self.Y = {} # dependent variable
        self.weights = {} # weights for each data source
        for i,dname in enumerate(self.dnames):
            #self.X[dname] = np.hstack((self.data[dname], _ZC)) # design matrix
            self.warning_codes[dname] = [] 
            self.z_pvalues[dname] = -1
            self.q_pvalues[dname] = -1
            self.Y[dname] = data_matrix[i] if len(self.dnames) > 1 else data_matrix
            self.weights[dname] = 1 # TODO: implement weights
        

    def run_zinq(self) -> tuple[pd.DataFrame]:
        """
        run Entire ZINQ pipeline and return dataframes with p-values
        """
        self.run_sanity_check()
        self.run_marginal_tests()
        return self.run_test_combination()


    def _check_all(self) -> dict[int: list[int]]:
        """
        sanity check a single data source
        """
        return {dname: self._check(dname) for dname in self.dnames}


    def _check(self, dname) -> list[int]:
        """
        sanity check a single data source
        """
        return [
            i for i,chk in enumerate([
                self._check_lib_confound(dname), # not techincally implemented
                self._check_all_zero(dname), 
                self._check_limited_non_zero(dname),
                self._check_perfect_separation(dname)
            ]) if chk]


    def _check_lib_confound(self, dname) -> bool:
        """
        check if library size is a confounder
        """
        # if only one dimension, return False
        if len(self.Y[dname].shape) == 1:
            return False

        # Assumed that self.Y[dname] is an array with library sizes for each sample
        lib_sizes = self.Y[dname].sum(axis=1)
        test_vars = self.C
        pval = None


        if self.binary:
            # Two unique groups expected
            responses = np.unique(test_vars)
            if len(responses) != 2:
                raise ValueError("Expecting two unique responses for a binary test.")

            # Boolean masks for library sizes in each group
            mask_0 = test_vars == responses[0]
            mask_1 = test_vars == responses[1]

            # Sum library sizes within each group
            lib_size_group_0 = lib_sizes[mask_0].sum()
            lib_size_group_1 = lib_sizes[mask_1].sum()

            # Perform t-test between summed library sizes of the two groups
            _, pval = ttest_ind(
                [lib_size_group_0],
                [lib_size_group_1]
            )

        else:  # quantitative
            # Perform correlation between overall summed library size and the test variables
            pval, _ = pearsonr([lib_sizes.sum()] * len(test_vars), test_vars)

        return pval < 0.05
    

    def _check_all_zero(self, dname) -> list[int]:
        """
        check if all read counts are zero
        """
        return self.Y[dname].sum() == 0
    

    def _check_limited_non_zero(self, dname, thresh=30) -> bool:
        """
        check if there are limited non-zero read counts
        
        Returns True if the total sum of counts is less than `thresh`.
        """
        # Sum the values for the given name and compare with the threshold `thresh`
        return (self.Y[dname]!=0).sum() < thresh
    

    def _check_perfect_separation(self, dname) -> bool:
        """
        Check if there is perfect separation w.r.t. the binary variable of interest.
        
        Perfect separation exists if one unique value in the test variable `self.C`
        corresponds only to zeros or only to non-zeros in `self.data[dname]`, and the opposite is true
        for the other unique value in `self.C`.
        """
        test_var = self.C
        uniq = np.unique(test_var)  # binary unique values of the test variable
        
        if len(uniq) != 2:
            raise ValueError("The test variable is not binary; there should be exactly two unique values.")

        # Indices and values for both groups
        idx_group_0 = np.where(test_var == uniq[0])[0]
        idx_group_1 = np.where(test_var == uniq[1])[0]
        values_group_0 = self.Y[dname][idx_group_0]
        values_group_1 = self.Y[dname][idx_group_1]

        # Check for perfect separation
        perfect_separation_group_0 = np.all(values_group_0 == 0) and np.all(values_group_1 != 0)
        perfect_separation_group_1 = np.all(values_group_1 == 0) and np.all(values_group_0 != 0)

        return perfect_separation_group_0 or perfect_separation_group_1


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
        # Print warning messages for each dataset
         # Print warning messages for each dataset
        print('\n\n'.join([
            f'Source `{dname}`, has the following warnings: \n' + 
            '\n\t'.join([f"{codes[dname].count(cnt)} samples where {code_map[cnt]}" for cnt in set(codes[dname]) if cnt in code_map])
            for dname in self.dnames]))

        self.warning_codes = codes
        return codes
    
    
    @staticmethod
    def _firth_regress(C : np.ndarray, x : np.ndarray) -> tuple[np.ndarray[float]]: # x 5 
        # betas, bse, fitll, stats, pvals 
        
        return firth_logistic_regression(C, x)
    

    def _get_XZ(self, dname) -> np.ndarray:
        """
        get the design matrix and covariates for a single data source
        """
        y_column_vector = self.Y[dname][:, np.newaxis] # should be C

        return np.hstack((self.Z, y_column_vector))

    
    def run_firth_regression(self, dname):
        """
        perform firth logistic regression on a single data source
        """
        # betas, bse, fitll, stats, pvals 
        print(self._get_XZ(dname).shape)
        # C should be Y
        return self._firth_regress(self.C, self._get_XZ(dname))

    
    def _get_quantile(self, dname) -> np.ndarray:
        """
        build quantile matrix for quantile regression
        """
        # filter out zeros
        yq = self.Y[dname][self.Y[dname] != 0]
        
        # calculate zero inflation rate
        zero_rate = (self.Y[dname] == 0).sum() / len(self.Y[dname])

        self.zr = zero_rate
        
        if self.count_data: # dither discrete data
            yq = dither(yq, type="right", value=1)
        
        return yq, zero_rate
        
       
    @staticmethod
    def _rank_score_test(c_star, betas, quantiles, m, width):
        """
        Rank score test for quantile regression.
        """
        quantiles = np.array(quantiles)
        # this is ripped straight from the R code
        rs = [np.sum((tau - betas[k]) * c_star) / np.sqrt(m) for k, tau in enumerate(quantiles)]
        if width == 1:
            cov_rs = quantiles * (1 - quantiles)
        else:
            cov_rs = np.zeros((width, width))
            for k in range(width - 1):
                for l in range(k + 1, width):
                    cov_rs[k, l] = np.min([quantiles[k], quantiles[l]]) - quantiles[k] * quantiles[l]
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
        

    def _quant_regress(self, C : np.ndarray, y : np.ndarray, yq : np.ndarray, zr : float) -> tuple[np.ndarray[float]]:
        """
        Perform quantile regression on a single data source.
        """# Extract the non-zero data from the original data array to match the mask_nonzero size
        y_not_zero = y != 0

        Z_nonzero = self.Z[y_not_zero, :]
        C_nonzero = C[y_not_zero]
        C_star = C_nonzero - Z_nonzero @ np.linalg.pinv(Z_nonzero.T @ Z_nonzero) @ Z_nonzero.T @ C_nonzero

        # Prepare DataFrame for quantile regression
        data_nonzero = pd.DataFrame({'C': C_nonzero, 'y': yq})

        # Perform quantile regression for each quantile
        model = smf.quantreg('y ~ C', data_nonzero)
        qpred0 = [self._fit_quant_model(model, tau) for tau in self.quantiles]
        m = len(yq) 
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
        run marginal tests for firth logistic and quantile regression
        components for each data source
        """
        for dname in self.dnames:
            self.z_pvalues[dname], self.q_pvalues[dname] = self._marginal_test(dname)


    @staticmethod
    def _combine_cauchy(z_pvals, q_pvals, meta_weights, taus, zero_rate):
        """
        Combine p-values with Cauchy combination test
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

            pval = 1 - cauchy.cdf(cauchy_transform)

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