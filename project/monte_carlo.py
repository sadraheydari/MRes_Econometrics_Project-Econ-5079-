import pandas as pd
from tqdm import tqdm, trange
from datetime import datetime

from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error  as mse
from sklearn.metrics import r2_score            as r2

from numpy import sqrt, abs
from numpy import zeros, array, append
from numpy.random import randn, random_integers, seed
seed(19)
from numpy.linalg import inv
from numpy.linalg import cholesky

from project.data_generating_process import DGP

from functools import lru_cache



@lru_cache(maxsize=256)
def generate_correlation_matrix(rho, p):
    R  = zeros((p,p))
    for i in range(p):
        for j in range(p):
            R[i, j] = rho ** (abs(i-j))
    return R


# @lru_cache(maxsize=128)
def generate_coefs(p, beta):
    return array([(beta[i] if i<len(beta) else random_integers(low=1, high=5)) for i in range(p)])


def generate_result(beta_arr, X, y_true, p, n_omit, d_time):
    y_pred = X @ beta_arr
    result = []
    for i in range(y_pred.shape[1]):
        y_p = y_pred[:, [i]]
        result += [mse(y_true, y_p), mae(y_true, y_p), r2(y_true, y_p)]
    result += [mse(beta_arr[:, 0], beta_arr[:, 1]), mse(beta_arr[:(p-n_omit), 0], beta_arr[:(p-n_omit), 2])]
    result += [abs((beta_arr[:, 0] - beta_arr[:, 1])).mean() * 100, abs((beta_arr[:(p - n_omit), 0] - beta_arr[:(p - n_omit), 2])).mean() * 100]
    return result + [d_time]


def run_single_monte_carlo(
        nMC: int = 10000,   # Number of monte carlio iterations
        n: int = 100,       # Number of data observations
        p: int = 3,         # Number of predictors
        rho: float = 0.9,   # Correlation coefficient (keep between -1 and 1)
        sigma2: float = 0.5,# Regression variance
        n_omit: int = 1,    # Number of predictors used in omitted model
        beta: tuple = (),   # Coeficieants. Code will generate randomly if left empty
) -> list:
    s_time = datetime.now()

    TRUE   = []
    OLS    = []
    BIASED = []

    R = generate_correlation_matrix(rho, p)
    # print(R)
    R_chol = cholesky(R).T
    # print(R_chol)
    beta = generate_coefs(p, beta)
    # print(beta)

    for _ in range(nMC):
        
        X = randn(n,p) @ R_chol
        y = (X @ beta).reshape((-1, 1)) + sqrt(sigma2) * randn(n,1)  

        X_est = X[:, :n_omit]
        
        beta_OLS            = inv(X.T @ X) @ (X.T @ y)
        beta_OLS_omitbias   = inv(X_est.T @ X_est) @ (X_est.T @ y)

        beta_TRUE = beta

        TRUE  .append(beta_TRUE                                         .tolist())
        OLS   .append(beta_OLS.reshape(-1)                              .tolist())
        BIASED.append(append(beta_OLS_omitbias, zeros((p-n_omit, 1)))   .tolist())

        # print(y)
        # print(X)
        # print(X_est)
        # print(beta_OLS)
        # print(beta_OLS_omitbias)
    
    # print(array([TRUE, OLS, BIASED]).mean(axis = 1).T)
    return generate_result(array([TRUE, OLS, BIASED]).mean(axis = 1).T, X, y, p, n_omit, d_time=(datetime.now() - s_time).total_seconds())


def run_multi_monte_carlo(
        nMC:    list,   # Number of monte carlio iterations
        n:      list,   # Number of data observations
        p:      list,   # Number of predictors
        rho:    list,   # Correlation coefficient (keep between -1 and 1)
        sigma2: list,   # Regression variance
        n_omit: list,   # Number of predictors used in omitted model
        beta_l: list = []
) -> pd.DataFrame:
    setup_list = []
    for inMP in nMC:
        for i_n in n:
            for ip in p:
                for irho in rho:
                    for i_s in sigma2:
                        for iomit in n_omit:
                            setup_list.append([inMP, i_n, ip, irho, i_s, iomit])

    result = []
    for setup in tqdm(setup_list):
        inMP, i_n, ip, irho, i_s, iomit = setup
        result.append(setup + run_single_monte_carlo(inMP, i_n, ip, irho, i_s, iomit, tuple(beta_l)))
    
    feature_col = [
        'true_mse', 'true_mae', 'true_r2',
        'ols_mse',  'ols_mae',  'ols_r2',
        'bias_mse', 'bias_mae', 'bias_r2',
        'ols_mse_diff', 'bias_mse_diff',
        'ols_percent_diff', 'bias_percent_diff', 'runtime'
    ]
    setup_cols =  ['nMC', 'n', 'p', 'rho', 'sigma2', 'n_omit']

    return pd.DataFrame(result, columns = setup_cols + feature_col)




class MC_Datasets:

    def __init__(self, n_iter: int = 100, show_progress: bool = False) -> None:
        self.show_progress = show_progress
        self.n_iter = n_iter


    def create_datasets(self, err_variance: float = 0.25, rho: float = 0.9, n_observ = 200, beta = []):
        self.DGP = DGP(err_variance = err_variance, rho = rho, coef_list=beta)
        self.datasets = [self.DGP.generate_data(T = n_observ) for _ in range(self.n_iter)]


    def iter(self):
        if self.show_progress:
            return tqdm(self.datasets)
        return self.datasets
    

    @staticmethod
    def average_dataframes(df_list: list):
        if len(df_list) == 0:
            return pd.DataFrame()
        if len(df_list) == 1:
            return df_list[0]
        
        df = df_list[0].copy()
        num_cols = df.describe().columns.values

        for c in df.columns:
            if c in num_cols:
                df[c] = array([d[c].values.tolist() for d in df_list]).mean(axis=0)

        return df
    

    


