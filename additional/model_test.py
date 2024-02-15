from numpy import array, append, sqrt
from numpy.random import randn, uniform
from numpy.linalg import cholesky
from datetime import datetime


class DGP:
    """
    Data Generation Process

    ...

    Attributes
    ----------
    meta    : dict
        contains the setting og the DGP: {
            `n_covariate`   : number of covariates used for the model, 
            `err_variance`  : varaince of the regression model, 
            `coef`          : coefficiant in regression model, 
            `rho`           : correlation factor
        }
    
    DGP1    : function

    DGP2    : function

    DGP3    : function

            
    Methods
    -------
    generate_data(T = 10_000):
        generates three datasets based on each DGP
    """
    
    def __init__(
            self, 
            n_covariate: int = 100, 
            err_variance : float = 1.0,

            n_relevant: int = 3, 
            coef_list: list = [], 
            rand_interval: tuple = (-5, 5),

            rho: float = 0.9
        ) -> None:

        self.__n_covariate  = n_covariate
        self.__err_variance = err_variance
        
        self.DGP1 = None
        self.__generate_DGP1(n_relevant, coef_list, rand_interval)
        
        self.DGP2 = None
        self.__generate_DGP2(rho)

        self.DGP3 = None
        self.__generate_DGP3()

        self.meta = {
            'n_covariate':  self.__n_covariate, 
            'err_variance': self.__err_variance,
            'coef': self.__beta,
            'rho': self.__rho
        }



    def __generate_DGP1(self, n_relevant, coef_list, rand_interval):
        coefs = [[(coef_list[i] if i < len(coef_list) else uniform(rand_interval[0], rand_interval[1]))] for i in range(n_relevant)]
        coefs += [[0] for _ in range(n_relevant, self.__n_covariate)]
        self.__beta = array(coefs)
        self.DGP1 = lambda x: (x @ self.__beta) + (sqrt(self.__err_variance) * randn(x.shape[0], 1))
    

    def __generate_DGP2(self, rho):
        self.__rho = rho
        self.__correlation_matrix = array([[rho ** abs(i - j) for j in range(self.__n_covariate)] for i in range(self.__n_covariate)])
        self.__chol = cholesky(self.__correlation_matrix).T
        self.DGP2 = lambda x: self.DGP1(x @ self.__chol)


    def __generate_DGP3(self):
        self.DGP3 = lambda x: self.DGP1(self.__generate_AR_X(x))

    
    def __generate_AR_X(self, X, random_residual = True):
        ARX = X[:, [0]].copy()
        
        u_vec = randn(X.shape[0], 1) * 0.5

        for _ in range(1, self.__n_covariate):
            if random_residual:
                u_vec = randn(X.shape[0], 1) * 0.5
            ARX = append(ARX, self.__rho * ARX[:, [-1]] + u_vec, axis = 1)
        return ARX
    

    def generate_data(self, T = 500):
        X1 = randn(T, self.__n_covariate)
        X2 = X1 @ self.__chol
        X3 = self.__generate_AR_X(X1)

        residuals = randn(T, 1) * sqrt(self.__err_variance)
        self.meta['n_obs'] = T
        
        return DGP_Result(
            X1, X1 @ self.__beta + randn(T, 1) * residuals,
            X2, X2 @ self.__beta + randn(T, 1) * residuals,
            X3, X3 @ self.__beta + randn(T, 1) * residuals,
            self.meta
        )


class DGP_Result:

    def __init__(self, DGP1_X, DGP1_y, DGP2_X, DGP2_y, DGP3_X, DGP3_y, meta) -> None:
        
        self.DGP1_X = DGP1_X
        self.DGP1_y = DGP1_y
        
        self.DGP2_X = DGP2_X
        self.DGP2_y = DGP2_y
        
        self.DGP3_X = DGP3_X
        self.DGP3_y = DGP3_y
        
        self.meta = meta
        self.time = str(datetime.now())


    def __repr__(self) -> str:
        return f"DGP [{self.time}] [err_var: {self.meta['err_variance']}\t| rho: {self.meta['rho']}\t| n_feature: {self.meta['n_covariate']}\t| n_sample: {self.DGP1_X.shape[0]}]" + f"\n\t\tbeta: [{' '.join(['%2f' % i for i in self.meta['coef'].reshape(-1)[:5]])}]"
    

    def __str__(self) -> str:
        return self.__repr__()
    

import pandas as pd
from tqdm import tqdm, trange
from datetime import datetime

from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error  as mse
from sklearn.metrics import r2_score            as r2

from numpy import sqrt, abs
from numpy import zeros, array, append
from numpy.random import randn
from numpy.linalg import inv
from numpy.linalg import cholesky

from functools import lru_cache


@lru_cache(maxsize=256)
def generate_correlation_matrix(rho, p):
    R  = zeros((p,p))
    for i in range(p):
        for j in range(p):
            R[i, j] = rho ** (abs(i-j))
    return R


@lru_cache(maxsize=128)
def generate_coefs(p, beta):
    return array([(beta[i] if i<len(beta) else randn(1)[0]) for i in range(p)])


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
    

from numpy import ndarray, e, exp, array, zeros
from statsmodels.regression.linear_model import OLS
from statsmodels.api import add_constant
from pandas import DataFrame
from tqdm import trange


class InfoTheoricModelAVG:

    def __init__(self, mc_dataset: MC_Datasets, n_estimator: int = 10, prob_pow: float = e) -> None:
        self.n_estimator = n_estimator
        self.mc_dataset = mc_dataset
        self.prob_pow = prob_pow


    def _get_dataset(self):
        return self.mc_dataset.datasets[0]
    

    def fit_all_combination(self, dgp: str, show_progress = False, dataset = None, metric='BIC') -> DataFrame:
        itr = trange if show_progress else range

        results = []

        if dataset is None:
            X = eval(f'self._get_dataset().{dgp}_X')
            y = eval(f'self._get_dataset().{dgp}_y')
        else:
            X = eval(f'dataset.{dgp}_X')
            y = eval(f'dataset.{dgp}_y')
        
        model_combinations = InfoTheoricModelAVG.generate_all_possible_combination(n = self.n_estimator)
        for i in itr(len(model_combinations)):
            model_string = ''.join([str(j) for j in model_combinations[i]])
            X_cmb = InfoTheoricModelAVG.generate_dataset(X, model_combinations[i])
            results.append([dgp, model_string] + InfoTheoricModelAVG.run_ols(X_cmb, y))
        
        result_df = DataFrame(results, columns=['Dataset', 'Combination', 'AIC', 'BIC', 'R2', 'Adj_R2'])
        result_df['prob'] = self.get_each_combination_probabitiy(result_df, metric)

        return result_df
    

    def get_each_combination_probabitiy(self, df: DataFrame, metric: str = 'BIC') -> ndarray:
        # Use formula from Kapetanios, Labhard and Price (2008)
        metric_vals = df[metric].values
        metric_vals = exp(-0.5 * (metric_vals - metric_vals.min()))
        return  metric_vals / metric_vals.sum()
    

    def MC_fit_all_combinations(self, dgp: str, **args) -> DataFrame:
        fit_df = []
        for dataset in self.mc_dataset.iter():
            fit_df += [self.fit_all_combination(dgp, show_progress=False, dataset=dataset)]
        return MC_Datasets.average_dataframes(fit_df)
    

    def __MC_get_variable_probability(self, df: DataFrame) -> ndarray:
        combination = array([[int(k) for k in s] for s in df['Combination']])
        prob = df['prob'].values.reshape((1, -1))
        return enumerate((prob @ combination).reshape(-1).tolist())
    

    def MC_get_best_models(self, metric: str = 'BIC'):
        best_models = []
        true_model_stat = []
        variable_prob = []
        for dgp in ('DGP1', 'DGP2', 'DGP3'):
            prob = self.MC_fit_all_combinations(dgp)
            variable_prob += [[dgp, f'x_{i+1}', p] for i, p in self.__MC_get_variable_probability(prob)]
            best_models += prob.sort_values('prob', ascending=False).iloc[[0]].values.tolist()
            true_model_stat += prob[prob['Combination'] == '1110000000'].values.tolist()
        return DataFrame(best_models, columns = prob.columns.to_list()), DataFrame(true_model_stat, columns = prob.columns.to_list()), DataFrame(variable_prob, columns=['Dataset', 'var', 'prob'])
    

    @staticmethod
    def generate_all_possible_combination(n = 10) -> list:
        if n == 0:
            return [[]]
        t = InfoTheoricModelAVG.generate_all_possible_combination(n-1)
        return [[0] + a for a in t] + [[1] + a for a in t]


    @staticmethod
    def generate_dataset(X: ndarray, selection_array: list) -> ndarray:
        cols = [0] + [i + 1 for i, s in enumerate(selection_array) if s]
        return add_constant(X)[:, cols]


    @staticmethod
    def run_ols(X: ndarray, y: ndarray) -> list:
        mdl = OLS(y, X).fit()
        return [mdl.aic, mdl.bic, mdl.rsquared_adj, mdl.rsquared]


class ITMA:

    def __init__(self, n_estimator: int = 10) -> None:
        self.n_estimator = n_estimator
        self.model_combinations = InfoTheoricModelAVG.generate_all_possible_combination(n = n_estimator)
        self.models = [None for _ in self.model_combinations]
        self.X = None
        self.y = None
        self.coefs_list = [None for _ in self.models]
        self.BICs = zeros(len(self.models))
        self.pi_weight = zeros((n_estimator, 1))
        self.coefs = None


    def __generate_beta(self, beta: ndarray, combination: list):
        res = []
        j = 1
        for c in combination:
            res.append(beta[j] if int(c) == 1 else 0)
            j += int(c)
        return res


    def __fit_single_model(self, X, y, combination):
        x_cmb = InfoTheoricModelAVG.generate_dataset(X, selection_array=combination)
        mdl = OLS(y, x_cmb).fit()
        beta = self.__generate_beta(mdl.params, combination)
        return mdl, beta, mdl.bic



    def fit(self, y: ndarray, X: ndarray):
        self.y = y
        self.X = X
        for i, cmb in enumerate(self.model_combinations):
            mdl, coef, bic = self.__fit_single_model(X, y, combination=cmb)
            self.models[i] = mdl
            self.coefs_list[i] = coef
            self.BICs[i] = bic
        delta = self.BICs - self.BICs.min()
        self.pi_weight = (exp(-0.5 * delta) / exp(-0.5 * delta).sum()).reshape((-1, 1))
        self.coefs = (array(self.coefs_list).T @ self.pi_weight)


    
    def predict(self, X: ndarray):
        return (X[:, :self.n_estimator] @ self.coefs).reshape(-1)

        
from project.monte_carlo import MC_Datasets
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from sklearn.linear_model import LinearRegression as OLS2
from sklearn.linear_model import LassoCV
import numpy as np
import pandas as pd


def coef_Norm2(b1, b2):
    n = min(len(b1), len(b2))
    return MSE(b1[:n], b2[:n])


class ProjectModel:

    def __init__(self, mc_datasets: MC_Datasets) -> None:
        self.mc_ds = mc_datasets

        self.beta = mc_datasets.datasets[0].meta['coef'].reshape(-1)
        self.rho  = mc_datasets.datasets[0].meta['rho']
        self.var  = mc_datasets.datasets[0].meta['err_variance']
        self.n_o  = mc_datasets.datasets[0].meta['n_obs']


        self.PCA3 = {'DGP1': [], 'DGP2': [], 'DGP3': []}
        self.PCA5 = {'DGP1': [], 'DGP2': [], 'DGP3': []}
        self.ITMA = {'DGP1': [], 'DGP2': [], 'DGP3': []}
        self.lasso = {'DGP1': [], 'DGP2': [], 'DGP3': []}


    def __fit_PCA(self, X, y, n_components):
        pca = PCA(n_components=n_components)
        x_pca = pca.fit_transform(X)
        model = OLS2()
        model.fit(x_pca, y.reshape(-1))
        y_pre = model.predict(x_pca)
        coef = pca.inverse_transform(model.coef_.reshape(-1))
        return [MSE(y, y_pre), R2(y, y_pre), coef_Norm2(self.beta, coef)]
    

    def __fit_ITMA(self, X, y):
        model = ITMA()
        model.fit(y, X)
        y_pre = model.predict(X)
        return [MSE(y, y_pre), R2(y, y_pre), coef_Norm2(self.beta, model.coefs.reshape(-1))]
    

    def __fit_Lasso(self, X, y):
        model = LassoCV(max_iter=10_000)
        model.fit(X, y.reshape(-1))
        y_pre = model.predict(X)
        return [MSE(y, y_pre), R2(y, y_pre), coef_Norm2(self.beta, model.coef_.reshape(-1))]


    def __agg_results(self, results: list):
        results = np.array(results)
        avg_res = results.mean(axis = 0).reshape(-1).tolist()
        var_res = results.var(axis=0).reshape(-1).tolist()
        return avg_res + var_res


    def fit(self):
        for dataset in self.mc_ds.iter():
            for dgp in ('DGP1', 'DGP2', 'DGP3'):
                X = eval(f'dataset.{dgp}_X')
                y = eval(f'dataset.{dgp}_y')
                self.PCA3[dgp] .append(self.__fit_PCA  (X, y, n_components=3))
                self.PCA5[dgp] .append(self.__fit_PCA  (X, y, n_components=5))
                # self.ITMA[dgp] .append(self.__fit_ITMA (X, y                ))
                # self.lasso[dgp].append(self.__fit_Lasso(X, y                ))
        
        results = []
        for dgp in ('DGP1', 'DGP2', 'DGP3'):
            results.append([dgp, 'PCA3',  self.rho, self.var, self.n_o] + self.__agg_results(self.PCA3 [dgp]))
            results.append([dgp, 'PCA5',  self.rho, self.var, self.n_o] + self.__agg_results(self.PCA5 [dgp]))
            # results.append([dgp, 'ITMA',  self.rho, self.var, self.n_o] + self.__agg_results(self.ITMA [dgp]))
            # results.append([dgp, 'LASSO', self.rho, self.var, self.n_o] + self.__agg_results(self.lasso[dgp]))
        
        return pd.DataFrame(results, columns = ['Dataset', 'Model', 'rho', 'err_var', 'n_obs', 'mse_avg', 'r2_avg', 'coef_diff_avg', 'mse_var', 'r2_var', 'coef_diff_var'])
            
            
from tqdm import trange


setups = []

for rho in [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95]:
    for var in [0, 0.01, 1, 100]:
        for n_obs in [500, 300, 100, 50]:
            setups.append((rho, var, n_obs))


result_dfs = []

for i in trange(len(setups)):
    rho, var, n_obs = setups[i]

    mc_ds = MC_Datasets(n_iter=100, show_progress=False)
    mc_ds.create_datasets(beta=[-5, 3, 2], err_variance=var, rho=rho, n_observ=n_obs)
    pr_model = ProjectModel(mc_ds)
    
    result_dfs.append(pr_model.fit())

    if (i+1) % 10 == 0:
        pd.concat(result_dfs, ignore_index=True).to_csv('compare.csv', index=False)

pd.concat(result_dfs, ignore_index=True).to_csv('compare.csv', index=False)