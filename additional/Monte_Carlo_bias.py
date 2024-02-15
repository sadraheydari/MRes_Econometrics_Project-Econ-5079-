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

# from project.data_generating_process import DGP



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
    

    



from numpy import ndarray, e, exp, array, power
from statsmodels.regression.linear_model import OLS
from statsmodels.api import add_constant
from pandas import DataFrame
from tqdm import trange
# from project.monte_carlo import MC_Datasets


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




import pandas as pd
import numpy as np
from tqdm import tqdm


setups = []

# for rho in [0.75, 0.8, 0.85, 0.9]:
    # for sig in np.array([0, 1, 10]):
for rho in [0.8]:
    for sig in np.array([0, 1]):
        setups.append((rho, sig**2))


result_mdl = []
result_prob = []
result_best = []

for rho, var in tqdm(setups):
    mc_dataset = MC_Datasets(show_progress=False, n_iter=100)
    mc_dataset.create_datasets(beta=[-5, 3, 2], err_variance=var, rho=rho)
    mdl = InfoTheoricModelAVG(mc_dataset=mc_dataset)
    b, t, p = mdl.MC_get_best_models()

    b['var'] = var
    b['rho'] = rho

    t['var'] = var
    t['rho'] = rho
    
    p['sigma'] = var
    p['rho'] = rho
    
    result_mdl.append(t)
    result_prob.append(p)
    result_best.append(b)

pd.concat(result_mdl,  ignore_index=True).to_csv('true.csv')
pd.concat(result_prob, ignore_index=True).to_csv('prob.csv')
pd.concat(result_best, ignore_index=True).to_csv('best.csv')