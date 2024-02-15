from project.monte_carlo import MC_Datasets
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression as OLS2
from sklearn.linear_model import LassoCV
from project.information_theoric_model import ITMA
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
        return [MSE(y, y_pre), R2(y, y_pre), coef_Norm2(self.beta, model.coef_.reshape(-1))]
    

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
                self.ITMA[dgp] .append(self.__fit_ITMA (X, y                ))
                self.lasso[dgp].append(self.__fit_Lasso(X, y                ))
        
        results = []
        for dgp in ('DGP1', 'DGP2', 'DGP3'):
            results.append([dgp, 'PCA3',  self.rho, self.var, self.n_o] + self.__agg_results(self.PCA3 [dgp]))
            results.append([dgp, 'PCA5',  self.rho, self.var, self.n_o] + self.__agg_results(self.PCA5 [dgp]))
            results.append([dgp, 'ITMA',  self.rho, self.var, self.n_o] + self.__agg_results(self.ITMA [dgp]))
            results.append([dgp, 'LASSO', self.rho, self.var, self.n_o] + self.__agg_results(self.lasso[dgp]))
        
        return pd.DataFrame(results, columns = ['Dataset', 'Model', 'rho', 'err_var', 'n_obs', 'mse_avg', 'r2_avg', 'coef_diff_avg', 'mse_var', 'r2_var', 'coef_diff_var'])
            
            
            
            

