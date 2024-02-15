from numpy import array, zeros
from pandas import DataFrame, concat
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error as mse
from statsmodels.regression.linear_model import OLS

from project.data_generating_process import DGP_Result
from project.monte_carlo import MC_Datasets


class MC_PCA:

    def __init__(self, mc_dataset: MC_Datasets) -> None:
        self.mc_dataset = mc_dataset
        self.PCAS = [{'DGP1': PCA().fit(ds.DGP1_X), 'DGP2': PCA().fit(ds.DGP2_X), 'DGP3': PCA().fit(ds.DGP3_X)} for ds in mc_dataset.iter()]
        
        self.__n = mc_dataset.datasets[0].meta['n_covariate']
        self.DGP1_X = array([ds.DGP1_X.tolist() for ds in self.mc_dataset.datasets])
        self.DGP2_X = array([ds.DGP2_X.tolist() for ds in self.mc_dataset.datasets])
        self.DGP3_X = array([ds.DGP3_X.tolist() for ds in self.mc_dataset.datasets])

        self.DGP1_y = array([ds.DGP1_y.tolist() for ds in self.mc_dataset.datasets])
        self.DGP2_y = array([ds.DGP2_y.tolist() for ds in self.mc_dataset.datasets])
        self.DGP3_y = array([ds.DGP3_y.tolist() for ds in self.mc_dataset.datasets])

        self.beta = self.mc_dataset.datasets[0].meta['coef']
        


    def __get_dgp_explained_var_ratio(self, dgp: str):
        return array([pca[dgp].explained_variance_ratio_.cumsum().tolist() for pca in self.PCAS]).mean(axis=0)


    def MC_get_explained_variance_ratio(self):
        return concat(
            [DataFrame({
                'Dataset': dgp,
                'n_component': list(range(self.__n)),
                'plot_cmp': [f'x_{i+1}' for i in range(self.__n)],
                'ratio': self.__get_dgp_explained_var_ratio(dgp),
                'rho': self.mc_dataset.datasets[0].meta['rho'],
                'var': self.mc_dataset.datasets[0].meta['err_variance'],
                'n_iter': self.mc_dataset.n_iter
            }) for dgp in ('DGP1', 'DGP2', 'DGP3')]
        )
    

    def __fit_single_ols(self, dgp: str, dataset: DGP_Result, pca: PCA):
        ols = OLS(eval(f'dataset.{dgp}_y'), pca.transform(eval(f'dataset.{dgp}_X'))).fit()
        return pca.inverse_transform(ols.params).tolist()


    def __fit_ols(self, dgp: str):
        coefs = array([self.__fit_single_ols(dgp, ds, pca[dgp]) for pca, ds in zip(self.PCAS, self.mc_dataset.datasets)]).mean(axis=0)
        coef_process = zeros((self.__n, 1))

        result = []
        X = eval(f'self.{dgp}_X')
        y = eval(f'self.{dgp}_y')

        for i in range(self.__n):
            coef_process[i][0] = coefs[i]
            y_pre = (X @ coef_process)

            res_coef_mse = mse(self.beta, coef_process)
            res_mse = ((y - y_pre)**2).sum(axis=0).mean()
            result.append([dgp, i+1, res_coef_mse, res_mse])
        
        return DataFrame(result, columns=['Dataset', 'n_component', 'coef_mse', 'mse'])
            


    def MC_fit(self):
        df = concat([self.__fit_ols(dgp) for dgp in ('DGP1', 'DGP2', 'DGP3')])
        df['rho'] = self.mc_dataset.datasets[0].meta['rho']
        df['var'] = self.mc_dataset.datasets[0].meta['err_variance']
        df['n_iter'] = self.mc_dataset.n_iter
        return df