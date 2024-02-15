from sklearn.linear_model import Lasso, LassoCV
from numpy import abs as nabs, array
from pandas import DataFrame, concat
from sklearn.metrics import mean_squared_error as mse
from project.monte_carlo import MC_Datasets


class Lasso_MC:

    def __init__(self, mc_dataset: MC_Datasets, lambdas: list) -> None:
        self.mc_dataset = mc_dataset
        self.lambdas = lambdas

    
    def fit_dgp(self, dgp: str) -> DataFrame:
        results = []
        df_cols = ['Dataset', 'lambda', 'score', 'mse', 'coef_mse', 'nzero_coef'] + [f'beta_{i}' for i in range(self.mc_dataset.datasets[0].meta['n_covariate'])]
        for dataset in self.mc_dataset.iter():
            X = eval(f'dataset.{dgp}_X')
            y = eval(f'dataset.{dgp}_y')
            a_results = []
            for a in self.lambdas:
                mdl = Lasso(max_iter=10_000, alpha=a).fit(X, y)
                mdl
                a_results += [[dgp, a, mdl.score(X, y), mse(y, mdl.predict(X)), mse(dataset.meta['coef'], mdl.coef_), (nabs(mdl.coef_) < 0.00005).sum()] + mdl.coef_.tolist()]
            results += [DataFrame(a_results, columns=df_cols)]
        return MC_Datasets.average_dataframes(results)
    

    def fit(self):
        return concat([self.fit_dgp(dgp) for dgp in ('DGP1', 'DGP2', 'DGP3')], ignore_index=True)


    def fit_dgp_cv(self, dgp: str) -> DataFrame:
        results = []
        for dataset in self.mc_dataset.iter():
            X = eval(f'dataset.{dgp}_X')
            y = eval(f'dataset.{dgp}_y')
            mdl = LassoCV(alphas=self.lambdas, max_iter=20_000).fit(X, y.reshape(-1)) 
            results += [[mdl.alpha_, mdl.score(X, y), mse(y, mdl.predict(X)), mse(dataset.meta['coef'], mdl.coef_), (nabs(mdl.coef_) < 0.00005).sum()] + mdl.coef_.tolist()]
            
        return [dgp] + array(results).mean(axis=0).tolist()


    def fit_cv(self):
        return DataFrame(
            [self.fit_dgp_cv(dgp) for dgp in ('DGP1', 'DGP2', 'DGP3')],
            columns = ['Dataset', 'lambda', 'score', 'mse', 'coef_mse', 'nzero_coef'] + [f'beta_{i}' for i in range(self.mc_dataset.datasets[0].meta['n_covariate'])]
        )
