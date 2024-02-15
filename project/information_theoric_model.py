from numpy import ndarray, e, exp, array, zeros
from statsmodels.regression.linear_model import OLS
from statsmodels.api import add_constant
from pandas import DataFrame
from tqdm import trange
from project.monte_carlo import MC_Datasets


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

        
        