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