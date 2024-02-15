clear all; clc;


nMC = 10000;  % Number of Monte Carlo iterations

% Storage matrices for estimates
TRUE   = [];
OLS    = [];
BIASED = [];

for iMC = 1:nMC  %%%% Start Monte Carlo iterations here
    
    %%%%% ========================| GENERATE ARTIFICIAL DATA |=================
    % Settings for Monte Carlo exercise
    n = 100;   % Number of data observations
    p = 3;     % Number of predictors

    % Generate regressors that might be correlated
    % first generate correlation matrix
    rho = 0.99;   % Correlation coefficient (keep between -1 and 1)
    R   = zeros(p,p);  % Space in memory for correlation matrix
    for i = 1:p
        for j = 1:p
            R(i,j) = rho^(abs(i-j));   
        end
    end
    
    % next generate correlated predictors from a Normal distribution
    X = randn(n,p)*chol(R);

    % generate regression coefficients. Here you can either set each parameter
    % yourself, e.g. beta = [1.2; -0.8] in the case of p=2 predictors, beta =
    % [2.1; -1.4; 3.6] in the case of p=3 predictors, etc. Here below I specify
    % a beta vector that complies with p=2. If you do not want to input each
    % value of beta every time (e.g. because you want to try p=20), then you
    % can generate this randomly (using a command of the form beta =
    % rand(p,1);)
    beta = [1.3; 0.9; -0.9];

    % Generate regression data y
    sigma2 = 1;   % Regression variance
    y = X*beta + sqrt(sigma2)*randn(n,1);   % These are my data y following the regression data generating process
    %%%%% =====================================================================


    % From here on, you have in your memory data X and y, and you can treat
    % them as a sample from a population. The only difference is that you now
    % know that y DOES come from a linear regression model with p predictors.
    % You can do various experiments. For example, if you use all p predictors
    % you can estimate the regression model with various values of n
    % (observations) and check how OLS becomes more precise as n increases. The
    % experiment we are going to do here is that of omitted variable bias.
    % Assume that you are given p predictors in X but you only use the first
    % one
    X_est = X(:,1);

    % This means that OLS estimates in the TRUE model (the one in the data
    % generating process above) is:
    beta_OLS  = (X'*X)\(X'*y);

    % ...while OLS estimates in our misspecified model with only one predictor
    % are
    beta_OLS_omitbias = (X_est'*X_est)\(X_est'*y);

    % The true parameters are known of course with precision, and are provided
    % from the DGP above
    beta_TRUE = beta;

    % Save results from i-th Monte Carlo iteration
    % These are p estimates for each of the nMC datasets
    TRUE = [TRUE, beta_TRUE];
    OLS =  [OLS, beta_OLS];
    BIASED = [BIASED, [beta_OLS_omitbias;zeros(p-1,1)]];

end % end Monte Carlo iterations

% The results we are interested in are the averages over the nMC iterations
disp(table(mean(TRUE,2),mean(OLS,2),mean(BIASED,2),'VariableNames', {'TRUE', 'OLS', 'BIASED'}))
