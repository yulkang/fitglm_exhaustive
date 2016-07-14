% Demo for fitglm_exhaustive().

%% Preparation
n = 1e4;
X = rand(n,3) - 0.5;
y = X(:,1) * 0.4 + X(:,3) * 0.6 + 0.1;
y = max(min(y, 10), -10);
p = exp(y) ./ (1 + exp(y));
ch = rand(size(p)) < p;

%% BIC
[mdl, info, mdls] = fitglm_exhaustive(X, ch, {'Distribution', 'binomial'}, ...
    'model_criterion', 'BIC')

%% cross-validation with default settings
[mdl, info, mdls] = fitglm_exhaustive(X, ch, {'Distribution', 'binomial'}, ...
    'model_criterion', 'crossval')

%% cross-validation with 20 simulations of 50% holdout
[mdl, info, mdls] = fitglm_exhaustive(X, ch, {'Distribution', 'binomial'}, ...
    'model_criterion', 'crossval', ...
    'crossval_method', 'HoldOut', ...
    'crossval_args', {0.5}, ...
    'n_sim', 20)

%% 5-fold cross-validation
[mdl, info, mdls] = fitglm_exhaustive(X, ch, {'Distribution', 'binomial'}, ...
    'model_criterion', 'crossval', ...
    'crossval_method', 'Kfold', ...
    'crossval_args', {5})
