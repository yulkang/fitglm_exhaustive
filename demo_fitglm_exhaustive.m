%% BIC
n = 1e4;
X = rand(n,3) - 0.5;
y = X(:,1) * 0.4 + X(:,3) * 0.6 + 0.1;
y = max(min(y, 10), -10);
p = exp(y) ./ (1 + exp(y));
ch = rand(size(p)) < p;
[mdl, info, mdls] = fitglm_exhaustive(X, ch, {'Distribution', 'binomial'}, ...
    'model_criterion', 'BIC');
disp(mdl);
disp(info);
disp(mdls);

%% Crossval
n = 1e4;
X = rand(n,3) - 0.5;
y = X(:,1) * 0.4 + X(:,3) * 0.6 + 0.1;
y = max(min(y, 10), -10);
p = exp(y) ./ (1 + exp(y));
ch = rand(size(p)) < p;
[mdl, info, mdls] = fitglm_exhaustive(X, ch, {'Distribution', 'binomial'}, ...
    'model_criterion', 'crossval');
disp(mdl);
disp(info);
disp(mdls);
