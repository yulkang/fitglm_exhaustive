function [mdl, info, mdls] = fitglm_exhaustive(X, y, glm_args, varargin)
% FITGLM_EXHAUSTIVE  Picks the best model among all 2^n_param possible models.
%
% USAGE
% -----
% [mdl, info, mdls] = FITGLM_EXHAUSTIVE(X, y, glm_args, ...)
%
% INPUT
% -----
% X: a matrix of independent variables, as given to glmfit or fitglm.
% y: a vector of the dependent variable, as given to glmfit or fitglm.
% glm_args: a cell array of the rest of the inputs for the fitglm.
%           For example, {'Distribution', 'binomial'}.
%
% OUTPUT
% ------
% mdl: the best GeneralizedLinearModel among the models.
% mdls: all models that are compared.
% info: struct about the results of the comparison
%     .param_incl : logical index of parameters chosen
%     .ic_min : minimum information criterion
%     .ic_min_ix : index of the model, starting from the null model.
%     .ic_all(k) : k-th model's IC.
%     .param_incl_all(k,m) : true if m-th parameter is included in k-th model.
%     .ic_all0{k}(s) : for model_criterion=crossval, 
%                      IC from s-th simulation of k-th model.
%     .ic_all_se(k) : standard error of mean of k-th model's ICs.
%
% info also contains the options:
%     .model_criterion
%     .must_include
%     .UseParallel
%     .group
%
% Name-value pair arguments
% -------------------------
% [...] = fitglm_exhaustive(..., 'OPTION1', OPTION1, ...)
%
% 'model_criterion'
%     'crossval' : cross validates using negative log likelihood
%     'AIC', 'AICc', 'BIC', 'BICc', 'CAIC' : see mdl.ModelCriterion
%     Default: 'BIC'
%
% 'must_include'
%     Numerical indices of columns to include.
%     Default: []
%
% 'crossval_args'
%     See crossval_glmfit
%
% 'UseParallel'
%     'auto'|'model'|'none'
%     Default: 'auto'
%
% 'group'
%     vector|empty
%     vector: When model_criterion is 'crossval', enables stratified sampling.
%             group(k) is the group number the k-th sample belongs to.
%             It must be an integer between 1 and the number of groups.
%     empty : All samples are in one group.
%
% EXAMPLE - BIC
% -------------
% n = 1e4;
% X = rand(n,3) - 0.5;
% y = X(:,1) * 0.4 + X(:,3) * 0.6 + 0.1;
% y = max(min(y, 10), -10);
% p = exp(y) ./ (1 + exp(y));
% ch = rand(size(p)) < p;
%
% [mdl, info, mdls] = fitglm_exhaustive(X, ch, {'Distribution', 'binomial'}, ...
%     'model_criterion', 'BIC')
%
% EXAMPLE - BIC, always including the second column
% -------------------------------------------------
% [mdl, info, mdls] = fitglm_exhaustive(X, ch, {'Distribution', 'binomial'}, ...
%     'model_criterion', 'BIC', 'must_include', 2)
%
% EXAMPLE - cross-validation with default settings
% ------------------------------------------------
% [mdl, info, mdls] = fitglm_exhaustive(X, ch, {'Distribution', 'binomial'}, ...
%     'model_criterion', 'crossval')
%
% EXAMPLE - cross-validation with 200 simulations of 50% holdout
% --------------------------------------------------------------
% [mdl, info, mdls] = fitglm_exhaustive(X, ch, {'Distribution', 'binomial'}, ...
%     'model_criterion', 'crossval', ...
%     'crossval_method', 'HoldOut', ...
%     'crossval_args', {0.5}, ...
%     'n_sim', 200)
%
% EXAMPLE - 5-fold cross-validation
% --------------------------------------------------------------
% [mdl, info, mdls] = fitglm_exhaustive(X, ch, {'Distribution', 'binomial'}, ...
%     'model_criterion', 'crossval', ...
%     'crossval_method', 'Kfold', ...
%     'crossval_args', {5})
%
% See also: demo_fitglm_exhaustive, fitglm, crossval_glmfit

% 2015 (c) Yul Kang. hk2699 at columbia dot edu.

    if ~exist('glm_args', 'var')
        glm_args = {};
    end
    S = varargin2S(varargin, {
        'model_criterion', 'BIC' 
        ... 'model_criterion'
        ... : 'crossval' : cross validates using negative log likelihood
        ... : 'AIC', 'AICc', 'BIC', 'BICc', 'CAIC' : see mdl.ModelCriterion
        ...
        'must_include', [] % Numerical indices of columns to include.
        'crossval_args', {} % See crossval_glmfit
        'UseParallel', 'auto' % 'auto'|'model'|'none' % 'auto': 
        'group', []
        ... group(k) 
        ... : an integer between 1 and #group that k-th sample belongs to.
        ... : as from [~, ~, group] = unique(matrix).
        });
    
    % Construct param_incl_all
    n_param = size(X, 2);
    n_model = 2 ^ n_param;
    param_incl_all = false(n_model, n_param);
    model_incl = true(n_model, 1);

    for i_model = n_model:-1:1
        param_incl = dec2bin(i_model - 1, n_param) == '1';
        if ~isempty(S.must_include) && any(~param_incl(S.must_include))
            model_incl(i_model) = false;
        end
        param_incl_all(i_model, :) = param_incl;
    end

    param_incl_all = param_incl_all(model_incl, :);
    n_model = size(param_incl_all, 1);

    % Determine UseParallel
    if strcmp(S.UseParallel, 'auto')
        if (strcmp(S.model_criterion, 'crossval') & n_model >= 2) ...
            || (n_model > 1e3)
            
            S.UseParallel = 'model';
        else
            S.UseParallel = 'none';
        end
    end

    % Fit
    [ic_all, ic_all0, param_incl_all, mdls] = ...
        fitglm_all(X, y, glm_args, param_incl_all, S);

    % Output
    ic_all_se = cellfun(@sem, ic_all0);

    [ic_min, ic_min_ix] = min(ic_all);
    mdl = mdls{ic_min_ix};

    param_incl = param_incl_all(ic_min_ix, :);

    info = packStruct(param_incl, ic_min, ic_min_ix, ...
        ic_all, param_incl_all, ...
        ic_all0, ic_all_se);
    info = copyFields(info, S);
end
function [ic_all, ic_all0, param_incl_all, mdls] = ...
        fitglm_all(X, y, glm_args, param_incl_all, S)
    
    n_model = size(param_incl_all, 1);

    ic_all = zeros(n_model, 1);
    ic_all0 = cell(n_model, 1);

    mdls = cell(n_model, 1);
    model_criterion = S.model_criterion;
    crossval_args = S.crossval_args;
    if isempty(S.group)
        group = ones(size(X, 1), 1);
%         [~, ~, group] = unique(X, 'rows');
    else
        group = S.group;
    end

    switch S.UseParallel
        case 'model'
            parfor i_model = 1:n_model
                param_incl = param_incl_all(i_model, :);

                [c_ic, c_ic0, c_mdl] = fitglm_unit(X, y, glm_args, param_incl, ...
                    model_criterion, crossval_args, group);

                ic_all(i_model) = c_ic;
                ic_all0{i_model} = c_ic0;

                mdls{i_model} = c_mdl;
            end
        otherwise
            for i_model = 1:n_model
                param_incl = param_incl_all(i_model, :);

                [c_ic, c_ic0, c_mdl] = fitglm_unit(X, y, glm_args, param_incl, ...
                    model_criterion, crossval_args, group);

                ic_all(i_model) = c_ic;
                ic_all0{i_model} = c_ic0;

                mdls{i_model} = c_mdl;
            end
    end
end
function [c_ic, c_ic0, c_mdl] = fitglm_unit(X, y, glm_args, param_incl, ...
    model_criterion, crossval_args, group)

    c_mdl = fitglm(X, y, glm_args{:}, ...
        'PredictorVars', find(param_incl));

    switch model_criterion
        case 'crossval'
            if verLessThan('matlab', '8.6')
                glm_args1 = [glm_args(:)', {'PredictorVars', find(param_incl)}];
                [c_ic, c_ic0] = crossval_glmfit(X, y, glm_args1, ...
                    'group', group, crossval_args{:});

                % Take negative log likelihood
                c_ic = -c_ic;
                c_ic0 = -c_ic0;
            else
                glm_args1 = [glm_args(:)', {'PredictorVars',find(param_incl)}];
                [c_ic, c_ic0] = crossval_glmfit(X, y, glm_args1, ...
                    'group', group, crossval_args{:});

                % Take negative log likelihood
                c_ic = -c_ic;
                c_ic0 = -c_ic0;
            end

        otherwise
            c_ic = c_mdl.ModelCriterion.(model_criterion);
            c_ic0 = c_ic;
    end
end