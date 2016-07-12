function [mdl, info, mdls] = fitglm_exhaustive(X, y, glm_args, varargin)
% Picks the best model among all 2^n_param possible models.
%
% [mdl, info, mdls] = fitglm_exhaustive(X, y, glm_args, varargin)
%
% OPTIONS:
% 'model_criterion', 'BIC'
% 'must_include', [] % Numerical indices of columns to include.
% 'crossval_args', {}
% 'UseParallel', 'model' % 'model'|'none'
%
% EXAMPLE:
% n = 1e4;
% X = rand(n,3) - 0.5;
% y = X(:,1) * 0.2 + X(:,3) * 0.5 + 0.1;
% y = max(min(y, 10), -10);
% p = exp(y) ./ (1 + exp(y));
% ch = rand(size(p)) < p;
% [mdl, info, mdls] = fitglm_exhaustive(X, ch, {'Distribution', 'binomial'}, ...
%     'model_criterion', 'BIC');
% disp(mdl);
%
% 2015 (c) Yul Kang. hk2699 at columbia dot edu.

    if ~exist('glm_args', 'var')
        glm_args = {};
    end
    S = varargin2S(varargin, {
        ... % 'model_criterion'
        ... % : 'crossval' : cross validates using negative log likelihood
        ... % : 'AIC', 'AICc', 'BIC', 'BICc', 'CAIC' : see mdl.ModelCriterion
        'model_criterion', 'BIC' 
        'must_include', [] % Numerical indices of columns to include.
        'crossval_args', {}
        'UseParallel', 'none' % 'model'|'none'
        'group', []
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
        [~, ~, group] = unique(X, 'rows');
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
                [c_ic, c_ic0] = yk.stat.crossval_glmfit(X, y, glm_args1, ...
                    'group', group, crossval_args{:});

                % Take negative log likelihood
                c_ic = -c_ic;
                c_ic0 = -c_ic0;
            else
                glm_args1 = [glm_args(:)', {'PredictorVars',find(param_incl)}];
                [c_ic, c_ic0] = yk.stat.crossval_glmfit(X, y, glm_args1, ...
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