clear; clc; close all;


param_domains = struct();
param_domains.lambda_v = [1e-4, 1e0];
param_domains.lambda_a = [1e-4, 1e0];
param_domains.k1       = [1e-1, 1e2];
param_domains.k2       = [1e-1, 1e2];

bounds = { ...
    [log10(param_domains.lambda_v(1)), log10(param_domains.lambda_v(2))], ...
    [log10(param_domains.lambda_a(1)), log10(param_domains.lambda_a(2))], ...
    [log10(param_domains.k1(1)),       log10(param_domains.k1(2))], ...
    [log10(param_domains.k2(1)),       log10(param_domains.k2(2))] ...
};
bounds_py = py.list();
for i = 1:numel(bounds)
    bounds_py.append(py.tuple(bounds{i}));
end

checkpoint_file = 'safeopt_checkpoint.mat';
if isfile(checkpoint_file)
    load(checkpoint_file, 'X_data', 'Y_data', 'iteration', 'use_dynamic_beta');
    fprintf('恢复检查点：已加载迭代次数 %d 的数据。\n', iteration);
    X_init = py.numpy.array(X_data);
    Y_init = py.numpy.array(Y_data);
else
    iteration = 0;
    use_dynamic_beta = true;
    initial_params = [ 
        0.1, 0.1,  1,    1;
        0.2, 0.05, 2,    1.5;
        0.05,0.1,  1.5,  2;
        0.5, 0.3,  5,    1;
        0.8, 0.6,  3,    4;
        0.3, 0.2,  2.5,  3;
        0.05,0.05,1,    0.5;
        0.1, 0.5,  4,    2;
        0.2, 0.3,  3,    3.5;
        0.01,0.5,  8,    5;
        0.5, 0.01, 0.5,  8;
        0.9, 0.9,  0.2,  0.2;
        0.05,0.9,  9,    9;
        0.5, 0.7,  0.3,  6;
        0.7, 0.1,  7,    0.5;
    ];
    initial_J = [ 
        0.50;
        0.55;
        0.48;
        0.60;
        0.52;
        0.58;
        0.45;
        0.50;
        0.53;
        0.90;
        1.10;
        1.50;
        0.95;
        1.20;
        1.00;
    ];
    X_data = log10(initial_params);
    Y_data = -initial_J;
    X_init = py.numpy.array(X_data);
    Y_init = py.numpy.array(Y_data);
end

safe_threshold_J = max(initial_J(1:9));
h = safe_threshold_J; 
fprintf('安全性能阈值设定为 J = %.3f。\n', h);
fmin_list = py.list({-h});

py.importlib.import_module('GPy');
gp_model = py.GPy.models.GPRegression(X_init, Y_init, pyargs('noise_var', 0.01^2));

py.importlib.import_module('safeopt');
if use_dynamic_beta
    py.eval("import math", py.dict());
    beta_func = py.eval("@(t) (2*1 + 300 * 0.5 * t * math.log(t) * math.log(t/0.05)**2)", py.dict());
    opt = py.safeopt.SafeOptSwarm(gp_model, fmin_list, bounds_py, pyargs('beta', beta_func));
else
    opt = py.safeopt.SafeOptSwarm(gp_model, fmin_list, bounds_py, pyargs('beta', 2.0));
end

if exist('iteration','var') && iteration > 0
    opt.t = int32(size(X_data,1));
end

while true
    iteration = iteration + 1;
    fprintf('\n========= 安全优化迭代 %d =========\n', iteration);
    x_next_py = opt.optimize();  
    x_next = double(x_next_py);
    actual_param = 10 .^ x_next;
    fprintf('建议评估参数: lambda_v=%.4f, lambda_a=%.4f, k1=%.4f, k2=%.4f (对数空间:%s)\n', ...
            actual_param(1), actual_param(2), actual_param(3), actual_param(4), mat2str(x_next));
    J_val = input('请输入上述参数下测得的性能指标 J: ');
    f_val = -J_val;
    new_X = py.numpy.array(x_next_py);
    new_Y = py.numpy.array(py.list({f_val}));
    opt.add_new_data_point(new_X, new_Y);
    X_data = [X_data; double(x_next)]; 
    Y_data = [Y_data; f_val];
    initial_J = [initial_J; J_val];

    save(checkpoint_file, 'X_data', 'Y_data', 'iteration', 'use_dynamic_beta');
    fprintf('迭代 %d 完成，数据已保存至检查点文件。\n', iteration);

    cont = input('继续下一次迭代吗？(Y/N): ', 's');
    if upper(cont) ~= 'Y'
        break;
    end
end

[x_best_py, f_best_py] = opt.get_maximum();
x_best = double(x_best_py);
f_best = double(f_best_py);
best_params = 10 .^ x_best;
fprintf('\n>> 已完成所有迭代。\n');
fprintf('当前模型估计的最优安全参数为: lambda_v=%.4f, lambda_a=%.4f, k1=%.4f, k2=%.4f\n', ...
        best_params(1), best_params(2), best_params(3), best_params(4));
fprintf('对应的性能指标 J 估计值为 %.4f (收益 f = %.4f)。\n', -f_best, f_best);
