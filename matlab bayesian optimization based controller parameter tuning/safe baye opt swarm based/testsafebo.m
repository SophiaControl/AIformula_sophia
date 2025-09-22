clear; clc; close all;

% === 安全贝叶斯优化实验脚本 ===

% 参数域定义（参考MATLAB bayesopt示例），使用对数10变换表示范围
param_domains = struct();
param_domains.lambda_v = [1e-4, 1e0];   % lambda_v参数原始范围
param_domains.lambda_a = [1e-4, 1e0];   % lambda_a参数原始范围
param_domains.k1       = [1e-1, 1e2];   % k1参数原始范围
param_domains.k2       = [1e-1, 1e2];   % k2参数原始范围

% 将参数范围取log10，作为GP优化空间边界
bounds = { ...
    [log10(param_domains.lambda_v(1)), log10(param_domains.lambda_v(2))], ...
    [log10(param_domains.lambda_a(1)), log10(param_domains.lambda_a(2))], ...
    [log10(param_domains.k1(1)),       log10(param_domains.k1(2))], ...
    [log10(param_domains.k2(1)),       log10(param_domains.k2(2))] ...
};
bounds_py = py.list();  % Python端的bounds列表
for i = 1:numel(bounds)
    bounds_py.append(py.tuple(bounds{i}));
end

% === 初始化或恢复 ===
checkpoint_file = 'safeopt_checkpoint.mat';
if isfile(checkpoint_file)
    % 恢复检查点：加载之前保存的数据，重建GP模型和SafeOpt状态
    load(checkpoint_file, 'X_data', 'Y_data', 'iteration', 'use_dynamic_beta');
    fprintf('恢复检查点：已加载迭代次数 %d 的数据。\n', iteration);
    % 将历史数据转换为Python numpy数组用于GP初始化
    X_init = py.numpy.array(X_data);
    Y_init = py.numpy.array(Y_data);
else
    % 全新运行：用户提供初始种子点参数及性能J
    iteration = 0;
    use_dynamic_beta = true;  % β_t模式：false=固定β, true=自适应β公式
    % 定义初始参数集合 (15个种子点示例，其中前9个假定为安全点)
    % 用户需根据实际情况填写 initial_params 和 initial_J
    initial_params = [ 
        % 15行，每行对应 lambda_v, lambda_a, k1, k2 的实际值 (示例占位)
        0.1, 0.1,  1,    1;    % 第1个安全点 (示例值)
        0.2, 0.05, 2,    1.5;  % 第2个安全点
        0.05,0.1,  1.5,  2;    % 第3个安全点
        0.5, 0.3,  5,    1;    % 第4个安全点
        0.8, 0.6,  3,    4;    % 第5个安全点
        0.3, 0.2,  2.5,  3;    % 第6个安全点
        0.05,0.05,1,    0.5;   % 第7个安全点
        0.1, 0.5,  4,    2;    % 第8个安全点
        0.2, 0.3,  3,    3.5;  % 第9个安全点
        0.01,0.5,  8,    5;    % 第10个点 (可能不安全)
        0.5, 0.01, 0.5,  8;    % 第11个点 (可能不安全)
        0.9, 0.9,  0.2,  0.2;  % 第12个点 (可能不安全)
        0.05,0.9,  9,    9;    % 第13个点 (可能不安全)
        0.5, 0.7,  0.3,  6;    % 第14个点 (可能不安全)
        0.7, 0.1,  7,    0.5;  % 第15个点 (可能不安全)
    ];
    initial_J = [ 
        % 对应上述15个参数点的观测性能J值 (用户需提供真实测量或仿真值)
        % 假设前9个点J均在安全阈值以内，以下为示例数据:
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
    % 转换初始数据到对数空间（GP输入），同时转换J为GP目标值
    X_data = log10(initial_params);    % X_data 为 N×4 矩阵 (对数尺度参数)
    Y_data = -initial_J;  % 若优化目标是降低误差J，这里取负值将其转为最大化的收益
    % （以上将误差最小化问题转为收益最大化问题，使SafeOpt可用:contentReference[oaicite:9]{index=9}）
    % 将初始数据转换为Python的numpy数组
    X_init = py.numpy.array(X_data);
    Y_init = py.numpy.array(Y_data);
end

% 确定安全性能阈值 h （使用已知安全点的最差表现作为阈值）
safe_threshold_J = max(initial_J(1:9));   % 假定前9个初始点为安全点:contentReference[oaicite:10]{index=10}
h = safe_threshold_J; 
fprintf('安全性能阈值设定为 J = %.3f。\n', h);
fmin_list = py.list({-h});  % SafeOpt参数：性能函数的安全下限（对收益f = -J而言为 -h）

% 构建GP模型 (使用GPy库)，核函数和超参由GPy自动选择
py.importlib.import_module('GPy');  % 确保导入GPy模块
% 创建高斯过程回归模型，假设观测噪声较小（如 0.01）
gp_model = py.GPy.models.GPRegression(X_init, Y_init, pyargs('noise_var', 0.01^2));

% 初始化 SafeOpt 优化器
py.importlib.import_module('safeopt');  % 导入safeopt模块
if use_dynamic_beta
    % 自适应 β_t 模式：定义一个Python函数或lambda用于计算β_t:contentReference[oaicite:11]{index=11}
    py.eval("import math", py.dict());  % 导入math以供计算
    beta_func = py.eval("@(t) (2*1 + 300 * 0.5 * t * math.log(t) * math.log(t/0.05)**2)", py.dict());
    % 上式实现 β_t = 2B + 300 * γ_t * [log(t/δ)]^3，其中 B=1, γ_t=0.5*t*log(t), δ=0.05:contentReference[oaicite:12]{index=12}
    opt = py.safeopt.SafeOptSwarm(gp_model, fmin_list, bounds_py, pyargs('beta', beta_func));
else
    % 固定 β 模式：例如取 β=2 （可根据需要调整）
    opt = py.safeopt.SafeOptSwarm(gp_model, fmin_list, bounds_py, pyargs('beta', 2.0));
end

% 如果从检查点恢复，需更新 SafeOpt 内部计数
if exist('iteration','var') && iteration > 0
    opt.t = int32(size(X_data,1));  % 将已观测点数量赋给 opt.t (若需要)
end

% === 优化迭代 ===
while true
    iteration = iteration + 1;
    fprintf('\n========= 安全优化迭代 %d =========\n', iteration);
    % 使用 SafeOpt 获取下一个评估的安全参数点:contentReference[oaicite:13]{index=13}
    x_next_py = opt.optimize();  
    x_next = double(x_next_py);  % 转换Python返回的numpy数组为MATLAB数组
    % 将 log 空间的参数转换回实际尺度供实验
    actual_param = 10 .^ x_next;
    fprintf('建议评估参数: lambda_v=%.4f, lambda_a=%.4f, k1=%.4f, k2=%.4f (对数空间:%s)\n', ...
            actual_param(1), actual_param(2), actual_param(3), actual_param(4), mat2str(x_next));
    % 提示用户输入该参数下的真实性能指标J值
    J_val = input('请输入上述参数下测得的性能指标 J: ');
    % 将新观测转换为收益值（负误差）
    f_val = -J_val;
    % 更新GP模型和SafeOpt优化器的数据
    new_X = py.numpy.array(x_next_py);
    new_Y = py.numpy.array(py.list({f_val}));
    opt.add_new_data_point(new_X, new_Y);
    % 将新数据追加到本地数据记录
    X_data = [X_data; double(x_next)]; 
    Y_data = [Y_data; f_val];
    initial_J = [initial_J; J_val];  %#ok<AGROW> 记录所有J值以更新阈值时使用（可选）
    % 可选：根据需要更新安全阈值h（例如如果允许逐步提高要求，可动态改变h）
    % h = ... (本实验中保持初始h不变)

    % 保存检查点数据（当前迭代结果）
    save(checkpoint_file, 'X_data', 'Y_data', 'iteration', 'use_dynamic_beta');
    fprintf('迭代 %d 完成，数据已保存至检查点文件。\n', iteration);

    % 判断停止条件（例如最大迭代次数或用户手动停止）
    cont = input('继续下一次迭代吗？(Y/N): ', 's');
    if upper(cont) ~= 'Y'
        break;
    end
end

% === 实验结束：输出最优安全参数 ===
[x_best_py, f_best_py] = opt.get_maximum();  % 获取当前GP下估计的最优安全点
x_best = double(x_best_py);
f_best = double(f_best_py);
best_params = 10 .^ x_best;  % 转换回实际参数值
fprintf('\n>> 已完成所有迭代。\n');
fprintf('当前模型估计的最优安全参数为: lambda_v=%.4f, lambda_a=%.4f, k1=%.4f, k2=%.4f\n', ...
        best_params(1), best_params(2), best_params(3), best_params(4));
fprintf('对应的性能指标 J 估计值为 %.4f (收益 f = %.4f)。\n', -f_best, f_best);
