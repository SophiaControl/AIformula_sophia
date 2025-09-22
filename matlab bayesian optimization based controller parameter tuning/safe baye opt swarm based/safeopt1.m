% safebo_manual_step_rebuild_sparse_seed.m
% 在“可运行”的 SafeOptSwarm 版本上：仅新增“初始安全集合稀疏化以选 seed”
% - f = -J；J<=3500 ⇔ f>=-3500；beta=2.0；log10 空间；swarm_size=100（按你之前偏好）
% - 稀疏化：log10 空间 L2 阈值 tau=0.2，只用于选 seed；其余历史仍全部回放

clear; clc; close all;
ckpt = 'checkpoint_safeopt.mat';

%% ===== 0) Python 环境与兼容补丁（保持你之前的一致做法） ===================
info = pyenv;
fprintf('Python: %s\n', info.Executable);

% 旧别名（你的环境需要，保留）
np = py.importlib.import_module('numpy');
py.setattr(np,'float',py.builtins.float);
py.setattr(np,'int',  py.builtins.int);
py.setattr(np,'bool', py.builtins.bool);

% collections.Sequence 兼容（Py3.10+）
collections     = py.importlib.import_module('collections');
collections_abc = py.importlib.import_module('collections.abc');
py.setattr(collections,'Sequence', py.getattr(collections_abc,'Sequence'));

% —— 你的最小补丁（不要删除）
if exist('safeopt_runtime_patch','file') == 2
    safeopt_runtime_patch();  % 仅修 safeopt 内部 zeros/zeros_like/empty/empty_like + np.bool/np.float
end

%% ===== 1) 恢复或初始化（仅保留 J<=3500 的初始样本） =======================
if exist(ckpt,'file')
    load(ckpt,'historyTbl','iter','N0','NMAX');
    fprintf('>>> 已加载 checkpoint_safeopt：完成 %d / %d 次评估。\n', iter, NMAX);
else
    % --- 目标总评估次数 ---
    NMAX = 30;

    % --------- 初始参数&J（下方仅保留 J<=3500 的为真正初始化；其余全部注释保留） ---------
    % idx  raw([lambda_v,lambda_a,k1,k2])                  J
    %  1   [0.0200, 0.2500, 0.7000,  50.0000]        ->  2076.345804   (安全✓)
    %  2   [0.0002, 0.0020, 0.0200,   0.3000]        ->  40000         (不安全✗)
    %  3   [0.4500, 1.2000, 8.0000,  80.0000]        ->  13205.66371   (不安全✗)
    %  4   [0.0240, 0.3000, 0.8400,  55.0000]        ->  2331.849311   (安全✓)
    %  5   [0.0180, 0.2000, 0.6000,  45.0000]        ->  4509.018205   (不安全✗)
    %  6   [0.0150, 0.2800, 0.6500,  60.0000]        ->  2226.183768   (安全✓)
    %  7   [0.0220, 0.2200, 0.7500,  48.0000]        ->  2019.882725   (安全✓)
    %  8   [0.0280, 0.2700, 0.5000,  58.0000]        ->  2171.833048   (安全✓)
    %  9   [0.0170, 0.2300, 0.8000,  40.0000]        ->  1887.649908   (安全✓)
    % 10   [0.0090, 0.0150, 4.5000,   2.5000]        ->  40000         (不安全✗)
    % 11   % [0.3000, 0.0200, 0.5000,   5.0000]      ->  % 40000        (原本就注释，保留)
    % 12   [0.0005, 1.4000, 0.0200,  90.0000]        ->  2714.915107   (安全✓)
    % 13   [0.0400, 0.1000, 6.0000,  60.0000]        ->  13316.85943   (不安全✗)
    % 14   [0.0800, 0.4000, 9.0000,  10.0000]        ->  40000         (不安全✗)
    % 15   [0.0050, 0.6000, 0.0500,  30.0000]        ->  2486.898993   (安全✓)

    % —— 实际用于初始化的“安全子集”（仅 J<=3500）：
    raw = [ ...
      0.0200  0.2500  0.7000   50.0000 ;   % 1  ✓
      0.0240  0.3000  0.8400   55.0000 ;   % 4  ✓
      0.0150  0.2800  0.6500   60.0000 ;   % 6  ✓
      0.0220  0.2200  0.7500   48.0000 ;   % 7  ✓
      0.0280  0.2700  0.5000   58.0000 ;   % 8  ✓
      0.0170  0.2300  0.8000   40.0000 ;   % 9  ✓
      0.3000  0.0200  0.5000    5.0000 ;
      0.0005  1.4000  0.0200   90.0000 ;   % 12 ✓
      0.0050  0.6000  0.0500   30.0000 ]; % 15 ✓

    J0 = [ ...
      2076.345804;   % 1
      2331.849311;   % 4
      2226.183768;   % 6
      2019.882725;   % 7
      2171.833048;   % 8
      1887.649908;   % 9
      2750.943025 ;
      2714.915107;   % 12
      2486.898993 ]; % 15

    N0   = size(raw,1);   % 9
    NMAX = 30;

    initTbl    = array2table(raw,'VariableNames',{'lambda_v','lambda_a','k1','k2'});
    historyTbl = [ table((1:N0)','VariableNames',{'iter'}), initTbl, table(J0,'VariableNames',{'J'}) ];
    iter = N0;

    save(ckpt,'historyTbl','iter','N0','NMAX','-v7.3');
    fprintf('>>> 首次初始化完成并写入 checkpoint_safeopt（安全起点 N0=%d）。\n', N0);
end

%% ===== 2) SafeOpt 配置（f = -J；thr = -3500；内部 log10 空间） ============
D            = 4;
beta_val     = 2.0;        % 默认保守度
swarm_size   = 150;        % 按你之前偏好
USE_SIM_TRUE_FUNC = false; % 实验：手动输入 J

% 变量边界（外部原始空间，与 BayesOpt 一致）
bounds_ext = [1e-4, 0.5 ;   % lambda_v
              1e-3, 1.5 ;   % lambda_a
              1e-2, 10  ;   % k1
              1e-1, 100 ];  % k2

% —— 内部使用 log10 空间（仅是参数坐标变换）
lb_int = log10(bounds_ext(:,1))';
ub_int = log10(bounds_ext(:,2))';
bounds_lit_int = sprintf('[(%g,%g),(%g,%g),(%g,%g),(%g,%g)]', ...
    lb_int(1),ub_int(1), lb_int(2),ub_int(2), lb_int(3),ub_int(3), lb_int(4),ub_int(4));

% 历史：外→内坐标；J→f（最简单）：
X_ext = historyTbl{:, {'lambda_v','lambda_a','k1','k2'}};
X_int = log10(X_ext);
J_hist = historyTbl.J(:);
f_hist = -J_hist;           % 关键：f = -J

% 安全阈值：J<=3500 ⇔ f>=-3500
thr = -3500;

%% ===== 2.1 初始安全集合的“稀疏化”用来选 seed（不丢历史） ==================
% 目标：在 log10 空间用 L2 阈值 tau 筛出更分散的子集，然后从该子集里选 seed
tau_sparse = 0.20;                  % 经验值（每维约 0.2 decade 的量级）
idx_sparse = greedy_sparsify_l2(X_int, tau_sparse);

% 从“稀疏子集”里选 seed：f 最大（J 最小）
[~, pos] = max(f_hist(idx_sparse));
idx_seed = idx_sparse(pos);

x0_int = X_int(idx_seed, :);
y0_f   = f_hist(idx_seed);

fprintf('初始安全集合稀疏化：保留 %d/%d；选 seed = #%d（J=%.4f）。\n', ...
    numel(idx_sparse), size(X_int,1), idx_seed, -y0_f);

%% ===== 3) Python 侧：创建 GPy + SafeOptSwarm，并回放历史 ================
noise_var_init = 50^2;  % = 2500，与 J 量级匹配

lines_create = { ...
    'import numpy as np, GPy, safeopt', ...
    sprintf('bounds = %s', bounds_lit_int), ...
    sprintf('D = %d', D), ...
    sprintf('x0 = np.array([[%g,%g,%g,%g]], dtype=float)', x0_int(1),x0_int(2),x0_int(3),x0_int(4)), ...
    sprintf('y0 = np.array([[%g]], dtype=float)', y0_f), ...
    'kernel = GPy.kern.RBF(input_dim=D, variance=1.0, lengthscale=np.ones(D), ARD=True)', ...
    sprintf('gp = GPy.models.GPRegression(x0, y0, kernel, noise_var=%g)', noise_var_init), ...
    sprintf('opt = safeopt.SafeOptSwarm(gp, fmin=[%g], bounds=bounds, beta=%g, swarm_size=%d)', thr, beta_val, swarm_size) ...
    };
pycode_create = sprintf('%s\n', lines_create{:});
pyrun(pycode_create, 'opt');
fprintf('SafeOptSwarm 已创建：f=-J，thr=%g（J<=3500），log10 参数空间，beta=%.2f，swarm=%d，noise_var=%g。\n', ...
    thr, beta_val, swarm_size, noise_var_init);

% 回放历史（仍然“全部回放”，不丢信息；只是在 seed 的选择上用稀疏化）
for i = 1:size(X_int,1)
    if i == idx_seed, continue; end
    xi = X_int(i,:);
    yi = f_hist(i);
    lines_add = { ...
        'import numpy as np', ...
        'x_np = np.asarray(xm, dtype=float).reshape(1,-1)', ...
        'y_np = np.array([[float(ym)]], dtype=float)', ...
        'opt.add_new_data_point(x_np, y_np)', ...
        'ret = None' };
    pycode_add = sprintf('%s\n', lines_add{:});
    tmp = pyrun(pycode_add, 'ret', 'xm', xi, 'ym', yi); %#ok<NASGU>
end
fprintf('已回放历史（安全）%d 条观测（包含稀疏外的点，以提高 GP 精度）。\n', size(X_int,1));

%% ===== 4) 主循环：每步取建议点 → 实验输入 J → 回写 → 每步学超参 ==========
for t = (historyTbl.iter(end)+1) : NMAX
    fprintf('\n========== 迭代 %d / %d ==========\n', t, NMAX);

    % 4.1 取下一点（log10 空间）
    lines_next = { ...
        'import numpy as np', ...
        'x_next = opt.optimize()', ...
        'x_list = np.asarray(x_next, dtype=float).ravel().tolist()' };
    pycode_next = sprintf('%s\n', lines_next{:});
    x_next_list = pyrun(pycode_next, 'x_list');
    x_next_int  = pylist_to_rowdouble(x_next_list);     % 1x4（内部）
    x_next_ext  = 10.^x_next_int;                       % 外部参数

    fprintf('SafeOpt 推荐参数（外部原始空间）：\n');
    fprintf('  lambda_v = %.8g\n  lambda_a = %.8g\n  k1 = %.8g\n  k2 = %.8g\n', x_next_ext(1),x_next_ext(2),x_next_ext(3),x_next_ext(4));

    % 4.2 获取实测 J（实验输入）
    if false
        warning('模拟模式仅用于占位测试。');
        J_meas = 2000 + 500*randn;   % 占位
    else
        J_meas = input('请输入该点的实测 J，然后回车： ');
        while isempty(J_meas) || ~isscalar(J_meas) || ~isnumeric(J_meas)
            J_meas = input('无效！请重新输入标量 J： ');
        end
    end
    f_meas = -J_meas;   % 关键：f = -J

    % 4.3 回写 SafeOpt（内部 log10 空间 + f）
    lines_add = { ...
        'import numpy as np', ...
        'x_np = np.asarray(xm, dtype=float).reshape(1,-1)', ...
        'y_np = np.array([[float(ym)]], dtype=float)', ...
        'opt.add_new_data_point(x_np, y_np)', ...
        'ret = None' };
    pycode_add = sprintf('%s\n', lines_add{:});
    tmp = pyrun(pycode_add, 'ret', 'xm', x_next_int, 'ym', f_meas); %#ok<NASGU>

    % 4.4 每步学习核超参（与原代码一致）
    pyrun([ ...
        "gp['rbf.variance'].constrain_bounded(1e-8, 1e8, warning=False)", newline, ...
        "gp['rbf.lengthscale'].constrain_bounded(1e-2, 2, warning=False)", newline, ...
        "gp['Gaussian_noise.variance'].constrain_bounded(1e-6, 200.0**2, warning=False)" ...
    ]);
    pyrun("gp.optimize(max_iters=80, messages=False)");

    % 4.5 追加到历史表（外部参数 + J）
    newRow = [ ...
        table(t,'VariableNames',{'iter'}), ...
        array2table(x_next_ext,'VariableNames',{'lambda_v','lambda_a','k1','k2'}), ...
        table(J_meas,'VariableNames',{'J'}) ];
    historyTbl = [historyTbl ; newRow];

    % 4.6 保存 checkpoint
    assignin('base','historyTbl',historyTbl);
    iter = t;  %#ok<NASGU>
    save(ckpt,'historyTbl','iter','N0','NMAX','-v7.3');
    fprintf('>> 已保存 checkpoint_safeopt（迭代 %d，J = %.4f）。可 Ctrl-C 中断。\n', t, J_meas);
end

%% ===== 5) 输出最优（按 J 最小） & SafeOpt 安全域最优（换算） =============
[bestJ, idx] = min(historyTbl.J);
fprintf('\n=========== 优化完成（SafeOptSwarm） ===========\n');
fprintf('最小 J = %.4f  (发生在第 %d 次)\n', bestJ, historyTbl.iter(idx));
disp('对应参数：'); disp(historyTbl(idx,2:5));

% SafeOpt 内部最优（f 最大）并换算回 J
lines_best = { ...
    'import numpy as np', ...
    'bx, by = opt.get_maximum()', ...
    'bx_list = np.asarray(bx, dtype=float).ravel().tolist()' };
pycode_best = sprintf('%s\n', lines_best{:});
bx_list = pyrun(pycode_best, 'bx_list');
best_x_int  = pylist_to_rowdouble(bx_list);
best_f      = pyrun('bx, by = opt.get_maximum(); float(by)', 'by'); best_f = double(best_f);

best_x_ext  = 10.^best_x_int;
best_J_est  = -best_f;   % f=-J
fprintf('（SafeOpt 安全域）f 最大对应：估计 J = %.6f，x = [%s]\n', best_J_est, num2str(best_x_ext,'%.6g '));

%% ===== 附：小工具 =====
function v = pylist_to_rowdouble(py_list_obj)
    % Python list -> MATLAB 1xN double
    v = cellfun(@double, cell(py_list_obj));
    v = reshape(v, 1, []);
end

function idx_sel = greedy_sparsify_l2(X, tau)
    % X: N×D（已在 log10 空间）；tau: L2 距离阈值
    % 简单贪心：按原顺序遍历（如需“J 最小优先”，可在外部按 f_hist 排序后传入）
    N = size(X,1);
    sel = false(1,N);
    for i = 1:N
        if ~any(sel)
            sel(i) = true; continue;
        end
        d = sqrt(sum((X(sel,:) - X(i,:)).^2, 2));
        if all(d >= tau)
            sel(i) = true;
        end
    end
    idx_sel = find(sel);
end
