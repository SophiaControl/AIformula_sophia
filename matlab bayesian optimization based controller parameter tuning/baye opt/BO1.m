% bo_manual_step_rebuild.m  ——  R2024b 适用 / 无 addpoints
% 每轮：用历史重建 BO → 拿 NextPoint → 人工输 J → 存盘续跑
% 初始化 J0 采用软惩罚：J0 = J_raw + 7000*(1 - q)，q∈{0(severe), 0.6(half), 0.8(near)}
% 注：与 SafeOpt 版本保持一致；保留你注释掉的那一行 raw（不参与初始化）。

clear; clc; close all;
ckpt = 'checkpointmatbo.mat';

%% ===== A. 恢复或初始化 ================================================
if exist(ckpt,'file')
    load(ckpt,'historyTbl','iter','N0','NMAX','optimVars');
    fprintf('>>> 已加载 checkpointmatbo：完成 %d / %d 次评估。\n', iter, NMAX);
else
    % --- 基本设置 ---
    N0   = 15;      % 先验条数
    NMAX = 35;      % 总评估次数
    optimVars = [
        optimizableVariable('lambda_v',[1e-4 0.5 ],'Transform','log')
        optimizableVariable('lambda_a',[1e-3 1.5 ],'Transform','log')
        optimizableVariable('k1'      ,[1e-2 10   ],'Transform','log')
        optimizableVariable('k2'      ,[1e-1 100 ],'Transform','log')];

    % --- 先验参数（与之前一致；第 11 行保留注释） ---
    raw = [ ...
      0.0200  0.2500  0.7000   50.0000 ;   % 1  success
      0.0002  0.0020  0.0200    0.3000 ;   % 2  fail-severe (q=0)
      0.4500  1.2000  8.0000   80.0000 ;   % 3  fail-half   (q=0.6)
      0.0240  0.3000  0.8400   55.0000 ;   % 4  success
      0.0180  0.2000  0.6000   45.0000 ;   % 5  fail-near   (q=0.8)
      0.0150  0.2800  0.6500   60.0000 ;   % 6  success
      0.0220  0.2200  0.7500   48.0000 ;   % 7  success
      0.0280  0.2700  0.5000   58.0000 ;   % 8  success
      0.0170  0.2300  0.8000   40.0000 ;   % 9  success
      0.0090  0.0150  4.5000    2.5000 ;   % 10 fail-severe (q=0)
      0.3000  0.0200  0.5000    5.0000 ; % 11（保留注释）
      0.0005  1.4000  0.0200   90.0000 ;   % 11' success  （行号 12）
      0.0400  0.1000  6.0000   60.0000 ;   % 12 fail-half   (q=0.6)
      0.0800  0.4000  9.0000   10.0000 ;   % 13 fail-severe (q=0)
      0.0050  0.6000  0.0500   30.0000 ];  % 14 success

    % --- J0（按 J0 = J_raw + 7000*(1-q) 计算并与 raw 对齐；第 11 行无） ---
    J0 = [
      2076.345804 ;   % 1  success → J0 = 2076.345804
      7638.2468534;   % 2  severe  → 638.2468534 + 7000
      6005.663709 ;   % 3  half    → 3205.663709 + 2800
      2331.849311 ;   % 4  success
      3909.018205 ;   % 5  near    → 4509.018205 + 1400
      2226.183768 ;   % 6  success
      2019.882725 ;   % 7  success
      2171.833048 ;   % 8  success
      1887.649908 ;   % 9  success
      7336.2574263;   % 10 severe  → 336.2574263 + 7000
      2750.943025 ;
      2714.915107 ;   % 11' success（对应 raw 第 12 行）
      6116.859428 ;   % 12 half    → 
      7490.1909914;   % 13 severe  → 490.1909914 + 7000
      2486.898993 ];  % 14 success

    initTbl = array2table(raw,'VariableNames',{'lambda_v','lambda_a','k1','k2'});
    historyTbl = [table((1:N0)','VariableNames',{'iter'}), initTbl, table(J0,'VariableNames',{'J'})];
    iter = N0;

    save(ckpt,'historyTbl','iter','N0','NMAX','optimVars','-v7.3');
    fprintf('>>> 首次初始化完成并写入 checkpointmatbo。\n');
end

%% ===== B. 主循环：重建 BO → NextPoint → 人工输 J =====================
for iter = iter+1 : NMAX
    fprintf('\n========== 迭代 %d / %d ==========\n', iter, NMAX);

    thisX = historyTbl{:, {'lambda_v','lambda_a','k1','k2'}};
    thisJ = historyTbl.J;

    BO = bayesopt(@(~)nan, optimVars, ...
        'InitialX',          array2table(thisX,'VariableNames',{'lambda_v','lambda_a','k1','k2'}), ...
        'InitialObjective',  thisJ, ...
        'MaxObjectiveEvaluations', size(thisX,1), ...
        'IsObjectiveDeterministic', true, ...
        'AcquisitionFunctionName', 'probability-of-improvement', ...
        'Verbose', 0);

    Xnext = BO.NextPoint;
    disp('BayesOpt 推荐参数：'); disp(Xnext);

    J = input('请输入实测 J，然后回车： ');
    while isempty(J) || ~isscalar(J) || ~isnumeric(J)
        J = input('无效！请重新输入标量 J： ');
    end

    newRow = [ table(iter,'VariableNames',{'iter'}), Xnext, table(J,'VariableNames',{'J'}) ];
    historyTbl = [historyTbl ; newRow];

    assignin('base','historyTbl',historyTbl);
    save(ckpt,'historyTbl','iter','N0','NMAX','optimVars','-v7.3');
    fprintf('>> 已保存 checkpointmatbo（迭代 %d，J = %.4f）。可 Ctrl-C 中断。\n', iter, J);
end

%% ===== C. 结束：输出最优 ===============================================
[bestJ, idx] = min(historyTbl.J);
fprintf('\n=========== 优化完成 ===========\n');
fprintf('最小 J = %.4f  (发生在第 %d 次)\n', bestJ, historyTbl.iter(idx));
disp('对应参数：'); disp(historyTbl(idx,2:5));
