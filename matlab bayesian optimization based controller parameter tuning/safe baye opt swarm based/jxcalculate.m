clear; clc; close all;

%% 0. 文件名 ---------------------------------------------------------------
fname   = 'lya_fixedpath_data062615.xlsx';
[folder, base, ~] = fileparts(fname);
outName = fullfile(folder, [base '_jx.xlsx']);

%% 1. 读表 -----------------------------------------------------------------
T = readtable(fname);

%% 2. “上一行参考”配对 -----------------------------------------------------
T_ref = T(1:end-1 ,:);   % 路径 (k-1)
T_car = T(2:end   ,:);   % 车辆 (k)

% 2a. 横向误差 e_lat(k)
e_lat = (T_car.Y_current - T_ref.Y).*cos(T_car.theta) ...
      - (T_car.X_current - T_ref.X).*sin(T_car.theta);

% 2b. 航向误差 e_head(k)
e_head = T_ref.theta - T_car.theta_current ;   % 参考 - 车辆

%% 3. 有效行（11 … N-50）
validIdx = 6 : height(T)-10;         % 对应 T 的行号
lat_abs  = abs(e_lat (validIdx-1));   % e_lat 比 T 少 1 行
head_abs = abs(e_head(validIdx-1));

%% 4. 绝对值加和 + 中位数归一化
% ---- 横向 ----
sum_lat = nansum(lat_abs);
med_lat = median(lat_abs,'omitnan');
if med_lat == 0
    J_lat = sum_lat;
else
    J_lat = sum_lat / med_lat;
end

% ---- 航向 ----
sum_head = nansum(head_abs);
med_head = median(head_abs,'omitnan');
if med_head == 0
    J_head = sum_head;
else
    J_head = sum_head / med_head;
end

% ---- 综合指标 ----
w  = 0.10;                  % 权重
Jx = J_lat + w * J_head;

%% 5. 控制器参数 -----------------------------------------------------------
LAM_V = T.LAM_V(1);
LAM_A = T.LAM_A(1);
K1    = T.K1(1);
K2    = T.K2(1);
%% 6. 汇总表 (1×5) ---------------------------------------------------------
T_out = table(Jx, J_lat, J_head, LAM_V, LAM_A, K1, K2);

%% 7. 写文件 --------------------------------------------------------------
writetable(T_out, outName, 'WriteVariableNames', true);
fprintf('Summary saved ➜ %s\n', outName);
