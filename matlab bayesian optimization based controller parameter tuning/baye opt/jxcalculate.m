clear; clc; close all;

fname   = 'lya_fixedpath_data072932.xlsx';
[folder, base, ~] = fileparts(fname);
outName = fullfile(folder, [base '_jx.xlsx']);

T = readtable(fname);

T_ref = T(1:end-1 ,:);
T_car = T(2:end   ,:);

e_lat = (T_car.Y_current - T_ref.Y).*cos(T_car.theta) ...
      - (T_car.X_current - T_ref.X).*sin(T_car.theta);

e_head = T_ref.theta - T_car.theta_current ;

validIdx = 6 : height(T)-10;
lat_abs  = abs(e_lat (validIdx-1));
head_abs = abs(e_head(validIdx-1));

sum_lat = nansum(lat_abs);
% Fixed baseline medians from lya_fixedpath_data062601.xlsx.
med_lat = 0.292235607029988;
if med_lat == 0
    J_lat = sum_lat;
else
    J_lat = sum_lat / med_lat;
end

sum_head = nansum(head_abs);
med_head = 0.0345732664586631;
if med_head == 0
    J_head = sum_head;
else
    J_head = sum_head / med_head;
end

w  = 0.10;
Jx = J_lat + w * J_head;

LAM_V = T.LAM_V(1);
LAM_A = T.LAM_A(1);
K1    = T.K1(1);
K2    = T.K2(1);
T_out = table(Jx, J_lat, J_head, LAM_V, LAM_A, K1, K2);

writetable(T_out, outName, 'WriteVariableNames', true);
fprintf('Summary saved ➜ %s\n', outName);
