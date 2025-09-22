exe = char(pyenv().Executable);     % ① 把 string → char

% 1) 查看 safeopt 是否已装
system(['"', exe, '" -m pip show safeopt']);

% % 2) 若未安装，就执行安装 / 升级
 system(['"', exe, '" -m pip install --upgrade safeopt GPy numpy scipy matplotlib']);
