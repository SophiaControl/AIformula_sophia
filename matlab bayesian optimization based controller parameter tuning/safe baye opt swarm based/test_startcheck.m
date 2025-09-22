function startcheck()
% 环境与依赖自检 + 兼容补丁（对 numpy/collections 旧别名）
% 说明：
% - 不再用 py.setattr（会报“没有名为 'setattr' 的模块或函数”）
% - 改用 pybuiltin('setattr', ...) 与 py.builtins.getattr
% - 若 Python 已加载，建议先 terminate(pyenv) 再设 pyenv

%% 0) 可选：指定并重启 Python 解释器
pyExe = "C:\Users\17396\AppData\Local\Programs\Python\Python310\python.exe";  % 按需改
pe = pyenv;
if pe.Status == "Loaded" && ~strcmpi(string(pe.Executable), pyExe)
    % 已加载但不是这套解释器 → 先终止再切版本
    terminate(pyenv);
end
pe = pyenv('Version', pyExe);
fprintf('Python: %s\n', pe.Executable);

%% 1) 刷新 import 缓存（可选）
try
    py.importlib.invalidate_caches();
catch
end

%% 2) 打补丁：numpy 旧别名（np.float/np.int/np.bool），collections.Sequence
% 这段就是你要的“兼容补丁”，只是把 py.setattr 改成了兼容写法
try
    np = py.importlib.import_module('numpy');
    % 把 numpy 去掉的旧别名映射到内建类型
    pybuiltin('setattr', np, 'float', py.builtins.float);
    pybuiltin('setattr', np, 'int',   py.builtins.int);
    pybuiltin('setattr', np, 'bool',  py.builtins.bool);
catch ME
    warning("numpy 兼容补丁跳过：%s", ME.message);
end

try
    collections     = py.importlib.import_module('collections');
    collections_abc = py.importlib.import_module('collections.abc');
    seq_cls = py.builtins.getattr(collections_abc, 'Sequence');
    pybuiltin('setattr', collections, 'Sequence', seq_cls);
catch ME
    warning("collections 兼容补丁跳过：%s", ME.message);
end

%% 3) 打印关键依赖版本（用 importlib.metadata，避免 __version__ 缺失）
mods = {'numpy','scipy','matplotlib','future','GPy','safeopt'};
fprintf('\n依赖版本：\n');
for k = 1:numel(mods)
    name = mods{k};
    v = pyver(name);
    fprintf('%-11s %s\n', name, v);
end

fprintf('\n检查完成。\n');

end

function v = pyver(pkg)
% 用 importlib.metadata 读取三方包版本；没有则返回占位
try
    md = py.importlib.import_module('importlib.metadata');
    v  = string(md.version(pkg));
catch
    v = "<not installed>";
end
end
