function startcheck()

pyExe = "C:\Users\17396\AppData\Local\Programs\Python\Python310\python.exe";
pe = pyenv;
if pe.Status == "Loaded" && ~strcmpi(string(pe.Executable), pyExe)
    terminate(pyenv);
end
pe = pyenv('Version', pyExe);
fprintf('Python: %s\n', pe.Executable);

try
    py.importlib.invalidate_caches();
catch
end

try
    np = py.importlib.import_module('numpy');
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
try
    md = py.importlib.import_module('importlib.metadata');
    v  = string(md.version(pkg));
catch
    v = "<not installed>";
end
end
