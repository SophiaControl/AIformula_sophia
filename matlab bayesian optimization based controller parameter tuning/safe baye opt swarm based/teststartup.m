pyenv('Version','C:\Users\17396\AppData\Local\Programs\Python\Python310\python.exe');
np = py.importlib.import_module('numpy');
% 兼容补丁（numpy 老别名）
py.setattr(np,'float',py.builtins.float);
py.setattr(np,'int',  py.builtins.int);
py.setattr(np,'bool', py.builtins.bool);
