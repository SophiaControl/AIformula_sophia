exe = char(pyenv().Executable);

system(['"', exe, '" -m pip show safeopt']);

 system(['"', exe, '" -m pip install --upgrade safeopt GPy numpy scipy matplotlib']);
