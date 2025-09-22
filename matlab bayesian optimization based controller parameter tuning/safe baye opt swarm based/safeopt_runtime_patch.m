function safeopt_runtime_patch()
% 仅为 safeopt.gp_opt / safeopt.swarm 注入 dtype 容错：
% 1) 覆盖 np.zeros / np.zeros_like / np.empty / np.empty_like（走 core.numeric，避免递归）
% 2) 只在 safeopt 模块内部把 np.bool / np.float 矫正为内建类型（bool / float）
%    —— 不改全局 numpy，避免影响其他库

try
    % 依赖可导入
    py.importlib.import_module('numpy');
    py.importlib.import_module('safeopt');

    pycode = strjoin({
        'import numpy as _np',
        'import safeopt.gp_opt as _gpo',
        'try:',
        '    import safeopt.swarm as _sw',
        'except Exception:',
        '    _sw = None',
        '',
        'def _dtype_fix(dt, fallback=float):',
        '    try:',
        '        if dt is None: return fallback',
        '        if isinstance(dt, type) or isinstance(dt, _np.dtype): return dt',
        '    except Exception: pass',
        '    return fallback',
        '',
        '# ---- wrappers：统一走 core.numeric，避免递归 ----',
        'def _zeros_safe(shape, dtype=None, order="C"):',
        '    return _np.core.numeric.zeros(shape, dtype=_dtype_fix(dtype), order=order)',
        'def _zeros_like_safe(a, dtype=None, order="C", subok=True, shape=None):',
        '    base = getattr(getattr(a, "dtype", None), "type", float)',
        '    return _np.core.numeric.zeros_like(a, dtype=_dtype_fix(dtype, base),',
        '                                       order=order, subok=subok, shape=shape)',
        'def _empty_safe(shape, dtype=None, order="C"):',
        '    return _np.core.numeric.empty(shape, dtype=_dtype_fix(dtype), order=order)',
        'def _empty_like_safe(a, dtype=None, order="C", subok=True, shape=None):',
        '    base = getattr(getattr(a, "dtype", None), "type", float)',
        '    return _np.core.numeric.empty_like(a, dtype=_dtype_fix(dtype, base),',
        '                                       order=order, subok=subok, shape=shape)',
        '',
        'def _apply_numpy_fixes(mod):',
        '    """只在 safeopt 模块内部修复 numpy 别名与函数入口"""',
        '    try:',
        '        npns = getattr(mod, "np", None)',
        '        if npns is not None:',
        '            # ① 修复 dtype 入口',
        '            setattr(npns, "zeros",       _zeros_safe)',
        '            setattr(npns, "zeros_like",  _zeros_like_safe)',
        '            setattr(npns, "empty",       _empty_safe)',
        '            setattr(npns, "empty_like",  _empty_like_safe)',
        '            # ② 纠正旧别名为内建类型(重要)：避免 np.bool 变成 False',
        '            try: setattr(npns, "bool",  bool)',
        '            except Exception: pass',
        '            try: setattr(npns, "float", float)',
        '            except Exception: pass',
        '    except Exception:',
        '        pass',
        '',
        '_apply_numpy_fixes(_gpo)',
        'if _sw is not None:',
        '    _apply_numpy_fixes(_sw)',
        '',
        '# 调试输出（可注释）：确认别名已修正为类型而非布尔值',
        'try:',
        '    # print("DEBUG gpo np.bool:", _gpo.np.bool, type(_gpo.np.bool))',
        '    # print("DEBUG gpo np.float:", _gpo.np.float, type(_gpo.np.float))',
        '    pass',
        'except Exception:',
        '    pass'
    }, newline);

    py.builtins.exec(pycode, py.dict());
    fprintf('[safeopt_patch] 已修复 gp_opt/swarm：zeros/zeros_like/empty/empty_like + np.bool/np.float。\n');

catch ME
    warning('[safeopt_patch] 注入失败：%s', ME.message);
end
end
