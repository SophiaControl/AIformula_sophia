pyrun(strjoin({
    'import numpy as np',
    'import collections, collections.abc as cabc',
    'setattr(np, "float", float)',
    'setattr(np, "int", int)',
    'setattr(np, "bool", bool)',
    'setattr(collections, "Sequence", getattr(cabc, "Sequence"))',
    'ret=True'
}, newline), 'ret');
