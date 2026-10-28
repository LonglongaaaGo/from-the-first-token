# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Supplied loading and numeric assertions for the prepared course exercises."""
from pathlib import Path
import importlib.util
import sys
import torch
_MODULES={}
def load(filename,symbol):
    if filename not in _MODULES:
        path=Path(__file__).with_name(filename)
        spec=importlib.util.spec_from_file_location('onelong_'+path.stem,path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module
        spec.loader.exec_module(module)
        _MODULES[filename]=module
    return getattr(_MODULES[filename],symbol)
def close(actual,expected,rtol=2e-5,atol=2e-6):
    torch.testing.assert_close(actual,expected,rtol=rtol,atol=atol,check_dtype=False)
def run(label,check):
    try:check()
    except NotImplementedError as error:
        print(f'Lesson {label}: learner TODO or an unfinished prerequisite. {error}')
        raise SystemExit(1) from None
    print(f'Lesson {label} checks passed.')
