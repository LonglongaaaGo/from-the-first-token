# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Verify the distributed files, fresh exercises and runnable CPU demonstration."""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

# Import checks must not change the package being verified.
sys.dont_write_bytecode = True


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_inventory(root):
    manifest = json.loads((root / 'content-manifest.json').read_text())
    actual = set()
    for directory, folders, filenames in os.walk(root):
        parent = Path(directory)
        # A Git clone, documented virtual environment and local reports are
        # generated reader state, not part of the distributed file inventory.
        folders[:] = [name for name in folders if name != '__pycache__'
                      and not (parent == root and name in {'.git', '.venv', 'outputs'})]
        for name in folders + filenames:
            path = parent / name
            if path.is_symlink():
                raise ValueError('Package content must not contain symbolic links')
            if path.is_file():
                actual.add(path.relative_to(root).as_posix())
    expected = set(manifest['files']) | {'content-manifest.json', 'SHA256SUMS'}
    if actual != expected:
        raise ValueError('Package inventory differs from the recorded release')
    for relative, item in manifest['files'].items():
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Out-of-scope manifest path')
        if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError('File content differs: ' + relative)
    sums = dict(line.split('  ', 1)[::-1]
                for line in (root / 'SHA256SUMS').read_text().splitlines())
    expected_sums = {name: item['sha256'] for name, item in manifest['files'].items()}
    expected_sums['content-manifest.json'] = digest(root / 'content-manifest.json')
    if sums != expected_sums:
        raise ValueError('Release checksum list differs from the manifest')
    return manifest


def check_runtime(root):
    import numpy
    import torch

    exercise_dir = root / 'exercises'
    paths = sorted(exercise_dir.glob('[0-9][0-9]_[0-9][0-9]_*.py'))
    if len(paths) != 70:
        raise ValueError('Expected 70 exercise scaffolds')
    sys.path.insert(0, str(exercise_dir))
    assigned = 0
    for index, path in enumerate(paths):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                body = body[1:]
            if len(body) == 1 and isinstance(body[0], ast.Raise):
                error = body[0].exc
                if isinstance(error, ast.Call) and isinstance(error.func, ast.Name) and error.func.id == 'NotImplementedError':
                    assigned += 1
        name = '_public_check_' + str(index)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    if assigned != 76:
        raise ValueError('Expected 76 unimplemented assigned functions')
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['OMP_NUM_THREADS'] = '1'
    command = [sys.executable, '-B', str(exercise_dir / '01_01_text_to_tokens.py'), '--check', 'encode']
    todo = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, timeout=30)
    if todo.returncode == 0 or 'NotImplementedError' not in todo.stderr:
        raise ValueError('First exercise must report its documented unimplemented starting point')
    generated = subprocess.run([sys.executable, '-B', str(root / 'examples/generate_teaching_data.py'), '--check'],
                               cwd=root, env=env, text=True, capture_output=True, timeout=30)
    if generated.returncode:
        raise ValueError('Teaching data reproducibility failed: ' + generated.stderr[-1500:])
    demo = subprocess.run([sys.executable, '-B', str(root / 'examples/train_bigram.py'), '--check'],
                          cwd=root, env=env, text=True, capture_output=True, timeout=60)
    if demo.returncode:
        raise ValueError('CPU demonstration failed: ' + demo.stderr[-1500:])
    return {'imported_exercises': len(paths), 'assigned_todos': assigned,
            'first_exercise_expected_todo': True,
            'teaching_data_reproduced': True,
            'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': numpy.__version__,
            'cpu_example_output': demo.stdout.strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory-only', action='store_true', help='Check files without importing dependencies')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = check_inventory(root)
    result = {'status': 'PASS', 'release': manifest['version'], 'files': len(manifest['files']) + 2}
    if not args.inventory_only:
        result.update(check_runtime(root))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
