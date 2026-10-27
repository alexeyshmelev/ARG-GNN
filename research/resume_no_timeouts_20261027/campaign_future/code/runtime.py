"""Readable task execution, exact artifact references and throttled resource checks."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
from .deadlines import elapsed_limit_exceeded

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
TITLE = 'AI Auto Research Pipeline Test'
GPU_UUIDS = {
    3: 'GPU-df11de7f-7a15-ebe2-085d-9c7d907f3786',
    4: 'GPU-6cf64ff3-d8c2-9201-b2b8-683029625d71',
    5: 'GPU-3db85505-d061-e45d-d154-e7e8c1cad497',
}


def path(value):
    p = Path(value)
    p = p if p.is_absolute() else ROOT / p
    p = p.resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError('Project-contained path required: ' + str(p))
    return p


def save(filename, obj):
    p = path(filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, p)
    return ref(p)


def ref(filename):
    p = path(filename)
    digest = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b''):
            digest.update(block)
    return dict(path=str(p.relative_to(ROOT)), sha256=digest.hexdigest())


def read(reference):
    if ref(reference['path']) != reference:
        raise ValueError('Artifact changed: ' + reference['path'])
    return json.loads(path(reference['path']).read_text())


def scope_bytes(directory):
    if directory.is_symlink():
        raise ValueError("Mutable scope cannot be a symbolic link")
    if directory.is_file():
        return directory.stat().st_size
    total = 0
    for current, dirs, files in os.walk(directory, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(current) / d).is_symlink()]
        for name in files:
            p = Path(current) / name
            if not p.is_symlink():
                try:
                    total += p.stat().st_size
                except FileNotFoundError:
                    pass
    return total


class ResourceCheck:
    def __init__(self, task):
        self.task = task
        self.started = time.monotonic()
        self.next_check = 0.0

    def __call__(self, stage):
        now = time.monotonic()
        prospective_bytes = stage if type(stage) is int else 0
        if prospective_bytes < 0:
            raise ValueError('Nonnegative write request required')
        if now < self.next_check and prospective_bytes == 0:
            return
        self.next_check = now + 30
        if elapsed_limit_exceeded(now - self.started, self.task['max_seconds']):
            raise RuntimeError('Task wall time limit reached')
        size = scope_bytes(path(self.task['output_directory']))
        if size + prospective_bytes > self.task['max_output_bytes']:
            raise RuntimeError('Task output reservation exceeded')
        import psutil
        root = psutil.Process()
        rss = getattr(root.memory_full_info(), 'pss', root.memory_info().rss)
        # Prefer PSS so forked data-loader pages are not counted multiple times.
        for child in root.children(recursive=True):
            try:
                rss += getattr(child.memory_full_info(), 'pss', child.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if rss > self.task['max_rss_bytes']:
            raise RuntimeError('Task aggregate RSS reservation exceeded')
        save(path(self.task['output_directory']) / 'resource_progress.json', dict(
            time_unix=time.time(), stage=stage, elapsed_seconds=now-self.started,
            output_bytes=size, aggregate_pss_or_fallback_rss_bytes=rss, pid=os.getpid(),
            physical_gpu=os.environ.get('ARGGNN_PHYSICAL_GPU')))


def run_task(filename):
    task = json.loads(path(filename).read_text())
    for source in task.get('source_refs', []):
        if ref(source['path']) != source:
            raise ValueError('Source changed before execution: ' + source['path'])
    import setproctitle
    setproctitle.setproctitle(TITLE)
    output = path(task['output_directory'])
    check = ResourceCheck(task)
    check('starting')
    call = task['call']
    if call.get('factory'):
        factory = call['factory']
        module = importlib.import_module(factory['module'])
        concrete = getattr(module, factory['function'])(**factory['kwargs'])
        # Interpreter cannot change after the parent selected the native runtime.
        if path(concrete['python']) != path(sys.executable):
            raise ValueError('Factory interpreter differs from scheduled runtime')
        call = dict(module=concrete['module'], function=concrete['function'], spec=concrete['spec'])
        save(output / 'concrete_call.json', call)
        for source in concrete.get('source_refs', []):
            if ref(source['path']) != source:
                raise ValueError('Concrete source changed: ' + source['path'])
    module = importlib.import_module(call['module'])
    function = getattr(module, call['function'])
    kwargs = dict(resource_check=check)
    if 'execution_check' in inspect.signature(function).parameters:
        kwargs['execution_check'] = lambda: check('execution_check')
    result = function(call['spec'], **kwargs)
    check.next_check = 0
    check('completed')
    save(output / 'result.json', dict(schema='paper_campaign_result_v1', value=result,
         finished_unix=time.time(), scientific_success=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    args = parser.parse_args()
    run_task(args.task)


if __name__ == '__main__':
    main()
