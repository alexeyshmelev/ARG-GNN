"""Authenticated, process-local bindings for qualified own preprocessing.

All targets are checked before the first assignment. Receipts identify the
owner, original callable, replacement and any fixed execution parameters.
"""
from functools import partial
import hashlib
import importlib
import inspect
from pathlib import Path

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
PARALLEL_RESERVE_BYTES = 8 * 2**30


def source_refs():
    base = ROOT / 'research/preprocessing_optimization_20261026'
    names = ('integration/hooks.py', 'integration/treeloom_resume.py',
             'tree/kernels.cpp', 'tree/kernels.py', 'tree/libstructural.so',
             'tree/build.sh', 'tree/observations.py', 'tree/process_observations.py',
             'tree/episodes.py', 'tree/cut.py', 'observed/parser.py', 'observed/adapter.py')
    return [_source_ref(base / name) for name in sorted(names)]


def _source_ref(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Project-contained callable source required')
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _callable_ref(function):
    keywords = None
    if isinstance(function, partial):
        if function.args:
            raise ValueError('Only explicit keyword execution configuration is supported')
        keywords, function = dict(function.keywords), function.func
    if not inspect.isfunction(function):
        raise ValueError('An ordinary source-defined preprocessing function is required')
    result = dict(module=function.__module__, qualname=function.__qualname__,
                  source_ref=_source_ref(inspect.getsourcefile(function)))
    if keywords is not None:
        result['partial_keywords'] = keywords
    return result


def _process4_encode_block(*args, max_rss_bytes, resource_check, **kwargs):
    """Admit four CPU children inside the existing complete-parent reservation."""
    import psutil
    from research.preprocessing_optimization_20261026.tree import process_observations
    resource_check('TreeLoom_admit_process4_encoder')
    process = psutil.Process()
    aggregate = 0
    for child in [process] + process.children(recursive=True):
        try:
            memory = child.memory_full_info()
            aggregate += getattr(memory, 'pss', memory.rss)
        except psutil.NoSuchProcess:
            continue
    if aggregate + PARALLEL_RESERVE_BYTES > max_rss_bytes:
        raise ValueError('Aggregate parent plus eight GiB child reserve exceeds the admitted task limit')
    return process_observations.encode_block(*args, **kwargs, resource_check=resource_check,
        preparation_workers=4, max_parallel_bytes=PARALLEL_RESERVE_BYTES,
        tree_chunk_size=128, max_chunk_bytes=64 * 2**20)


def _install(specifications):
    prepared, receipt = [], []
    for owner_name, attribute, original_name, original_attribute, replacement in specifications:
        owner = importlib.import_module(owner_name)
        original_module = importlib.import_module(original_name)
        actual = getattr(owner, attribute)
        original = getattr(original_module, original_attribute)
        before = _callable_ref(actual)
        expected_path = _source_ref(original_module.__file__)
        if (actual is not original or before['module'] != original_name
                or before['qualname'] != original_attribute or before['source_ref'] != expected_path):
            raise ValueError('Unexpected preprocessing binding: ' + owner_name + '.' + attribute)
        prepared.append((owner, attribute, actual, replacement))
        receipt.append(dict(owner_module=owner_name, attribute=attribute,
                            owner_source_ref=_source_ref(owner.__file__), before=before,
                            after=_callable_ref(replacement)))
    # Roll back only our own assignments if an installation/verification fails.
    try:
        for owner, attribute, before, replacement in prepared:
            setattr(owner, attribute, replacement)
        verify_bindings(receipt)
    except BaseException:
        for owner, attribute, before, replacement in prepared:
            setattr(owner, attribute, before)
        raise
    return receipt


def verify_bindings(receipt):
    if not isinstance(receipt, list) or not receipt:
        raise ValueError('Explicit installed binding receipt required')
    seen = set()
    for row in receipt:
        key = (row['owner_module'], row['attribute'])
        if key in seen:
            raise ValueError('Duplicate installed binding')
        seen.add(key)
        owner = importlib.import_module(row['owner_module'])
        actual = getattr(owner, row['attribute'])
        descriptor = row['after']
        expected = getattr(importlib.import_module(descriptor['module']), descriptor['qualname'])
        target = actual.func if isinstance(actual, partial) else actual
        if (target is not expected or _callable_ref(actual) != descriptor
                or _source_ref(owner.__file__) != row['owner_source_ref']):
            raise ValueError('Installed preprocessing binding changed: ' + '.'.join(key))
    return True


def install_tree(preparation_workers=1, *, max_rss_bytes=None):
    from research.preprocessing_optimization_20261026.tree import observations, episodes, cut
    if type(preparation_workers) is not int or preparation_workers not in (1, 4):
        raise ValueError('Only qualified serial or four-process preprocessing is supported')
    encoder = observations.encode_block
    if preparation_workers == 4:
        if type(max_rss_bytes) is not int or max_rss_bytes <= PARALLEL_RESERVE_BYTES:
            raise ValueError('Explicit aggregate task memory limit required for four CPU workers')
        encoder = partial(_process4_encode_block, max_rss_bytes=max_rss_bytes)
    american = 'research.benchmark_rebuild_20261024.american.treeloom_speed.static_preparation_v2'
    recovery = 'research.repairs.benchmark_recovery_20261020.runtime'
    european_observations = 'research.reviews.TreeLoom_preparation_reuse_source_20261017_v1.optimized_observations_v1'
    compatibility = 'research.reviews.H060_isolated_hold_compatibility_source_20261017_v1'
    return _install([
        (american, 'encode_block', recovery + '.observations', 'encode_block', encoder),
        (american, 'build_episodes', recovery + '.episodes', 'build_episodes', episodes.build_episodes),
        (american, 'prepare_cut_plan_v1', recovery + '.cuts', 'prepare_cut_plan_v1', cut.prepare_cut_plan_v1),
        (european_observations, 'encode_block', european_observations, 'encode_block', encoder),
        (compatibility + '.episode_preparation_v1', 'build_clade_episode_arrays_v1',
         compatibility + '.episode_preparation_v1', 'build_clade_episode_arrays_v1', episodes.build_clade_episode_arrays_v1),
        (compatibility + '.cut_preparation_v1', 'prepare_cut_plan_v1',
         compatibility + '.cut_preparation_v1', 'prepare_cut_plan_v1', cut.prepare_cut_plan_v1),
    ])


def install_observed():
    from research.preprocessing_optimization_20261026.observed.adapter import load_relate_observation
    owner = 'arggnn.benchmarking.european_treeloom'
    return _install([(owner, 'load_relate_observation', owner, 'load_relate_observation', load_relate_observation)])
