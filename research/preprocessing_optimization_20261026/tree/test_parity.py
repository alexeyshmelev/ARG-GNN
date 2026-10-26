"""Bounded CPU regression suite comparing candidates with unchanged originals."""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from types import SimpleNamespace
import numpy as np
import torch
import setproctitle
setproctitle.setproctitle('AI Auto Research Pipeline Test')
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from arggnn.mutation_routed_tree_observations_v1 import NativeObservedTreeV1
from research.reviews.H060_isolated_hold_compatibility_source_20261017_v1 import observations_v1 as original_layout
from research.reviews.H060_isolated_hold_compatibility_source_20261017_v1 import episode_preparation_v1 as original_episodes
from research.reviews.H060_isolated_hold_compatibility_source_20261017_v1 import cut_preparation_v1 as original_cut
from research.reviews.TreeLoom_preparation_reuse_source_20261017_v1 import optimized_observations_v1 as original_observations
from research.preprocessing_optimization_20261026.tree import kernels, observations, episodes, cut
from research.repairs.benchmark_recovery_20261020.runtime import observations as current_observations
from research.repairs.benchmark_recovery_20261020.runtime import episodes as current_episodes
from research.repairs.benchmark_recovery_20261020.runtime import cuts as current_cut
from research.reviews.workflow_progress_diagnosis_20261018.treeloom import indices_v1 as current_indices
OUT = Path('research/preprocessing_optimization_20261026/tree')
COUNTS = {}

def increment(name):
    COUNTS[name] = COUNTS.get(name, 0) + 1

def exact(a, b, path='root'):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), path
        assert a.detach().numpy().tobytes() == b.detach().numpy().tobytes(), path
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a: exact(a[key], b[key], path + '.' + str(key))
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)): exact(x, y, path + '.' + str(i))
    else:
        assert a == b, (path, a, b)

def tree(rng, h, *, isolated=False, unary=True, star=False):
    ages = np.zeros(h)
    if h > 2:
        ages[1] = 350.
        ages[2] = 351.
    p = [-1] * h
    clock = ages.tolist()
    active = list(range(h))
    if not isolated:
        while len(active) > 1:
            k = len(active) if star else int(rng.integers(2, min(5, len(active)) + 1))
            indexes = rng.choice(len(active), k, replace=False)
            children = [active[i] for i in indexes]
            active = [u for i, u in enumerate(active) if i not in set(indexes)]
            u = len(p)
            p.append(-1)
            clock.append(max(clock[c] for c in children) + float(rng.integers(1, 301)))
            for c in children: p[c] = u
            if unary and rng.random() < .4:
                v = len(p)
                p.append(-1)
                clock.append(clock[u] + float(rng.integers(1, 100)))
                p[u] = v
                u = v
            active.append(u)
        if unary and rng.random() < .5:
            u = len(p)
            p.append(-1)
            clock.append(clock[active[0]] + 1.)
            p[active[0]] = u
    p = np.asarray(p, dtype=np.int64)
    clock = np.asarray(clock)
    permutation = rng.permutation(len(p))
    inverse = np.empty(len(p), dtype=np.int64)
    inverse[permutation] = np.arange(len(p))
    new_parent = p[permutation].copy()
    mask = new_parent >= 0
    new_parent[mask] = inverse[new_parent[mask]]
    return NativeObservedTreeV1(new_parent, clock[permutation], inverse[:h], np.arange(h, dtype=np.int64))

def packed(trees):
    parents, samples, times, offset = [], [], [], 0
    for tr in trees:
        p = tr.parent.copy()
        p[p>=0] += offset
        parents.append(p)
        samples.append(tr.sample_node+offset)
        times.append(tr.node_time)
        offset += len(p)
    return np.concatenate(parents), np.stack(samples), np.concatenate(times)

def bank(trees, spans):
    p, s, tm = packed(trees)
    depth=np.concatenate([original_layout._layout_with_isolated_samples(tr)[5] for tr in trees])
    return SimpleNamespace(downward_levels=tuple(torch.from_numpy(level) for level in kernels._level_arrays(depth)[1:]),parent=torch.from_numpy(p), sample_index=torch.from_numpy(s),
        node_time=torch.from_numpy(tm.astype(np.float32)), sample_age=torch.from_numpy(trees[0].node_time[trees[0].sample_node].astype(np.float32)),
        spans=torch.from_numpy(spans.copy()), site_hold=torch.empty(0, dtype=torch.long))

def compare_cut(b, isolated):
    kwargs = dict(expected_haplotypes=b.sample_index.shape[1], isolated_hold_indices=isolated)
    old, old_meta = original_cut.prepare_cut_plan_v1(b, **kwargs)
    new, new_meta = cut.prepare_cut_plan_v1(b, **kwargs)
    exact(old.tensors(), new.tensors(), 'cut.tensors')
    for f in dataclasses.fields(old):
        if f.name not in ('source_stamps', 'tensor_stamps') and not isinstance(getattr(old, f.name), torch.Tensor):
            exact(getattr(old, f.name), getattr(new, f.name), 'cut.'+f.name)
    exact(old_meta, new_meta, 'cut.metadata')
    active, active_meta = current_cut.prepare_cut_plan_v1(b, **kwargs)
    exact(active.tensors(), new.tensors(), 'current_cut.tensors')
    exact(active_meta, new_meta, 'current_cut.metadata')
    increment('cut_complete')
    increment('current_cut_complete')

def compare_sequence(trees):
    isolated = tuple(i for i, tr in enumerate(trees) if len(tr.parent)>1 and np.all(tr.parent == -1))
    for tr in trees:
        exact(original_layout._layout_with_isolated_samples(tr), kernels._layout_with_isolated_samples(tr), 'layout')
        increment('layout_complete')
    p, s, _ = packed(trees)
    spans = np.asarray([1. + i * .25 for i in range(len(trees))], dtype=np.float32)
    kwargs = dict(isolated_hold_indices=isolated)
    exact(original_episodes._forest_index(p, s, **kwargs), kernels._forest_index(p, s, **kwargs), 'forest')
    exact(original_episodes.build_clade_episode_arrays_v1(p, s, spans, **kwargs),
        episodes.build_clade_episode_arrays_v1(p, s, spans, **kwargs), 'episodes')
    increment('episodes_complete')
    b=bank(trees, spans)
    active_arrays, active_meta = current_episodes.build_clade_episode_arrays_v1(p,s,spans,
        _forest_index=lambda parent,samples: current_indices.forest_for_bank(b,parent,samples,isolated))
    exact((active_arrays,active_meta), episodes.build_clade_episode_arrays_v1(p,s,spans,**kwargs), 'current_episodes')
    active_obj, active_meta = current_episodes.build_episodes(b, max_payload_bytes=original_episodes.DEFAULT_PAYLOAD_CAP, **kwargs)
    new_obj, new_meta = episodes.build_episodes(b, max_payload_bytes=original_episodes.DEFAULT_PAYLOAD_CAP, **kwargs)
    for field in dataclasses.fields(active_obj):
        a,bvalue=getattr(active_obj,field.name),getattr(new_obj,field.name)
        if field.name=='flags':
            exact(a.node_flags,bvalue.node_flags,'current_episode_object.flags')
        else:exact(a,bvalue,'current_episode_object.'+field.name)
    exact(active_meta,new_meta,'current_episode_object.metadata')
    increment('current_episodes_complete')
    compare_cut(b, isolated)

def compare_encode(trees, rng):
    h = len(trees[0].sample_node)
    intervals = np.column_stack((np.arange(len(trees))*10., (np.arange(len(trees))+1)*10.))
    positions = np.asarray([i*10.+j for i, tr in enumerate(trees) if not np.all(tr.parent==-1) for j in (0., 3., 9.)])
    k = rng.integers(1, 5, len(positions), dtype=np.int64)
    g = np.asarray([rng.integers(-1, int(v), h) for v in k], dtype=np.int64).reshape(-1,h)
    labels = np.arange(h, dtype=np.int64) % 3
    labels[-1] = -1
    args = (trees, intervals, g, k, positions, labels, trees[0].node_time[trees[0].sample_node])
    old, old_meta = original_observations.encode_block(*args, num_classes=3)
    new, new_meta = observations.encode_block(*args, num_classes=3)
    for f in dataclasses.fields(old): exact(getattr(old, f.name), getattr(new, f.name), 'encoded.'+f.name)
    exact(old_meta, new_meta, 'encoded.metadata')
    active, active_meta = current_observations.encode_block(*args, num_classes=3)
    for f in dataclasses.fields(active): exact(getattr(active, f.name), getattr(new, f.name), 'current_encoded.'+f.name)
    exact(active_meta,new_meta,'current_encoded.metadata')
    increment('encoded_complete')
    increment('current_encoded_complete')

def invalid_layouts(rng):
    base = tree(rng, 5)
    cases = []
    for label, field, value in (
        ('empty', 'parent', np.empty(0,dtype=np.int64)),
        ('parent_oob', 'parent', np.full(len(base.parent),999,dtype=np.int64)),
        ('parent_negative', 'parent', np.full(len(base.parent),-2,dtype=np.int64)),
        ('float_parent', 'parent', base.parent.astype(float)),
        ('nonfinite_time', 'node_time', np.full(len(base.parent),np.nan)),
        ('negative_time', 'node_time', np.full(len(base.parent),-1.)),
        ('nonpositive_edges', 'node_time', np.zeros(len(base.parent))),
        ('duplicate_sample', 'sample_node', np.zeros(5,dtype=np.int64)),
        ('duplicate_identity', 'sample_id', np.zeros(5,dtype=np.int64)),
        ('sample_oob', 'sample_node', np.full(5,999,dtype=np.int64))):
        cases.append((label, dataclasses.replace(base, **{field:value})))
    cases.append(('isolated_missing_sample', NativeObservedTreeV1(np.full(5,-1,dtype=np.int64),np.zeros(5),np.arange(4),np.arange(4))))
    cases.append(('hidden_component', NativeObservedTreeV1(np.array([-1,0,-1]), np.array([3.,2.,1.]),np.array([1]),np.array([0]))))
    for label, tr in cases:
        errors=[]
        for fun in (original_layout._layout_with_isolated_samples, kernels._layout_with_isolated_samples):
            try: fun(tr)
            except Exception as exc: errors.append(type(exc).__name__)
            else: errors.append(None)
        assert errors[0] is not None and errors[0] == errors[1], (label,errors)
        increment('invalid_layout')

def main():
    start=time.perf_counter()
    rng=np.random.default_rng(90127)
    singleton=tree(rng,1)
    exact(original_layout._layout_with_isolated_samples(singleton),kernels._layout_with_isolated_samples(singleton),'singleton')
    increment('singleton_layout')
    invalid_layouts(rng)
    for i in range(200):
        h=int(rng.integers(3,80))
        trees=[tree(rng,h,isolated=(j%5==2),star=(i%7==0)) for j in range(int(rng.integers(1,7)))]
        if i%3==0: trees.insert(1,trees[0])
        compare_sequence(trees)
        if i<25: compare_encode(trees,rng)
    for h in (3,4,63,64,65,127,128,129,257,1600):
        tr=tree(rng,h)
        compare_sequence([tr,tr,tree(rng,h,isolated=True),tr,tr,tree(rng,h)])
    report=dict(status='pass',counts=COUNTS,seconds=time.perf_counter()-start,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        threads=1,device='cpu',integer_and_float_outputs='bitwise_identical',
        original_sources_unchanged=True)
    (OUT/'parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
