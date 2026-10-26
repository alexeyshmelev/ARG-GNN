"""Independent small CPU tests for current-route structural boundary semantics."""
from pathlib import Path
from types import SimpleNamespace
import dataclasses
import hashlib
import json
import time
import numpy as np
import torch
from research.repairs.benchmark_recovery_20261020.runtime import episodes as baseline_episodes, cuts as baseline_cut
from research.preprocessing_optimization_20261026.tree import episodes as candidate_episodes, cut as candidate_cut

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/preprocessing_optimization_20261026/review'
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def tree(h, variant=0, isolated=False):
    ages=np.resize(np.array([0.,np.nextafter(np.float32(350),np.float32(0)),350.,np.nextafter(np.float32(350),np.float32(1000)),351.],dtype=np.float32),h).astype(np.float64)
    parents=[-1]*h
    times=ages.tolist()
    active=list(range(h))
    if variant:active.reverse()
    if not isolated:
        while len(active)>1:
            next_level=[]
            for at in range(0,len(active),2):
                children=active[at:at+2]
                if len(children)==1:
                    next_level.extend(children)
                    continue
                node=len(parents)
                parents.append(-1)
                times.append(max(times[c] for c in children)+1.)
                for child in children:parents[child]=node
                if node%3==0:
                    parents[node]=node+1
                    parents.append(-1)
                    times.append(times[node]+1.)
                    node+=1
                next_level.append(node)
            active=next_level
    parents=np.array(parents,dtype=np.int64)
    times=np.array(times,dtype=np.float32)
    permutation=np.random.default_rng(913+h+variant).permutation(len(parents))
    inverse=np.empty(len(parents),dtype=np.int64)
    inverse[permutation]=np.arange(len(parents))
    p=parents[permutation].copy()
    nonroot=p>=0
    p[nonroot]=inverse[p[nonroot]]
    return p,times[permutation],inverse[:h],ages.astype(np.float32)

def make_bank(sequence):
    ps,ts,ss=[],[],[]
    offset=0
    for p,tm,s,ages in sequence:
        p=p.copy()
        p[p>=0]+=offset
        ps.append(p);ts.append(tm);ss.append(s+offset)
        offset+=len(p)
    p=np.concatenate(ps)
    depth=np.empty(len(p),dtype=np.int64)
    for u in range(len(p)):
        d=0;at=u
        while p[at]>=0:
            at=p[at];d+=1
        depth[u]=d
    levels=tuple(torch.from_numpy(np.flatnonzero(depth==i)) for i in range(1,int(depth.max())+1))
    return SimpleNamespace(parent=torch.from_numpy(p),node_time=torch.from_numpy(np.concatenate(ts)),
        sample_index=torch.from_numpy(np.stack(ss)),sample_age=torch.from_numpy(sequence[0][3]),
        spans=torch.tensor([float(i+1)/4 for i in range(len(sequence))],dtype=torch.float32),
        site_hold=torch.empty(0,dtype=torch.int64),downward_levels=levels)

def exact(x,y):
    if torch.is_tensor(x):
        assert x.dtype==y.dtype and x.shape==y.shape and x.numpy().tobytes()==y.numpy().tobytes()
    elif isinstance(x,np.ndarray):
        assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
    elif isinstance(x,dict):
        assert x.keys()==y.keys()
        for k in x:exact(x[k],y[k])
    elif isinstance(x,(list,tuple)):
        assert len(x)==len(y)
        for a,b in zip(x,y):exact(a,b)
    else:assert x==y

def episode_signature(fn,bank,isolated):
    obj,meta=fn(bank,max_payload_bytes=2**28,isolated_hold_indices=isolated)
    value={f.name:getattr(obj,f.name) for f in dataclasses.fields(obj)}
    value['flags']=value['flags'].node_flags
    return value,meta

def cut_signature(fn,bank,isolated):
    obj,meta=fn(bank,expected_haplotypes=bank.sample_index.shape[1],isolated_hold_indices=isolated)
    value={f.name:getattr(obj,f.name) for f in dataclasses.fields(obj) if f.name not in ('source_stamps','tensor_stamps')}
    return value,meta

def accepted(fn):
    try:
        fn()
        return True,None
    except Exception as exc:return False,(type(exc).__name__,str(exc))

start=time.perf_counter()
valid=0
for h in [3,4,63,64,65,127,128,129,257,1600]:
    a,b,z=tree(h),tree(h,1),tree(h,isolated=True)
    for sequence,isolated in [([a],()),([a,a,z,a,b,a],(2,)),([z,z],(0,1))]:
        bank=make_bank(sequence)
        exact(episode_signature(baseline_episodes.build_episodes,bank,isolated),episode_signature(candidate_episodes.build_episodes,bank,isolated))
        exact(cut_signature(baseline_cut.prepare_cut_plan_v1,bank,isolated),cut_signature(candidate_cut.prepare_cut_plan_v1,bank,isolated))
        valid+=1
boundary=[]
base=make_bank([tree(3),tree(3)])
for label,levels in [('missing depth levels',()),('reversed depth levels',tuple(reversed(base.downward_levels))),('root in depth levels',(torch.nonzero(base.parent<0).flatten(),)+base.downward_levels)]:
    bank=SimpleNamespace(**dict(vars(base),downward_levels=levels))
    for name,old,new,signature in [('episodes',baseline_episodes.build_episodes,candidate_episodes.build_episodes,episode_signature),('cut',baseline_cut.prepare_cut_plan_v1,candidate_cut.prepare_cut_plan_v1,cut_signature)]:
        old_ok,old_error=accepted(lambda:signature(old,bank,()))
        new_ok,new_error=accepted(lambda:signature(new,bank,()))
        boundary.append(dict(case=label,entry=name,baseline_accepted=old_ok,candidate_accepted=new_ok,baseline_error=old_error,candidate_error=new_error))
files=[Path(__file__),ROOT/'research/preprocessing_optimization_20261026/tree/kernels.cpp',ROOT/'research/preprocessing_optimization_20261026/tree/kernels.py',ROOT/'research/preprocessing_optimization_20261026/tree/episodes.py',ROOT/'research/preprocessing_optimization_20261026/tree/cut.py',ROOT/'research/preprocessing_optimization_20261026/tree/libstructural.so']
report=dict(status='valid_inputs_pass_boundary_review_required' if any(x['baseline_accepted']!=x['candidate_accepted'] for x in boundary) else 'pass',valid_complete_sequences=valid,bank_level_boundary=boundary,seconds=time.perf_counter()-start,CPU_only=True,no_training=True,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
(HERE/'tree_adversarial.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
