"""CPU-only scheduling, cleanup, RNG and prediction-wrapper differential tests."""
from __future__ import annotations
from contextlib import ExitStack
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import textwrap
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
import weakref
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/preprocessing_optimization_20261026/observed'
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import setproctitle
setproctitle.setproctitle('AI Auto Research Pipeline Test')
torch.set_num_threads(1)
from research.preprocessing_optimization_20261026.observed import prediction as new, prefetch
from research.repairs.benchmark_recovery_20261025.european import ap_prediction_chunked_gru as old
RESULT={'status':'running','tests':{}}


def source_identity():
    original=(ROOT/'src/arggnn/benchmarking/european_treeloom.py').read_text()
    start=original.index('def prepare_observation(')
    end=original.find('\n\ndef ',start+1)
    expected=original[start:end if end>=0 else None].rstrip()
    for a,b in {
      'research.reviews.TreeLoom_preparation_reuse_source_20261017_v1.optimized_observations_v1':'research.preprocessing_optimization_20261026.tree.observations',
      'research.reviews.H060_isolated_hold_compatibility_source_20261017_v1.episode_preparation_v1':'research.preprocessing_optimization_20261026.tree.episodes',
      'research.reviews.H060_isolated_hold_compatibility_source_20261017_v1.cut_preparation_v1':'research.preprocessing_optimization_20261026.tree.cut',
    }.items(): expected=expected.replace(a,b)
    actual=(HERE/'preparation.py').read_text()
    assert actual[actual.index('def prepare_observation('):].rstrip()==expected
    old_source=Path(old.__file__).read_text()
    new_source=Path(new.__file__).read_text()
    for begin,end in [('    complete = original._fit_completion(', '    cache,predictions = OrderedDict(),{}'),
                      ('logits = predict_logits(', 'check("AP_chunked_GRU_heldout_prediction_sealed")'),
                      ('    torch.cuda.synchronize(0)', '\n\ndef prediction_factory(')]:
        a=old_source[old_source.index(begin):old_source.index(end,old_source.index(begin))+len(end)]
        b=new_source[new_source.index(begin):new_source.index(end,new_source.index(begin))+len(end)]
        # Whitespace indentation of the complete unchanged numerical loop differs.
        assert [x.strip() for x in a.splitlines()]==[x.strip() for x in b.splitlines()],begin
    assert new.predict_logits is old.predict_logits
    assert new.install_model_execution is old.install_model_execution
    required=new.requirements()['source_refs']
    assert len({r['path'] for r in required})==len(required)
    for r in required:
        assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
    RESULT['tests']['unchanged_preparation_and_prediction_source']=True


def rng_state():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


def same_rng(a,b):
    assert a[0]==b[0]
    assert a[1][0]==b[1][0] and np.array_equal(a[1][1],b[1][1]) and a[1][2:]==b[1][2:]
    assert torch.equal(a[2],b[2])


def schedules():
    baseline_threads={t.ident for t in threading.enumerate()}
    before=rng_state()
    ncases=0
    for delay in [(0,0,0,0),(.002,0,.006,.001),(0,.005,0,.004)]:
        for admissions in [(True,True,True),(False,False,False),(True,False,True),(False,True,False)]:
            opened=[]
            refs=[]
            live_peak=[0]
            class Payload:
                def __init__(self,index):
                    self.values=np.arange(10,dtype=np.float32)+index
            def load(parent,check):
                check('start')
                opened.append(parent)
                time.sleep(delay[parent])
                check('end')
                value=Payload(parent)
                refs.append(weakref.ref(value))
                live_peak[0]=max(live_peak[0],sum(r() is not None for r in refs))
                return value
            decisions=iter(admissions)
            outputs=[]
            with prefetch.OneParentLookahead(range(4),load,resource_check=prefetch.SerializedCheck(lambda _:None),
                    admit_overlap=lambda:next(decisions)) as stream:
                assert opened==[]
                for parent in range(4):
                    value=stream.take(parent)
                    outputs.append(value.values.copy())
                    stream.start_next()
                    time.sleep(.002)
                    del value
            assert opened==list(range(4))
            assert all(np.array_equal(v,np.arange(10,dtype=np.float32)+i) for i,v in enumerate(outputs))
            gc.collect()
            assert not any(r() is not None for r in refs)
            assert live_peak[0]<=2
            ncases+=1
    same_rng(before,rng_state())
    assert {t.ident for t in threading.enumerate()}==baseline_threads
    RESULT['tests']['ordered_schedule_no_leak_and_rng_cases']=ncases


def failures_and_boundaries():
    active={t.ident for t in threading.enumerate()}
    for failure_at in [0,1,2]:
        seen=[]
        def load(parent,check):
            seen.append(parent)
            if parent==failure_at:raise LookupError('parent '+str(parent))
            return parent
        caught=None
        try:
            with prefetch.OneParentLookahead(range(3),load,resource_check=lambda _:None,admit_overlap=lambda:True) as stream:
                for parent in range(3):
                    assert stream.take(parent)==parent
                    stream.start_next()
        except LookupError as error:
            caught=str(error)
        assert caught=='parent '+str(failure_at)
        assert seen==list(range(failure_at+1))
        assert stream.closed and stream.future is None
    entered=threading.Event()
    cancelled=threading.Event()
    def cooperative(parent,check):
        if parent==0:return 0
        entered.set()
        try:
            while True:
                check('active background')
                time.sleep(.001)
        except prefetch.PrefetchCancelled:
            cancelled.set()
            raise
    try:
        with prefetch.OneParentLookahead([0,1],cooperative,resource_check=lambda _:None,admit_overlap=lambda:True) as stream:
            stream.take(0)
            assert stream.start_next()
            assert entered.wait(1)
            raise ArithmeticError('prediction error must remain primary')
    except ArithmeticError as error:
        assert str(error)=='prediction error must remain primary'
    assert cancelled.is_set() and stream.future is None
    with prefetch.OneParentLookahead([],lambda p,c:None,resource_check=lambda _:None,admit_overlap=lambda:True) as empty:
        assert not empty.start_next()
    with prefetch.OneParentLookahead(['x'],lambda p,c:p,resource_check=lambda _:None,admit_overlap=lambda:True) as one:
        try:one.take('z')
        except ValueError:pass
        else:raise AssertionError('Reorder accepted')
        assert one.take('x')=='x'
        assert not one.start_next()
    one.close()
    env=dict(preparation_headroom_bytes=128*2**30,prediction_headroom_bytes=64*2**30,max_rss_bytes=512*2**30)
    assert new.combined_headroom_available(320*2**30,env)
    assert not new.combined_headroom_available(320*2**30+1,env)
    assert new.combined_headroom_available(0,env)
    assert not new.combined_headroom_available(512*2**30,env)
    assert {t.ident for t in threading.enumerate()}==active
    RESULT['tests']['exceptions_cancellation_empty_order_memory_boundaries']=True


def callback_serialization():
    state={'active':0,'peak':0,'calls':0}
    def unsafe(stage):
        state['active']+=1
        state['peak']=max(state['peak'],state['active'])
        time.sleep(.001)
        state['calls']+=1
        state['active']-=1
    wrapped=prefetch.SerializedCheck(unsafe)
    threads=[threading.Thread(target=lambda:[wrapped('check') for _ in range(20)]) for _ in range(8)]
    for t in threads:t.start()
    for t in threads:t.join()
    assert state==dict(active=0,peak=1,calls=160)
    RESULT['tests']['resource_callbacks_serialized']=state


class Writer:
    def __init__(self):self.values={};self.pending=False
    def npz(self,name,**values):
        self.values[name]={k:v.copy() for k,v in values.items()}
        return {'path':name,'sha256':hashlib.sha256(b''.join(v.tobytes() for v in values.values())).hexdigest()}
    def json(self,name,value):
        self.values[name]=deepcopy(value)
        return {'path':name,'sha256':'fixture'}


class Model:
    output_classes=('A','B','C')
    def to(self,**kwargs):return self


def wrapper_fixture(module,enabled):
    parents=['p0','p1','p2']
    writer=Writer()
    loaded=[]
    sampled=[]
    inputs={p:{'inference_ref':{'path':p+'/observed.json','sha256':'fixture'},
               'observed_vcf_ref':{'path':p+'/observed.vcf','sha256':'fixture'},
               'sampled_registry_ref':{'path':p+'/registry.json','sha256':'fixture'}} for p in parents}
    pred={'protocol_id':module.PROTOCOL,'memory_envelope':dict(max_rss_bytes=512*2**30,preparation_headroom_bytes=128*2**30,prediction_headroom_bytes=64*2**30),
          'observation_cache_bytes':0,'heldout_inputs':inputs,'fit_completion_ref':{'path':'fit.json'},'sampling_plan_ref':{'path':'plan.json'},'max_metadata_bytes':2**20}
    spec={'schema':module.SCHEMA,'requirements':{},'prediction_spec':pred,'gpu_uuid_map':{'fixture':'fixture'}}
    if module is new:spec['prefetch_next_parent']=enabled
    complete={'observation_barrier_ref':{'path':'barrier.json'},'checkpoint_ref':{'path':'checkpoint.pt'}}
    def load(parent,check):
        check('fixture observed input')
        loaded.append(parent)
        time.sleep([.001,.004,.002][int(parent[-1])])
        return SimpleNamespace(focal_hold=torch.tensor([0,1]),focal_identity_sha256=parent),dict(parent=parent)
    def logits(model,bundle,metadata,**kwargs):
        time.sleep(.003)
        return torch.tensor([[1.,2.,3.],[2.,0.,1.]],dtype=torch.float32)+int(metadata['parent'][-1])
    def records(ref,**kwargs):
        sampled.append(kwargs['parent_id'])
        return {'record_ids':np.array([4,8],dtype=np.int64)}
    fake_torch=SimpleNamespace(load=lambda *a,**k:{},float32=torch.float32,isfinite=torch.isfinite,cuda=SimpleNamespace(synchronize=lambda _:None))
    before=rng_state()
    with ExitStack() as stack:
        def set_(obj,name,value):stack.enter_context(patch.object(obj,name,value))
        set_(module,'requirements',lambda:{})
        set_(module.original,'_validate',lambda *a,**k:{'train_parent_ids':['train0','train1','train2'],'test_parent_ids':parents})
        set_(module.original,'_fit_completion',lambda *a,**k:complete)
        set_(module.original,'_validate_saved',lambda *a,**k:None)
        set_(module.c,'read_ref',lambda r:{'sampling_plan_ref':pred['sampling_plan_ref'],'parents':{p:{} for p in ['train0','train1','train2']},'all_original_TRAIN_observations_sealed_before_targets':True,'private_truth_opened':False})
        set_(module.european,'_bound',lambda r:SimpleNamespace(stat=lambda:SimpleNamespace(st_size=16)))
        set_(module.european,'build_model',lambda p:Model())
        set_(module.european,'_sampled_records',records)
        set_(module,'install_model_execution',lambda m:m)
        set_(module,'predict_logits',logits)
        set_(module.baseline,'setup',lambda s,c:(fake_torch,c,writer,{'gpu_uuid':'fixture'}))
        set_(module.memory,'_admit_parent_headroom',lambda *a,**k:None)
        set_(module.memory,'_current_rss_bytes',lambda:20*2**30)
        if module is old:set_(module.original,'_observed_loader',lambda spec,cache,check,fit:lambda p:load(p,check))
        else:set_(module,'_load_observed_parent',lambda spec,p,check,torch:load(p,check))
        module.run_predictions(spec,resource_check=lambda _:None)
    same_rng(before,rng_state())
    assert loaded==sampled==parents
    return writer.values


def wrapper_equivalence():
    expected=wrapper_fixture(old,False)
    for enabled in [False,True]:
        actual=wrapper_fixture(new,enabled)
        assert expected.keys()==actual.keys()
        for path,a in expected.items():
            b=actual[path]
            if path.endswith('.npz'):
                assert a.keys()==b.keys()
                for key in a:
                    assert a[key].dtype==b[key].dtype and a[key].shape==b[key].shape and a[key].tobytes()==b[key].tobytes()
            elif path.startswith('products/'):
                a=deepcopy(a);b=deepcopy(b)
                # The only changed product metadata is the truthful wrapper source.
                a.pop('prediction_wrapper_source_ref');b.pop('prediction_wrapper_source_ref')
                assert a==b,(path,a,b)
            else:assert a==b,path
    RESULT['tests']['complete_wrapper_fixed_logits_product_parity']=dict(prefetch_disabled=True,prefetch_enabled=True,real_model_execution=False)


def main():
    source_identity();schedules();failures_and_boundaries();callback_serialization();wrapper_equivalence()
    RESULT.update(status='pass',no_scientific_training=True,no_gpu_used=True,
                  full_parent_execution_qualification_still_required=True,
                  dependency_refs=new.requirements()['source_refs'],
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'prediction.py',HERE/'preparation.py',HERE/'prefetch.py']})
    (HERE/'prefetch_qualification.json').write_text(json.dumps(RESULT,indent=2)+'\n')
    print(json.dumps(RESULT,indent=2))

if __name__=='__main__':main()
