"""Bounded CPU state-machine and serial/concurrent exact-codec qualification."""
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
import gc
import json
import pickle
from pathlib import Path
import random
import threading
import time
from types import SimpleNamespace
import numpy as np
import torch
from .adaptive_prefetch_v2 import AdaptiveMemoryGovernor, AdaptiveCompactParents, EncoderCredits

BASE=Path('research/efficiency_20261027/execution')


def rejects(function, error=(ValueError, RuntimeError)):
    try:function()
    except error:return
    raise AssertionError('Required rejection did not occur')


def wait_until(predicate, timeout=3.):
    deadline=time.monotonic()+timeout
    while not predicate():
        assert time.monotonic()<deadline, 'Bounded test wait expired'
        time.sleep(.002)


def fake_pool(schedule, prepare=None, *, rss=lambda:0, maximum=1000, reserve=100, events=None):
    parents=set(schedule)
    governor=AdaptiveMemoryGovernor(maximum_bytes=maximum,rss_bytes=rss,
        resource_check=lambda stage:None,reserve_bytes=reserve)
    credits=EncoderCredits(maximum_bytes=maximum,allowance_bytes=10,
        family_memory=lambda:dict(bytes=0,recursive_children=0))
    def pack(bundle,meta,**kwargs):return dict(bundle=bundle,metadata=meta,charged_bytes=5,read_peak_upper_bound_bytes=7)
    def restore(record,**kwargs):
        assert 7 in governor.active.values() and 'exact_restore' in governor.labels.values()
        return record['bundle'],record['metadata']
    pool=AdaptiveCompactParents(schedule=schedule,start_cursor=0,parent_bounds={p:10 for p in parents},
        input_refs={p:{} for p in parents},external_arrays={},max_cache_bytes=10*len(parents),
        governor=governor,credits=credits,resource_check=lambda stage:None,
        event=lambda name,data:(events.append((name,data)) if events is not None else None),
        prepare=prepare or (lambda parent,**kwargs:(parent,dict(parent=parent))),pack=pack,restore=restore)
    return pool


def queue_checks():
    b_release=threading.Event(); c_ready=threading.Event(); events=[]
    def prepare(parent,*,resource_check):
        if parent=='B':assert b_release.wait(3)
        if parent=='C':c_ready.set()
        resource_check('prepare')
        return parent,dict(parent=parent)
    pool=fake_pool(['A','B','C','B','A','D'],prepare,events=events)
    original_result=Future.result
    def checked_result(future,*args,**kwargs):
        assert not pool.lock._is_owned() and not pool.governor.condition._is_owned()
        return original_result(future,*args,**kwargs)
    Future.result=checked_result
    order=[]
    try:
        rejects(lambda:pool.load('B'))
        rejects(lambda:pool.durable_checkpoint(parent='A',completed_updates=1))
        for index,parent in enumerate(pool.schedule):
            bundle,meta=pool.load(parent)
            assert bundle==meta['parent']==parent
            order.append(parent)
            with pool.training_phase():
                if index==0:
                    assert c_ready.wait(3)
                    wait_until(lambda:pool.futures['C'].done())
                    assert not pool.futures['B'].done() and pool.cursor==0
                    assert pool.states['C']=='ready' and pool.preparations['B']==1
                    b_release.set()
                pool.durable_checkpoint(parent=parent,completed_updates=index+1)
        assert order==list(pool.schedule) and pool.cursor==6
        assert pool.preparations==dict(A=1,B=1,C=1,D=1)
        assert pool.restorations==dict(A=2,B=2,C=1,D=1)
        assert sum(row[0]=='observation_prefetch_started' for row in events)==4
        return dict(out_of_order_completion_preserves_schedule=True,repeated_parents_prepare_once=True,
                    future_waits_outside_locks=True,exact_restore_reserve_checked=True,
                    cursor_moves_only_after_durable_callback=True,receipt=pool.receipt())
    finally:
        b_release.set()
        pool.close()
        Future.result=original_result


def adaptive_checks():
    rss=[190]; release=threading.Event(); started=[]
    def prepare(parent,*,resource_check):
        started.append(parent)
        if parent!='A':
            while not release.wait(.002):resource_check('held_test_producer')
        return parent,{}
    pool=fake_pool(['A','B','C','D','E'],prepare,rss=lambda:rss[0],maximum=600,reserve=128)
    try:
        pool.load('A')
        with pool.training_phase():
            wait_until(lambda:len(started)==3)
            assert pool.credits.snapshot()['active_child_credits']==2
            assert pool.governor.snapshot()['active_reserved_bytes']==384
            release.set()
            wait_until(lambda:all(f.done() for f in pool.futures.values()))
            pool.durable_checkpoint(parent='A',completed_updates=1)
        rss[0]=300;release.clear()
        pool.load('B')
        with pool.training_phase():
            wait_until(lambda:'D' in started)
            assert 'E' not in started and pool.states['E']=='absent'
            assert pool.credits.snapshot()['active_child_credits']==1
            assert pool.governor.optional_deferrals>0
            release.set()
            pool.durable_checkpoint(parent='B',completed_updates=2)
        return dict(two_to_one_actual_RSS_admission=True,optional_deferral_does_not_abort_training=True)
    finally:
        release.set();pool.close()


def governor_checks():
    rss=[100];gov=AdaptiveMemoryGovernor(maximum_bytes=600,rss_bytes=lambda:rss[0],
        resource_check=lambda stage:None,reserve_bytes=300)
    occupied=threading.Event();release=threading.Event();entered=threading.Event();errors=[]
    def producer():
        with gov.phase('test_producer'):
            occupied.set();assert release.wait(3)
    def foreground():
        try:
            with gov.phase('foreground'):entered.set()
        except BaseException as error:errors.append(str(error))
    thread=threading.Thread(target=producer);thread.start();assert occupied.wait(3)
    front=threading.Thread(target=foreground);front.start()
    wait_until(lambda:gov.foreground_waiters==1)
    assert gov.acquire('optional_small',1,optional=True) is None
    release.set();thread.join(3);front.join(3)
    assert entered.is_set() and not errors
    gov.close()
    # Actual RSS must fall after eviction, not just retained-byte accounting.
    rss=[595];gov=AdaptiveMemoryGovernor(maximum_bytes=600,rss_bytes=lambda:rss[0],
        resource_check=lambda stage:None,reserve_bytes=100)
    class Cache:
        def __init__(self,physical):self.once=True;self.physical=physical
        def evict_one_unleased(self):
            if not self.once:return False
            self.once=False
            if self.physical:rss[0]-=20
            return True
    gov.cache=Cache(True)
    with gov.phase('exact_restore',7):assert gov.eviction_checks==1
    rss[0]=595;gov.cache=Cache(False)
    rejects(lambda:gov.acquire('retained_allocation',7))
    gov.close()
    # A nested layout waits for another producer, never for its own training reserve.
    gov=AdaptiveMemoryGovernor(maximum_bytes=700,rss_bytes=lambda:100,
        resource_check=lambda stage:None,reserve_bytes=300)
    occupied.clear();release.clear()
    def other():
        with gov.phase('producer'):
            occupied.set()
            wait_until(lambda:gov.foreground_waiters==1)
    with gov.phase('training'):
        thread=threading.Thread(target=other);thread.start();assert occupied.wait(3)
        with gov.phase('layout_build_or_bind',50):pass
        thread.join(3)
        rejects(lambda:gov.acquire('impossible_own_nested',400))
    gov.close()
    # Closure wakes an admission waiter while the existing phase is retained.
    gov=AdaptiveMemoryGovernor(maximum_bytes=600,rss_bytes=lambda:100,
        resource_check=lambda stage:None,reserve_bytes=300)
    occupied.clear();release.clear();errors.clear();entered.clear()
    thread=threading.Thread(target=producer);thread.start();assert occupied.wait(3)
    front=threading.Thread(target=foreground);front.start();wait_until(lambda:gov.foreground_waiters==1)
    gov.close();front.join(3);assert errors and not entered.is_set()
    release.set();thread.join(3)
    return dict(foreground_priority=True,layout_eviction_requires_actual_RSS_drop=True,
                nested_layout_waits_only_other_owners=True,close_wakes_admission_waiters=True)


def failure_checks():
    calls=[]
    def fail(parent,**kwargs):calls.append(parent);raise RuntimeError('Expected synthetic failure')
    pool=fake_pool(['A'],fail)
    try:
        rejects(lambda:pool.load('A'));rejects(lambda:pool.load('A'))
        assert calls==['A'] and pool.states['A']=='failed'
    finally:pool.close()
    entered=threading.Event()
    def cooperative(parent,*,resource_check):
        entered.set()
        while True:
            resource_check('cooperative_stop');time.sleep(.002)
    pool=fake_pool(['A','B'],cooperative)
    # Force a queued future to test cancellation, without changing production worker count.
    pool.executor.shutdown()
    pool.executor=ThreadPoolExecutor(max_workers=1)
    pool._submit('A',optional=True);assert entered.wait(3)
    pool._submit('B',optional=True)
    pool.close()
    assert pool.futures['B'].cancelled() and pool.futures['A'].done()
    assert not pool.credits.active and not pool.governor.active and not pool.pending
    return dict(failed_parent_never_retried=True,queued_future_cancelled=True,
                running_producer_cooperatively_joined=True,all_credits_released=True)


def credit_checks():
    ledger=EncoderCredits(maximum_bytes=100,allowance_bytes=10,
        family_memory=lambda:dict(bytes=85,recursive_children=0))
    one=ledger.acquire('A');assert one is not None and ledger.acquire('B') is None
    rejects(ledger.before_encoder)
    with ledger.owner(one):assert ledger.before_encoder()=='A'
    ledger.release(one);two=ledger.acquire('B');assert two is not None;ledger.release(two)
    return dict(outstanding_family_credit_charged_atomically=True,unowned_encoder_rejected=True,
                synthetic_measurements_not_real_child_overhead=True)


def joint_family_checks():
    # Units are GiB; no allocation is performed in this boundary test.
    for live, expected_two in ((200,False),(192,True)):
        measured=[live]
        governor=AdaptiveMemoryGovernor(maximum_bytes=592,rss_bytes=lambda:measured[0],
            resource_check=lambda stage:None,reserve_bytes=128)
        credits=EncoderCredits(maximum_bytes=592,allowance_bytes=8,
            family_memory=lambda:dict(bytes=measured[0],recursive_children=0))
        governor.bind_credits(credits)
        with governor.phase('foreground_training'):
            first=governor.acquire('parent_A',optional=True,credits=credits,parent='A')
            assert first is not None
            second=governor.acquire('parent_B',optional=True,credits=credits,parent='B')
            assert (second is not None)==expected_two
            if second is not None:
                measured[0]=193
                with credits.owner(first[1]):rejects(credits.before_encoder)
                measured[0]=live
                governor.release(second[0],credits=credits,credit=second[1])
            with credits.owner(first[1]):assert credits.before_encoder()=='A'
            governor.release(first[0],credits=credits,credit=first[1])
        governor.close()
    return dict(joint_200_plus_384_plus_16_rejected=True,
                joint_192_plus_384_plus_16_admitted=True,
                encoder_rechecks_all_outstanding_phases_and_credits=True)


def rng():return pickle.dumps((random.getstate(),np.random.get_state())),torch.get_rng_state().numpy().tobytes()


def codec_checks(output):
    from arggnn.benchmarking import contracts as c
    from research.efficiency_20261027.integration_candidate import training_v3 as training
    from research.reviews.workflow_progress_diagnosis_20261018.treeloom.resident_pool_v2 import (
        exact_signature,pack_bundle,restore_bundle)
    from research.preprocessing_optimization_20261026.integration.hooks import install_tree,verify_bindings
    from .adaptive_fixture_v2 import fixture
    # Only serial native encoding inside each preparation thread in this bounded
    # CPU proof. Eight-child memory/cadence remain a separate ROOT qualification.
    bindings=install_tree(preparation_workers=1)
    job=c.read_json('.runtime/benchmarking/workflow_20261019_v8/jobs/AM_treeloom_fit.json')
    registry=training.fresh.FreshParentRegistry.from_ref(job['call']['spec']['training_spec']['parent_registry_ref'])
    runtime=training.fresh.FreshTreeLoom(registry)
    parents=registry.ids('train')[:3]
    before=rng();inputs={};states={};serial={}
    for parent in parents:
        directory=output/parent;directory.mkdir()
        state,refs=fixture(directory,SimpleNamespace(ids=lambda split,p=parent:[p]),runtime)
        states[parent]=state;inputs[parent]=refs
    def prepare(parent,*,resource_check):
        return runtime.prepare_observation(states[parent],split='train',resource_check=resource_check,
            **inputs[parent],max_nodes=100000,max_holds=4,max_episode_payload_bytes=16*2**20)
    for parent in parents:
        bundle,meta=prepare(parent,resource_check=lambda stage:None)
        signature=exact_signature(bundle,meta)
        record=pack_bundle(bundle,meta,input_refs=inputs[parent])
        restored,new_meta=restore_bundle(record,input_refs=inputs[parent],verify_integrity=True)
        assert exact_signature(restored,new_meta)==signature
        serial[parent]=signature
        del bundle,meta,record,restored,new_meta
    gov=AdaptiveMemoryGovernor(maximum_bytes=8*2**30,rss_bytes=lambda:0,
        resource_check=lambda stage:None,reserve_bytes=256*2**20)
    credits=EncoderCredits(maximum_bytes=8*2**30,allowance_bytes=1,
        family_memory=lambda:dict(bytes=0,recursive_children=0))
    schedule=[parents[0],parents[1],parents[2],parents[1]]
    pool=AdaptiveCompactParents(schedule=schedule,start_cursor=0,parent_bounds={p:128*2**20 for p in parents},
        input_refs=inputs,external_arrays={},max_cache_bytes=384*2**20,governor=gov,credits=credits,
        resource_check=lambda stage:None,event=lambda *args:None,prepare=prepare)
    try:
        for index,parent in enumerate(schedule):
            restored,meta=pool.load(parent)
            assert exact_signature(restored,meta)==serial[parent]
            with pool.training_phase():
                pool.durable_checkpoint(parent=parent,completed_updates=index+1)
            del restored,meta
        assert pool.preparations==dict.fromkeys(parents,1) and rng()==before
        verify_bindings(bindings)
        return dict(serial_concurrent_prepare_pack_restore_exact=True,arrays_metadata_provenance_exact=True,
                    global_Python_NumPy_Torch_RNG_unchanged=True,parent_records=3,
                    native_holds_each=2,haplotypes_each=1600,focals_each=10000,
                    sites_each=4,encoder_processes_per_parent=0,
                    real_eight_child_process_family_qualification=False,pool=pool.receipt())
    finally:pool.close()


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    output=BASE/'adaptive_prefetch_CPU_v3';output.mkdir(exist_ok=False)
    started=time.monotonic()
    result=dict(status='passed',queue=queue_checks(),adaptive=adaptive_checks(),
        governor=governor_checks(),failure=failure_checks(),credits=credit_checks(),joint_family=joint_family_checks(),
        codec=codec_checks(output),GPU_initialized=torch.cuda.is_initialized(),
        elapsed_seconds=time.monotonic()-started,
        limitations=['Synthetic bounded observations; not complete real-parent throughput or memory qualification.',
                    'Real two-parent/eight-child family, cadence and reaping require ROOT admission.'])
    assert not result['GPU_initialized']
    with (BASE/'ADAPTIVE_PREFETCH_CPU_v3.json').open('x') as handle:json.dump(result,handle,indent=2)
    print(json.dumps(dict(status=result['status'],elapsed_seconds=result['elapsed_seconds'])))


if __name__=='__main__':main()
