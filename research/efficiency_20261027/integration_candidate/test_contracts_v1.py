"""CPU-only concurrency, real layout rebind, immutable schedule and RNG checks."""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import ast
import gc
import json
import threading
import time
import torch
from arggnn.benchmarking import contracts as c
from research.repairs.TreeLoom_lazy_compact_prefetch_20261024_v1 import training as original
from research.repairs.TreeLoom_lazy_compact_prefetch_20261024_v1 import cpu_smoke as smoke
from research.reviews.workflow_progress_diagnosis_20261018.treeloom.resident_pool_v2 import pack_bundle,restore_bundle
from research.efficiency_20261027.root.compare_host_parallelism import CHECKPOINT,JOB
from . import training_v1 as candidate
from .memory_v1 import LayoutMemoryGovernor, GovernedLayoutCache, GovernedLargeLayoutCache, LayoutAwareParents
from . import entrypoint_v1

BASE=c.ROOT/'research/efficiency_20261027/integration_candidate'


def need(ok,message):
    if not ok: raise AssertionError(message)


def expect_failure(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Expected rejection')


def memory_checks():
    state={'rss':0}
    governor=LayoutMemoryGovernor(maximum_bytes=100,rss_bytes=lambda:state['rss'],
        resource_check=lambda stage:None,reserve_bytes=60)
    admitted=threading.Event()
    started=threading.Event()
    def waiter():
        started.set()
        with governor.phase('other_thread'):
            admitted.set()
    with governor.phase('training'):
        thread=threading.Thread(target=waiter)
        thread.start()
        need(started.wait(1) and not admitted.wait(.03),'Overlapping phase waits for real reservation')
        expect_failure(lambda: enter_phase(governor,50))
    thread.join(2)
    need(not thread.is_alive() and admitted.is_set(),'Other thread resumes after release')
    need(not governor.active and not governor.owners,'All reservations released')
    cache=GovernedLayoutCache(governor,max_bytes=100000)
    cache.entries['leased']=dict(base_charge=20,bindings={},leases=1,lease_charge=0)
    cache.entries['protected']=dict(base_charge=20,bindings={},leases=0,lease_charge=0)
    cache.entries['old']=dict(base_charge=20,bindings={},leases=0,lease_charge=0)
    cache._protected_keys.add('protected')
    need(cache.evict_one_unleased() and set(cache.entries)=={'leased','protected'},
         'Only unleased and unprotected entries may be evicted')
    need(not cache.evict_one_unleased(),'Active leases/protected rebind survive')
    cache._protected_keys.clear()
    cache.entries['leased']['leases']=0
    state['rss']=95
    expect_failure(lambda: enter_phase(governor,10))
    need(not cache.entries and state['rss']==95,'Evicted accounting never substitutes for actual RSS')
    state['rss']=0
    try:
        with governor.phase('exception',5):
            raise RuntimeError('expected')
    except RuntimeError:
        pass
    need(not governor.active,'Exception releases active reserve')
    governor.close()
    expect_failure(lambda: enter_phase(governor,0))
    return dict(overlap_wait_and_resume=True,nested_self_wait_rejected=True,
        leases_and_pending_rebind_protected=True,actual_RSS_required_after_eviction=True,
        exception_release=True,closed_governor_rejected=True)


def enter_phase(governor,amount):
    with governor.phase('test',amount):
        pass


def actual_cache_check(output,registry,holds_per_block):
    output=output/str(holds_per_block)
    output.mkdir()
    runtime=original.fresh.FreshTreeLoom(registry)
    before=smoke.rng_state()
    (bundle,metadata),refs=smoke.fixture(output,registry,runtime)
    governor=LayoutMemoryGovernor(maximum_bytes=8*2**30,rss_bytes=original._current_rss_bytes,
        resource_check=lambda stage:None,reserve_bytes=128*2**20)
    cache=(GovernedLayoutCache if holds_per_block==128 else GovernedLargeLayoutCache)(governor,max_bytes=64*2**20)
    with governor.phase('complete_parent_training'):
        lease=cache.acquire(bundle,metadata,holds_per_block=holds_per_block)
        need(cache.snapshot()['builds']==1 and cache.snapshot()['active_executor_leases']==1,'Actual initial layout build')
        need(not cache.evict_one_unleased(),'Real active executor layout cannot be evicted')
        lease.release()
        record=pack_bundle(bundle,metadata,input_refs=refs,external_arrays=None,resource_check=lambda stage:None)
        restored,restored_metadata=restore_bundle(record,input_refs=refs,resource_check=lambda stage:None)
        rebound=cache.acquire(restored,restored_metadata,holds_per_block=holds_per_block)
        need(cache.snapshot()['restored_content_binds']==1 and cache.snapshot()['builds']==1,
             'Exact restored topology must reuse plan rather than rebuild')
        rebound.release()
    need(smoke.state_equal(before,smoke.rng_state()),'Layout and observation/cache work preserve global RNG')
    snapshot=cache.snapshot()
    cache.clear()
    governor.close()
    return dict(real_CPU_layout_build_and_exact_restore_rebind=True,global_RNG_preserved=True,snapshot=snapshot)


def schedule_check(registry,arguments,barrier):
    def governed_pool(**kwargs):
        governor=LayoutMemoryGovernor(maximum_bytes=kwargs['max_rss_bytes'],
            rss_bytes=kwargs['rss_bytes'],resource_check=kwargs['resource_check'],reserve_bytes=kwargs['phase_reserve_bytes'])
        GovernedLayoutCache(governor,max_bytes=64*2**20)
        return LayoutAwareParents(layout_governor=governor,**kwargs)
    with patch.object(smoke,'training',candidate),patch.object(smoke,'PipelinedCompactParents',governed_pool):
        return smoke.schedule_check(registry,arguments,barrier)


def source_contracts():
    old=ast.parse(Path(original.__file__).read_text())
    new=ast.parse(Path(candidate.__file__).read_text())
    functions=lambda tree:{node.name:ast.dump(node,include_attributes=False)
        for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    old_functions,new_functions=functions(old),functions(new)
    changed=sorted(name for name in old_functions if old_functions[name]!=new_functions.get(name))
    need(changed==['_validate_spec','build_spec','requirements','run_training'],'Only explicit construction/requirements change; all scientific/checkpoint functions identical')
    need(candidate.SETTINGS==original.SETTINGS and candidate.LIMITS==original.LIMITS,'Original complete scientific contract')
    return dict(changed_functions=changed,train_loop_and_checkpoint_RNG_validators_AST_identical=True)


def exit_checks():
    transition=dict(predecessor_pid=12345,predecessor_start_ticks='987',predecessor_actual_wait_ref={})
    receipt=dict(actual_wait=True,owned_group_absent=True,pid=12345,process_start_ticks='987',actual_returncode=-15)
    with patch.object(entrypoint_v1,'read',return_value=receipt),patch.object(entrypoint_v1,'identity',return_value=None),patch.object(entrypoint_v1,'owned_group_members',return_value=[]):
        entrypoint_v1.validate_exit(transition)
        for mutation in ({'actual_wait':False},{'owned_group_absent':False},{'pid':12346},{'process_start_ticks':'988'}):
            with patch.object(entrypoint_v1,'read',return_value=dict(receipt,**mutation)):
                expect_failure(lambda:entrypoint_v1.validate_exit(transition))
        with patch.object(entrypoint_v1,'identity',return_value='987'):
            expect_failure(lambda:entrypoint_v1.validate_exit(transition))
        with patch.object(entrypoint_v1,'owned_group_members',return_value=[12346]):
            expect_failure(lambda:entrypoint_v1.validate_exit(transition))
    return dict(actual_wait_identity_and_empty_group_required=True,controlled_stop_returncode_retained=True)


def main():
    need(not torch.cuda.is_initialized(),'CPU-only')
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    output=BASE/'cpu_contracts_v2'
    output.mkdir(exist_ok=False)
    saved=torch.load(c.ROOT/CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)
    registry=original.fresh.FreshParentRegistry.from_ref(saved['parent_registry_ref'])
    arguments=c.read_json(JOB)['call']['spec']['arguments']
    barrier=c.read_ref(saved['observation_barrier_ref'])
    result=dict(status='passed',CPU_only=True,source_contracts=source_contracts(),memory=memory_checks(),
        actual_cache={str(size):actual_cache_check(output,registry,size) for size in (128,512,1024)},schedule=schedule_check(registry,arguments,barrier),
        predecessor_exit=exit_checks(),no_CUDA_initialization=not torch.cuda.is_initialized())
    (output/'completed.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
