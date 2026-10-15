"""Persistent bounded C3 observation adaptation; original providers stay frozen."""
from __future__ import annotations
import copy,multiprocessing as mp,os,resource,tempfile,time
from pathlib import Path
from research.reviews.C3_GNN_prefetch_provider_20261015_v1.abstract_ipc_bootstrap_v1 import ensure_abstract_ipc_before_torch_v1
IPC_BOOTSTRAP=ensure_abstract_ipc_before_torch_v1()
if mp.current_process().name!="MainProcess":os.environ["CUDA_VISIBLE_DEVICES"]=""
import torch
from arggnn.american_c3_common_v1 import require,ref,local
from arggnn.american_c3_gnn_stream_train_isolated_v2 import SealedObservationProviderV1
from research.reviews.C3_GNN_prefetch_isolated980_root_20261015_v2.prefetch_provider_isolated_v2 import (
    PrefetchSealedObservationProviderV1,_check_pins,_rss,PINS)
from .persistent_workers_v1 import PersistentOrderedWorkersV1,context_sha
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2"
TRAIN=tuple("american_c3_short_train_%03d"%i for i in range(20))

class _PersistentCPUReader:
    def __init__(self,*,parent_plan_refs,worker_tmp_directory,max_block_bytes,worker_rss_cap_bytes):
        require(set(parent_plan_refs)==set(TRAIN) and not torch.cuda.is_initialized(),"All20 observation-only refs; CPU worker")
        os.environ["CUDA_VISIBLE_DEVICES"]="";torch.set_num_threads(1);ensure_abstract_ipc_before_torch_v1();_check_pins()
        self.refs=copy.deepcopy(parent_plan_refs);self.block_cap=max_block_bytes;self.rss_cap=worker_rss_cap_bytes
        directory=local(Path(worker_tmp_directory)/("persistent_prefetch_worker_%d"%os.getpid()))
        directory.mkdir(parents=True,exist_ok=False);os.environ["TMPDIR"]=str(directory);tempfile.tempdir=str(directory)
        self.directory=str(directory);self.provider=None;self.context=None;self.last_serial=0
    def bind(self,ctx):
        require(self.context is None and self.provider is None,"One active observation parent per worker")
        value=copy.deepcopy(ctx);digest=value.pop("context_sha256")
        require(digest==context_sha(value) and set(value)=={"parent_id","plan_ref","blocks","seal_record","context_serial"}
            and value["parent_id"] in TRAIN and value["plan_ref"]==self.refs[value["parent_id"]]
            and type(value["context_serial"])is int and value["context_serial"]>self.last_serial,"Authenticated exact registered context")
        _check_pins()
        p=SealedObservationProviderV1(value["plan_ref"],parent_id=value["parent_id"],worker_tmp_directory=self.directory,cache_bytes=0)
        require(value["blocks"]==len(p) and self.block_cap>=p.plan["feature_tensor_byte_planning_bound_per_block"],"Frozen full physical plan and block bound")
        seal=value["seal_record"]
        if seal is not None:
            require(seal["schema"]=="american_c3_observation_blocks_sealed_v1"
                and seal["parent_id"]==value["parent_id"] and seal["plan_ref"]==value["plan_ref"]
                and seal["observation_ref"]==p.plan["observation_ref"] and seal["focal_registry_ref"]==p.plan["focal_registry_ref"]
                and seal["blocks"]==len(p) and len(seal["block_feature_sha256"])==len(p)
                and seal["all_blocks_sealed_before_truth"] is True and seal["all_trees_and_samples_retained"] is True,"Exact actual parent observation seal")
            p.seal_record=copy.deepcopy(seal)
        self.provider=p;self.context=copy.deepcopy(ctx);self.last_serial=value["context_serial"]
    def __call__(self,index,*,context_sha256,parent_id,plan_ref):
        require(self.context is not None and self.provider is not None and context_sha256==self.context["context_sha256"]
            and parent_id==self.context["parent_id"] and plan_ref==self.context["plan_ref"]
            and type(index)is int and 0<=index<len(self.provider),"Exact registered parent/plan/block/context")
        started=time.perf_counter();block,digest,size=self.provider._read(index)
        require(size<=self.block_cap and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=self.rss_cap
            and not torch.cuda.is_initialized(),"Original checked features within CPU-only worker caps")
        return block,digest,size,dict(read_and_original_hash_seconds=time.perf_counter()-started)
    def release(self,digest):
        require(self.context is not None and digest==self.context["context_sha256"],"Exact state-clear context")
        self.provider.release();self.provider=None;self.context=None
    def close(self):
        if self.provider is not None:self.provider.release()
        self.provider=None;self.context=None

def _open_reader(**kwargs):return _PersistentCPUReader(**kwargs)

class _PersistentManager:
    def __init__(self,options,resource_check):
        self.options=dict(options);self.resource_check=resource_check;self.refs={};self.pool=None;self.active_parent=None
    def register(self,provider):
        parent=provider.plan["parent_id"];require(parent in TRAIN and parent not in self.refs,"One original provider per parent")
        self.refs[parent]=copy.deepcopy(provider.plan["plan_ref"])
    def check(self,event,pids):
        self.resource_check(event);rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        for pid,start in pids:
            owned=_rss(pid,start);require(owned<=self.options["worker_rss_cap_bytes"],"Exact worker RSS cap");rss+=owned
        require(rss<=self.options["aggregate_rss_cap_bytes"],"Main plus every exact persistent worker RSS cap")
    def activate(self,provider):
        parent=provider.plan["parent_id"]
        if self.pool is None:
            require(set(self.refs)==set(TRAIN),"All20 metadata constructors before first worker")
            self.pool=PersistentOrderedWorkersV1(workers=self.options["num_workers"],opener=_open_reader,
                opener_kwargs=dict(parent_plan_refs=copy.deepcopy(self.refs),worker_tmp_directory=provider.worker_tmp_directory,
                    max_block_bytes=self.options["max_block_bytes"],worker_rss_cap_bytes=self.options["worker_rss_cap_bytes"]),
                check=self.check,task_timeout_seconds=self.options["task_timeout_seconds"],drain_timeout_seconds=self.options["drain_timeout_seconds"])
        require(not self.pool.closed,"Persistent pool cannot reopen/retry")
        if self.active_parent is None:
            self.pool.bind(dict(parent_id=parent,plan_ref=copy.deepcopy(provider.plan["plan_ref"]),
                blocks=len(provider),seal_record=copy.deepcopy(provider.seal_record)))
            self.active_parent=parent
        else:require(self.active_parent==parent,"Release previous original parent before switching")
        return self.pool
    def release(self,provider,*,failed):
        if self.pool is None:return None
        if failed:
            result=self.pool.close(failed=True);self.active_parent=None;return dict(failed_pool_drain=result)
        if self.pool.closed:return None
        if self.active_parent is None:return None
        require(self.active_parent==provider.plan["parent_id"],"Exact active provider release")
        row=self.pool.release_parent();self.active_parent=None;return row
    def close(self,*,failed):
        if self.pool is None:return dict(drained=True,failed=bool(failed),worker_pids=[],worker_identities=[],exitcodes=[],max_inflight=self.options["num_workers"])
        result=self.pool.close(failed=failed);self.active_parent=None;return result

class PersistentSealedObservationProviderV1(PrefetchSealedObservationProviderV1):
    def __init__(self,*args,persistent_manager,**kwargs):
        super().__init__(*args,**kwargs);self.manager=persistent_manager;self.parent_releases=[]
        self.manager.register(self)
    def _pool(self):return self.manager.activate(self)
    def _release(self,*,failed):
        try:
            row=self.manager.release(self,failed=failed)
            if row is not None:self.parent_releases.append(copy.deepcopy(row))
        finally:
            self.engine=None;self.last_index=None;self.direction=1
            SealedObservationProviderV1.release(self)
    def diagnostics(self):
        return dict(schema="C3_persistent_observation_prefetch_diagnostics_v1",counters=dict(self.counters),
            bounds=dict(self.options),cache_bytes=self.cache_bytes,
            latest_parent_release=copy.deepcopy(self.parent_releases[-1]) if self.parent_releases else None,
            persistent_workers=True,worker_process_drain_at_factory_finalization=True,
            worker_time_excludes_discarded_prefetch=True,speedup_claimed=False)

class PersistentObservationProviderFactoryV1:
    def __init__(self,*,operational_caps,resource_check):
        keys={"num_workers","max_block_bytes","worker_rss_cap_bytes","aggregate_rss_cap_bytes","task_timeout_seconds","drain_timeout_seconds"}
        require(isinstance(operational_caps,dict) and set(operational_caps)==keys
            and operational_caps["num_workers"] in (2,4) and callable(resource_check),"Six root-bound persistent worker caps")
        ensure_abstract_ipc_before_torch_v1();_check_pins()
        self.options=copy.deepcopy(operational_caps);self.resource_check=resource_check
        self.manager=_PersistentManager(self.options,resource_check);self.providers={}
        paths=[REL+"/persistent_provider_v1.py",REL+"/persistent_workers_v1.py",
            "research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/prefetch_provider_isolated_v2.py",
            "research/reviews/C3_GNN_prefetch_provider_20261015_v1/bounded_workers_v1.py",
            "research/reviews/C3_GNN_prefetch_provider_20261015_v1/abstract_ipc_bootstrap_v1.py",
            "research/reviews/C3_GNN_prefetch_provider_20261015_v1/worker_entry_integration_api_v1.py",*PINS]
        self.execution_binding=dict(schema="C3_observation_provider_operational_v1",source_refs=[ref(p) for p in paths],
            IPC_bootstrap=dict(IPC_BOOTSTRAP),num_workers=self.options["num_workers"],max_inflight=self.options["num_workers"],
            persistent_one_pool=True,cache_requires_full_block=True,cached_features_rehashed=True,
            observation_equations_unchanged=True,checkpoint_parent_gradient_unchanged=True,
            bounds=copy.deepcopy(self.options),registered_parent_plan_refs={})
    def __call__(self,plan_ref,*,parent_id,worker_tmp_directory,cache_bytes):
        require(parent_id not in self.providers,"One provider per frozen original parent")
        value=PersistentSealedObservationProviderV1(plan_ref,parent_id=parent_id,worker_tmp_directory=worker_tmp_directory,
            cache_bytes=cache_bytes,persistent_manager=self.manager,resource_check=self.resource_check,**self.options)
        self.providers[parent_id]=value
        self.execution_binding["registered_parent_plan_refs"][parent_id]=copy.deepcopy(plan_ref)
        return value
    def finalize(self,*,training_failed):
        errors=[]
        for parent,provider in self.providers.items():
            try:provider.release()
            except BaseException as error:errors.append(dict(parent_id=parent,error=type(error).__name__+": "+str(error)[:1000]))
        drain=self.manager.close(failed=bool(training_failed or errors))
        require(drain["drained"] is True and len(drain["worker_pids"])==len(drain["exitcodes"])
            and all(type(x)is int for x in drain["exitcodes"]),"Every persistent worker genuinely joined")
        releases=copy.deepcopy(self.manager.pool.release_receipts) if self.manager.pool is not None else []
        if not training_failed:
            require(not errors and len(self.providers)==20 and len(releases)==620
                and not drain["failed"] and all(x==0 for x in drain["exitcodes"]),"Full20seal+600update release contract and normal final pool drain")
        return dict(schema="C3_GNN_prefetch_all_worker_drains_v1",source_ref=ref(REL+"/persistent_provider_v1.py"),
            training_failed=bool(training_failed),errors=errors,parent_release_receipts=releases,parent_release_count=len(releases),
            global_drain=copy.deepcopy(drain),drain_count=1 if self.manager.pool is not None else 0,
            worker_pids=copy.deepcopy(drain["worker_pids"]),all_workers_joined=True,
            all_workers_zero_exit=all(x==0 for x in drain["exitcodes"]),persistent_one_pool=True,
            exact_owned_Process_object_joins=True,outer_process_tree_supervision_still_required=True)
