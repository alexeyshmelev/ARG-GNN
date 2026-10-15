"""Focused synthetic CPU cache test: no observation payload, model, worker or guard."""
import pathlib,sys,importlib.util,json,hashlib,types,copy
from collections import OrderedDict
ROOT=pathlib.Path("/workspace-SR003.nfs2/ARG-GNN");sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from research.reviews.C3_GNN_prefetch_provider_20261015_v1.abstract_ipc_bootstrap_v1 import ensure_abstract_ipc_before_torch_v1
ensure_abstract_ipc_before_torch_v1()
import numpy as np,torch
from arggnn import american_c3_gnn_stream_train_isolated_v2 as stream
from arggnn.american_c3_gnn_dataset_isolated_v2 import BANK_FIELDS,BLOCK_TENSORS
from research.reviews.C3_GNN_prefetch_isolated980_root_20261015_v2.prefetch_provider_isolated_v2 import _buffer_feature_identity
HERE=ROOT/"research/reviews/C3_GNN_immutable_RAM_cache_source_20261015_v1"
sp=importlib.util.spec_from_file_location("_focused_RAM_cache",HERE/"ram_cache_provider_v1.staged.py")
ram=importlib.util.module_from_spec(sp);sp.loader.exec_module(ram)
checks=[]
def ck(n,v):
    if not v:raise AssertionError(n)
    checks.append(n)
def rejects(n,f):
    try:f()
    except (ValueError,RuntimeError,AssertionError):checks.append(n);return
    raise AssertionError(n)
parent=stream.TRAIN[0];pr=dict(path="synthetic_plan.json",sha256="a"*64)
records=dict(parent_id=np.asarray(parent),class_order=np.asarray(["EUR","EAS","AFR"]),
    record_ids=np.arange(10000,dtype=np.int64),sample_nodes=np.arange(10000,dtype=np.int64),
    sample_indices=600+np.arange(10000,dtype=np.int64)%1000,positions_bp0=np.arange(10000,dtype=np.float64))
class Block(dict):
    def __getattr__(self,k):return self[k]
    def clone(self):
        def cloned(x):return tuple(cloned(v) for v in x) if isinstance(x,tuple) else x.clone() if torch.is_tensor(x) else copy.deepcopy(x)
        return Block({k:cloned(v) for k,v in self.items()})
def block(i):
    b=Block()
    for name in BANK_FIELDS+BLOCK_TENSORS:
        b[name]=(torch.arange(3,dtype=torch.long),) if name in ("upward_levels","upward_children","upward_parent_rank","downward_levels") else torch.arange(3,dtype=torch.float32)
    rows=np.arange(i*5000,(i+1)*5000,dtype=np.int64)
    b.update(focal_rows=torch.as_tensor(rows),focal_hold=torch.zeros(5000,dtype=torch.long),
        focal_sample_column=torch.as_tensor(records["sample_indices"][rows]),
        parent_id=parent,plan_ref=copy.deepcopy(pr),block_index=i,start_tree=i,stop_tree=i+1,
        sequence_length_bp=2000000.,complete_parent=False)
    return b
class Dataset:
    def __init__(self):self._state=None;self._pid=None
    def __len__(self):return 2
class Manager:
    def __init__(self,blocks):self.blocks=blocks;self.reads=0;self.active=False;self.releases=[];self.binds=0
    def activate(self,p):
        if not self.active:self.active=True;self.binds+=1
        return self
    def take(self,i,direction,skip):
        self.reads+=1;b=self.blocks[i];digest,size=_buffer_feature_identity(b)
        return b,digest,size,dict(read_and_original_hash_seconds=0.)
    def release(self,p,*,failed):
        if not self.active:return None
        self.active=False;r=dict(parent_id=parent,failed=failed);self.releases.append(r);return r
def make():
    p=ram.ImmutableRAMObservationProviderV1.__new__(ram.ImmutableRAMObservationProviderV1)
    p.plan=dict(parent_id=parent,plan_ref=pr,observation_ref=dict(path="synthetic_observed.json",sha256="b"*64),
        focal_registry_ref=dict(path="synthetic_focals.npz",sha256="c"*64),trees=2,sites=0,haplotypes=1600,
        focal_records=10000,class_order=["EUR","EAS","AFR"],blocks=[dict(start_tree=i,stop_tree=i+1,nodes=3,sites=0) for i in range(2)])
    p.dataset=Dataset();p.ram_budget=ram._RAMBudgetV1(ram.CACHE_CAPS);p.cache=OrderedDict();p.used=0
    p.cache_bytes=2*2**30;p.tensor_byte_bound=ram.tensor_byte_bound_v1(p.plan)
    p._sealing=False;p._sealed_cache_complete=False;p._cache_poisoned=False;p.seal_record=None
    p.options=dict(max_block_bytes=128*2**20);p.manager=Manager([block(0),block(1)])
    p.counters=dict(cache_hits=0,cache_misses=0,delivered_worker_blocks=0,delivered_worker_read_and_hash_seconds=0.,
        demand_wait_seconds=0.,transport_hash_seconds=0.,cache_hash_seconds=0.)
    p.engine=None;p.direction=1;p.last_index=None;p.parent_releases=[]
    return p
original_registry=stream.registry
try:
    stream.registry=lambda r:copy.deepcopy(records)
    p=make()
    for b in p.manager.blocks:ck("original_and_zero_copy_digest_equal",stream._feature_identity(b)==_buffer_feature_identity(b))
    baseline=stream.SealedObservationProviderV1.__new__(stream.SealedObservationProviderV1)
    baseline.plan=copy.deepcopy(p.plan);baseline.dataset=Dataset();baseline.cache=OrderedDict();baseline.used=0;baseline.seal_record=None
    baseline._read=types.MethodType(lambda obj,i:(p.manager.blocks[i],*_buffer_feature_identity(p.manager.blocks[i])),baseline)
    old_seal=baseline.seal(resource_check=lambda event:None)
    seal=p.seal(resource_check=lambda event:None)
    ck("required_original_seal_byte_identical",json.dumps(seal,sort_keys=True)==json.dumps(old_seal,sort_keys=True))
    ck("seal_retains_all_complete_blocks",len(p.cache)==2 and p._sealed_cache_complete and p.used==p.ram_budget.used)
    ck("seal_releases_worker_parent",len(p.manager.releases)==1 and p.manager.active is False)
    reads=p.manager.reads
    for i in [0,1,1,0]:
        b=p.get(i);ck("cache_tensor_digest_equal_to_required_seal",_buffer_feature_identity(b)[0]==seal["block_feature_sha256"][i])
        clone=b.clone();clone["genotype"][0]+=1
        ck("caller_clone_leaves_cached_bank_unchanged",_buffer_feature_identity(b)[0]==seal["block_feature_sha256"][i])
    ck("reused_banks_never_rebuilt",p.manager.reads==reads and p.counters["cache_misses"]==0)
    p.release();ck("step_release_retains_banks_and_clears_worker_context",len(p.cache)==2 and len(p.manager.releases)==2 and not p.manager.active)
    p.get(0);ck("next_context_reuses_original_CPU_object",p.cache[0][0] is p.manager.blocks[0] and p.manager.reads==reads)
    p.cache[0][0]["genotype"][0]+=1
    rejects("cached_tensor_mutation_fails_closed",lambda:p.get(0))
    ck("mutation_poison_clears_ownership",p._cache_poisoned and not p.cache and p.ram_budget.used==0)
    rejects("poisoned_cache_cannot_reopen",lambda:p.get(1))
    p=make();p.seal(resource_check=lambda event:None);p.cache[0][0]["plan_ref"]=dict(path="wrong",sha256="d"*64)
    rejects("cached_plan_identity_mutation_fails_closed",lambda:p.get(0))
    budget=ram._RAMBudgetV1(ram.CACHE_CAPS);budget.reserve(parent,0,2*2**30)
    rejects("per_parent_cap_exact_fail_stop",lambda:budget.reserve(parent,1,1))
    rejects("duplicate_cache_ownership_rejected",lambda:budget.reserve(parent,0,1))
    budget.release_parent(parent);ck("explicit_final_ownership_clear",budget.used==0 and not budget.entries)
    p=make()
    rejects("unsealed_get_rejected",lambda:p.get(0))
finally:stream.registry=original_registry
receipt=dict(schema="C3_immutable_RAM_cache_focused_synthetic_fixture_v1",status="PASS",checks=checks,check_count=len(checks),
    fixture_source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
    RAM_provider_source_sha256=hashlib.sha256((HERE/"ram_cache_provider_v1.staged.py").read_bytes()).hexdigest(),
    synthetic_blocks=2,synthetic_focals=10000,actual_observation_payload_opened=False,ancestry_truth_opened=False,
    model_forward_or_update=False,worker_process_spawn=False,GPU_probe=False,guard_execution=False,actual_training=False)
(HERE/"focused_cache_fixture_receipt.v2.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(dict(status=receipt["status"],check_count=len(checks)),sort_keys=True))
