"""Exact complete-parent TRAIN gradients with checkpointed whole-tree blocks.

Root supplies execution/resource authority. This module owns no launcher, lease,
storage accounting, worker process or CLI. CPU boundary/focal tensors keep their
autograd edges. Their storage grows with block/focal count, not all-node GPU
activations. Actual full-training memory/time remain unqualified.
"""
from __future__ import annotations
from collections import OrderedDict
from typing import Protocol,runtime_checkable
import hashlib,json,resource
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .american_c3_common_v1 import require,CLASSES
from .american_c3_evaluate_v1 import registry
from .american_c3_gnn_dataset_isolated_v2 import AmericanC3WholeTreeDatasetV1,BANK_FIELDS,BLOCK_TENSORS
from .american_c3_gnn_model_v1 import AmericanC3WholeTreeGNNV1
from .american_c3_gnn_targets_v1 import TRAIN,TrainFocalTargetVectorV1,identity_sha

BASELINE="C3_typed_gated_upward_recipient_excluded_cavity_v1"

@runtime_checkable
class C3BackboneV1(Protocol):
    """Audited baseline interface; a later H043 layer requires its own audit."""
    hidden_dim:int
    class_order:tuple
    num_classes:int
    def encode_bank(self,bank):...
    def scan_context(self,context,geometry,*,direction,hidden=None):...
    def classify_context(self,context,forward,reverse):...
    def checkpoint_contract(self):...

def require_baseline_v1(model,*,extension=None):
    require(extension is None and type(model) is AmericanC3WholeTreeGNNV1
        and isinstance(model,C3BackboneV1) and model.class_order==CLASSES,
        "Only audited typed C3 baseline accepted; H043 extension disabled")

def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def _tensors(value):
    if torch.is_tensor(value):yield value
    elif isinstance(value,(tuple,list)):
        for item in value:yield from _tensors(item)
    else:raise ValueError("Only observation tensors/level tuples in feature bank")

def _feature_identity(block):
    h=hashlib.sha256();size=0
    for name in BANK_FIELDS+BLOCK_TENSORS:
        h.update(name.encode())
        for t in _tensors(block[name]):
            require(t.device.type=="cpu" and t.layout==torch.strided,"CPU complete-tree tensors required")
            a=t.detach().contiguous().numpy();size+=a.nbytes
            h.update(str(a.dtype).encode());h.update(json.dumps(list(a.shape)).encode());h.update(a.tobytes())
    h.update(json.dumps({k:getattr(block,k) for k in
        ("parent_id","block_index","start_tree","stop_tree","sequence_length_bp","complete_parent")},
        sort_keys=True).encode())
    return h.hexdigest(),size

def validate_plan_v1(plan,*,parent_id,length_bp):
    cfg=plan["config"];blocks=plan["blocks"]
    require(plan["schema"]=="american_c3_whole_tree_block_index_v1"
        and plan["parent_id"]==parent_id and plan["sequence_length_bp"]==length_bp
        and plan["class_order"]==list(CLASSES) and plan["haplotypes"]==1600
        and plan["original_reference_counts"]==[200,200,200]
        and plan["effective_reference_counts"]==[200,200,200]
        and plan["focal_records"]==10000 and plan["all_trees_and_samples_retained"] is True
        and plan["targets_opened"] is False and blocks and plan["trees"]>0
        and cfg["class_order"]==list(CLASSES) and cfg["expected_haplotypes"]==1600
        and cfg["input_reference_class_mask"] in (None,[True,True,True])
        and cfg["input_reference_sample_indices"] is None and cfg["allow_missing"] is False
        and cfg["donor_age_split"]==350.,"Exact complete original C3 baseline plan/mask/cardinality")
    cursor=0
    for b in blocks:
        require(type(b["start_tree"]) is int and type(b["stop_tree"]) is int
            and b["start_tree"]==cursor and b["stop_tree"]>cursor and b["nodes"]>0 and b["sites"]>=0
            and b["stop_tree"]-b["start_tree"]<=cfg["max_trees_per_block"]
            and b["nodes"]<=cfg["max_nodes_per_block"] and b["sites"]<=cfg["max_sites_per_block"],
            "Every original whole tree must occur in one contiguous bounded block")
        cursor=b["stop_tree"]
    require(cursor==plan["trees"] and sum(b["sites"] for b in blocks)==plan["sites"],
        "Complete original tree/site plan coverage")

class SealedObservationProviderV1:
    """One parent CPU Dataset and bounded LRU; no targets or private paths."""
    def __init__(self,plan_ref,*,parent_id,worker_tmp_directory,cache_bytes):
        require(parent_id in TRAIN and type(cache_bytes) is int and cache_bytes>=0,
            "TRAIN-only bounded CPU observation provider")
        self.dataset=AmericanC3WholeTreeDatasetV1(plan_ref,direction="forward",
            worker_tmp_directory=worker_tmp_directory)
        self.plan=dict(self.dataset.plan,plan_ref=dict(self.dataset.plan_ref))
        validate_plan_v1(self.plan,parent_id=parent_id,length_bp=2000000)
        self.cache_bytes=cache_bytes;self.cache=OrderedDict();self.used=0;self.seal_record=None
    def __len__(self):return len(self.dataset)
    def _read(self,index):
        block=self.dataset[index];spec=self.plan["blocks"][index]
        require(block.parent_id==self.plan["parent_id"] and block.plan_ref==self.plan["plan_ref"]
            and block.block_index==index and block.start_tree==spec["start_tree"]
            and block.stop_tree==spec["stop_tree"] and block.sequence_length_bp==2000000,
            "Same exact physical observation block on every recomputation")
        require(torch.equal(block.query_column,torch.arange(600,1600))
            and torch.equal(block.reference_class,torch.tensor([2]*200+[0]*200+[1]*200+[-1]*1000)),
            "Original AFR/EUR/EAS/reference roster and unknown query columns only")
        rows=block.focal_rows
        require(rows.dtype==torch.long and rows.ndim==1 and block.focal_hold.shape==rows.shape
            and block.focal_sample_column.shape==rows.shape
            and (rows.numel()==0 or bool(((rows>=0)&(rows<10000)).all()))
            and bool(((block.focal_sample_column>=600)&(block.focal_sample_column<1600)).all())
            and bool(((block.focal_hold>=0)&(block.focal_hold<block.stop_tree-block.start_tree)).all()),
            "Exact prespecified query readout indices")
        digest,size=_feature_identity(block)
        if self.seal_record is not None:
            require(digest==self.seal_record["block_feature_sha256"][index],
                "Observation features changed after all-parent seal; no continuation")
        return block,digest,size
    def get(self,index):
        require(type(index) is int and 0<=index<len(self),"Physical block index")
        if index in self.cache:
            block,digest,size=self.cache.pop(index)
            # Cached tensors are immutable by contract; callers only transfer clones.
            actual,_=_feature_identity(block);require(actual==digest,"CPU feature cache mutated")
            self.cache[index]=(block,digest,size);return block
        block,digest,size=self._read(index)
        if size<=self.cache_bytes:
            while self.cache and self.used+size>self.cache_bytes:
                _,(_,_,n)=self.cache.popitem(last=False);self.used-=n
            self.cache[index]=(block,digest,size);self.used+=size
        return block
    def seal(self,*,resource_check):
        require(self.seal_record is None and callable(resource_check),"One observation seal before targets")
        records=registry(self.plan["focal_registry_ref"])
        require(records["parent_id"].item()==self.plan["parent_id"] and len(records["record_ids"])==10000
            and bool(np.isin(records["sample_indices"],np.arange(600,1600)).all()),
            "Exact original common query focal registry")
        seen=np.zeros(10000,dtype=np.int64);hashes=[];max_bytes=0
        for i in range(len(self)):
            resource_check("observation_seal_block")
            block,digest,size=self._read(i);rows=block.focal_rows.numpy()
            np.add.at(seen,rows,1);hashes.append(digest);max_bytes=max(max_bytes,size)
            require(np.array_equal(block.focal_sample_column.numpy(),records["sample_indices"][rows]),
                "Focal global-row/sample-column identity")
            del block
        require(bool((seen==1).all()),"Every common focal covered exactly once before any truth")
        self.seal_record=dict(schema="american_c3_observation_blocks_sealed_v1",
            parent_id=self.plan["parent_id"],plan_ref=self.plan["plan_ref"],
            observation_ref=self.plan["observation_ref"],focal_registry_ref=self.plan["focal_registry_ref"],
            focal_identity_sha256=identity_sha(records),blocks=len(self),trees=self.plan["trees"],
            focal_records=10000,block_feature_sha256=hashes,largest_feature_block_bytes=max_bytes,
            observation_only=True,all_blocks_sealed_before_truth=True,all_trees_and_samples_retained=True)
        self.release();return dict(self.seal_record)
    def release(self):
        self.cache.clear();self.used=0;self.dataset._state=None;self.dataset._pid=None

def _parameter_sha(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def checkpointed_complete_parent_step_v1(model,provider,targets,optimizer,*,device,
        resource_check,gradient_clip=1.,focal_chunk_size=1024,first_update=False):
    """One all-focal mean CE and optimizer update, with exact cross-block BPTT.

    Encoder+directional GRU closures reload only complete sealed blocks. Their
    hidden/focal outputs are copied to CPU WITH autograd edges. No detach/reset,
    finite halo, pruning, block loss or block optimizer step is substituted.
    Re-encoding the deterministic local encoder in each direction gives the
    same complete-parent objective with all recurrent gradient paths. FP32
    reductions and max-pooling tie subgradients can depend on block grouping;
    bitwise dense-gradient equivalence is not claimed.
    """
    require_baseline_v1(model)
    require(isinstance(provider,SealedObservationProviderV1) and provider.seal_record is not None
        and isinstance(targets,TrainFocalTargetVectorV1) and targets.parent_id==provider.plan["parent_id"]
        and targets.focal_identity_sha256==provider.seal_record["focal_identity_sha256"]
        and targets.labels_cpu.dtype==torch.long and targets.labels_cpu.shape==(10000,)
        and bool(((targets.labels_cpu>=0)&(targets.labels_cpu<3)).all())
        and type(focal_chunk_size) is int and focal_chunk_size>0 and gradient_clip==1.
        and callable(resource_check),"Sealed complete TRAIN parent/external target contract")
    actual=next(model.parameters()).device;requested=torch.device(device)
    require(actual==requested and all(p.dtype==torch.float32 and p.device==actual for p in model.parameters()),
        "Root-transferred FP32 model on one explicit admitted device")
    resource_check("complete_parent_begin");model.train();optimizer.zero_grad(set_to_none=True)
    before=_parameter_sha(model) if first_update else None
    if actual.type=="cuda":torch.cuda.reset_peak_memory_stats(actual)
    dummy=torch.zeros((),device=actual,dtype=torch.float32,requires_grad=True)
    n=10000;d=model.hidden_dim
    # Only CPU boundary tensors enter checkpoint save slots, except scalar dummy.
    def scan(direction):
        hidden=torch.zeros((1,1600,d),dtype=torch.float32,device="cpu")
        rows_parts=[];local_parts=[];history_parts=[]
        order=range(len(provider)) if direction=="forward" else range(len(provider)-1,-1,-1)
        for index in order:
            cpu=provider.get(index);rows=cpu.focal_rows.clone();del cpu
            def evaluate(_dummy,state,physical=index,scan_direction=direction):
                resource_check("checkpoint_block_or_recompute")
                block=provider.get(physical).clone().to(actual)
                context=model.encode_bank(block.to_model_inputs())
                history,next_state=model.scan_context(context,block.genomic_geometry,
                    direction=scan_direction,hidden=state.to(actual))
                focal_history=history[block.focal_hold,block.focal_sample_column].to("cpu")
                local=(context[block.focal_hold,block.focal_sample_column].to("cpu")
                    if scan_direction=="forward" else context.new_empty((0,d)).to("cpu"))
                return next_state.to("cpu"),local,focal_history
            hidden,local,history=checkpoint(evaluate,dummy,hidden,use_reentrant=False,preserve_rng_state=True)
            rows_parts.append(rows);history_parts.append(history)
            if direction=="forward":local_parts.append(local)
        joined=torch.cat(rows_parts);permutation=torch.argsort(joined)
        require(torch.equal(joined[permutation],torch.arange(n)),"Exact one-to-one global focal row join")
        histories=torch.cat(history_parts).index_select(0,permutation)
        locals_=torch.cat(local_parts).index_select(0,permutation) if local_parts else None
        return locals_,histories
    try:
        local,forward=scan("forward");_,reverse=scan("reverse")
        losses=[]
        for start in range(0,n,focal_chunk_size):
            stop=min(start+focal_chunk_size,n);labels=targets.labels_cpu[start:stop]
            def classify(_dummy,x,f,r,y):
                resource_check("focal_classifier_or_recompute")
                logits=model.classify_context(x.to(actual),f.to(actual),r.to(actual))
                return F.cross_entropy(logits,y.to(actual),reduction="sum").to("cpu")
            losses.append(checkpoint(classify,dummy,local[start:stop],forward[start:stop],
                reverse[start:stop],labels,use_reentrant=False,preserve_rng_state=True))
        loss=torch.stack(losses).sum()/n
        require(bool(torch.isfinite(loss)),"Finite complete-parent mean CE")
        resource_check("complete_parent_backward");loss.backward()
        grads=[p.grad for p in model.parameters() if p.requires_grad]
        require(all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
            and any(bool((g!=0).any()) for g in grads),"Finite gradients through every baseline parameter and nonzero global gradient")
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),gradient_clip,error_if_nonfinite=True)
        require(bool(torch.isfinite(norm)),"Finite unclipped gradient norm")
        resource_check("complete_parent_optimizer");optimizer.step()
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),"Finite updated FP32 parameters")
        after=_parameter_sha(model) if first_update else None
        if first_update:require(before!=after,"First actual optimizer update must change fresh parameters")
        resource_check("complete_parent_end")
        memory=dict(process_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
            CUDA_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(actual)) if actual.type=="cuda" else None,
            CUDA_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(actual)) if actual.type=="cuda" else None,
            CPU_cache_cap_bytes=provider.cache_bytes,CPU_boundary_state_graph_scales_with_blocks=True)
        return dict(parent_id=targets.parent_id,records=n,loss=float(loss.detach()),gradient_norm=float(norm),
            condition="fresh_C3_supervised_short_independent_complete_parents",training_length_bp=2000000,
            whole_parent_optimizer_updates=1,forward_blocks=len(provider),reverse_blocks=len(provider),
            checkpointed_encoder_and_GRU=True,block_reset_or_truncated_BPTT=False,
            detached_recurrent_history=False,all_trees_and_samples_retained=True,
            bitwise_dense_gradient_equivalence_claimed=False,
            first_real_update_checks=(dict(finite_loss=True,all_parameter_gradients_finite=True,
                global_gradient_nonzero=True,updated_parameters_finite=True,parameter_sha_before=before,
                parameter_sha_after=after,checked_inside_full_training=True) if first_update else None),
            memory=memory,paper_native_training_equivalent=False)
    finally:
        # Backward completes before releasing source state needed by recomputation.
        optimizer.zero_grad(set_to_none=True);provider.release()
