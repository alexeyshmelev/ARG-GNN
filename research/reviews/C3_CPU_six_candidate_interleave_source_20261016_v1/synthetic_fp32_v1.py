"""Tiny synthetic CPU FP32 equivalence, never reads scientific artifacts."""
from __future__ import annotations
from pathlib import Path
import ast,copy,hashlib,json,math,os,sys,time,types
import numpy as np
import torch
from arggnn.american_c3_common_v1 import require
from arggnn.american_c3_gnn_model_v1 import AmericanC3WholeTreeGNNV1
from arggnn.american_c3_gnn_dataset_isolated_v2 import (
    AmericanC3WholeTreeDatasetV1, AmericanC3BlockV1, make_dataloader,
    shutdown_dataloader)
from arggnn.american_c3_gnn_observations_isolated_v2 import encode_block
from arggnn.mutation_routed_tree_observations_v1 import NativeObservedTreeV1
from arggnn.american_c3_evaluate_v1 import verify_prediction
from .six_candidate_scan_v1 import interleaved_six_focal_predictions_v1

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/C3_CPU_six_candidate_interleave_source_20261016_v1"
OLD=ROOT/"src/arggnn/american_c3_gnn_run_isolated_v2.py"
OLD_SHA="3115019fdfe7cb39e19aeb84fc806827fc61138a57e47cbfd5a9e9392052916f"
INTERVALS=np.asarray([[0,2],[2,5],[5,6],[6,10],[10,12]],dtype=np.float32)
POSITIONS=np.asarray([.5,1.5,2.5,4.,6.5,9.,10.5,11.5],dtype=np.float32)
GENOTYPES=np.asarray([[0,1,0,1,0,1],[1,0,1,0,1,0],
    [0,0,1,1,0,1],[1,1,0,0,1,0],[1,0,0,1,1,0],
    [0,1,1,0,0,1],[1,1,1,0,1,0],[0,0,0,1,0,1]],dtype=np.int8)
AGES=np.asarray([0,400,0,400,0,0],dtype=np.float32)
LABELS=np.asarray([0,1,2,-1,-1,-1],dtype=np.int64)
FOCAL_TREES=np.asarray([0,1,3,4],dtype=np.int64)
SAMPLES=np.asarray([3,4,5,3],dtype=np.int64)
RECORDS=dict(record_ids=np.asarray([11,23,37,41],dtype=np.int64),
    sample_nodes=SAMPLES.copy(),sample_indices=SAMPLES.copy(),
    positions_bp0=np.asarray([1.,3.,8.,11.],dtype=np.float32),
    parent_id=np.asarray("tiny_synthetic_parent"),
    class_order=np.asarray(["EUR","EAS","AFR"]))


def tree(i):
    parents=[
        [6,6,7,7,8,8,9,9,9,-1],
        [6,7,7,8,8,8,9,9,9,-1],
        [-1]*6,
        [6,6,6,6,6,6,-1],
        [7,6,6,7,8,8,9,9,9,-1]][i]
    p=np.asarray(parents,dtype=np.int64)
    clocks=AGES.copy() if len(p)==6 else np.concatenate((
        AGES,np.asarray([1000] if len(p)==7 else [600,700,800,1000],dtype=np.float32)))
    return NativeObservedTreeV1(p,clocks,np.arange(6,dtype=np.int64),np.arange(6,dtype=np.int64))


class TinyDataset(AmericanC3WholeTreeDatasetV1):
    def __init__(self,bounds,direction,counts,*,broken=False):
        self.bounds=bounds
        self.direction=direction
        self.counts=counts
        self.broken=broken
        self.plan_ref=dict(path="synthetic_only/plan.json",sha256="0"*64)
        self.plan=dict(parent_id="tiny_synthetic_parent",class_order=["EUR","EAS","AFR"],
            trees=5,blocks=bounds,focal_registry_ref=dict(path="synthetic_only/registry.npz",sha256="1"*64),
            observation_ref=dict(path="synthetic_only/observation.json",sha256="2"*64),
            sequence_length_bp=12)
    def __len__(self):
        return len(self.bounds)
    def __getitem__(self,index):
        self.counts[self.direction]+=1
        actual=index if self.direction=="forward" else len(self.bounds)-1-index
        a,b=self.bounds[actual]
        lo,hi=np.searchsorted(POSITIONS,[INTERVALS[a,0],INTERVALS[b-1,1]])
        bank,receipt=encode_block([tree(i) for i in range(a,b)],INTERVALS[a:b],
            GENOTYPES[lo:hi],np.full(hi-lo,2,dtype=np.int8),POSITIONS[lo:hi],
            LABELS,AGES,num_classes=3,age_split=350.,query_columns=np.asarray([3,4,5]))
        spans=INTERVALS[:,1]-INTERVALS[:,0]
        previous=np.asarray([0.,2.5,2.,2.5,3.],dtype=np.float32)
        following=np.asarray([2.5,2.,2.5,3.,0.],dtype=np.float32)
        geometry=np.stack((np.log1p(spans[a:b]),np.log1p(previous[a:b]),np.log1p(following[a:b])),-1)/math.log(2000001.)
        rows=np.flatnonzero((FOCAL_TREES>=a)&(FOCAL_TREES<b))
        if self.broken and self.direction=="forward" and actual==0:
            rows=np.concatenate((rows,rows[:1]))
        return AmericanC3BlockV1(bank,parent_id="tiny_synthetic_parent",plan_ref=self.plan_ref,
            block_index=actual,start_tree=a,stop_tree=b,sequence_length_bp=12,
            complete_parent=a==0 and b==5,
            genomic_geometry=torch.as_tensor(geometry,dtype=torch.float32),
            focal_rows=torch.as_tensor(rows,dtype=torch.long),
            focal_hold=torch.as_tensor(FOCAL_TREES[rows]-a,dtype=torch.long),
            focal_sample_column=torch.as_tensor(SAMPLES[rows],dtype=torch.long),
            observations_receipt=receipt,worker_pid=os.getpid())


def loaders(bounds,counts,broken=False):
    return [make_dataloader(TinyDataset(bounds,d,counts,broken=broken),
        num_workers=0,pin_memory=False) for d in ("forward","reverse")]


def reference():
    raw=OLD.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==OLD_SHA,"Exact unchanged reference arithmetic source")
    node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)
              and n.name=="_streamed_focal_predictions_impl")
    namespace=dict(torch=torch,np=np,require=require,
        AmericanC3WholeTreeGNNV1=AmericanC3WholeTreeGNNV1,
        AmericanC3WholeTreeDatasetV1=AmericanC3WholeTreeDatasetV1,
        shutdown_dataloader=shutdown_dataloader,
        registry=lambda record:RECORDS,verify_prediction=verify_prediction)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
        str(OLD)+":unchanged_synthetic_registry_only", "exec"),namespace)
    return namespace[node.name]


def floating_tensors(value):
    if torch.is_tensor(value):
        if value.is_floating_point():
            require(value.dtype==torch.float32,"Every floating computation tensor is FP32")
    elif isinstance(value,(tuple,list)):
        for x in value:floating_tensors(x)
    elif isinstance(value,dict):
        for x in value.values():floating_tensors(x)


def bank_snapshot(bank):
    return {name:tuple(x.clone() for x in value) if isinstance(value,tuple) else value.clone()
            for name,value in vars(bank).items()}


def unchanged_bank(bank,snapshot):
    for name,before in snapshot.items():
        now=getattr(bank,name)
        pairs=zip(now,before) if isinstance(before,tuple) else [(now,before)]
        require(all(torch.equal(a,b) for a,b in pairs),"No candidate may mutate shared feature tensors")


def capture(model,events,counters):
    encode=model.encode_bank
    scan=model.scan_context
    classify=model.classify_context
    def encoded(bank):
        rng=torch.random.get_rng_state().clone()
        floating_tensors(vars(bank))
        before=bank_snapshot(bank)
        out=encode(bank)
        unchanged_bank(bank,before)
        require(torch.equal(rng,torch.random.get_rng_state()),"Graph encoding consumes no RNG")
        counters["unchanged_feature_checks"]+=1
        floating_tensors(out)
        events.append(("context",out.clone()))
        return out
    def scanned(context,geometry,*,direction,hidden=None):
        rng=torch.random.get_rng_state().clone()
        out,state=scan(context,geometry,direction=direction,hidden=hidden)
        floating_tensors((out,state))
        require(torch.equal(rng,torch.random.get_rng_state()),"GRU eval consumes no RNG")
        events.append(("scan_"+direction,(out.clone(),state.clone())))
        return out,state
    def classified(context,forward,reverse):
        rng=torch.random.get_rng_state().clone()
        out=classify(context,forward,reverse)
        floating_tensors(out)
        require(torch.equal(rng,torch.random.get_rng_state()),"Readout consumes no RNG")
        events.append(("logits",out.clone()))
        return out
    model.encode_bank=encoded
    model.scan_context=scanned
    model.classify_context=classified


def compare(a,b):
    if torch.is_tensor(a):
        require(a.shape==b.shape and a.dtype==b.dtype==torch.float32,"Same FP32 tensor shape/type")
        require(torch.allclose(a,b,atol=1e-6,rtol=1e-5),"Actual FP32 tensors match")
        return float((a-b).abs().max()) if a.numel() else 0.
    return max((compare(x,y) for x,y in zip(a,b)),default=0.)


def run_case(bounds,hidden):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(611003+hidden)
        models=[AmericanC3WholeTreeGNNV1(hidden_dim=hidden,expected_haplotypes=6,
            site_chunk_size=2,checkpoint_sites=True).eval() for _ in range(6)]
    old_models=copy.deepcopy(models)
    before=[{k:v.clone() for k,v in m.state_dict().items()} for m in models]
    old_events=[[] for _ in models]
    new_events=[[] for _ in models]
    counters=dict(unchanged_feature_checks=0)
    for m,e in zip(old_models,old_events):capture(m,e,counters)
    for m,e in zip(models,new_events):capture(m,e,counters)
    original=reference()
    old_counts=dict(forward=0,reverse=0)
    rng=torch.random.get_rng_state().clone()
    old=[]
    for model in old_models:
        a,b=loaders(bounds,old_counts)
        old.append(original(model,a,b,device="cpu"))
    serial_rng=torch.random.get_rng_state().clone()
    require(not torch.equal(rng,serial_rng),"Original DataLoader advances CPU RNG")
    torch.random.set_rng_state(rng)
    new_counts=dict(forward=0,reverse=0)
    guard_calls=[]
    a,b=loaders(bounds,new_counts)
    new=interleaved_six_focal_predictions_v1(models,a,b,records=RECORDS,
        runtime_guard=guard_calls.append)
    require(torch.equal(serial_rng,torch.random.get_rng_state()),
            "Successful interleaving preserves original12iterator RNG advancement")
    maximum=0.
    event_counts={}
    for k,((p,r),(q,s)) in enumerate(zip(old,new)):
        require([x[0] for x in old_events[k]]==[x[0] for x in new_events[k]],
                "Each model has the exact original numerical call order")
        for (kind,x),(_,y) in zip(old_events[k],new_events[k]):
            maximum=max(maximum,compare(x,y))
            event_counts[kind]=event_counts.get(kind,0)+1
        require(set(p)==set(q) and all(np.array_equal(p[name],q[name]) for name in p),
                "Every original focal identity and FP32 posterior is exactly equal")
        require(all(r[name]==s[name] for name in r),
                "Every original arithmetic receipt field is retained exactly")
        require(all(torch.equal(before[k][name],value) for name,value in models[k].state_dict().items()),
                "Eval never mutates model parameters or state")
    require(sum(old_counts.values())==6*sum(new_counts.values())==12*len(bounds),
            "Sixfold observed feature-build reduction with all model work retained")
    require(counters["unchanged_feature_checks"]==24*len(bounds),
            "Every old and interleaved model/block input checked for contamination")
    return dict(hidden=hidden,blocks=len(bounds),candidates=6,trees=5,
        focal_records=4,old_feature_builds=old_counts,new_feature_builds=new_counts,
        unchanged_feature_checks=counters["unchanged_feature_checks"],
        per_candidate_numerical_call_order_equal=True,model_states_unchanged=True,
        original_CPU_RNG_advancement_equal=True,all_focal_arrays_exactly_equal=True,
        original_receipt_fields_equal=True,max_tensor_abs_difference=maximum,
        compared_tensor_calls=event_counts,guard_calls=len(guard_calls))


def negative_checks():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(611007)
        models=[AmericanC3WholeTreeGNNV1(hidden_dim=4,expected_haplotypes=6,
            site_chunk_size=2).eval() for _ in range(6)]
    passed=[]
    def rejects(name,ms,broken=False,fail_guard=False):
        count=dict(forward=0,reverse=0)
        a,b=loaders([(0,2),(2,3),(3,4),(4,5)],count,broken=broken)
        def guard(event):
            if fail_guard and event=="before_original_candidate_block":
                raise ValueError("synthetic deadline reached")
        try:
            interleaved_six_focal_predictions_v1(ms,a,b,records=RECORDS,runtime_guard=guard)
        except ValueError:
            passed.append(name)
        else:raise ValueError("Expected rejection: "+name)
        require(getattr(a,"_iterator",None) is None and getattr(b,"_iterator",None) is None,
                "Zero-worker synthetic error paths leave no loader processes")
    rejects("five_candidates",models[:5])
    rejects("duplicate_model_identity",[models[0]]*6)
    alias=copy.deepcopy(models)
    alias[1].site_mlp[0].weight=alias[0].site_mlp[0].weight
    rejects("shared_candidate_parameter_storage",alias)
    rejects("duplicate_focal_row",models,broken=True)
    rejects("deadline_guard_propagates",models,fail_guard=True)
    return passed


def constructor_rng_check():
    from .prediction_bridge_v1 import new_candidate_models_v1
    from arggnn.american_c3_gnn_shared30mb_isolated_v2 import MODEL_KWARGS
    before=torch.random.get_rng_state().clone()
    with torch.device("cpu"):
        original=AmericanC3WholeTreeGNNV1(**MODEL_KWARGS)
    expected=torch.random.get_rng_state().clone()
    torch.random.set_rng_state(before)
    candidates=new_candidate_models_v1()
    require(torch.equal(expected,torch.random.get_rng_state()),
            "Six constructors retain exactly the original one-constructor RNG advancement")
    require(all(torch.equal(original.state_dict()[k],v)
                for k,v in candidates[0].state_dict().items()),
            "First constructor initialization remains identical before strict loading")
    require(len({id(m)for m in candidates})==6,"Six private candidate model objects")
    return dict(original_single_constructor_RNG_advancement_equal=True,
        first_initialization_equal=True,additional_five_constructor_draws_isolated=True,
        actual_checkpoint_payloads_loaded=False)


def main():
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","Synthetic CPU-only visibility")
    require(Path.cwd()==ROOT,"All synthetic work is repository-contained")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(),"No synthetic CUDA context")
    started=time.monotonic()
    cases=[run_case([(0,2),(2,3),(3,4),(4,5)],32),
           run_case([(0,5)],4)]
    negatives=negative_checks()
    constructors=constructor_rng_check()
    result=dict(schema="C3_six_candidate_interleave_synthetic_FP32_v1",
        status="PASS_SYNTHETIC_ONLY",torch=str(torch.__version__),
        torch_threads=torch.get_num_threads(),CUDA_initialized=torch.cuda.is_initialized(),
        actual_scientific_payloads_or_checkpoints_or_truth_opened=False,
        training_or_optimization_performed=False,all_floating_Torch_tensors="float32",
        real_full30Mb_runtime_memory_fit_or_speedup_measured=False,
        reference_source_sha256=OLD_SHA,cases=cases,negative_checks=negatives,
        constructor_RNG=constructors,
        elapsed_seconds=time.monotonic()-started)
    target=HERE/"synthetic_result.v1.json"
    with target.open("x") as stream:json.dump(result,stream,sort_keys=True,indent=2)
    print(json.dumps(result,sort_keys=True))
    return 0

if __name__=="__main__":raise SystemExit(main())
