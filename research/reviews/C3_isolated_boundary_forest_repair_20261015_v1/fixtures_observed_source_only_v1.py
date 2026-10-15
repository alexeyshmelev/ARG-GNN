"""Observation-only CPU functional fixtures; no training, processes, guards or GPU."""
from __future__ import annotations
import ast
import dataclasses
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=ROOT/"research/reviews/C3_isolated_boundary_forest_repair_20261015_v1"
sys.path.insert(0,str(ROOT/"src"))
os.environ["CUDA_VISIBLE_DEVICES"]=""
import numpy as np
import torch
import tskit
torch.set_num_threads(1)
from arggnn import american_c3_gnn_observations_v1 as original
from arggnn.mutation_routed_tree_observations_v1 import NativeObservedTreeV1
from arggnn.mutation_routed_whole_tree_gnn_v1 import _FLOAT_FIELDS,_INDEX_FIELDS,_LEVEL_FIELDS

STAGED=OUT/"american_c3_gnn_observations_isolated_v2.staged.py"
spec=importlib.util.spec_from_file_location("arggnn._c3_isolated_source_fixture",STAGED)
repaired=importlib.util.module_from_spec(spec);spec.loader.exec_module(repaired)
checks=[]
def check(name,condition):
    if not condition:raise AssertionError(name)
    checks.append(name)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def checked_json(ref):
    path=ROOT/ref["path"]
    check("hash:"+ref["path"],sha(path)==ref["sha256"])
    return json.loads(path.read_text())
def rejected(name,tree):
    try:repaired._layout_with_isolated_samples(tree)
    except ValueError:checks.append(name)
    else:raise AssertionError(name)
def same_bank(a,b):
    for field in dataclasses.fields(a):
        x,y=getattr(a,field.name),getattr(b,field.name)
        if isinstance(x,tuple):
            if len(x)!=len(y) or any(not torch.equal(u,v) for u,v in zip(x,y)):return False
        elif not torch.equal(x,y):return False
    return True
def bank_sha(bank):
    h=hashlib.sha256();size=0
    for field in dataclasses.fields(bank):
        h.update(field.name.encode())
        value=getattr(bank,field.name)
        for tensor in value if isinstance(value,tuple) else (value,):
            a=tensor.contiguous().numpy()
            h.update(str(a.dtype).encode());h.update(json.dumps(list(a.shape)).encode())
            h.update(memoryview(a.reshape(-1)).cast("B"));size+=a.nbytes
    return h.hexdigest(),size
def validate_bank(bank,trees,intervals,g,x,receipt):
    n=sum(len(t.parent) for t in trees);h=g.shape[1];nt=len(trees)
    check("bank_complete_nodes:"+str(intervals[0][0]),len(bank.parent)==n==receipt["nodes"])
    check("bank_full_sample_axis:"+str(intervals[0][0]),tuple(bank.sample_index.shape)==(nt,h))
    check("bank_all_sites:"+str(intervals[0][0]),torch.equal(bank.genotype,torch.from_numpy(g.copy())))
    check("bank_dtype:"+str(intervals[0][0]),all(getattr(bank,k).dtype==torch.float32 for k in _FLOAT_FIELDS) and all(getattr(bank,k).dtype==torch.long for k in _INDEX_FIELDS) and bank.genotype.dtype==torch.int8 and bank.donor_call_mask.dtype==torch.bool and all(t.dtype==torch.long for k in _LEVEL_FIELDS for t in getattr(bank,k)))
    check("bank_finite:"+str(intervals[0][0]),all(bool(torch.isfinite(getattr(bank,k)).all()) for k in _FLOAT_FIELDS))
    check("bank_shapes:"+str(intervals[0][0]),tuple(bank.edge_geometry.shape)==(n,4) and tuple(bank.branch_support_true.shape)==(n,5) and tuple(bank.log_donor_allele_frequency.shape)==(len(g),int(max(k_actual.max(initial=1),1)),6) and tuple(bank.donor_call_mask.shape)==(len(g),6))
    check("bank_original_spans:"+str(intervals[0][0]),torch.equal(bank.spans,torch.tensor([b-a for a,b in intervals],dtype=torch.float32)))
    expected=np.searchsorted(np.asarray(intervals)[:,1],x,side="right")
    check("bank_site_hold:"+str(intervals[0][0]),torch.equal(bank.site_hold,torch.from_numpy(expected)))
    cover=torch.cat(bank.upward_levels)
    check("bank_all_nodes_in_upward:"+str(intervals[0][0]),torch.equal(cover.sort().values,torch.arange(n)))
    check("bank_support_sham_copy:"+str(intervals[0][0]),torch.equal(bank.branch_support_true,bank.branch_support_sham))
    offset=0;isolates=0
    for hold,tree in enumerate(trees):
        p=np.asarray(tree.parent);count=len(p);sl=slice(offset,offset+count)
        expected=p.copy();expected[p>=0]+=offset
        check("native_parent:"+str(intervals[0][0])+":"+str(hold),torch.equal(bank.parent[sl],torch.from_numpy(expected)))
        check("native_samples:"+str(intervals[0][0])+":"+str(hold),torch.equal(bank.sample_index[hold],torch.from_numpy(tree.sample_node+offset)))
        check("native_times:"+str(intervals[0][0])+":"+str(hold),torch.equal(bank.node_time[sl],torch.tensor(tree.node_time,dtype=torch.float32)))
        if count>1 and np.all(p==-1):
            isolates+=1
            expected_support=torch.zeros((count,5));expected_support[:,4]=1
            check("isolated_support:"+str(hold),torch.equal(bank.branch_support_true[sl],expected_support))
            check("isolated_geometry:"+str(hold),bool((bank.edge_geometry[sl]==0).all()))
            check("isolated_no_downward:"+str(hold),all(not bool(((v>=offset)&(v<offset+count)).any()) for v in bank.downward_levels))
        offset+=count
    return dict(trees=nt,nodes=n,sites=len(g),isolated_forests=isolates,feature_bank_sha256=bank_sha(bank)[0],feature_bank_bytes=bank_sha(bank)[1],interval_bp=[intervals[0][0],intervals[-1][1]],all_float_tensors="torch.float32",sample_shape=list(bank.sample_index.shape))

# Static equality protects the original connected-tree implementation.
old_ast=ast.parse((ROOT/"src/arggnn/american_c3_gnn_observations_v1.py").read_text())
new_ast=ast.parse(STAGED.read_text())
old_fn={n.name:ast.dump(n,include_attributes=False) for n in old_ast.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
new_fn={n.name:ast.dump(n,include_attributes=False) for n in new_ast.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
check("donor_frequencies_AST_unchanged",old_fn["donor_frequencies"]==new_fn["donor_frequencies"])
check("native_tree_AST_unchanged",old_fn["native_tree"]==new_fn["native_tree"])
check("original_layout_function_import_unchanged",repaired._layout is original._layout)
check("original_support_function_import_unchanged",repaired.branch_concordance_support_v1 is original.branch_concordance_support_v1)
check("only_one_function_added",set(new_fn)-set(old_fn)=={"_layout_with_isolated_samples"})

inv=checked_json(dict(path=str((OUT/"observed_train20_root_inventory.v1.json").relative_to(ROOT)),sha256=sha(OUT/"observed_train20_root_inventory.v1.json")))
isolated_rows=[]
for row in inv["parents"]:
    manifest=checked_json(row["observation_ref"]);tr=manifest["tree_ref"]
    check("tree_hash:"+row["parent_id"],sha(ROOT/tr["path"])==tr["sha256"])
    ts=tskit.load(ROOT/tr["path"]);nodes=ts.samples()
    for forest in row["multiroot_or_rootless_trees"]:
        tree=repaired.native_tree(ts.at_index(forest["tree_index"]),nodes)
        before=[a.copy() for a in dataclasses.astuple(tree)]
        p,t,sn,sid,post,depth,height,descendant,duration=repaired._layout_with_isolated_samples(tree)
        check("all_isolate_semantics:"+row["parent_id"]+":"+str(forest["tree_index"]),len(p)==1600 and np.all(p==-1) and np.all(depth==0) and np.all(height==0) and np.all(descendant==1) and np.all(duration==0) and np.array_equal(sn,np.arange(1600)) and np.array_equal(sid,np.arange(1600)) and np.array_equal(np.sort(post),np.arange(1600)))
        check("input_arrays_immutable:"+row["parent_id"]+":"+str(forest["tree_index"]),all(np.array_equal(a,b) for a,b in zip(before,dataclasses.astuple(tree))))
        isolated_rows.append(dict(parent_id=row["parent_id"],tree_index=forest["tree_index"],nodes=len(p),interval_bp=forest["interval_bp"]))

# Malformed all-isolated and general forests remain rejected.
def native(p,t,sn,sid):
    return NativeObservedTreeV1(np.asarray(p),np.asarray(t),np.asarray(sn),np.asarray(sid))
for name,tree in [
    ("reject_missing_sample",native([-1,-1],[0.,0.],[0],[0])),
    ("reject_duplicate_sample",native([-1,-1],[0.,0.],[0,0],[0,1])),
    ("reject_duplicate_identity",native([-1,-1],[0.,0.],[0,1],[0,0])),
    ("reject_float_sample",native([-1,-1],[0.,0.],[0.,1.],[0,1])),
    ("reject_negative_identity",native([-1,-1],[0.,0.],[0,1],[-1,1])),
    ("reject_negative_time",native([-1,-1],[-1.,0.],[0,1],[0,1])),
    ("reject_nonfinite_time",native([-1,-1],[float("nan"),0.],[0,1],[0,1])),
    ("reject_out_of_range_sample",native([-1,-1],[0.,0.],[0,2],[0,1])),
    ("reject_general_multiroot_forest",native([-1,-1,0],[1.,0.,0.],[1,2],[0,1])),
    ("reject_cycle",native([1,0],[1.,2.],[0,1],[0,1])),
    ("reject_empty",native(np.array([],dtype=np.int64),[],np.array([],dtype=np.int64),np.array([],dtype=np.int64))),
]:rejected(name,tree)

first=inv["parents"][0]
manifest=checked_json(first["observation_ref"])
roster=checked_json(manifest["roster_ref"])["samples"]
positions=np.load(ROOT/manifest["positions_ref"]["path"],allow_pickle=False)
check("position_hash",sha(ROOT/manifest["positions_ref"]["path"])==manifest["positions_ref"]["sha256"])
check("genotype_hash",sha(ROOT/manifest["genotypes_ref"]["path"])==manifest["genotypes_ref"]["sha256"])
genotype=np.load(ROOT/manifest["genotypes_ref"]["path"],mmap_mode="r",allow_pickle=False)
check("alleles_hash",sha(ROOT/manifest["alleles_ref"]["path"])==manifest["alleles_ref"]["sha256"])
with (ROOT/manifest["alleles_ref"]["path"]).open() as f:allele_count=np.asarray([len(json.loads(line)) for line in f],dtype=np.int8)
labels=np.asarray([-1 if r["observed_reference_class"] is None else r["observed_reference_class"] for r in roster],dtype=np.int64)
ages=np.asarray([r["age_generations"] for r in roster],dtype=np.float64)
ts=tskit.load(ROOT/manifest["tree_ref"]["path"]);sample_nodes=ts.samples()
plan_path=ROOT/"experiments/results/C3_short_TRAIN_focal_plans_v1/american_c3_short_train_000/block_index.json"
plan=json.loads(plan_path.read_text())
check("exact_failed_parent_plan_observation",plan["observation_ref"]==first["observation_ref"])
check("actual_original_plan_all_trees_retained",plan["all_trees_and_samples_retained"] is True and plan["trees"]==ts.num_trees)
fixture_blocks=[]
for index in [0,1,len(plan["blocks"])-1]:
    block=plan["blocks"][index];trees=[];intervals=[]
    for tree_index in range(block["start_tree"],block["stop_tree"]):
        tree=ts.at_index(tree_index);trees.append(repaired.native_tree(tree,sample_nodes));intervals.append(tuple(tree.interval))
    lo,hi=np.searchsorted(positions,[intervals[0][0],intervals[-1][1]])
    g=np.asarray(genotype[lo:hi]);k_actual=allele_count[lo:hi];x=positions[lo:hi]
    args=(trees,intervals,g,k_actual,x,labels,ages)
    kwargs=dict(num_classes=3,age_split=350.,query_columns=np.arange(600,1600,dtype=np.int64))
    if index==0:
        try:original.encode_block(*args,**kwargs)
        except ValueError as exc:
            check("original_actual_failed_block_same_error",str(exc)=="One complete observed native tree and unique sampled identities required")
        else:raise AssertionError("Original failed block unexpectedly passed")
    layout_calls=[];support_calls=[]
    old_layout=repaired._layout;old_support=repaired.branch_concordance_support_v1
    def spy_layout(tree):layout_calls.append(id(tree));return old_layout(tree)
    def spy_support(tree,*a):support_calls.append(id(tree));return old_support(tree,*a)
    repaired._layout=spy_layout;repaired.branch_concordance_support_v1=spy_support
    try:bank,receipt=repaired.encode_block(*args,**kwargs)
    finally:repaired._layout=old_layout;repaired.branch_concordance_support_v1=old_support
    ordinary=[id(t) for t in trees if not (len(t.parent)>1 and np.all(t.parent==-1))]
    check("connected_layout_delegation:"+str(index),layout_calls==ordinary)
    check("connected_support_delegation:"+str(index),support_calls==ordinary)
    result=validate_bank(bank,trees,intervals,g,x,receipt)
    check("actual_plan_node_site_counts:"+str(index),receipt["nodes"]==block["nodes"] and receipt["sites"]==block["sites"])
    if index==1:
        original_bank,original_receipt=original.encode_block(*args,**kwargs)
        check("connected_block_all_tensor_bytes_identical",same_bank(bank,original_bank) and bank_sha(bank)==bank_sha(original_bank))
        check("connected_block_receipt_identical",receipt==original_receipt)
        result["original_connected_block_bank_sha256"]=bank_sha(original_bank)[0]
    result.update(block_index=index,start_tree=block["start_tree"],stop_tree=block["stop_tree"],ordinary_layout_calls=len(layout_calls),ordinary_support_calls=len(support_calls))
    fixture_blocks.append(result)
    del bank
receipt=dict(schema="C3_isolated_encoder_source_functional_fixture_v1",observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status="PASS",check_count=len(checks),checks=checks,staged_source_ref=dict(path=str(STAGED.relative_to(ROOT)),sha256=sha(STAGED)),actual_failed_block_original_raises=True,all40_actual_isolated_layouts=isolated_rows,actual_TRAIN000_blocks=fixture_blocks,exact_parent_plan_ref=dict(path=str(plan_path.relative_to(ROOT)),sha256=sha(plan_path)),training_updates=0,model_forward_or_optimizer_used=False,GPU_or_process_probe_used=False,ancestry_truth_opened=False,guard_or_accounting_mutation=False,canonical_source_modified=False)
rp=OUT/"functional_fixture_receipt.v1.json";rp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(dict(status=receipt["status"],check_count=len(checks),all40_isolated_layouts=len(isolated_rows),actual_blocks=fixture_blocks,receipt_ref=dict(path=str(rp.relative_to(ROOT)),sha256=sha(rp))),sort_keys=True),flush=True)
