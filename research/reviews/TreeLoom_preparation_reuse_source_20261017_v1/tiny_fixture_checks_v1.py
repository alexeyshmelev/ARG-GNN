"""Bounded synthetic CPU preparation/codec/source checks; no model or real data."""
from __future__ import annotations
import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import torch
from research.reviews.frozen_benchmark_training_ready_20261017_v1 import american_full30mb_frozen_v1 as frozen
from research.reviews.frozen_benchmark_training_ready_20261017_v1 import american_full30mb_prepare_v1 as baseline
from research.reviews.H060_isolated_hold_compatibility_source_20261017_v1.observations_v1 import native_tree as old_native
from research.reviews.frozen_benchmark_training_ready_20261017_v1.american_full30mb_train_v1 import epoch_order_v1
from . import optimized_preparation_v1 as optimized
from . import optimized_observations_v1 as observation
from . import cache_codec_v1 as codec
from . import cache_policy_v1 as policy
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/TreeLoom_preparation_reuse_source_20261017_v1"


def need(ok,message):
    if not ok:raise AssertionError(message)


def fixture_state_v1():
    # Reuse only construction of the existing small parity fixture. Reduce to
    # three markers per nonempty hold and add one connected zero-site hold.
    path=ROOT/"research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1/parity_v1.py"
    tree=ast.parse(path.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=="synthetic_parent_v1")
    body=[]
    for node in function.body:
        if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="fixture"for t in node.targets):break
        body.append(node)
    function.body=body+[ast.Return(ast.Name(id="state",ctx=ast.Load()))]
    ast.fix_missing_locations(function)
    text=ast.unparse(function).replace("if at != 2:","if at not in (0, 2):")
    text=text.replace("right - 0.25, 100)","right - 0.25, 3)")
    ns=dict(np=np,frozen=frozen)
    exec(compile(text,str(path)+"#tiny-construction-only","exec"),ns)
    state=ns["synthetic_parent_v1"]()
    s,h=state["genotype"].shape
    state["genotype"]=((np.arange(s)[:,None]+np.arange(h)[None,:])%2).astype(np.int8)
    state["allele_count"]=np.full(s,2,dtype=np.int8)
    need(s==15 and state["ts"].num_trees==7,"Bounded15SNP/7hold synthetic fixture only")
    return state


def equal_bundle_v1(a,b):
    av,_,_=codec.tensors_v1(a);bv,_,_=codec.tensors_v1(b)
    need(set(av)==set(bv),"Every original bank/episode/cut/focal tensor retained")
    for name in av:
        x,y=av[name].numpy(),bv[name].numpy()
        need(x.dtype==y.dtype and x.shape==y.shape
            and x.tobytes(order="C")==y.tobytes(order="C"),"Exact tensor bytes: "+name)
    return len(av)


def run_v1():
    torch.set_num_threads(1)
    state=fixture_state_v1()
    workspace=observation.native_workspace_v1(state["ts"])
    times=np.asarray(state["ts"].nodes_time,dtype=np.float64)
    for tree in state["ts"].trees():
        old=old_native(tree,state["nodes"])
        new=observation.native_tree(tree,state["nodes"],node_time=times,workspace=workspace)
        for name in("parent","node_time","sample_node","sample_id"):
            x,y=getattr(old,name),getattr(new,name)
            need(x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes(),
                "Original local native node order/metadata: "+name)
    fake=lambda label:dict(path="SYNTHETIC_RAM_ONLY/"+label,sha256="0"*64)
    item=dict(observation_ref=fake("observations"),focal_registry_ref=fake("focals"),
        native_preparation_ref=fake("native"))
    args=dict(split="train",**item,max_nodes=100000,max_holds=100,max_episode_payload_bytes=8*2**20)
    old,old_meta=baseline.prepare_from_loaded_observation_v1(state,**args)
    new,new_meta=optimized.prepare_from_loaded_observation_v1(state,**args)
    count=equal_bundle_v1(old,new)
    need(codec.encoded(old_meta)==codec.encoded(new_meta),"Exact complete recipe metadata")
    source_ref=fake("SOURCE_MANIFEST")
    key,identity=codec.identity_key_v1("american_c3_train_000",item,source_ref)
    descriptor=codec.describe_v1(new,new_meta,key,identity)[2]
    buffer=io.BytesIO()
    written=codec.write_archive_v1(buffer,new,new_meta,key,identity)
    need(written==descriptor and buffer.tell()<=descriptor["serialized_upper_bound_bytes"],"Actual tiny archive within serialization bound")
    restored,replay_meta=codec.restore_v1(io.BytesIO(buffer.getvalue()),key,identity,
        max_tensor_bytes=descriptor["tensor_nbytes"],max_archive_bytes=descriptor["serialized_upper_bound_bytes"])
    need(equal_bundle_v1(old,restored)==count and codec.encoded(old_meta)==codec.encoded(replay_meta),
        "Bitwise exact replay with fresh pointer/version stamps")
    failures=0
    for bad_key,tensor_cap,archive_cap in(
        ("f"*64,descriptor["tensor_nbytes"],descriptor["serialized_upper_bound_bytes"]),
        (key,descriptor["tensor_nbytes"]-1,descriptor["serialized_upper_bound_bytes"]),
        (key,descriptor["tensor_nbytes"],len(buffer.getvalue())-1)):
        try:codec.restore_v1(io.BytesIO(buffer.getvalue()),bad_key,identity,
            max_tensor_bytes=tensor_cap,max_archive_bytes=archive_cap)
        except ValueError:failures+=1
        else:raise AssertionError("Wrong identity/size must fail without fallback")
    # Synthetic policy records are RAM fixtures, not actual capacity evidence.
    m=dict(parent_id="american_c3_train_000",actual_full_parent_measurement=True,
        preparation_source_manifest_ref=source_ref,sequence_length_bp=30000000,haplotypes=1600,
        native_focals=10000,truth_opened=False,roundtrip_all_tensor_bytes_verified=True,
        actual_original_owned_RSS_cap_enforced=True,
        actual_measured_worker_peak_RSS_bytes=10**7,actual_measured_replay_worker_peak_RSS_bytes=10**7,
        cold_prepare_seconds=10.,actual_verified_replay_seconds=1.,**descriptor)
    cap=descriptor["per_entry_charge_upper_bound_bytes"]+65536
    capacity=dict(root_authorizer="root_agent",actual_current_observation=True,source_only_estimate=False,
        actual_admissible_new_bytes=cap,cache_reservation_bytes=cap,project_limit_bytes=10**12,
        peak_owned_RSS_cap_bytes=384*2**30)
    choice=policy.select_fixed_subset_v1({m["parent_id"]:m},capacity,source_manifest_ref=source_ref)
    need(choice["selected_parent_ids"]==[m["parent_id"]]and choice["all20_cached"]is False,
        "Measured gate selects only bounded fixed subset, never pretends all20 fit")
    order=[parent for epoch in range(1,31)for parent in epoch_order_v1(epoch)]
    need(len(order)==600 and order.count(m["parent_id"])==30
        and choice["per_selected_parent_avoided_repreparations"]==29,"Fixed subset has real cycle reuse in unchanged600order")
    capacity["actual_admissible_new_bytes"]=65536
    empty=policy.select_fixed_subset_v1({m["parent_id"]:m},capacity,source_manifest_ref=source_ref)
    need(empty["selected_parent_ids"]==[]and empty["cache_charge_upper_bound_bytes"]==0,"Insufficient size gives no cache fit/speedup claim")
    # Same exact training schedule and learned training statements.
    orig=ast.parse((ROOT/"research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1/worker_v1.py").read_text())
    succ=ast.parse((ROOT/HERE/"worker_v1.py").read_text())
    get=lambda tree:next(n for n in ast.walk(tree)if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=="epoch")
    need(ast.dump(get(orig),include_attributes=False)==ast.dump(get(succ),include_attributes=False),
        "Epoch/parent order, truthjoin, all600optimizer updates/checkpoints untouched")
    return dict(schema="TreeLoom_preparation_reuse_tiny_RAM_source_checks_v1",status="PASS",
        synthetic_holds=7,synthetic_haplotypes=1600,synthetic_sites=15,
        exact_native_metadata_all_holds=True,connected_and_isolated_zero_SNP_holds_checked=True,
        all_tensor_fields_bitwise_equal=count,global_recipe_metadata_equal=True,
        exact_archive_replay=True,wrong_identity_or_size_rejections=failures,
        tiny_tensor_nbytes=descriptor["tensor_nbytes"],tiny_archive_actual_bytes=len(buffer.getvalue()),
        tiny_archive_upper_bound_bytes=descriptor["serialized_upper_bound_bytes"],
        unchanged600training_loop_AST=True,fixed_subset_cycle_reuses=29,
        all20_cache_fit_claimed=False,actual_capacity_measured=False,
        real_observation_or_truth_payloads_read=False,model_or_optimizer_executed=False,
        resource_or_accounting_operations=False)


if __name__=="__main__":
    print(json.dumps(run_v1(),sort_keys=True,indent=2))
