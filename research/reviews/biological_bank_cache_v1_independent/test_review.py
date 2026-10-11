from __future__ import annotations
import copy
from dataclasses import fields, is_dataclass, replace
from datetime import datetime, timezone, timedelta
import hashlib, json, os, random, traceback
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from arggnn import biological_bank_cache_v1 as cachemod
from arggnn import biological_candidate as frozen
from arggnn import biological_mechanism_controls_v1 as controls
from arggnn.anchor_kernel import AnchorKernelBank, AnchorOperators, prepare_anchor_kernel_bank
from arggnn.kron_interface import LeafInterfaceBank, QueryInterfaceCache, KronDiagnostics

OUT = Path("research/reviews/biological_bank_cache_v1_independent")
FIX = OUT / "handmade"
FIX.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.use_deterministic_algorithms(True)
results = []
limits = {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "GPU_context_created": False, "real_banks_opened": False, "ancestry_targets_opened": False}

def ref(p):
    return {"path": str(p), "sha256": frozen.sha(p)}

def save_json(name, value):
    p = FIX / name
    p.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return ref(p)

def expect_raises(action, text=None, typ=ValueError):
    try:
        action()
    except typ as error:
        if text is not None:
            assert text in str(error), (text, str(error))
        return str(error)
    raise AssertionError("Expected " + typ.__name__)

def test(name, action):
    try:
        details = action()
        result = {"name": name, "status": "pass", "details": details}
    except Exception as error:
        result = {"name": name, "status": "fail", "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
    results.append(result)
    print(json.dumps(result), flush=True)
    (OUT / "test_results.json").write_text(json.dumps({"scope": limits, "checks": results}, sort_keys=True, indent=2) + "\n")

def exact(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor)
        assert a.device == b.device and a.dtype == b.dtype and a.shape == b.shape and a.stride() == b.stride()
        assert torch.equal(a, b)
    elif is_dataclass(a):
        assert type(a) is type(b)
        for field in fields(a):
            exact(getattr(a, field.name), getattr(b, field.name))
    elif isinstance(a, dict):
        assert list(a) == list(b)
        for key in a:
            exact(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    elif isinstance(a, float):
        assert a.hex() == b.hex()
    else:
        assert a == b

def rng():
    return cachemod._rng_v1("cpu")

cfg = json.loads(Path("configs/biological_mechanism_controls_v1.json").read_text())
n, q, r = 280, 184, 96
ids = torch.arange(1000, 1000+n)
query_ids, reference_ids = ids[:q].clone(), ids[q:].clone()
response = (n * torch.eye(n, dtype=torch.float64) - torch.ones((n,n), dtype=torch.float64))[None]
normalized = response / (n-1)
cqq, cqr = normalized[:, :q, :q].clone(), normalized[:, :q, q:].clone()
eigenvalues, eigenvectors = torch.linalg.eigh(cqq)
qcache = QueryInterfaceCache(
    normalized_response=normalized, normalization_scale=torch.tensor([n-1.], dtype=torch.float64),
    cqq=cqq, cqr=cqr, eigenvalues=eigenvalues, eigenvectors=eigenvectors,
    query_indices=torch.arange(q), reference_indices=torch.arange(q,n),
    query_ids=query_ids, reference_ids=reference_ids,
    interval_weights=torch.ones(1, dtype=torch.float64), bag_cqq=cqq[0].clone(),
    bag_eigenvalues=eigenvalues[0].clone(), bag_eigenvectors=eigenvectors[0].clone())
leaf_x = torch.arange(n*47, dtype=torch.float64).reshape(1,n,47)/10000
leaf_x[:,:,4] = 0
leaf_x[:,:120,4] = 1
diagnostic = KronDiagnostics(1., 0., 0., 0., 0., 0., 0.)
leaf = LeafInterfaceBank(
    response=response, sample_ids=ids, boundary_nodes=torch.arange(n)[None],
    leaf_x=leaf_x, feature_columns=tuple(i for i in range(48) if i != 7),
    intervals=torch.tensor([[0.,1.]], dtype=torch.float64), query_node_ids=query_ids,
    positions=torch.tensor([.25,.75], dtype=torch.float64), target_graph=torch.zeros(2,dtype=torch.long),
    window_bounds=torch.tensor([0.,1.],dtype=torch.float64), diagnostics=(diagnostic,),
    resistance_scale=1., time_scale=1., query_cache=qcache)
prepared = prepare_anchor_kernel_bank(leaf, num_anchors=32, num_scales=3)
factors = torch.tensor([1.,1.15,1.4],dtype=torch.float64)[None,:,None,None]
age = AnchorOperators(cqq[:,None]*factors, cqr[:,None]*factors,
    eigenvalues[:,None]*factors[:,:,:,0], eigenvectors[:,None].expand(-1,3,-1,-1))
matched_cqq = torch.diag_embed(age.cqq.diagonal(dim1=-2,dim2=-1).mean(-1)[...,None].expand(-1,-1,q))
matched = AnchorOperators(matched_cqq, age.cqr,
    matched_cqq.diagonal(dim1=-2,dim2=-1), torch.eye(q,dtype=torch.float64)[None,None].expand(1,3,-1,-1))
prepared = replace(prepared, operators={**prepared.operators, "age":age, "matched_scalar_age":matched})
parent = {
    "parent_id":"independent_handmade_train_000", "split":"train",
    "dynamics_slots":np.arange(q), "reference_slots":np.arange(q,n), "query_slots":np.arange(120),
    "positions":np.array([500000.,1500000.]), "observation":{"nodes":ids.numpy().copy()}}

def make_tile(index, suffix="", schema="biological_candidate_full_age_bank_v1"):
    p=FIX / ("bank_" + str(index) + suffix + ".pt")
    torch.save({"schema":schema, "fixed_prepared":prepared, "age_bank":{"handmade_only":True,"extra_archive_field":torch.arange(13,dtype=torch.float64)}}, p)
    return {"tile_id":index, "bank":ref(p), "position_indexes":[0,1],
        "original_tree_ids":[0], "exact_serialization_roundtrip_checked":True,
        "all_original_age_cache_fields_retained":True}

tiles=[make_tile(0),make_tile(1)]
cfgref=save_json("scientific_config.json",cfg)
observedref=save_json("observed_manifest.json",{"handmade_only":True,"truth_opened":False})
manifestref=save_json("prepared_manifest.json",{"status":"complete_observed_only_Q184_R96_actualQ120","truth_opened":False,
    "parents":[{"parent_id":parent["parent_id"],"tiles":tiles}]})

def new_cache(cpu=64_000_000_000,gpu=32_000_000_000, manifest=manifestref, config=cfgref, observed=observedref, **kwargs):
    return cachemod.BiologicalBankCacheV1(manifest,config,observed,cpu_budget_bytes=cpu,gpu_budget_bytes=gpu,**kwargs)

original = cachemod.BASE_LOAD_BANK(tiles[0],cfg,parent,"cpu")
SIZE = cachemod.resident_bytes_v1(original)

def fields_hits_sha_rng():
    cache=new_cache()
    before=rng()
    first=cache(tiles[0],cfg,parent,"cpu")
    second=cache(tiles[0],cfg,parent,torch.device("cpu:0"))
    exact(original, first);exact(original,second)
    assert first is second and rng()==before
    assert cache.stats["CPU_hits"]==1 and cache.stats["cold_original_loads"]==1
    assert cache.stats["bank_sha_verified_bytes"]==2*Path(tiles[0]["bank"]["path"]).stat().st_size
    assert first.bank.positions.dtype==torch.float64 and first.bank.query_node_ids.dtype==torch.int64
    assert first.bank.leaf_x.dtype==torch.float32
    cache.close()
    assert cache.closed and not cache.cpu and not cache.gpu and cache.cpu_used==cache.gpu_used==0
    return {"complete_nested_fields_equal":True,"hit_object_reused":True,"RNG_unchanged":True,"CPU_resident_bytes":SIZE,"file_bytes_per_request":Path(tiles[0]["bank"]["path"]).stat().st_size}
test("complete_original_fields_CPU_hit_SHA_RNG", fields_hits_sha_rng)

def aliases():
    baseline=frozen.load_bank
    for error in (False,True):
        cache=new_cache()
        try:
            with cachemod.installed_cache_v1(cache):
                assert frozen.load_bank is cache and controls.load_bank is cache
                frozen.load_bank(tiles[0],cfg,parent,"cpu")
                controls.load_bank(tiles[0],cfg,parent,"cpu")
                if error:raise RuntimeError("handmade deliberate consumer failure")
        except RuntimeError as exc:
            assert error and "deliberate" in str(exc)
        assert frozen.load_bank is baseline and controls.load_bank is baseline
        assert cache.closed and cache.cpu_used==cache.gpu_used==0
    cache=new_cache()
    with patch.object(controls,"load_bank",lambda *a:None):
        expect_raises(lambda:cachemod.installed_cache_v1(cache).__enter__(),"already replaced")
    cache.close()
    return {"both_aliases_scoped":True,"success_and_error_restored":True,"stacked_alias_rejected":True}
test("scoped_two_aliases_and_error_cleanup", aliases)

def cpu_budget():
    cache=new_cache(cpu=SIZE,gpu=0)
    a=cache(tiles[0],cfg,parent,"cpu");b=cache(tiles[1],cfg,parent,"cpu")
    assert cache.cpu_used<=SIZE and list(cache.cpu)==[(parent["parent_id"],1)]
    assert cache.stats["CPU_evictions"]==1
    cache(tiles[0],cfg,parent,"cpu")
    assert cache.stats["cold_original_loads"]==3 and cache.stats["CPU_peak_resident_bytes"]<=SIZE
    cache.close()
    for budget in (0,SIZE-1):
        cache=new_cache(cpu=budget,gpu=0)
        a=cache(tiles[0],cfg,parent,"cpu");b=cache(tiles[0],cfg,parent,"cpu")
        exact(a,b);assert a is not b and not cache.cpu
        assert cache.stats["CPU_oversized_bypass"]==2 and cache.cpu_used==0
        cache.close()
    return {"whole_entry_LRU_eviction":True,"oversize_and_zero_budget_original_uncached":True,"bounded_at_actual_small_bank_size":SIZE}
test("CPU_bounds_LRU_oversize",cpu_budget)

def memory_account():
    base=torch.arange(24,dtype=torch.float32)
    views={"base":base,"same":base,"slice":base[3:8],"expanded":base[:1].expand(200)}
    assert cachemod.resident_bytes_v1(views)==base.untyped_storage().nbytes()
    expected=3*base.untyped_storage().nbytes()+200*4
    assert cachemod.transfer_upper_bytes_v1(views)==expected
    return {"shared_storage_counted_once":True,"expanded_logical_copy_included":True,"resident_bytes":96,"transfer_upper_bytes":expected}
test("storage_dedup_and_conservative_transfer_bound",memory_account)

def simulated_gpu_lru():
    method=AnchorKernelBank.to
    calls=[]
    def simulated_to(bank,device,dtype=None):
        if torch.device(device).type=="cuda":
            calls.append(str(device))
            return method(bank,"cpu",dtype)
        return method(bank,device,dtype)
    upper=cachemod.transfer_upper_bytes_v1(original)
    with patch.object(AnchorKernelBank,"to",simulated_to):
        cache=new_cache(cpu=2*SIZE,gpu=upper)
        a=cache(tiles[0],cfg,parent,"cuda");b=cache(tiles[0],cfg,parent,"cuda:0")
        assert a is b and cache.stats["GPU_hits"]==1
        cache(tiles[1],cfg,parent,"cuda")
        assert cache.stats["GPU_evictions"]>=1 and cache.gpu_used<=upper
        assert cache.stats["bank_sha_verified_bytes"]==3*Path(tiles[0]["bank"]["path"]).stat().st_size
        cache.close()
        cache=new_cache(cpu=SIZE,gpu=upper)
        cache(tiles[0],cfg,parent,"cuda");cache(tiles[1],cfg,parent,"cpu")
        assert not cache.gpu and cache.gpu_used==0 and cache.stats["GPU_evictions"]==1
        cache.close()
        for cpu,gpu in ((SIZE,upper-1),(0,upper),(SIZE,0)):
            cache=new_cache(cpu=cpu,gpu=gpu)
            a=cache(tiles[0],cfg,parent,"cuda");b=cache(tiles[0],cfg,parent,"cuda")
            assert not cache.gpu and cache.gpu_used==0 and cache.stats["GPU_oversized_bypass"]==2
            exact(a,b)
            cache.close()
    return {"GPU_transfer_simulated_on_CPU_only":True,"real_CUDA_not_tested":True,"GPU_hit_LRU_CPU_eviction_and_oversize_paths":True,"small_transfer_upper_bytes":upper,"transfer_calls":len(calls)}
test("simulated_GPU_cache_control_flow_only",simulated_gpu_lru)

def hit_changed_bytes():
    tile=make_tile(3,"_tamper")
    m=save_json("tamper_manifest.json",{"status":"complete_observed_only_Q184_R96_actualQ120","truth_opened":False,"parents":[{"parent_id":parent["parent_id"],"tiles":[tile]}]})
    cache=new_cache(manifest=m)
    with cachemod.installed_cache_v1(cache):
        cache(tile,cfg,parent,"cpu")
        with Path(tile["bank"]["path"]).open("ab") as stream:stream.write(b"independent deliberate trailing-byte mutation")
        expect_raises(lambda:cache(tile,cfg,parent,"cpu"),"Missing/changed bound file")
    assert cache.closed and cache.cpu_used==0
    return {"full_file_append_detected_on_hit":True,"failure_fixture_retained":tile["bank"]["path"]}
test("changed_full_file_bytes_rejected_on_hit",hit_changed_bytes)

def identity_schema_roles_dtype():
    cache=new_cache()
    cache(tiles[0],cfg,parent,"cpu")
    cases=[]
    for field in ("split","positions","dynamics_slots","reference_slots","query_slots","observation"):
        changed=copy.deepcopy(parent)
        if field=="split":changed[field]="val"
        elif field=="observation":changed[field]["nodes"][0]+=1
        elif field=="positions":changed[field]=changed[field].astype(np.float32)
        else:changed[field]=changed[field][::-1].copy()
        cases.append(expect_raises(lambda changed=changed:cache(tiles[0],cfg,changed,"cpu"),"Parent role"))
    changedtile=copy.deepcopy(tiles[0]);changedtile["original_tree_ids"]=[1]
    expect_raises(lambda:cache(changedtile,cfg,parent,"cpu"),"Parent/tile/file role")
    changedcfg=copy.deepcopy(cfg);changedcfg["seeds"]=[999]
    expect_raises(lambda:cache(tiles[0],changedcfg,parent,"cpu"),"Changed scientific")
    expect_raises(lambda:cache(tiles[0],cfg,parent,"cpu",dtype=torch.float64),"float32")
    for device in ("cuda:1","cpu:1","meta"):
        expect_raises(lambda device=device:cache(tiles[0],cfg,parent,device))
    cache.close()
    badtile=make_tile(4,"_bad_schema",schema="deliberately_wrong_handmade_schema")
    m=save_json("bad_schema_manifest.json",{"status":"complete_observed_only_Q184_R96_actualQ120","truth_opened":False,"parents":[{"parent_id":parent["parent_id"],"tiles":[badtile]}]})
    cache=new_cache(manifest=m)
    expect_raises(lambda:cache(badtile,cfg,parent,"cpu"),"Exact full bank schema")
    cache.close()
    return {"split_positions_roles_node_ID_dtype_device_tile_config_schema_rejected":True,"role_cases":len(cases)}
test("identity_schema_split_roles_dtype_device",identity_schema_roles_dtype)

def mutation_guard():
    cache=new_cache()
    with cachemod.installed_cache_v1(cache):
        bank=cache(tiles[0],cfg,parent,"cpu")
        bank.bank.leaf_x.add_(1)
        expect_raises(lambda:cache(tiles[0],cfg,parent,"cpu"),"mutated in memory")
        expect_raises(lambda:cache.close(),"mutated in memory")
    assert cache.closed and not cache.cpu and not cache.gpu
    return {"ordinary_tensor_mutation_rejected":True,"cleanup_even_if_guard_fails":True}
test("mutation_detection_and_guard_error_cleanup",mutation_guard)

def immutable_binding_identity():
    localcfg=save_json("identity_config.json",cfg)
    cache=new_cache(config=localcfg)
    text=Path(localcfg["path"]).read_text()
    Path(localcfg["path"]).write_text(text)
    expect_raises(lambda:cache(tiles[0],cfg,parent,"cpu"),"file identity changed")
    cache.close()
    tile=make_tile(5,"_identity")
    m=save_json("bank_identity_manifest.json",{"status":"complete_observed_only_Q184_R96_actualQ120","truth_opened":False,"parents":[{"parent_id":parent["parent_id"],"tiles":[tile]}]})
    cache=new_cache(manifest=m)
    cache(tile,cfg,parent,"cpu")
    bankpath=Path(tile["bank"]["path"]);content=bankpath.read_bytes()
    replacement=bankpath.with_suffix(".replacement");replacement.write_bytes(content);replacement.replace(bankpath)
    expect_raises(lambda:cache(tile,cfg,parent,"cpu"),"Immutable bank file identity changed")
    cache.close()
    return {"same_bytes_pinned_binding_rewrite_rejected":True,"same_bytes_bank_inode_replacement_rejected":True}
test("immutable_file_identity_after_full_SHA",immutable_binding_identity)

def model_checks():
    checks=[]
    cache=new_cache()
    with cachemod.installed_cache_v1(cache):
        baseline=cachemod.BASE_LOAD_BANK(tiles[0],cfg,parent,"cpu")
        cached=cache(tiles[0],cfg,parent,"cpu")
        exact(baseline,cached)
        before_bank=frozen.fingerprint_sha(baseline)
        for name in controls.CONTROL_NAMES:
            torch.manual_seed(71)
            first=controls.model_factory(name,cfg);second=copy.deepcopy(first)
            opt1=torch.optim.AdamW(frozen.optimizer_groups(first,cfg),lr=cfg["training"]["learning_rate"])
            opt2=torch.optim.AdamW(frozen.optimizer_groups(second,cfg),lr=cfg["training"]["learning_rate"])
            before=rng();logits=[];grads=[]
            for model,bank,opt in ((first,baseline,opt1),(second,cached,opt2)):
                values=model(bank)
                weights=torch.linspace(-.7,1.1,values.numel(),dtype=values.dtype).reshape_as(values)
                objective=(values*weights).mean();objective.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                logits.append(values.detach().clone());grads.append([p.grad.detach().clone() for p in model.parameters()])
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                opt.step()
            assert torch.equal(*logits)
            assert all(torch.equal(a,b) for a,b in zip(*grads))
            assert frozen.fingerprint_sha(first.state_dict())==frozen.fingerprint_sha(second.state_dict())
            assert frozen.fingerprint_sha(opt1.state_dict())==frozen.fingerprint_sha(opt2.state_dict())
            assert frozen.fingerprint_sha(baseline)==frozen.fingerprint_sha(cached)==before_bank
            assert rng()==before
            check={"model":name,"original_parameter_count":sum(p.numel() for p in first.parameters()),
                "full_fields_logits_VJP_clipped_AdamW_RNG_exact":True,"target_free_disposable_unit_step":True}
            checks.append(check)
            print(json.dumps({"model_check":check}),flush=True)
            del first,second,opt1,opt2,logits,grads
    return checks
test("seven_unchanged_original_models_handmade_CPU_parity",model_checks)

def capability_checks():
    shipped=json.loads(Path("configs/biological_bank_cache_v1.json").read_text())
    cacheref=ref(Path("configs/biological_bank_cache_v1.json"))
    sourcemap={**shipped["source_sha256"],"src/arggnn/biological_bank_cache_v1.py":frozen.sha(Path("src/arggnn/biological_bank_cache_v1.py"))}
    reviewref=save_json("admission_contract_review_only.json",{"status":"qualified_handmade_contract_fixture_only","actual_admission":False,"fixture_mode":False,
        "artifact_sha256":{"src/arggnn/biological_bank_cache_v1.py":sourcemap["src/arggnn/biological_bank_cache_v1.py"],"configs/biological_bank_cache_v1.json":cacheref["sha256"]},
        "warning":"Fresh handmade admission-validator test only. Never authorize real work with this file."})
    allmanifest=save_json("all32_handmade_identity_only_manifest.json",{"status":"complete_observed_only_Q184_R96_actualQ120","truth_opened":False,
        "parents":[{"parent_id":p,"tiles":[{**copy.deepcopy(tiles[0]),"tile_id":i} for i in range(8)]} for p in frozen.PARENTS]})
    now=datetime.now(timezone.utc)
    base_memory={"schema":"biological_bank_cache_memory_reservation_v1","status":"active","job_id":"independent_handmade_contract_only",
        "CPU_cache_bytes":shipped["CPU_budget_bytes"],"GPU_cache_bytes":shipped["GPU_budget_bytes"],
        "created_at_utc":(now-timedelta(minutes=1)).isoformat(),"expires_at_utc":(now+timedelta(hours=1)).isoformat(),
        "original_model_optimizer_runtime_headroom_preserved":True,"handmade_only":True}
    memref=save_json("admission_contract_memory_only.json",base_memory)
    refs={"cache_config":cacheref,"review":reviewref,"original_config":shipped["original_config"],"observed_manifest":observedref,
        "prepared_manifest":allmanifest,"memory_reservation":memref}
    admission={"schema":"biological_bank_cache_admission_v1","status":"active","phase":"profile","job_id":base_memory["job_id"],
        "created_at_utc":base_memory["created_at_utc"],"expires_at_utc":base_memory["expires_at_utc"],
        "refs":refs,"source_sha256":sourcemap,"handmade_only":True,"warning":"Admission-validation test only, no run/profile/fit allowed"}
    def check_record(name,value,phase="profile",failure=None):
        record=save_json(name+".json",value)
        if failure is not None:
            return expect_raises(lambda:cachemod.admit_cache_v1(record,phase=phase),failure)
        obj,_,_=cachemod.admit_cache_v1(record,phase=phase);obj.close()
        return record
    accepted=check_record("handmade_profile_contract_only",admission)
    expect_raises(lambda:cachemod.admit_cache_v1(accepted,phase="fit"),"distinct root cache phase")
    bad=copy.deepcopy(admission);bad["expires_at_utc"]=(now-timedelta(seconds=1)).isoformat()
    check_record("expired_profile",bad,failure="distinct root cache phase")
    bad=copy.deepcopy(admission);bad["source_sha256"]["src/arggnn/biological_candidate.py"]="0"*64
    check_record("wrong_source",bad,failure="Complete exact cache source pins")
    bad=copy.deepcopy(admission);bad["refs"]["unbound_extra_role"]=observedref
    check_record("extra_ref_role",bad,failure="Exact root cache reference roles")
    badmem={**base_memory,"CPU_cache_bytes":1}
    bad=copy.deepcopy(admission);bad["refs"]["memory_reservation"]=save_json("wrong_memory_reservation.json",badmem)
    check_record("wrong_memory",bad,failure="memory reservation")
    badreview={"status":"qualified_handmade_only","actual_admission":False,"fixture_mode":True,"artifact_sha256":{
        "src/arggnn/biological_bank_cache_v1.py":sourcemap["src/arggnn/biological_bank_cache_v1.py"],
        "configs/biological_bank_cache_v1.json":cacheref["sha256"]}}
    bad=copy.deepcopy(admission);bad["refs"]["review"]=save_json("forbidden_fixture_review.json",badreview)
    check_record("forbidden_fixture_review_admission",bad,failure="Fresh independent")
    fit=copy.deepcopy(admission);fit["phase"]="fit"
    gate={"schema":"biological_bank_cache_profile_gate_v1","status":"qualified_execution_only","cache_config_ref":cacheref,
        "cache_source_sha256":sourcemap["src/arggnn/biological_bank_cache_v1.py"],
        "scientific_refs":{k:refs[k] for k in ("original_config","observed_manifest","prepared_manifest")},
        "all_fields_logits_VJP_optimizer_RNG_parity":True,"measured_profile_acceptable":False,
        "profile_evidence_ref":observedref}
    fit["refs"]["profile_gate"]=save_json("unqualified_fit_gate.json",gate)
    check_record("unqualified_fit",fit,phase="fit",failure="measured one-tile execution profile")
    return {"fresh_profile_contract_handmade_only":True,"separate_phase_expiry_source_roles_memory_review_and_fit_gate_fail_closed":True,
        "actual_profile_or_fit_entrypoints_not_invoked":True,"configured_CPU_bytes":shipped["CPU_budget_bytes"],"configured_GPU_bytes":shipped["GPU_budget_bytes"]}
test("source_capability_contract_handmade_only",capability_checks)

summary={"schema":"biological_bank_cache_v1_independent_tests","scope":limits,"handmade_dimensions":{"graphs":1,"boundary":280,"queries":184,"references":96,"actual_query_flags":120,"anchors":32,"branches":3,"positions":2},
    "checks":results,"all_passed":all(result["status"]=="pass" for result in results),
    "source_sha256":frozen.sha(Path("src/arggnn/biological_bank_cache_v1.py")),
    "config_sha256":frozen.sha(Path("configs/biological_bank_cache_v1.json")),
    "scientific_speed_accuracy_claims":False}
(OUT / "test_results.json").write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
print(json.dumps({"all_passed":summary["all_passed"],"tests":len(results),"source_sha256":summary["source_sha256"],"config_sha256":summary["config_sha256"]}),flush=True)
raise SystemExit(0 if summary["all_passed"] else 1)

