
"""Handmade CPU and synthetic-metadata checks only; no actual observations/targets."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime,timezone,timedelta
import ast,hashlib,importlib.util,io,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
import torch
from arggnn import census_tsinfer_fit_v1 as fit,census_tsinfer_device_profile_v1 as profile
ROOT=fit.ROOT
torch.set_num_threads(1)
fixture_spec=importlib.util.spec_from_file_location("handmade_tsinfer_only",ROOT/"research/reviews/census_tsinfer_observation_v1_author/test_census_tsinfer_observation_v1.py")
fixture=importlib.util.module_from_spec(fixture_spec);fixture_spec.loader.exec_module(fixture)

def synthetic_gate():
    """Ephemeral stand-ins, deliberately never published as root evidence."""
    cfg=fit.bridge.load_config_v1()
    sources=dict(cfg["source_sha256"],**{fit.bridge.SOURCE:fit._hash(fit.bridge.SOURCE)})
    refs={k:{"path":"research/reviews/census_tsinfer_fit_v1_author/handmade_metadata/"+k+".json","sha256":"a"*64} for k in fit.GATE_REFS}
    cells=[]
    for p in fit.PARENTS:
        for tile in range(8):
            trees=446 if p.endswith("train_001") and tile==7 else 550 if p.endswith("train_000") and tile==1 else 500
            row={"parent_id":p,"split":"train" if "_train_" in p else "val","tile_id":tile,"num_trees":trees,"certificate_sha256":hashlib.sha256((p+str(tile)).encode()).hexdigest(),
                "inner_certificate_sha256":"b"*64,"lineage":{"handmade_synthetic_metadata":True},"structural":{},"event_collision_counts":{},
                "raw_ref":{"path":cfg["inspection_output_prefix"]+"handmade-inspect/"+p+f"/tile_{tile:02d}.whole_observed.pt","sha256":"c"*64},
                "producer_ref":{"path":cfg["inspection_output_prefix"]+"handmade-inspect/"+p+f"/tile_{tile:02d}.producer.json","sha256":"d"*64},
                "status":"inspected_pending_root_freeze"}
            cells.append(row)
    selection={"minimum":{"num_trees":446,"parent_id":"bio_paths_pilot_train_001","split":"train","tile_id":7},
        "maximum":{"num_trees":550,"parent_id":"bio_paths_pilot_train_000","split":"train","tile_id":1}}
    base={"schema":"census_tsinfer_observation_evidence_v1","source_sha256":sources,"metadata_input_sha256":cfg["metadata_input_sha256"],
        "observed_artifact_sha256":cfg["observed_artifact_sha256"],"fixture_mode":False,"expected_cells":32,"failed_cells":[],
        "biological_targets_opened":False,"fit_performed":False,"GPU_used":False,"feature_semantic_adaptation":False,"old_Relate_arm_unchanged":True,
        "prespecified_TRAIN_numerical_cells":selection,"explicit_test_standin":True}
    inspect={**deepcopy(base),"phase":"inspect","job_id":"handmade-inspect","status":"inspection_complete_pending_root_freeze","cells":cells}
    hashes={f"{c['parent_id']}:{c['tile_id']}":c["certificate_sha256"] for c in cells}
    closure={r["path"]:r["sha256"] for c in cells for r in (c["raw_ref"],c["producer_ref"])}
    initial={"float32":"e"*64,"float64":"f"*64}
    native={"schema":"census_tsinfer_observation_admission_v1","status":"active","authorizer":"root_agent","phase":"numerical","job_id":"handmade-numerical",
        "query_truth_authorized":False,"fit_authorized":False,"source_sha256":sources,"observed_artifact_sha256":cfg["observed_artifact_sha256"],
        "inspection_manifest":refs["inspection_evidence"],"refs":{"config":{"path":fit.bridge.CONFIG,"sha256":fit._hash(fit.bridge.CONFIG)}},
        "expected_certificates":hashes,"derived_artifact_sha256":closure,"initial_state_sha256_by_dtype":initial,"group_permutation":list(range(184))}
    freeze={"schema":"root_census_tsinfer_inspection_freeze_v1","status":"complete","authorizer":"root_agent","inspection_evidence":refs["inspection_evidence"],
        "actual_observation_cells":32,"raw_and_producer_objects":64,"source_sha256":sources,"observed_artifact_sha256":cfg["observed_artifact_sha256"],
        "initialization_model_seed_dtype_runtime_unchanged":True,"raw_payload_decoded":False,"models_created":False,"targets_accessed":False,"GPU_used":False,
        "numerical_or_fit_authorized_by_this_freeze":False,"expected_certificates":hashes,"derived_artifact_sha256":closure,
        "initial_state_sha256_by_dtype":initial,"group_permutation":list(range(184)),"prespecified_TRAIN_numerical_cells":selection}
    lineage={"schema":"root_census_tsinfer_source_lineage_v1","status":"complete","authorizer":"root_agent",
        "inspection_ref":refs["inspection_evidence"],"freeze_ref":refs["inspection_freeze"],"source_sha256":sources,
        "observed_artifact_sha256":cfg["observed_artifact_sha256"],"metadata_input_sha256":cfg["metadata_input_sha256"],
        "source_and_original_count_lineage_rechecked":True,"model_or_label_outcomes_used_for_selection":False,
        "arrays_decoded":False,"models_created":False,"targets_accessed":False,"GPU_used":False}
    numerical=[]
    ordered=sorted(cells,key=lambda c:(0 if (c["parent_id"],c["tile_id"])==("bio_paths_pilot_train_001",7) else 1 if (c["parent_id"],c["tile_id"])==("bio_paths_pilot_train_000",1) else 2,c["parent_id"],c["tile_id"]))
    for before in ordered:
        row=deepcopy(before);row["status"]="numerically_qualified";row["native_oracles"]={"direct_native_census_groups":True,"float64_native_dates":True}
        row["qualification"]={"status":"qualified","performed_controls":[[d,c] for d in ("float64","float32") for c in fit.CONTROLS],
            "checks":[{"dtype":d,"control":c,"initial_state_sha256":initial[d],"finite_disposable_AdamW":True,"forest_output_VJP_parity":True,
                "sample_readout_target_column_order":True} for d in ("float64","float32") for c in fit.CONTROLS]}
        numerical.append(row)
    evidence={**deepcopy(base),"phase":"numerical","job_id":native["job_id"],"status":"numerical_complete_no_fitting_or_predictive_claim",
        "admission_ref":refs["numerical_admission"],"cells":numerical}
    terminal={"schema":"census_tsinfer_numerical_root_terminal_binding_v1","status":"complete","returncode":0,
        "evidence_ref":refs["numerical_evidence"],"admission_ref":refs["numerical_admission"],
        "inspection_ref":refs["inspection_evidence"],"freeze_ref":refs["inspection_freeze"],"job_id":native["job_id"]}
    return evidence,native,inspect,freeze,terminal,refs,lineage

def test_source_only_imports_and_entire_fit2_loop_AST():
    code="import sys;from arggnn import census_tsinfer_fit_v1 as f,census_tsinfer_device_profile_v1 as p;f.source_report_v1();p.source_report();assert not any(k in sys.modules for k in ('torch','numpy','tskit'))"
    subprocess.run([sys.executable,"-B","-c",code],check=True,cwd=ROOT,env={**os.environ,"CUDA_VISIBLE_DEVICES":"","PYTHONDONTWRITEBYTECODE":"1"})
    fit.source_report_v1();profile.source_report()

def test_synthetic_gate_positive_is_metadata_only():
    values=synthetic_gate()
    assert fit.require_terminal_numerical_v1(*values)==values[1]

@pytest.mark.parametrize("fault",[
 "old_schema","incomplete","duplicate","fixture","target","GPU","feature_change","source","observation","metadata","terminal","freeze_schema","freeze_count","missing_materialization",
 "changed_materialization","changed_inner","changed_structural","changed_lineage","changed_checks","changed_dtype","changed_control","changed_initial","changed_forest","changed_query_order",
 "missing_oracle","changed_permutation","changed_extrema","wrong_execution_order","no_lineage","lineage_not_rechecked","lineage_used_labels"])
def test_reject_incomplete_or_foreign_numerical(fault):
    e,n,i,f,t,r,l=deepcopy(synthetic_gate())
    if fault=="old_schema":e["schema"]="census_native_qualification_evidence_v1"
    elif fault=="incomplete":e["cells"].pop()
    elif fault=="duplicate":e["cells"][-1]=deepcopy(e["cells"][0])
    elif fault=="fixture":e["fixture_mode"]=True
    elif fault=="target":e["biological_targets_opened"]=True
    elif fault=="GPU":e["GPU_used"]=True
    elif fault=="feature_change":e["feature_semantic_adaptation"]=True
    elif fault=="source":e["source_sha256"]={}
    elif fault=="observation":n["observed_artifact_sha256"]={}
    elif fault=="metadata":e["metadata_input_sha256"]={}
    elif fault=="terminal":t["returncode"]=1
    elif fault=="freeze_schema":f["schema"]="root_external_census_inspection_freeze_v1"
    elif fault=="freeze_count":f["raw_and_producer_objects"]=63
    elif fault=="missing_materialization":n["derived_artifact_sha256"].pop(next(iter(n["derived_artifact_sha256"])))
    elif fault=="changed_materialization":e["cells"][0]["raw_ref"]["sha256"]="0"*64
    elif fault=="changed_inner":e["cells"][0]["inner_certificate_sha256"]="0"*64
    elif fault=="changed_structural":e["cells"][0]["structural"]={"changed":True}
    elif fault=="changed_lineage":e["cells"][0]["lineage"]={}
    elif fault=="changed_checks":e["cells"][0]["qualification"]["checks"].pop()
    elif fault=="changed_dtype":e["cells"][0]["qualification"]["checks"][0]["dtype"]="float32"
    elif fault=="changed_control":e["cells"][0]["qualification"]["checks"][0]["control"]="local"
    elif fault=="changed_initial":e["cells"][0]["qualification"]["checks"][0]["initial_state_sha256"]="0"*64
    elif fault=="changed_forest":e["cells"][0]["qualification"]["checks"][0]["forest_output_VJP_parity"]=False
    elif fault=="changed_query_order":e["cells"][0]["qualification"]["checks"][0]["sample_readout_target_column_order"]=False
    elif fault=="missing_oracle":e["cells"][0]["native_oracles"]={}
    elif fault=="changed_permutation":n["group_permutation"][-1]=0
    elif fault=="changed_extrema":f["prespecified_TRAIN_numerical_cells"]={}
    elif fault=="wrong_execution_order":e["cells"][0],e["cells"][2]=e["cells"][2],e["cells"][0]
    elif fault=="no_lineage":l=None
    elif fault=="lineage_not_rechecked":l["source_and_original_count_lineage_rechecked"]=False
    elif fault=="lineage_used_labels":l["model_or_label_outcomes_used_for_selection"]=True
    with pytest.raises((ValueError,KeyError)):fit.require_terminal_numerical_v1(e,n,i,f,t,r,l)

@pytest.fixture(scope="module")
def handmade():
    fit._science_v1()
    ts,roles,*_=fixture.handmade(full=True)
    capsule=fit.bridge_capsule_v1()
    outer,raw=capsule["certify_native_tile_v1"](ts,roles,fixture.positions(),[0.,6400.],0,fixture.BINDINGS,fixture.LINEAGE,fixture.CFG)
    cfg=fit.load_config_v1()
    return fit.ConsumerCertificateV1(outer),cfg,fit.bridge.load_config_v1()["numerical_policy"],raw

def test_exact_bridge_capsule_and_CPU_visibility_guard_preserved(handmade,monkeypatch):
    cert,cfg,qcfg,raw=handmade
    capsule=fit.bridge_capsule_v1()
    before=fit.bridge._dependencies_v1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","handmade_no_actual_GPU")
    with pytest.raises(ValueError,match="empty CUDA"):fit.bridge._dependencies_v1()
    fit.validate_census_biological_certificate_v1(cert,cert.outer.certificate_sha256)
    assert fit.bridge._dependencies_v1 is before
    assert "query_ancestry" not in cert.outer.lineage
    mask=torch.isin(raw["sequence"]["node_ids"],torch.arange(1,369,2))
    assert torch.count_nonzero(raw["sequence"]["x"][mask,:4])==0

def test_handmade_two_tree_five_control_CPU_profile_preserves_initial(handmade):
    cert,cfg,qcfg,raw=handmade
    model=fit.CensusTreeGNNV1(**cfg["model"]).float()
    initial=fit.fingerprint_sha256_v1(model.state_dict())
    checks=profile.compare_certified_tile_v1(cert,cert.outer.certificate_sha256,model,torch.arange(184),qcfg,"cpu",lambda:None)
    assert [c["control"] for c in checks]==list(fit.CONTROLS)
    assert all(c["output_shape"]==[64,120,4] and c["output_max_absolute_difference"]==0 and c["VJP_max_absolute_difference"]==0 for c in checks)
    assert all(c["selected_device"]["peak_CUDA_reserved_bytes"] is None and c["finite_disposable_AdamW"] for c in checks)
    assert fit.fingerprint_sha256_v1(model.state_dict())==initial

@pytest.mark.parametrize("control",fit.CONTROLS)
def test_handmade_exact_fit_step_and_raw_prediction(handmade,control):
    cert,cfg,qcfg,raw=handmade
    states=fit.matched_states_v1(cfg,101,"cpu");state=states[control]
    target=np.zeros((64,120,4),dtype=np.float64);target[:,:,0]=100.
    before=fit.fingerprint_sha256_v1(state["model"].state_dict())
    result=fit.train_step_v1(state,cert,cert.outer.certificate_sha256,target,control,torch.arange(184),cfg,"cpu")
    assert result["steps"]==1 and np.isfinite(result["loss"])
    assert fit.fingerprint_sha256_v1(state["model"].state_dict())!=before
    logits,p,ids=fit.raw_prediction_v1(state,cert,cert.outer.certificate_sha256,control,torch.arange(184),cfg)
    assert logits.shape==p.shape==(64,120,4) and ids.tolist()==list(range(120))
    assert np.allclose(p.sum(-1),1,atol=1e-6)

def test_outer_mutation_and_inner_only_certificate_rejected(handmade):
    cert,*_=handmade
    with pytest.raises(ValueError):fit.validate_census_biological_certificate_v1(cert.outer.native_certificate,cert.outer.native_certificate.certificate_sha256)
    changed=fit.ConsumerCertificateV1(replace(cert.outer,lineage={**cert.outer.lineage,"namespace":"other_ARG"}))
    with pytest.raises(ValueError):fit.validate_census_biological_certificate_v1(changed,cert.outer.certificate_sha256)

class MemoryReader:
    def __init__(self,values):self.values=values;self.calls=[]
    def read(self,ref):self.calls.append(ref["path"]);return self.values[ref["path"]]

@pytest.mark.parametrize("coverage,expected",[
 ([117.,1998898.],[117.,1998898.]),([19.,1999991.],[19.,1999991.]),
 ([273.,1999681.],[300.,1999681.]),([116.,1999724.],[116.,1999700.])])
def test_original100bp_center_support_and_original_target_IDs(coverage,expected):
    _,roles,original,*_=fixture.handmade(full=True)
    ref={"path":"explicit_handmade_source","sha256":"a"*64};reader=MemoryReader({ref["path"]:json.dumps(original).encode()})
    loaded=(None,roles,fixture.positions(),coverage[0],coverage,{"source":ref},{})
    binding={"query_sample_indexes":list(range(120)),"query_raw_node_ids":[r["observed_node_id"] for r in original[:120]]}
    parent=fit.scoring_parent_v1({"parent_id":"handmade_val","split":"val"},{},loaded,reader,binding)
    assert parent["edges"][[0,-1]].tolist()==expected
    assert float(np.diff(parent["edges"]).sum())*64==(expected[1]-expected[0])*64
    changed={**binding,"query_raw_node_ids":list(reversed(binding["query_raw_node_ids"]))}
    with pytest.raises(ValueError):fit.scoring_parent_v1({"parent_id":"handmade_val","split":"val"},{},loaded,reader,changed)

def test_VAL_target_requires_seal_and_exact_original_query_ids():
    parent={"parent_id":"handmade_val","split":"val","raw_query_nodes":np.arange(120)}
    b=io.BytesIO();np.savez(b,intervals_bp=np.array([[0.,2000000.]],dtype=np.float64),query_raw_node_ids=np.arange(120),
        native_paths_1based=np.ones((1,120),dtype=np.int64),source_classes_0based=np.zeros((1,120),dtype=np.int64))
    target={"path":"explicit_handmade_target","sha256":"a"*64};seal={"path":"explicit_handmade_seal","sha256":"b"*64}
    reader=MemoryReader({target["path"]:b.getvalue(),seal["path"]:b"explicit handmade sealed bytes"})
    with pytest.raises(ValueError):fit.ExternalTargetV1(parent,{"truth_npz":target},"VAL_after_sealed_prediction",reader)
    assert reader.calls==[]
    v=fit.ExternalTargetV1(parent,{"truth_npz":target},"VAL_after_sealed_prediction",reader,seal)
    assert reader.calls==[seal["path"],target["path"]]
    assert v.c4.shape==(1,120) and not hasattr(v,"native_paths_1based")
    with pytest.raises(ValueError):fit.ExternalTargetV1(parent,{"truth_npz":target},"TRAIN_loss",reader)

def test_memory_and_logical_jobs():
    now=datetime.now(timezone.utc)
    a={"job_id":"handmade","created_at_utc":(now-timedelta(seconds=10)).isoformat(),"expires_at_utc":(now+timedelta(seconds=100)).isoformat()}
    cfg={"GPU_total_reservation_mib":16384,"CPU_total_reservation_bytes":64000000000}
    memory={"schema":"census_tsinfer_consumer_memory_v1","status":"active","job_id":"handmade","external_watchdog":True,**cfg,
        "created_at_utc":(now-timedelta(seconds=5)).isoformat(),"expires_at_utc":(now+timedelta(seconds=50)).isoformat()}
    fit.memory_v1(memory,a,cfg)
    for k,v in (("job_id","other"),("external_watchdog",False),("GPU_total_reservation_mib",1),("expires_at_utc",(now-timedelta(seconds=1)).isoformat())):
        with pytest.raises(ValueError):fit.memory_v1({**memory,k:v},a,cfg)

def test_inactive_admissions_cannot_open_actual_sources(tmp_path):
    for mod,schema in ((fit,"census_tsinfer_fit_admission_v1"),(profile,"census_tsinfer_device_profile_admission_v1")):
        path=tmp_path/(schema+".json");data=json.dumps({"schema":schema,"status":"inactive"}).encode();path.write_bytes(data)
        ref={"path":str(path),"sha256":hashlib.sha256(data).hexdigest()}
        with pytest.raises((ValueError,KeyError)):
            (mod.admit_fit_v1 if mod is fit else mod.admit_profile)(ref)

def test_capsule_loader_and_materialized_certificate_roundtrip(tmp_path):
    fit._science_v1();capsule=fit.bridge_capsule_v1();cfg=fit.bridge.load_config_v1()
    ts,roles,original,derived,mapping=fixture.handmade(full=True)
    path=tmp_path/"explicit_handmade.trees";ts.dump(path)
    values={}
    def put(name,data):
        ref={"path":"research/reviews/census_tsinfer_fit_v1_author/explicit_handmade_memory/"+name,"sha256":hashlib.sha256(data).hexdigest()}
        values[ref["path"]]=data;return ref
    nref=put("observed.trees",path.read_bytes());sref=put("source.json",json.dumps(original).encode());dref=put("derived.json",json.dumps(derived).encode())
    b=io.BytesIO();np.savez(b,positions_bp=fixture.positions(),bin_edges_bp=np.arange(0.,2000001.,100.,dtype=np.float64));gref=put("grid.npz",b.getvalue())
    h={"parent_id":"bio_paths_pilot_train_000","split":"train","inferred_tree":nref,"original_observed_sample_manifest":sref,
        "observed_sample_manifest":dref,"sample_index_to_inferred_tree_node_id":mapping,
        "coordinate_contract":{"fixed_grid":gref,"coverage_original_bp0":[0.,6400.],"local_to_original_offset_bp":0.,
        "original_parent_interval_bp0":[0,2000000],"normalization_length_bp":2000000,
        "physical_coordinate_transform":"physical_original = local_tree_position + coverage_left","retained_fixed100bp_indexes":list(range(64))}}
    h["completion_provenance"]=put("completion.json",json.dumps(h).encode())
    pin={"parent_id":h["parent_id"],"split":"train","namespace":"explicit_handmade_not_actual_ARG",
        "ARGformer_dispatcher_ref":sref,"ARGformer_qualification_ref":sref,"ARGformer_prepared_manifest_ref":sref}
    class Reader(fit.ConsumerReaderV1):
        def read(self,record,**kwargs):
            assert hashlib.sha256(values[record["path"]]).hexdigest()==record["sha256"]
            return values[record["path"]]
    reader=Reader();loaded=fit.load_parent_v1(h,{"pin":pin},{},{},reader,tmp_path)
    ts2,roles2,positions,offset,coverage,refs,lineage=loaded
    raw,*_=capsule["extract_native_tsinfer_tile_v1"](ts2,roles2,positions,coverage,0,cfg)
    rawref=put("tile.whole_observed.pt",capsule["raw_sequence_bytes_v1"](raw,cfg))
    producer=capsule["_producer_payload_v1"](raw,lineage,0,cfg,rawref);pref=put("tile.producer.json",fit.bridge.json_bytes(producer))
    full=dict(lineage,raw_ref=rawref,producer_ref=pref)
    outer,_=capsule["certify_native_tile_v1"](ts2,roles2,positions,coverage,0,
        {"native_observed_ts":refs["native"],"serialized_whole_observed":rawref,"observed_roster":refs["derived"],"observed_grid":refs["grid"],"producer_receipt":pref},full,cfg,cached_raw=raw)
    previous={"raw_ref":rawref,"producer_ref":pref,"lineage":outer.lineage,"num_trees":2,"inner_certificate_sha256":outer.native_certificate.certificate_sha256}
    prepared,*_=fit.prepare_tile_v1(h,{"pin":pin},{"tile_id":0,"frozen":previous},loaded,
        {"expected_certificates":{h["parent_id"]+":0":outer.certificate_sha256}},{},reader)
    assert prepared.outer.certificate_sha256==outer.certificate_sha256
    assert len(prepared.native_trees)==2 and prepared.provenance["split"]=="train"
    corrupt=deepcopy(raw);corrupt["sequence"]["x"][0,5]+=.1
    with pytest.raises(ValueError):
        capsule["certify_native_tile_v1"](ts2,roles2,positions,coverage,0,
            {"native_observed_ts":refs["native"],"serialized_whole_observed":rawref,"observed_roster":refs["derived"],"observed_grid":refs["grid"],"producer_receipt":pref},full,cfg,cached_raw=corrupt)

@pytest.mark.parametrize("control",fit.CONTROLS)
def test_wrapper_matches_unchanged_inner_model_output_and_VJP(handmade,control):
    from arggnn.census_biological_bank_v1 import forward_certified_census_biological_v1 as original
    cert,cfg,qcfg,raw=handmade
    torch.manual_seed(41);a=fit.CensusTreeGNNV1(**cfg["model"]).float();b=deepcopy(a)
    kwargs={"group_permutation":torch.arange(184)} if control=="permuted_groups" else {}
    left=original(a,cert.outer.native_certificate,expected_certificate_sha256=cert.outer.native_certificate.certificate_sha256,
        control=control,readout_chunk_size=128,**kwargs)
    right=fit.forward_certified_census_biological_v1(b,cert,expected_certificate_sha256=cert.outer.certificate_sha256,
        control=control,readout_chunk_size=128,**kwargs)
    assert torch.equal(left.logits,right.logits) and torch.equal(left.scored_query_ids,right.scored_query_ids)
    ga=torch.autograd.grad(left.logits.double().square().mean(),tuple(a.parameters()),allow_unused=True)
    gb=torch.autograd.grad(right.logits.double().square().mean(),tuple(b.parameters()),allow_unused=True)
    assert all((x is None and y is None) or (x is not None and y is not None and torch.equal(x,y)) for x,y in zip(ga,gb))

def test_exclusive_and_shared_logical_job_attribution(monkeypatch):
    a={"job_id":"handmade-profile"}
    monkeypatch.setenv("ARGGNN_JOB_NAME","handmade-profile");monkeypatch.delenv("ARGGNN_LOGICAL_JOB_NAME",raising=False)
    profile.require_logical_job_v1({"launcher_metadata":{"job":"handmade-profile"}},a)
    fit.require_logical_job_v1({"launcher_metadata":{"job":"handmade-profile"}},a)
    monkeypatch.setenv("ARGGNN_JOB_NAME","handmade-pool");monkeypatch.setenv("ARGGNN_LOGICAL_JOB_NAME","handmade-profile")
    profile.require_logical_job_v1({"launcher_metadata":{"job":"handmade-pool"}},a)
    fit.require_logical_job_v1({"launcher_metadata":{"job":"handmade-pool"}},a)
    monkeypatch.setenv("ARGGNN_LOGICAL_JOB_NAME","other")
    with pytest.raises(ValueError):profile.require_logical_job_v1({"launcher_metadata":{"job":"handmade-pool"}},a)
    with pytest.raises(ValueError):fit.require_logical_job_v1({"launcher_metadata":{"job":"handmade-pool"}},a)

def test_readonly_v4_storage_first_pins_and_consumer_bindings(monkeypatch):
    from arggnn import project_budget_v4 as budget
    cfg=fit.load_config_v1();now=datetime.now(timezone.utc)
    a={"job_id":"handmade-fit","output_directory":cfg["output_prefix"]+"handmade-fit","refs":{"runtime":{"path":"explicit_handmade_runtime","sha256":"a"*64}}}
    reserve={"output":str(ROOT/a["output_directory"]),"auxiliary_scopes":[],"reserved_bytes":cfg["output_reservation_bytes"],
        "consumer_bindings":[{"path":fit.SOURCE,"sha256":fit._hash(fit.SOURCE)},{"path":fit.CONFIG,"sha256":fit._hash(fit.CONFIG)},a["refs"]["runtime"]]}
    storage={"schema":"census_tsinfer_consumer_storage_v1","status":"active","job_id":a["job_id"],"output_directory":a["output_directory"],
        "reserved_new_bytes":cfg["output_reservation_bytes"],"mode":"project_budget_v4","reservation":reserve,
        "initial_ledger_pins":{"legacy_ledger_sha256":"b"*64,"supplemental_ledger_sha256":"c"*64},
        "created_at_utc":(now-timedelta(seconds=10)).isoformat(),"expires_at_utc":(now+timedelta(seconds=60)).isoformat()}
    calls=[]
    def stub(root,reservation,**kw):
        calls.append((root,reservation,kw));return {"legacy_ledger_mutated":False,"supplemental_ledger_mutated":False}
    monkeypatch.setattr(budget,"check_reservation",stub)
    assert fit.storage_v1(storage,a,cfg,fit.SOURCE,fit.CONFIG)==storage
    assert calls[0][2]["initial_ledger_pins"]==storage["initial_ledger_pins"] and calls[0][2]["boundary"] is True
    for bad in ({**storage,"reserved_new_bytes":1},{**storage,"initial_ledger_pins":{}},{**storage,"reservation":{**reserve,"consumer_bindings":[]}}):
        with pytest.raises(ValueError):fit.storage_v1(bad,a,cfg,fit.SOURCE,fit.CONFIG)
    assert len(calls)==1
    monkeypatch.setattr(budget,"check_reservation",lambda *args,**kwargs:{"legacy_ledger_mutated":True,"supplemental_ledger_mutated":False})
    with pytest.raises(ValueError):fit.storage_v1(storage,a,cfg,fit.SOURCE,fit.CONFIG)

def test_canonical_jobs_reject_traversal_and_unbounded_names():
    assert fit.canonical_job_v1("handmade-seed101_r001")
    for value in ("","../other","nested/job","white space",None,1):assert not fit.canonical_job_v1(value)

def test_fresh_resources_and_logical_job_precede_opaque_reads_and_science():
    for module,name in ((fit,"admit_fit_v1"),(profile,"admit_profile")):
        tree=ast.parse(Path(module.__file__).read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        calls=[(n.lineno,ast.unparse(n.func)) for n in ast.walk(node) if isinstance(n,ast.Call)]
        def first(suffix):return min(line for line,func in calls if func.endswith(suffix))
        assert first("require_logical_job_v1")<first("verify_observation_closure_v1")
        assert first("storage_v1")<first("verify_observation_closure_v1")<first("runtime_v1")
        assert first("memory_v1")<first("verify_observation_closure_v1")
    source=ast.get_source_segment(Path(fit.__file__).read_text(),next(n for n in ast.parse(Path(fit.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="gate_from_reader_v1"))
    assert 'native["observed_artifact_sha256"]' not in source and 'native["derived_artifact_sha256"]' not in source
