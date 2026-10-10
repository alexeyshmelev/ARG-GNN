"""Synthetic DEV-only tensors with explicit mocked cohort/producer/factory admission.

The positive fixtures replay handmade main/secondary evidence. They are not real
completed development or temperature artifacts. No held-out input or CUDA use.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from arggnn import validation_temperature as temperature
from arggnn import confirmatory_lock as provenance
from test_secondary_development import evidence as secondary_evidence, admitted as secondary_admitted
from test_secondary_development import secondary_runs as make_secondary_runs

ROOT=Path(__file__).resolve().parents[1]


def write_json(root,path,value):
    p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(temperature.canonical(value))
    return {"path":path,"sha256":temperature.sha(p.read_bytes())}


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    patches=pytest.MonkeyPatch()
    base=secondary_evidence.__wrapped__(tmp_path_factory)
    fixture=secondary_admitted.__wrapped__(base,patches,SimpleNamespace(node=SimpleNamespace(name="validation_temperature_shared_fixture")))
    runs=make_secondary_runs.__wrapped__(fixture)
    root=runs["root"]
    for path in temperature.CODE:
        dest=root/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/path).read_bytes())
    secondary_receipt=temperature.secondary.synthesize_secondary_selection(runs["config_path"],runs["bound"].sha256,
        run_receipts=runs["run_refs"],project_root=root)
    secondary_ref=write_json(root,"temperature_fixture/secondary_selection.json",secondary_receipt["values"])
    amendment=write_json(root,"temperature_fixture/scope_amendment.json",
        json.loads((ROOT/"research/protocols/development_scope_amendment_v2.json").read_text()))
    calls=[]
    def complete_main_admission(*args,**kwargs):
        assert kwargs["output_directory"] is None and len(kwargs["family_configs"])==10
        assert {ref["family_id"] for ref in kwargs["family_configs"]}==set(temperature.combined.FAMILIES)
        calls.append("mocked_complete_main_producer_admission")
        return SimpleNamespace(plan=deepcopy(runs["plan"]))
    patches.setattr(temperature.combined,"bind_development_selection",complete_main_admission)
    def primary_factory_admission(capture,bundle,selection):
        # The source/factory proof is deliberately replaced for handmade counts/state.
        assert bundle["choice"] in (selection["candidate_choice"],selection["strongest_required_baseline"])
        calls.append("mocked_primary_factory_admission")
    patches.setattr(provenance,"validate_primary_bundle",primary_factory_admission)
    refs={"main_plan":runs["plan_ref"],"main_selection":runs["receipt_ref"],
          "secondary_config":{"path":str(runs["config_path"].relative_to(root)),"sha256":runs["bound"].sha256},
          "secondary_selection":secondary_ref,"amendment":amendment}
    yield {**runs,"refs":refs,"calls":calls}
    patches.undo()


@pytest.fixture
def case(study,request):
    directory=study["root"]/"temperature_cases"/request.node.name
    directory.mkdir(parents=True)
    return {**study,"directory":directory,"refs":deepcopy(study["refs"])}


def admit(case):
    r=case["refs"]
    return temperature.admit_temperature_fitting(r["main_plan"],r["main_selection"],r["secondary_config"],
        r["secondary_selection"],r["amendment"],project_root=case["root"])


def test_complete_admission_replays_both_selections_before_numerical_loader(case,monkeypatch):
    events=[]
    original_main=provenance.replay_main;original_secondary=provenance.replay_secondary
    def replay_main(*args,**kwargs):
        events.append("main_replay");return original_main(*args,**kwargs)
    def replay_secondary(*args,**kwargs):
        events.append("secondary_replay");return original_secondary(*args,**kwargs)
    monkeypatch.setattr(provenance,"replay_main",replay_main);monkeypatch.setattr(provenance,"replay_secondary",replay_secondary)
    monkeypatch.setattr(temperature,"_numerical_inputs",lambda *args:(_ for _ in ()).throw(AssertionError("Premature numerical input read")))
    result=admit(case)
    assert events==["main_replay","secondary_replay"] and len(result.methods)==4
    assert result.selection["matrix_counts"]=={"success":240,"failed":0,"missing":0}
    assert all(row["choice"] is not None for row in result.secondary_selection["controls"].values())
    assert result.plan["splits"]["test"]==0


@pytest.mark.parametrize("which",["main","secondary","producer"])
def test_incomplete_admission_never_loads_fit_inputs(case,monkeypatch,which):
    calls=[]
    monkeypatch.setattr(temperature,"_numerical_inputs",lambda *args:calls.append("forbidden"))
    def refuse(*args,**kwargs):raise ValueError("Explicit synthetic admission refusal")
    if which=="main":monkeypatch.setattr(provenance,"replay_main",refuse)
    elif which=="secondary":monkeypatch.setattr(provenance,"replay_secondary",refuse)
    else:monkeypatch.setattr(temperature.combined,"bind_development_selection",refuse)
    with pytest.raises(ValueError,match="admission refusal"):admit(case)
    assert calls==[]


@pytest.mark.parametrize("slot",["main_plan","main_selection","secondary_config","secondary_selection","amendment"])
def test_final_path_rejected_before_capture_read(case,monkeypatch,slot):
    case["refs"][slot]={"path":"data/final_test/hidden.json","sha256":"a"*64}
    reads=[]
    original=temperature.main._Capture.read
    def watch(self,ref,**kwargs):
        reads.append(ref["path"]);return original(self,ref,**kwargs)
    monkeypatch.setattr(temperature.main._Capture,"read",watch)
    with pytest.raises(ValueError):admit(case)
    assert reads==[]


def test_scalar_fit_is_equal_parent_not_coordinate_weighted_and_unchanged():
    z1=np.zeros((1,1,4));z1[...,0]=2.
    z2=np.repeat(z1,100,axis=0);y1=np.zeros((1,1),dtype=np.int64);y2=np.ones((100,1),dtype=np.int64)
    before=[x.copy() for x in (z1,z2,y1,y2)]
    fitted=temperature.fit_scalar_temperature([z1,z2],[y1,y2])
    assert fitted["temperature"]==pytest.approx(2/math.log(3),rel=1e-6)
    assert fitted["external_parent_count"]==2
    assert fitted["calibrated_equal_parent_nll"]<=fitted["raw_equal_parent_nll"]
    assert len(fitted["evaluations"])>0
    for a,b in zip((z1,z2,y1,y2),before):np.testing.assert_array_equal(a,b)


def test_flat_logits_choose_raw_temperature_and_uncapped_loss():
    zeros=np.zeros((2,3,4));labels=np.zeros((2,3),dtype=np.int64)
    flat=temperature.fit_scalar_temperature([zeros],[labels])
    assert flat["temperature"]==1.
    wrong=zeros.copy();wrong[...,0]=-1000.
    fitted=temperature.fit_scalar_temperature([wrong],[labels])
    assert fitted["raw_equal_parent_nll"]>1000 and fitted["calibrated_equal_parent_nll"]>50
    assert fitted["bound_selected"] and fitted["temperature"]==pytest.approx(20.)


@pytest.mark.parametrize("bad",["nan","shape","labels","classes"])
def test_bad_raw_arrays_cannot_fit(bad):
    z=np.zeros((2,2,4));y=np.zeros((2,2),dtype=np.int64)
    if bad=="nan":z[0,0,0]=np.nan
    elif bad=="shape":z=z[:1]
    elif bad=="labels":y=y.astype(np.float64)
    else:z=z[...,:3]
    with pytest.raises(ValueError):temperature.fit_scalar_temperature([z],[y])


def test_complete_actual_synthetic_fits_emit_lock_compatible_manifest_preserving_inputs(case):
    admitted=admit(case)
    files=[bundle[name] for method in admitted.methods.values() for bundle in method["seed_bundles"]
           for name in ("checkpoint","development_prediction") if bundle[name] is not None]
    before={ref["path"]:temperature.sha((case["root"]/ref["path"]).read_bytes()) for ref in files}
    result=temperature.fit_validation_temperatures(admitted,output_directory=case["directory"]/"fits")
    assert len(result["values"]["records"])==12 and result["values"]["final_test_access"] is False
    assert Path(result["path"]).read_bytes()==result["bytes"]
    for ref in files:assert temperature.sha((case["root"]/ref["path"]).read_bytes())==before[ref["path"]]
    for row in result["values"]["records"]:
        assert row["validation_parent_ids"]==admitted.selection["ordered_parent_ids"]
        evidence=json.loads((case["root"]/row["fit_evidence"]["path"]).read_text())
        assert evidence["unchanged_model_weights"] and evidence["source_binding"]==admitted.plan["source_binding"]
        assert len(evidence["validation_parents"])==16
        record=json.loads((case["root"]/evidence["fit_record_artifacts"][0]["path"]).read_text())
        assert record["fit_policy"]==temperature.FIT_POLICY and record["external_parent_count"]==16
        assert .05<=row["temperature"]<=20. and record["calibrated_equal_parent_nll"]<=record["raw_equal_parent_nll"]+1e-12
    summary=provenance.validate_temperature_manifest(admitted.capture,result["values"],
        {key:admitted.methods[key] for key in ("candidate","strongest_required_baseline")},
        {key:value for key,value in admitted.methods.items() if key not in ("candidate","strongest_required_baseline")},
        admitted.selection,admitted.secondary_selection,admitted.references,admitted.plan)
    assert len(summary)==12
    with pytest.raises(ValueError,match="Fresh exclusive"):
        temperature.fit_validation_temperatures(admitted,output_directory=case["directory"]/"fits")


def test_in_memory_fit_policy_drift_blocks_emission(case):
    admitted=admit(case);admitted.fit_policy["temperature_bounds"]=[.01,100.]
    with pytest.raises(ValueError,match="In-memory"):temperature.fit_validation_temperatures(admitted,output_directory=case["directory"]/"never")
    assert not (case["directory"]/"never").exists()


def test_selected_prediction_hash_drift_blocks_before_output(case):
    admitted=admit(case);ref=admitted.methods["candidate"]["seed_bundles"][0]["development_prediction"]
    path=case["root"]/ref["path"];before=path.read_bytes()
    try:
        path.write_bytes(b"Explicit synthetic corrupt selected prediction")
        with pytest.raises(ValueError):temperature.fit_validation_temperatures(admitted,output_directory=case["directory"]/"never")
        assert not (case["directory"]/"never").exists()
    finally:path.write_bytes(before)


def test_external_label_mismatch_cannot_fit(case):
    admitted=admit(case);labels=case["cases"]["val"][0].labels;before=labels.clone()
    try:
        labels.fill_(1)
        with pytest.raises(ValueError,match="raw validation NLL"):
            temperature.fit_validation_temperatures(admitted,output_directory=case["directory"]/"never")
        assert not (case["directory"]/"never").exists()
    finally:labels.copy_(before)


def test_optimizer_failure_preserves_trace_and_no_manifest(case,monkeypatch):
    admitted=admit(case)
    def failing(objective,**kwargs):
        objective(0.)
        return SimpleNamespace(success=False,x=0.,message="Explicit synthetic convergence failure",nfev=1,nit=500)
    monkeypatch.setattr(temperature,"minimize_scalar",failing)
    output=case["directory"]/"failed"
    with pytest.raises(temperature.TemperatureFitFailure):
        temperature.fit_validation_temperatures(admitted,output_directory=output)
    assert not (output/"manifest.json").exists()
    failure=json.loads((output/"failure.json").read_text())
    assert failure["optimization_record"]["evaluations"] and failure["manifest_emitted"] is False


def test_active_external_label_drift_rejects_after_first_objective(case,monkeypatch):
    admitted=admit(case);labels=case["cases"]["val"][0].labels;before=labels.clone()
    original=temperature.minimize_scalar
    def drift(objective,**kwargs):
        objective(0.);labels.fill_(1);return original(objective,**kwargs)
    monkeypatch.setattr(temperature,"minimize_scalar",drift)
    output=case["directory"]/"failed"
    try:
        with pytest.raises(temperature.TemperatureFitFailure,match="labels drift"):
            temperature.fit_validation_temperatures(admitted,output_directory=output)
        assert not (output/"manifest.json").exists() and not list(output.glob("*_fit.json"))
    finally:labels.copy_(before)


@pytest.mark.parametrize("failure",["write","check","race"])
def test_fit_artifact_publication_cleans_only_owned_pending(tmp_path,monkeypatch,failure):
    path=tmp_path/"synthetic_non_executable_record.json"
    if failure=="write":
        original=temperature.tempfile.NamedTemporaryFile
        class Broken:
            def __init__(self,handle):self.handle=handle;self.name=handle.name
            def __enter__(self):return self
            def fileno(self):return self.handle.fileno()
            def __exit__(self,*args):return self.handle.__exit__(*args)
            def write(self,value):self.handle.write(value[:7]);raise OSError("Synthetic fit-record write interruption")
        monkeypatch.setattr(temperature.tempfile,"NamedTemporaryFile",lambda **kw:Broken(original(**kw)))
    def check():
        if failure=="check":raise ValueError("Synthetic source drift")
        if failure=="race":path.write_bytes(b"Existing raced external bytes")
    with pytest.raises((ValueError,OSError)):temperature._publish(path,b"Explicit synthetic non-executable record",check)
    assert not list(tmp_path.glob(".val_temperature_pending_*"))
    if failure=="race":assert path.read_bytes()==b"Existing raced external bytes"
    else:assert not path.exists()


def test_no_actual_admission_cannot_emit(case):
    with pytest.raises(ValueError,match="Genuine"):
        temperature.fit_validation_temperatures(SimpleNamespace(),output_directory=case["directory"]/"never")
    assert not (case["directory"]/"never").exists()


@pytest.mark.parametrize("mutation",["same_inode","foreign_inode"])
def test_staged_fit_bytes_or_foreign_inode_cannot_publish(tmp_path,mutation):
    path=tmp_path/"never_corrupt_fit.json";foreign=None
    def check():
        nonlocal foreign
        pending=next(tmp_path.glob(".val_temperature_pending_*"))
        if mutation=="same_inode":pending.write_bytes(b"Corrupt staged fit bytes")
        else:
            pending.unlink();pending.write_bytes(b"Foreign replacement fit bytes");foreign=pending
    with pytest.raises(ValueError,match="(stat drift|inode changed|byte/hash drift)"):
        temperature._publish(path,b"Correct explicit synthetic non-executable fit record",check)
    assert not path.exists()
    if mutation=="foreign_inode":assert foreign.read_bytes()==b"Foreign replacement fit bytes"
    else:assert not list(tmp_path.glob(".val_temperature_pending_*"))


@pytest.mark.parametrize("mutation",["same_inode","foreign_inode"])
def test_published_fit_drift_cleans_owned_and_preserves_foreign_output(tmp_path,monkeypatch,mutation):
    path=tmp_path/"never_corrupt_fit.json";original=temperature.os.link
    def link(source,destination,**kwargs):
        original(source,destination,**kwargs)
        if mutation=="same_inode":Path(destination).write_bytes(b"Corrupt linked fit bytes")
        else:
            Path(destination).unlink();Path(destination).write_bytes(b"Foreign published replacement bytes")
    monkeypatch.setattr(temperature.os,"link",link)
    with pytest.raises(ValueError,match="(inode changed|byte/hash drift)"):
        temperature._publish(path,b"Correct explicit synthetic non-executable fit record",lambda:None)
    assert not list(tmp_path.glob(".val_temperature_pending_*"))
    if mutation=="foreign_inode":assert path.read_bytes()==b"Foreign published replacement bytes"
    else:assert not path.exists()
