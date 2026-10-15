"""Metadata/source-only H048 root IO checks. Never configure or call any guard."""
from pathlib import Path
from types import SimpleNamespace
import ast,contextlib,copy,hashlib,io,json,sys
sys.path[:0]=["/workspace-SR003.nfs2/ARG-GNN","/workspace-SR003.nfs2/ARG-GNN/src"]
from research.reviews.H048_GPU45_grid_root_source_v1 import root_common_v1 as c
from research.reviews.H048_GPU45_grid_root_source_v1 import root_driver as driver
from research.reviews.H048_GPU45_grid_root_source_v1 import context_factory_v1 as context
from arggnn import train_marker_grid_reset_chain_v1 as w
HERE=c.ROOT/c.HERE
def denied(call):
    try:call()
    except (ValueError,KeyError):return
    raise AssertionError("Expected denial")
cfg=c.read(c.ref(HERE/"config_inactive.json"))
assert cfg["execution_enabled"] is False
active=dict(cfg,execution_enabled=True,status="root_frozen_protocol_before_admission",
    root_resource_caps_are_proposals=False)
w.validate_protocol(active)
counts=c.read(cfg["count_manifest_ref"]);prep=c.read(cfg["preparation_completion_ref"])
manifest=c.read(cfg["source_manifest_ref"])
assert manifest["schema"]=="H048_grid_frozen_source_manifest_v1"
for name,digest in manifest["source_sha256"].items():assert c.ref(name)["sha256"]==digest
for r in manifest["metadata_refs"].values():assert c.ref(r["path"])==r
assert all(manifest["source_sha256"][r["path"]]==r["sha256"] for r in w.SCIENTIFIC_REFS)
assert cfg["producer_ref"]==c.ref(w.SOURCE)
assert c.CUDA_CAP==50000000000 and c.RSS_CAP==512000000000 and c.OUTPUT_CAP==100000000
assert c.read(c.AUTH_REF)["max_project_cuda_reserved_bytes"]==c.CUDA_CAP
assert "actual980_policy" in manifest["metadata_refs"]
plans={}
for phase in ("train","evaluate"):
    plan=c.count_plan_v1(counts,phase,prep);plans[phase]=plan
    assert plan["parent_count"]==(32 if phase=="train" else 8)
    assert plan["all_phase_grid_plus_marker_cache_upper_bytes"]<=cfg["grid_cache_cap_bytes"]
    assert plan["all_phase_marker_tensor_bytes"]<=cfg["marker_cache_cap_bytes"]
    assert plan["all96_original_grid_probability_bytes"]<=cfg["RAM_prediction_cap_bytes"]
    for pid,row in plan["parents"].items():
        bank=SimpleNamespace(parent=SimpleNamespace(numel=lambda n=row["native_nodes"]:n),
            genotype=SimpleNamespace(shape=(row["sites"],952)),sample_index=SimpleNamespace(shape=(row["native_holds"],952)),
            query_column=SimpleNamespace(shape=(686,)),spans=range(row["native_holds"]))
        declared=w.grid_upper_bounds_v1(bank,range(row["grid_centers"]))
        proof=c.preparation_check_v1(pid,bank,declared,cfg,counts,prep,rss_reader=lambda:1000000000)
        assert proof["full_position_history_and_queries_preserved"]
        denied(lambda:c.preparation_check_v1(pid,bank,declared,cfg,counts,prep,rss_reader=lambda:c.RSS_CAP))
bad=copy.deepcopy(counts);bad["parents"][w.TRAIN[0]]["haplotypes"]=951
denied(lambda:c.count_plan_v1(bad,"train",prep))
# Metadata-only future TRAIN lineage fixture; no real completion/checkpoint read.
def rr(path):return dict(path=path,sha256="f"*64)
gpu=4;output="experiments/results/H048_GPU4_grid_reset_train_seed101_v1"
inputref=rr(c.HERE+"/admitted_train/launch_input.json");configref=rr(c.HERE+"/admitted_train/config.json")
runtime=rr(c.HERE+"/admitted_train/runtime.json")
complete=rr(output+"/completed.json");launchref=rr(c.HERE+"/admitted_train/launch.json");waitref=rr(c.HERE+"/admitted_train/actual_wait.json")
groups={str(at):{a:rr(output+"/"+a+".pass_%02d.pt"%at) for a in w.ARMS} for at in w.PASSES}
train=dict(schema="H048_paired_train_completion_v1",status="completed_fixed30pass_paired_TRAIN",
    hypothesis_id="H048",arms=list(w.ARMS),steps={a:960 for a in w.ARMS},
    VAL_truth_read=False,outer_truth_read=False,same_VAL_is_development_evidence=True,
    input_ref=inputref,config_ref=configref,producer_ref=c.ref(w.SOURCE),checkpoint_groups=groups,
    preparation_completion_ref=cfg["preparation_completion_ref"])
launch=dict(launcher_pid=12345,input_ref=inputref,phase="train",physical_gpu=4)
wait=dict(actual_direct_launcher_wait=True,returncode=0,logging_error=None,supervisor_error=None,
    completed_ref=complete,launcher_pid=12345,phase="train",physical_gpu=4)
old=dict(phase="train",physical_gpu=4,output_directory=output,hypothesis_id="H048",
    producer_ref=c.ref(w.SOURCE),refs=dict(config=configref,runtime=runtime),
    preparation_completion_ref=cfg["preparation_completion_ref"])
fixtures={r["path"]:v for r,v in ((complete,train),(launchref,launch),(waitref,wait),(inputref,old))}
reader=SimpleNamespace(decode_json=lambda r:fixtures[r["path"]])
spec=dict(phase="evaluate",train_completion_ref=complete,train_launch_ref=launchref,train_actual_wait_ref=waitref)
lineage=c.validate_eval_train_io_v1(reader,spec)
assert lineage["train_checkpoint_groups"]==groups
wait["returncode"]=1
denied(lambda:c.validate_eval_train_io_v1(reader,spec));wait["returncode"]=0
train["steps"][w.ARMS[0]]=959
denied(lambda:c.validate_eval_train_io_v1(reader,spec));train["steps"][w.ARMS[0]]=960
old["producer_ref"]=dict(c.ref(w.SOURCE),sha256="0"*64)
denied(lambda:c.validate_eval_train_io_v1(reader,spec))
# Default CLI must be inert. Replace mutation entrypoints with fail-fast mocks.
driver.load_guard_v1=lambda **kwargs:(_ for _ in ()).throw(AssertionError("Guard configured"))
driver.admit=lambda *args:(_ for _ in ()).throw(AssertionError("Admission called"))
driver.launch=lambda *args:(_ for _ in ()).throw(AssertionError("Launch called"))
with contextlib.redirect_stdout(io.StringIO()) as stream:assert driver.main([])==0
assert "source_only_no_admission_or_execution" in stream.getvalue()
assert "torch" not in sys.modules
for p in HERE.glob("*.py"):ast.parse(p.read_text())
assert not (HERE/"admitted_train").exists() and not (HERE/"admitted_evaluate").exists()
receipt=dict(schema="H048_root_source_static_receipt_v1",status="passed_metadata_source_checks",
    actual_guard_loaded_or_configured=False,reservation_or_journal_mutated=False,scientific_launch=False,
    actual_payload_checkpoint_truth_opened=False,Torch_imported=False,
    all111_transitive_sources_bound=len(manifest["source_sha256"])==111,
    active980_and_memo2_sources_pinned=True,original50GB_GPU45_authority_unchanged=True,
    exact_original32TRAIN8VAL_and_H044_preparation=True,all40_actual_count_bounds_checked=True,
    TRAIN960_EVAL_lineage_rejects_failed_wait_incomplete_steps_and_changed_producer=True,
    default_rootCLI_inert=True,
    train_grid_plus_marker_cache_upper_bytes=plans["train"]["all_phase_grid_plus_marker_cache_upper_bytes"],
    eval_all96_probability_bytes=plans["evaluate"]["all96_original_grid_probability_bytes"],
    output_scope_forecast_bytes={k:v["total_output_scope_upper_bound_bytes"] for k,v in plans.items()},
    source_manifest_ref=cfg["source_manifest_ref"],producer_ref=cfg["producer_ref"],
    actual_guard_roster_and_GPU_availability="root_rechecks_at_real_admission_and_launch")
(HERE/"source_static_receipt_v1.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
