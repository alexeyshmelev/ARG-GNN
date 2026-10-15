"""Source/pure-metadata H046 integration checks; no guard/data/device actions."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]
assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
from arggnn import train_continuous_clade_flank_v1 as worker
assert "torch" not in worker.__dict__
checks=["worker_import_inert"]
def check(name,ok):
    if not bool(ok):raise AssertionError(name)
    checks.append(name)
def rejects(name,fn):
    try:fn()
    except (ValueError,TypeError,KeyError):checks.append(name)
    else:raise AssertionError("Expected rejection: "+name)
def require(ok,message):
    if not ok:raise ValueError(message)
def extract(path,names,namespace):
    tree=ast.parse(path.read_text(),str(path))
    body=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names]
    assert len(body)==len(names)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),"exec"),namespace)
cfg=json.loads((HERE/"config_inactive.json").read_text())
worker.validate_protocol(cfg)
check("inactive_config_exact_scientific_protocol",cfg["status"]=="source_only_inactive_awaiting_root_review"
    and cfg["root_resource_caps_are_proposals"] is True)
for name in ("root_common_v1.py","context_factory_v1.py","root_driver.py","worker_cli.py"):
    compile((HERE/name).read_text(),str(HERE/name),"exec")
    checks.append("syntax_"+name)
namespace=dict(require=require,OUTPUT_CAP=100000000,LOG_CAP=4000000,RSS_CAP=512000000000,
    current_RSS_bytes_v1=lambda:0)
extract(HERE/"root_common_v1.py",{"count_plan_v1","preparation_check_v1","validate_train_lineage_v1"},namespace)
ids=worker.TRAIN+worker.VAL
counts=dict(schema="root_H043_original40_observed_count_manifest_v1",
    counts_only_no_labels_or_model_metrics=True,outer_included=False,
    parents={p:dict(native_holds=3,native_nodes=10000,haplotypes=952) for p in ids})
preparation=dict(parents={p:dict(counts=dict(sites=12,native_holds=3,haplotypes=952,
    queries=686,scoring_centers=20000)) for p in ids})
plans={phase:namespace["count_plan_v1"](counts,phase,preparation) for phase in ("train","evaluate")}
check("count_derived_phase_output_plans_below100MB",
    all(p["total_output_scope_upper_bound_bytes"]<=100000000 for p in plans.values()))
check("complete96_dense_grid_prediction_bound_correct",
    plans["evaluate"]["all96_original_grid_probability_bytes"]==21073920000
    and plans["evaluate"]["all96_original_grid_probability_bytes"]<cfg["RAM_prediction_cap_bytes"])
check("retained_CPU_local_features_explicit",
    plans["train"]["parents"][ids[0]]["CPU_checkpoint_local_feature_upper_bound_bytes"]==548800000)
bad=copy.deepcopy(preparation);bad["parents"][ids[0]]["counts"]["scoring_centers"]=20001
rejects("out_of_grid_scope_rejected",lambda:namespace["count_plan_v1"](counts,"train",bad))
worker.science()
import torch
torch.set_num_threads(1)
bank=SimpleNamespace(genotype=torch.zeros((12,952),dtype=torch.int8),
    sample_index=torch.zeros((3,952),dtype=torch.long),parent=torch.zeros(10000,dtype=torch.long))
declared=worker.flank_upper_bounds_v1(bank,20000)
proof=namespace["preparation_check_v1"](ids[0],bank,declared,cfg,counts,preparation,rss_reader=lambda:1000000)
check("source_prebuild_uses_actual_S_T_N_P_and_injected_RSS",
    proof["raw_bound_bytes"]==24*10000+4*3+40*20000 and proof["full_logits_materialized"])
rejects("source_prebuild_RSS_overflow_rejected",lambda:namespace["preparation_check_v1"](
    ids[0],bank,declared,cfg,counts,preparation,rss_reader=lambda:512000000000))
wrong=dict(declared,native_nodes=1)
rejects("source_prebuild_misdeclared_counts_rejected",lambda:namespace["preparation_check_v1"](
    ids[0],bank,wrong,cfg,counts,preparation,rss_reader=lambda:0))
def ref(name):return dict(path="synthetic/"+name,sha256="0"*64)
prep_ref,wait_ref=ref("preparation.json"),ref("prep_wait.json")
spec=dict(phase="train",preparation_completion_ref=prep_ref,preparation_actual_wait_ref=wait_ref,
    refs=dict(observation_cache=ref("cache.json")),observation_entry_refs={p:ref(p+".json") for p in ids},
    observation_cache_source_refs={"synthetic/source.py":ref("source.py")})
from arggnn.prepare_marker_resolution_observations_v1 import COORDINATE_KEYS
prep=dict(preparation,schema="H044_marker_observation_preparation_v1",
    status="completed_all40_observation_only_sidecars",hypothesis_id="H044",
    coordinate_keys=list(COORDINATE_KEYS),artifact_format="H044_all40_batched_coordinate_binding_v2",
    observation_cache_ref=spec["refs"]["observation_cache"],entry_refs=spec["observation_entry_refs"],
    observation_source_refs=spec["observation_cache_source_refs"],observations_only=True,targets_read=False)
wait=dict(actual_direct_launcher_wait=True,returncode=0,completed_ref=prep_ref,owned_group_drained=True)
records={prep_ref["path"]:prep,wait_ref["path"]:wait}
reader=SimpleNamespace(decode_json=lambda r:records[r["path"]])
ns=dict(require=require)
extract(HERE/"context_factory_v1.py",{"completed_preparation_v1"},ns)
check("immutable_original_all40_preparation_direct_wait_accepted",
    ns["completed_preparation_v1"](reader,spec,cfg) is prep)
wait["returncode"]=1
rejects("preparation_failed_direct_wait_rejected",lambda:ns["completed_preparation_v1"](reader,spec,cfg))
wait["returncode"]=0;prep["targets_read"]=True
rejects("preparation_truth_read_rejected",lambda:ns["completed_preparation_v1"](reader,spec,cfg))
prep["targets_read"]=False
completion_ref,launch_ref,train_wait_ref,input_ref=map(ref,("train.json","launch.json","wait.json","input.json"))
train=dict(schema="H046_paired_train_completion_v1",status="completed_fixed30pass_paired_TRAIN",
    hypothesis_id="H046",arms=list(worker.ARMS),steps={a:960 for a in worker.ARMS},
    VAL_truth_read=False,outer_truth_read=False,same_VAL_is_development_evidence=True,input_ref=input_ref)
launch=dict(launcher_pid=123,phase="train",physical_gpu=4,input_ref=input_ref)
train_wait=dict(actual_direct_launcher_wait=True,returncode=0,completed_ref=completion_ref,
    launcher_pid=123,phase="train",physical_gpu=4,logging_error=None,supervisor_error=None)
records.update({completion_ref["path"]:train,launch_ref["path"]:launch,train_wait_ref["path"]:train_wait})
evalspec=dict(phase="evaluate",train_completion_ref=completion_ref,train_launch_ref=launch_ref,
    train_actual_wait_ref=train_wait_ref)
namespace["validate_train_lineage_v1"](reader,evalspec);checks.append("exact_actual_same_launcher_train_lineage")
train_wait["launcher_pid"]=124
rejects("different_launcher_wait_rejected",lambda:namespace["validate_train_lineage_v1"](reader,evalspec))
driver=(HERE/"root_driver.py").read_text()
check("root_freeze_precedes_admission_mutations",
    driver.index('cfg.get("status")=="root_frozen_protocol_before_admission"')<driver.index("packet.mkdir()"))
check("reservation_uses_unmemoized_path",driver.index("load_guard_v1(consumer=False)")<driver.index("apply_reserve_v1"))
check("original_reservation_immutable_pins_preserved",
    'initial_ledger_pins=receipt["initial_ledger_pins"]' in driver)
check("phase_GPU_choice_explicit_and_paired_one_GPU","gpu in (4,5)" in driver
    and 'CLI_MODULE="research.reviews.H046_GPU45_flank_root_source_v1.worker_cli"' in driver)
check("completed_v3_guard_memo_originalv4_pins_unchanged",
    cfg["helper_refs"]["full_reservation_adapter"]["sha256"]=="ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef"
    and cfg["transaction_memo_adapter_ref"]["sha256"]=="c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364"
    and cfg["original_v4_ref"]["sha256"]=="a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47")
check("no_admitted_capability_created",not any(HERE.glob("admitted*")))
receipt=dict(schema="H046_inactive_root_packet_static_checks_v1",status="passed",
    checks=checks,check_count=len(checks),source_only=True,synthetic_metadata_only=True,
    guard_operations=False,CUDA_operations=False,actual_count_corpus_or_data_reads=False,
    source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
        for name in ("root_common_v1.py","context_factory_v1.py","root_driver.py","worker_cli.py")},
    synthetic_max_grid_plans=plans,check_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
print(json.dumps(dict(status=receipt["status"],check_count=len(checks),source_sha256=receipt["source_sha256"]),indent=2))
if "--write-receipt" in sys.argv:
    (HERE/"static_checks_receipt.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+chr(10))
