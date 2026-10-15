"""Source/metadata-only H044 integration checks. No project module imports."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
FOLDER=ROOT/"research/reviews/H044_marker_resolution_root_v1"
checks=[]
def require(ok,name):
    if not ok:raise AssertionError(name)
    checks.append(name)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
owned_sources=[
    "src/arggnn/marker_resolution_readout_v1.py",
    "src/arggnn/prepare_marker_resolution_observations_v1.py",
    "src/arggnn/train_marker_resolution_readout_v1.py",
    "research/reviews/H044_marker_resolution_root_v1/context_factory_v1.py",
    "research/reviews/H044_marker_resolution_root_v1/root_driver.py",
    "src/arggnn/prepare_marker_resolution_observations_v2.py",
    "research/reviews/H044_marker_resolution_root_v1/context_factory_v2.py",
    "research/reviews/H044_marker_resolution_root_v1/root_driver_v2.py",
    "src/arggnn/marker_resolution_memory_bounds_v2.py"]
trees={}
for path in owned_sources:
    text=(ROOT/path).read_text();compile(text,path,"exec");trees[path]=ast.parse(text)
require(len(trees)==9,"all nine scientific/preparation/worker/context/driver sources compile")
frozen={
    "src/arggnn/marker_resolution_readout_v1.py":"27ecd30683024def2d408bf16250102b3252d0ae112687b9664fca086b141310",
    "src/arggnn/prepare_marker_resolution_observations_v1.py":"d849533b827f82a44e943f5da9ef2496d1509bcca9f193f202add13fdbdad952",
    "research/reviews/H044_marker_resolution_root_v1/context_factory_v1.py":"ce46a83c529b1751da4e24254712e3bc3a49b5f87f72e97aa0c6bc7bdd9cefed",
    "research/reviews/H044_marker_resolution_root_v1/root_driver.py":"8700c63ba515c5ae2e9d37c5efc1188b8596a1f37a20ca8ff8e507f6c0836318",
    "research/reviews/H044_marker_resolution_root_v1/preparation_source_manifest.json":"a17ff54a14e3d51863911f7d476f680ae82619d2809a0b9870869e529e807df2",
    "src/arggnn/prepare_marker_resolution_observations_v2.py":"b5e188d7961533a892b4dd27683f8e0fe3a0cae36ed9033127545a666d2d2372",
    "research/reviews/H044_marker_resolution_root_v1/context_factory_v2.py":"245f9dcdff47efeacba6a3aa40377375466f9d7abe746b6bd492d021dfa0b2fe",
    "research/reviews/H044_marker_resolution_root_v1/root_driver_v2.py":"8e97d4e605c2c1908c1d841272692f59865f861dc9f4a82ae22945fb9037d55f",
    "research/reviews/H044_marker_resolution_root_v1/preparation_source_manifest_v2.json":"7402d96449ef027eec5ef90cb089000360aa6cf0bc636cb4d7b0b147845959da"}
require(all(digest(ROOT/p)==h for p,h in frozen.items()),"root-frozen preparation/core/driver/context/manifest unchanged")
core_manifest=json.loads((ROOT/"research/reviews/H044_marker_resolution_source_v1/source_manifest.json").read_text())
require(all(digest(ROOT/r["path"])==r["sha256"] for r in core_manifest["unchanged_upstream_source_refs"]),
    "all frozen original scientific source pins unchanged")
manifest=json.loads((FOLDER/"source_manifest.json").read_text())
require(all(digest(ROOT/p)==h for p,h in manifest["source_sha256"].items()),"paired full source closure matches actual source bytes")
for r in manifest["metadata_refs"].values():
    require(digest(ROOT/r["path"])==r["sha256"],"metadata closure "+r["path"])
metadata_names=["preparation_interface.json","preparation_source_manifest.json","prepare_config_inactive.json",
    "prepare_admission_request_inactive.json","preparation_api_checks_receipt.json","source_manifest.json",
    "config_inactive.json","train_admission_request_inactive.json","evaluate_admission_request_inactive.json",
    "preparation_interface_v2.json","preparation_source_manifest_v2.json","prepare_config_inactive_v2.json",
    "prepare_admission_request_inactive_v2.json","preparation_wiring_checks_receipt_v2.json",
    "memory_interface_checks_receipt_v2.json","batched_interface_checks_receipt_v2.json"]
for name in metadata_names:json.loads((FOLDER/name).read_text())
require(True,"all owned informational JSON metadata parse")
cfg=json.loads((FOLDER/"config_inactive.json").read_text())
old=json.loads((ROOT/"research/reviews/H042_GPU45_clade_flags_root_source_v1/config.json").read_text())
require(all(cfg[k]==old[k] for k in ("model","metrics","selection","learning_rate","weight_decay",
    "AdamW_betas","AdamW_epsilon","gradient_clip_norm","seed","dtype","passes","updates_per_pass",
    "updates_per_arm","checkpoint_passes","TRAIN_parent_ids","VAL_parent_ids","observation_cache_source_refs")),
    "original model/optimization/schedule/selection/metrics/source protocol retained")
require(cfg["arms"]==["localized","tree_pooled"] and cfg["parameters_per_arm"]==31428
    and cfg["base_parameters"]==30788 and cfg["added_parameters"]==640
    and cfg["readout"]==dict(query_block=64,center_block=2048,checkpoint_readout=True),
    "exact matched H044 arms/readout/parameter contract")
require(all(cfg[k] is None for k in ("CPU_bank_cap_bytes","CPU_target_cap_bytes","flag_cache_cap_bytes",
    "flag_preparation_cap_bytes","GPU_residency_cap_bytes","GPU_flag_cap_bytes","RAM_prediction_cap_bytes",
    "marker_cache_cap_bytes","marker_preparation_cap_bytes","output_cap_bytes",
    "CPU_coordinate_bulk_cap_bytes","CPU_binding_bulk_cap_bytes")),
    "unknown actual caps remain inactive and unset")
source=(ROOT/"src/arggnn/train_marker_resolution_readout_v1.py").read_text()
worker=trees["src/arggnn/train_marker_resolution_readout_v1.py"]
defs={n.name:n for n in worker.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
segment=lambda name:ast.get_source_segment(source,defs[name])
train=segment("run_train_v1")
require(train.index("load_observations_v1(")<train.index("reader.truth_gate=True")<train.index("original_mass_v1("),
    "complete observation recipes precede TRAIN truth")
require(train.count("original_mass_v1(")==1 and train.index("mass_cpu=original_mass_v1(")<train.index("for arm in ARMS:"),
    "one complete target construction shared between paired arm updates")
require("native_hold_train_mass" not in source and "core.original_grid_span_nll_v1(logits,mass," in source,
    "original100bp physical target mass and grid NLL retained")
require("models,initialization=core.make_matched_pair_v1(" in train
    and "len(o.state)==0" in train and "models[a].parameters()" in train,
    "matched complete state and independent reset AdamW over all parameters")
evaluate=segment("run_evaluate_v1")
require(evaluate.index("ram.verify()")<evaluate.index("reader.truth_gate=True")<evaluate.index("validation.score("),
    "all96 dense products verified before any VAL score/target gate")
require("validation=FullPanelPrimaryValidationV1(" in evaluate
    and "h038.selection_key_v1(" in evaluate and "validation.full_final(" in evaluate,
    "original eight-VAL primary selection and per-arm selected diagnostics")
dense=segment("DenseRAMMatrixV1")
require('(len(parent["positions"]),686,4)' in dense and "return value" in dense
    and "return value[" not in dense,"dense original grid returned directly without hold compression")
require('self.bytes+4*len(parent["positions"])*686*4<=' in dense and
    '12*len(parent["positions"])*686*4<=' in segment("original_mass_v1"),
    "target and prediction allocations have count-derived preflight caps")
require("all96_RAM_seals_before_any_VAL_truth=True" in evaluate and "outer_truth_read=False" in evaluate,
    "completion carries product barrier and outer-unread scope")
driver_path="research/reviews/H044_marker_resolution_root_v1/root_driver.py"
driver=(ROOT/driver_path).read_text()
ddefs={n.name:n for n in trees[driver_path].body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
launch=ast.get_source_segment(driver,ddefs["launch"])
require('storage["initial_ledger_pins"]=' not in launch and
    'reservation=receipt["value"]' in driver and 'initial_ledger_pins=receipt["initial_ledger_pins"]' in driver,
    "original reserve value and initial pins preserved without launch rebasing")
require(launch.index("rc=proc.wait()")<launch.index("log_ref=save(")
    and launch.index('require(drained,')<launch.index("log_ref=save(")
    and 'rglob(' not in driver,"logs durable only after direct wait and owned-group drain without live scans")
require("memo.install(" not in driver and "memo.install(" in (FOLDER/"context_factory_v1.py").read_text(),
    "memo2 confined to consumer context")
for original,names in {
    "research/reviews/American_C3_native_model_root_v1/root_native_driver_v1.py":
        ("_stat","_tree","_signal_owned","_absent_group"),
    "src/arggnn/american_c3_native_run_v1.py":("_RAMLog","_subreaper","_children")}.items():
    text=(ROOT/original).read_text()
    original_defs={n.name:n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    require(all(ast.get_source_segment(driver,ddefs[n])==ast.get_source_segment(text,original_defs[n]) for n in names),
        "owned-process helpers copied unchanged from "+original)
prep_source=(ROOT/"src/arggnn/prepare_marker_resolution_observations_v2.py").read_text()
prep_tree=trees["src/arggnn/prepare_marker_resolution_observations_v2.py"]
prep_fn=next(n for n in prep_tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_context_v1")
prep_run=ast.get_source_segment(prep_source,prep_fn)
require(prep_run.index("read_full_cache_index_v1(")<prep_run.index('rows=selected_rows_v1(ctx["registry"])'),
    "actual v2 run_context wires original48 cache validation before40 local selection")
receipt=json.loads((FOLDER/"preparation_wiring_checks_receipt_v2.json").read_text())
require(receipt["status"]=="passed" and receipt["events"]==[["original_cache_validation",48],["selected_development_rows",40]]
    and receipt["prepared_parents"]==40 and receipt["original_IO_and_strict952_wrapper_mocked"] is True,
    "bounded actual run_context wiring qualification receipt is scoped and complete")
require('self.reader.register_packet_v1(key,packet)' in dense and
    dense.count('self.ctx["output"].save(')==1,"96 prediction packets use one unchanged original writer commit")
require('return super().read(record,**kwargs)' in segment("H044PhaseReaderV1")
    and 'record==self.packet_refs[path]' in segment("H044PhaseReaderV1"),
    "virtual reader intercepts exact known packet bytes and inherits all other phase gates")
require('coordinates_ref=preparation["coordinates_ref"]' in segment("MarkerCacheV1")
    and 'entry["coordinates_member_selector"]' in segment("MarkerCacheV1"),
    "bulk coordinates and bindings decoded once with exact parent/member selectors")
require('history.append(row);print(' in train and 'output.save("TRAIN_pass_' not in train
    and '.primary_metrics.json' not in evaluate and '.selected_full_diagnostics.json' not in evaluate,
    "optional pass and metric receipts consolidated without changing checkpoints or science")
require('memory_bounds.logical_bounds_v2(' in segment("MarkerCacheV1")
    and 'logical_bounds_v2(sites,holds,centers' in prep_run
    and cfg["H042_exported_state_channels"]==96
    and cfg["head_checkpoint_selected_state_and_feature_upper_bytes"]==6585600000,
    "corrected96-channel resource helper used by preparation and worker")
batch=json.loads((FOLDER/"batched_interface_checks_receipt_v2.json").read_text())
require(batch["status"]=="passed" and batch["matrix_writer_calls"]==1 and batch["products"]==96
    and batch["original_target_class_executed"] is True and batch["synthetic_target_archive_only"] is True,
    "bounded original-target/virtual-packet compatibility qualification remains scoped")
result=dict(schema="H044_marker_static_integration_checks_v1",status="passed",checks=checks,
    source_refs={p:dict(path=p,sha256=digest(ROOT/p)) for p in owned_sources},
    project_modules_imported=False,actual_observations_targets_checkpoints_runtime_guards_GPU_queried=False,
    native_model_constructed=False,execution_admission_or_launch=False)
print(json.dumps(result,sort_keys=True))
