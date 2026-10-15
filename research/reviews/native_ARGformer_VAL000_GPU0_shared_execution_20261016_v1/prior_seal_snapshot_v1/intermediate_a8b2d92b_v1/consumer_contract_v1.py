"""Standard-library C3 native capability/config contracts, with no execution."""
from __future__ import annotations
from pathlib import Path
import hashlib,importlib.util,json,os,sys
from arggnn.american_c3_common_v1 import ROOT,CLASSES,require,local,ref,bound
REL='research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1'
SOURCE=REL+"/consumer_contract_v1.py"
MAIN=REL+"/entry_postMLM_v1.py"
AUTHOR_ROOT=".runtime/author_sources/argformer_f7545be7bca0"
AUTHOR_MANIFEST_REF=dict(path="research/reviews/American_C3_native_consumer_source_v1/author_source_manifest.json",
    sha256="bed570db438f2c4981ebf606460dc4ff72a97b148ba371e3873ab0141376736a")
NATIVE_GUARD_REF=dict(path='research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_native_guard_v1.py',
    sha256='3a08d3bd84adc95aafd332bc224fdb7aa38730d6c2f48dee8108944251b7ee69')
LAUNCHER_REF=dict(path=REL+"/launcher_postMLM980_v1.py",sha256=None)

PROJECT_SOURCES=("src/arggnn/american_c3_common_v1.py",SOURCE,
    "src/arggnn/american_c3_native_run_v1.py","src/arggnn/american_c3_native_child_v1.py",
    "src/arggnn/american_c3_native_concat_v1.py","src/arggnn/american_c3_paths_v1.py",
    "src/arggnn/american_c3_evaluate_v1.py",'research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_native_guard_v1.py',
    "src/arggnn/__init__.py")
REGISTRY_REF=dict(path="research/reviews/American_C3_short_train2mb_wave_source_v1/parent_registry.inactive.json",
    sha256="c75c111bc48f94d164fe9d6dfe4f8796290b03d1093aca88602049ed47f80e37")
ORIGINAL_REGISTRY_REF=dict(path="research/reviews/ARGformer_American_C3_implementation_20261015_v1/parent_seed_registry_inactive.json",
    sha256="608da56508757646a0584cb653f7f6af4c910f05d560d6da1b99c0c1ea691f9a")
ORIGINAL_H041_REGISTRATION_REF=dict(path="research/reviews/American_C3_native_model_root_v1/H041_original_registration_snapshot.json",
    sha256="724365c9af8b1955ee0e2632e443e568989cb91ed2c8b28653d4034ce858e78c")
ORIGINAL_TRAIN_OBSERVATION_REF=dict(path="experiments/results/American_C3_observed_full30mb_supervisor_repair_20261015_v1/observed/manifest.json",
    sha256="c3d58a8d10d901a162f406e2e218a10e17e5f1617b3c626503d72bcbd1b4ccee")
VIEWS=("mlm","contrastive","query_paths")
SPLITS=("train","val")
GPU_PHASES=("fit","embeddings")
CPU_PHASES=()
DEFAULTS=dict(schema="american_c3_native_consumer_config_v1",execution_enabled=False,
    parent_id=None,phase=None,author_commit="f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596",
    class_order=list(CLASSES),regime="fresh_per_ARG_transductive_MLM_then_known_reference_contrastive",
    training_config_plumbing_correction=False,uniform_class_weight_fallback_preserved=True,
    MLM=dict(epochs=5,batch=512,microbatch=512,precision="amp_bf16",seed=17),
    contrastive=dict(epochs=1,batch=4096,microbatch=4096,precision="fp32",seed=36,temperature=.30),
    embeddings=dict(batch=256,max_samples=0,pooling="head_norm",layer=None,include_splits_one_at_a_time=True),
    retrieval=dict(topk=1,q_batch=1024,e_chunk=100000),
    child_log_ram_bytes=4194304,guard_poll_seconds=1.,termination_grace_seconds=20.,
    phase_write_budget_bytes=None,short_tmp_directory=None,max_preparation_file_bytes=None,
    max_checkpoint_files=None,max_checkpoint_bytes=None,max_embedding_input_bytes=None)
PROJECT_SOURCES=tuple(dict.fromkeys((*PROJECT_SOURCES,'research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/shared_native_runtime_entry_v1.py',"src/arggnn/american_c3_native_contract_v1.py",MAIN,SOURCE,REL+"/supervisor_v1.py",REL+"/native_child_v1.py",'research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/post_mlm_bridge_v1.py','research/reviews/C3_native_post_MLM_root_handoff_20261015_v1/ram_downstream_v1.py')))
DEFAULTS["checkpoint_retention_IO_override"]=1

def metadata(record,*,cap=1048576):
    p=bound(record);require(p.stat().st_size<=cap,"Bounded SHA-bound JSON metadata")
    return json.loads(p.read_text())
def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def no_private_refs(value):
    if isinstance(value,dict):
        for k,v in value.items():
            require(not any(x in k.lower() for x in ("truth_ref","truth_npz","oracle_ref")),
                "Native consumer accepts observation/reference inputs only")
            if k=="path":
                require(isinstance(v,str) and not any("truth" in x.lower() or "oracle" in x.lower() for x in Path(v).parts),
                    "Private paths forbidden in native capability")
            no_private_refs(v)
    elif isinstance(value,(list,tuple)):
        for v in value:no_private_refs(v)
def source_manifest_v1(*,verify_bytes=True):
    m=metadata(AUTHOR_MANIFEST_REF)
    require(m["schema"]=="american_c3_pinned_author_source_closure_v1"
        and m["author_commit"]==DEFAULTS["author_commit"] and m["author_root"]==AUTHOR_ROOT,
        "Exact pinned unchanged author source closure")
    require(len({r["path"] for r in m["refs"]})==len(m["refs"])
        and all(r["path"].startswith(AUTHOR_ROOT+"/") for r in m["refs"]),"Closed author source paths")
    if verify_bytes:
        for record in m["refs"]:bound(record)
    return m
def validate_config(cfg,phase):
    require(set(cfg)==set(DEFAULTS) and cfg["execution_enabled"] is True and cfg["phase"]==phase
        and phase in GPU_PHASES+CPU_PHASES,"Explicit root-activated exact native phase configuration")
    varying={"execution_enabled","parent_id","phase","child_log_ram_bytes","guard_poll_seconds",
        "termination_grace_seconds","phase_write_budget_bytes","short_tmp_directory",
        "max_preparation_file_bytes","max_checkpoint_files","max_checkpoint_bytes","max_embedding_input_bytes"}
    require(all(cfg[k]==v for k,v in DEFAULTS.items() if k not in varying),
        "Pinned native numerical settings; no corpus/batch/weighting/architecture fallback")
    require(isinstance(cfg["parent_id"],str) and cfg["parent_id"],"One explicit independent ARG parent")
    require(type(cfg["phase_write_budget_bytes"]) is int and cfg["phase_write_budget_bytes"]>0,
        "Root must bound/admit complete actual phase growth")
    for key in ("max_preparation_file_bytes","max_checkpoint_files","max_checkpoint_bytes","max_embedding_input_bytes"):
        require(type(cfg[key]) is int and cfg[key]>0,"Root must fill positive actual input/serialization bounds")
    require(type(cfg["child_log_ram_bytes"]) is int and 0<cfg["child_log_ram_bytes"]<=16777216
        and 0<float(cfg["guard_poll_seconds"])<=2 and 0<float(cfg["termination_grace_seconds"])<=60,
        "Bounded RAM logs and responsive native supervision")
    no_private_refs(cfg)
def validate_active_input(record,spec,source,*,GPU):
    require(spec.get("schema")=="american_c3_native_active_input_v1"
        and spec.get("root_authorizer")=="root_agent" and spec.get("status")=="active"
        and spec.get("execution_authorized") is True and spec.get("GPU_authorized") is GPU
        and spec.get("main_source_ref")==ref(source) and spec.get("author_source_manifest_ref")==AUTHOR_MANIFEST_REF
        and spec.get("actual_launcher_ref")== (ref(LAUNCHER_REF["path"]) if GPU else None),
        "Exact active root canonical source/phase/launcher capability")
    no_private_refs(spec)
    require(Path.cwd().resolve()==ROOT,"Canonical repository execution directory")
    for path in PROJECT_SOURCES:
        require(spec["source_sha256"].get(path)==ref(path)["sha256"],"Complete actual project consumer source closure")
    require(spec["root_context_module_ref"]["path"] in spec["source_sha256"]
        and spec["source_sha256"][spec["root_context_module_ref"]["path"]]==spec["root_context_module_ref"]["sha256"],
        "Actual root callback source is in admitted source closure")
    for path,sha in spec["source_sha256"].items():bound(dict(path=path,sha256=sha))
    closure=metadata(spec["transitive_source_closure_ref"])
    require(closure.get("source_sha256")==spec["source_sha256"]
        and closure.get("author_source_manifest_ref")==AUTHOR_MANIFEST_REF
        and closure.get("scientific_input_refs")=={k:v for k,v in spec["refs"].items() if k!="storage"},
        "Transitive actual source closure metadata must be reserved")
    cfg=metadata(spec["refs"]["config"]);validate_config(cfg,spec["phase"])
    require(cfg["parent_id"]==spec["parent_id"],"One parent in active input/config")
    storage=metadata(spec["refs"]["storage"])
    require(storage["job_id"]==spec["job_id"] and storage["output_directory"]==spec["output_directory"],
        "Actual existing storage scope/job identity")
    needed=[ref(p) for p in PROJECT_SOURCES]+[record,AUTHOR_MANIFEST_REF,ORIGINAL_REGISTRY_REF,ORIGINAL_H041_REGISTRATION_REF,
        spec["refs"]["config"],spec["refs"]["runtime"],spec["root_context_module_ref"],
        spec["transitive_source_closure_ref"]]
    # The active input may be transitively bound to avoid its storage-ref cycle.
    bindings=storage["reservation"]["consumer_bindings"]
    require(all(r in bindings for r in needed if r!=record)
        and spec["input_binding_mode"]=="actual_SHA_input_through_root_context",
        "All source/config/runtime/callback/transitive metadata reserved by existing framework")
    source_manifest_v1()
    return cfg,storage
def load_root_context(record,spec,*,native_child=False,CPU=False):
    context_ref=spec["root_context_module_ref"];path=bound(context_ref)
    loaded=importlib.util.spec_from_file_location("_american_c3_actual_root_context",path)
    module=importlib.util.module_from_spec(loaded);loaded.loader.exec_module(module)
    name="build_native_child_context_v1" if native_child else ("build_cpu_context_v1" if CPU else "build_context_v1")
    ctx=getattr(module,name)(record,spec)
    require(ctx.get("root_authorizer")=="root_agent" and ctx.get("admitted") is True
        and ctx.get("authority_validated_by_existing_guard") is True
        and ctx.get("upstream_inputs_complete_and_drained") is True
        and callable(ctx.get("fast_guard")) and ctx.get("fast_guard_scans_or_writes_scope") is False,
        "Root must inject existing fast title/lease/resource authority, with no active-scope scan/write")
    if not native_child:
        require(callable(ctx.get("boundary_guard")) and callable(getattr(ctx.get("writer"),"save",None)),
            "Existing admitted output writer and quiescent boundary guard required")
        require(str(local(ctx["writer"].path))==str(local(spec["output_directory"])),
            "Writer owns exact admitted phase output")
    return ctx
def preparation_v1(spec):
    prep=metadata(spec["refs"]["preparation"]);pcfg=metadata(spec["refs"]["preparation_config"])
    require(prep.get("status")=="completed_native_C3_path_preparation"
        and prep["parent_id"]==spec["parent_id"] and prep["class_order"]==list(CLASSES)
        and prep["contrastive_labels_reference_only"] is True
        and prep["query_embedding_view_prespecified_focals_only"] is True
        and prep["query_ancestry_truth_opened"] is False and prep["query_focal_count"]==10000
        and prep["observation_ref"]==pcfg["observation_ref"] and pcfg["query_focal_count"]==10000,
        "Exact C3 observation/full-path/focal preparation and actual input config")
    require(pcfg["schema"]=="american_c3_native_path_preparation_v1" and pcfg["execution_enabled"] is True
        and set(prep["native_dataset_refs"])=={v+"/"+s for v in VIEWS for s in SPLITS}
        and set(prep["full_path_id_order_refs"])=={v+"/"+s for v in VIEWS for s in SPLITS},
        "Complete three views x two native splits")
    original=metadata(ORIGINAL_REGISTRY_REF);registration=metadata(ORIGINAL_H041_REGISTRATION_REF)
    registered={r["parent_id"]:r for r in original["parents"]}
    expected=dict(parent_id="american_c3_val_000",split="val",ancestry_seed=31001,
        mutation_seed=1031001,path_sampling_seed=93001)
    require(original["schema"]=="american_c3_parent_split_seeds_v1"
        and registered.get("american_c3_val_000")==expected
        and registration["schema"]=="american_c3_original_H041_registration_snapshot_v1"
        and registration["original_parent_registry_ref"]==ORIGINAL_REGISTRY_REF
        and registration["registered_H041"]["id"]=="H041"
        and registration["registered_H041"]["registered_utc"]=="2026-10-14T23:17:57.401114+00:00"
        and spec["parent_id"]=="american_c3_val_000" and spec["split"]=="val"
        and spec["engineering_TRAIN_only"] is False and spec["heldout_parent_label_preserved"] is True
        and pcfg["path_sampling_seed"]==expected["path_sampling_seed"]
        and prep["native_split_seed"]==42 and prep["native_train_fraction"]==.9,
        "Original registered VAL000 identity/held-out label and unchanged release path split")
    preparation_dir=local(Path(spec["refs"]["preparation"]["path"]).parent)
    for key,value in prep["native_dataset_refs"].items():
        view,split=key.split("/")
        require(local(value["path"])==preparation_dir/view/split/("raw_"+split+"_sequences_and_vocab.pkl"),
            "Only exact original raw full-path datasets, never an injected tokenized replacement")
        require(not (preparation_dir/view/split/("tokenized_"+split+"_sequences_and_vocab.pkl")).exists(),
            "No shadow tokenized file may override the admitted raw corpus")
    return prep,pcfg,preparation_dir
def counts_v1(spec,prep):
    counts=metadata(spec["refs"]["counts"])
    require(counts.get("schema")=="american_c3_native_actual_counts_v1"
        and counts.get("status")=="completed_C3_native_observation_only_counts"
        and counts.get("source_ref")==ref("src/arggnn/american_c3_native_concat_v1.py")
        and counts.get("preparation_ref")==spec["refs"]["preparation"]
        and counts.get("preparation_config_ref")==spec["refs"]["preparation_config"]
        and counts.get("parent_id")==prep["parent_id"] and counts.get("class_order")==list(CLASSES)
        and counts.get("native_training_eligible") is True and counts.get("haplotypes")==1600
        and counts.get("query_focal_count")==10000 and counts.get("query_ancestry_truth_opened") is False
        and set(counts["datasets"])==set(prep["native_dataset_refs"]),
        "Actual source-bound complete per-parent counts before GPU fit")
    for key,item in counts["datasets"].items():
        require(item["dataset_ref"]==prep["native_dataset_refs"][key]
            and item["row_order_ref"]==prep["full_path_id_order_refs"][key] and item["rows"]>0,
            "Actual row counts bound to original complete raw paths and full-ID sidecars")
        bound(item["dataset_ref"]);bound(item["row_order_ref"])
    require(counts["MLM_optimizer_updates"]==5*(counts["datasets"]["mlm/train"]["rows"]//512)>0
        and counts["contrastive_optimizer_updates"]==counts["datasets"]["contrastive/train"]["rows"]//4096>0
        and sum(counts["datasets"]["query_paths/"+s]["rows"] for s in SPLITS)==10000,
        "Exact release drop_last updates and all common query rows")
    for split in SPLITS:bound(counts["reference_label_refs"][split])
    return counts
def fit_completion_v1(record,spec):
    value=metadata(record)
    require(value.get("status")=="completed_C3_fresh_native_fit"
        and value["parent_id"]==spec["parent_id"] and value["preparation_ref"]==spec["refs"]["preparation"]
        and value["counts_ref"]==spec["refs"]["counts"] and value["author_source_manifest_ref"]==AUTHOR_MANIFEST_REF
        and value["class_order"]==list(CLASSES) and value["uniform_class_weight_fallback_preserved"] is True
        and value["training_config_plumbing_correction"] is False and value["query_ancestry_truth_opened"] is False,
        "Fresh completed same-ARG native pipeline, no cross-parent model reuse")
    for mode,epochs in (("mlm",5),("contrastive",1)):
        phase=value["phases"][mode];receipt=metadata(phase["child_completion_ref"]);wait=metadata(phase["actual_wait_ref"])
        require(wait["returncode"]==0 and wait["actual_direct_child_wait"] is True and wait["process_group_ESRCH"] is True
            and wait["child_completion_ref"]==phase["child_completion_ref"]
            and receipt["completed_epochs"]==epochs and receipt["stage"]==mode
            and receipt["checkpoint_ref"]==phase["checkpoint_ref"] and receipt["parent_id"]==spec["parent_id"],
            "Actual child wait0/drain and completed final Composer checkpoint")
        bound(phase["checkpoint_ref"])
    mlm=metadata(value["phases"]["mlm"]["child_completion_ref"])
    contrastive=metadata(value["phases"]["contrastive"]["child_completion_ref"])
    require(mlm["initializer_checkpoint_ref"] is None
        and contrastive["initializer_checkpoint_ref"]==value["phases"]["mlm"]["checkpoint_ref"],
        "Fresh per-ARG MLM followed by exact same-ARG MLM initialization")
    counts=counts_v1(spec,metadata(spec["refs"]["preparation"]))
    require(mlm["actual_optimizer_updates"]==counts["MLM_optimizer_updates"]
        and contrastive["actual_optimizer_updates"]==counts["contrastive_optimizer_updates"],
        "Exact actual release updates in both native fit phases")
    return value
