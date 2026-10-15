"""Standard-library C3 native capability/config contracts, with no execution."""
from __future__ import annotations
from pathlib import Path
import hashlib,importlib.util,json,os,sys
from arggnn.american_c3_common_v1 import ROOT,CLASSES,require,local,ref,bound
AUTHOR_ROOT=".runtime/author_sources/argformer_f7545be7bca0"
AUTHOR_MANIFEST_REF=dict(path="research/reviews/American_C3_native_consumer_source_v1/author_source_manifest.json",
    sha256="bed570db438f2c4981ebf606460dc4ff72a97b148ba371e3873ab0141376736a")
NATIVE_GUARD_REF=dict(path="src/arggnn/native_argformer_exec.py",
    sha256="84ed405379ac56511c2f4722c130e8b884226018bbf8e2949ea2bb4dceeb99d0")
import amendment_v7 as amendment
LAUNCHER_REF=ref("research/reviews/American_C3_native_MLM_amendment_v7/launcher_v7.py")
REGISTRY_REF=dict(path="research/reviews/American_C3_short_train2mb_wave_source_v1/parent_registry.inactive.json",
    sha256="c75c111bc48f94d164fe9d6dfe4f8796290b03d1093aca88602049ed47f80e37")
ORIGINAL_REGISTRY_REF=dict(path="research/reviews/ARGformer_American_C3_implementation_20261015_v1/parent_seed_registry_inactive.json",
    sha256="608da56508757646a0584cb653f7f6af4c910f05d560d6da1b99c0c1ea691f9a")
ORIGINAL_H041_REGISTRATION_REF=dict(path="research/reviews/American_C3_native_model_root_v1/H041_original_registration_snapshot.json",
    sha256="724365c9af8b1955ee0e2632e443e568989cb91ed2c8b28653d4034ce858e78c")
ORIGINAL_TRAIN_OBSERVATION_REF=dict(path="experiments/results/American_C3_observed_full30mb_supervisor_repair_20261015_v1/observed/manifest.json",
    sha256="c3d58a8d10d901a162f406e2e218a10e17e5f1617b3c626503d72bcbd1b4ccee")

REL="research/reviews/American_C3_native_MLM_amendment_v7"
SOURCE=REL+"/consumer_contract_v1.py"
MAIN="research/reviews/American_C3_native_MLM_amendment_v7/entry_v7.py"
PREP_REF=dict(path="experiments/results/American_C3_native_paths_train000_full30mb_seed92001_v3/completed.json",sha256="3cc49fbf55bd4b4b3859a93ce1123e9218f407f7c7299ad5029be81f59c7c6fa")
GPU_PHASES=("qualify_mlm","mlm")
CPU_PHASES=()
DEFAULTS=dict(schema="american_c3_native_mlm_consumer_config_v1",execution_enabled=False,
 parent_id="american_c3_train_000",phase=None,author_commit="f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596",
 MLM=dict(epochs=5,batch=512,microbatch=512,precision="amp_bf16",seed=17),
 child_log_ram_bytes=4194304,guard_poll_seconds=1.,termination_grace_seconds=20.,
 phase_write_budget_bytes=None,short_tmp_directory=None,max_checkpoint_files=None,max_checkpoint_bytes=None)
PROJECT_SOURCES=("src/arggnn/american_c3_common_v1.py",MAIN,SOURCE,REL+"/supervisor_v1.py",REL+"/native_child_v1.py",
 REL+"/qualifier_v1.py",REL+"/terminal_bounds_v1.py",REL+"/corpus_auth_v2.py",REL+"/owned_lifecycle_v2.py","research/reviews/American_C3_native_MLM_full_source_v5/pidfd_backend_v1.py","src/arggnn/native_argformer_exec.py","src/arggnn/__init__.py")
def validate_config(cfg,phase):
    require(set(cfg)==set(DEFAULTS) and cfg["execution_enabled"] is True and phase in GPU_PHASES and cfg["phase"]==phase,
        "Explicit root exact MLM-only phase config")
    varying={"execution_enabled","phase","child_log_ram_bytes","guard_poll_seconds","termination_grace_seconds",
        "phase_write_budget_bytes","short_tmp_directory","max_checkpoint_files","max_checkpoint_bytes"}
    require(all(cfg[k]==v for k,v in DEFAULTS.items() if k not in varying),"Pinned full5ep/512/seed17 model/data settings")
    require(cfg["phase_write_budget_bytes"]==12*2**30 and cfg["max_checkpoint_files"]==2
        and cfg["max_checkpoint_bytes"]==2950918528 and 0<cfg["child_log_ram_bytes"]<=16777216
        and 0<float(cfg["guard_poll_seconds"])<=2 and 0<float(cfg["termination_grace_seconds"])<=60,
        "Exact source-based scope/checkpoint and responsive supervision caps")
    no_private_refs(cfg)
def counts_v1(spec,prep):
    require(spec["refs"]["preparation"]==PREP_REF and prep["unique_paths"]==25574587
        and prep["vocabulary_nodes"]==106769 and prep["max_raw_path_nodes"]==29,"Exact completed v3 metadata counts")
    n=int(prep["native_train_fraction"]*prep["unique_paths"])
    require(n==23017128 and 5*(n//512)==224775,"Full release five-epoch update arithmetic")
    return dict(MLM_optimizer_updates=224775,train_rows=n,val_rows=2557459,vocabulary=106776)

VIEWS=("mlm","contrastive","query_paths")
SPLITS=("train","val")
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

def validate_active_input(record,spec,source,*,GPU):
    require(spec.get("schema")=="american_c3_native_active_input_v1"
        and spec.get("root_authorizer")=="root_agent" and spec.get("status")=="active"
        and spec.get("execution_authorized") is True and spec.get("GPU_authorized") is GPU
        and spec.get("main_source_ref")==ref(source) and spec.get("author_source_manifest_ref")==AUTHOR_MANIFEST_REF
        and spec.get("actual_launcher_ref")== (LAUNCHER_REF if GPU else None),
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
    require(closure.get("schema")=="native_MLM_execution_source_closure_v7"
        and closure.get("source_sha256")==spec["source_sha256"]
        and closure.get("original_storage_ref")==amendment.ORIGINAL_STORAGE_REF,
        "Separate actual execution source closure and original immutable allocation")
    cfg=metadata(spec["refs"]["config"]);validate_config(cfg,spec["phase"])
    require(cfg["parent_id"]==spec["parent_id"],"One parent in active input/config")
    storage=metadata(spec["refs"]["storage"])
    require(storage["job_id"]==spec["job_id"] and storage["output_directory"]==spec["output_directory"],
        "Actual existing storage scope/job identity")
    amendment.validate_execution(record,spec,source)
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
    require(spec["refs"]["preparation"]==PREP_REF,"Only completed full v3 corpus")
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
    registry=metadata(REGISTRY_REF)
    registered={r["parent_id"]:r for r in registry["parents"]+registry["deferred_heldout_30Mb_original_records"]}
    # Explicit root-authorized pre-admission roster correction: original
    # registered full TRAIN000 is engineering TRAIN, never a held-out label.
    if spec["parent_id"]=="american_c3_train_000":
        original=metadata(ORIGINAL_REGISTRY_REF);registration=metadata(ORIGINAL_H041_REGISTRATION_REF)
        old=next(r for r in original["parents"] if r["parent_id"]=="american_c3_train_000")
        require(original["schema"]=="american_c3_parent_split_seeds_v1"
            and old==dict(parent_id="american_c3_train_000",split="train",ancestry_seed=21001,
                mutation_seed=1021001,path_sampling_seed=92001)
            and registration["schema"]=="american_c3_original_H041_registration_snapshot_v1"
            and registration["original_parent_registry_ref"]==ORIGINAL_REGISTRY_REF
            and registration["registered_H041"]["id"]=="H041"
            and registration["registered_H041"]["registered_utc"]=="2026-10-14T23:17:57.401114+00:00"
            and prep["observation_ref"]==ORIGINAL_TRAIN_OBSERVATION_REF
            and pcfg["observation_ref"]==ORIGINAL_TRAIN_OBSERVATION_REF,
            "Only genuinely registered original full30Mb TRAIN000/observed manifest; no held-out relabelling")
        registered=dict(registered,american_c3_train_000=old)
    require(spec["parent_id"] in registered
        and pcfg["path_sampling_seed"]==registered[spec["parent_id"]]["path_sampling_seed"]
        and prep["native_split_seed"]==42 and prep["native_train_fraction"]==.9,
        "Authoritative original independent parent/path seed and unchanged release path split")
    preparation_dir=local(Path(spec["refs"]["preparation"]["path"]).parent)
    for key,value in prep["native_dataset_refs"].items():
        view,split=key.split("/")
        require(local(value["path"])==preparation_dir/view/split/("raw_"+split+"_sequences_and_vocab.pkl"),
            "Only exact original raw full-path datasets, never an injected tokenized replacement")
        require(not (preparation_dir/view/split/("tokenized_"+split+"_sequences_and_vocab.pkl")).exists(),
            "No shadow tokenized file may override the admitted raw corpus")
    return prep,pcfg,preparation_dir
