"""Bind a completed fresh observed-tree development cohort, never generate it."""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import itertools
import json
from pathlib import Path
import re

from .resources import ROOT
from .development_execution import DETERMINISTIC_POLICY

SPLITS = {"train":64,"val":16,"test":0}
SEEDS = [101,202,303]
ADAPTER = {"reference_columns":[0,1,2,3],"time_column":6,"branch_column":7,
           "time_scale":10000,"kernel_scales":[.05,.1,.2]}
PATHS = {"oracle":"data/ancient_development64_v1/manifest.json",
    "inferred":"data/ancient_inferred_development64_v1/manifest.json",
    "boundary":"data/boundary_inferred_development64_v1/manifest.json",
    "age":"data/age_inferred_development64_v1/manifest.json"}
COMPLETION = "experiments/results/fresh_development_preparation_v1/completion.json"
PREPARATION_FREEZE = "configs/fresh_development_execution_freeze_v1.json"
PLAN = "configs/argmix_development_search_plan_v2.json"
CORE = ("development_execution.py","argmix_development_bindings.py","train_argmix_development.py","argformer_control.py",
        "study_models.py","models.py","data.py","train_study.py","simulate.py",
        "reference_encoder.py","genealogy_baselines.py","train.py","metrics.py",
        "confirmatory_metrics.py","resources.py")
SELECTION = "minimum equal-parent mean validation float64 uncalibrated raw-logit NLL; earliest epoch on exact tie"
STATUS = "independent_paper_equation_common_input_adaptation"


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def valid_sha(value):
    if not isinstance(value,str) or re.fullmatch(r"[0-9a-f]{64}",value) is None or value=="0"*64:
        raise ValueError("A factual nonzero lowercase SHA256 is required")
    return value


def safe_id(value):
    if not isinstance(value,str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}",value) is None:
        raise ValueError("Unsafe ID: require a bounded ASCII filename component")
    return value


def development_path(value):
    """Confine all paths before opening them, and block any held-out namespace."""
    if not isinstance(value,(str,Path)) or not str(value) or "\\" in str(value):
        raise ValueError("Invalid development path")
    path=Path(value)
    if ".." in path.parts:
        raise ValueError("Traversal is forbidden in development paths")
    root=ROOT.resolve()
    path=(path if path.is_absolute() else root/path).resolve()
    if path==root or root not in path.parents:
        raise ValueError("Path escapes project or names its root")
    components=path.relative_to(root).parts
    if any(any(token in component.lower() for token in ("confirmatory","final_test","final-test"))
           or component.lower()=="test" for component in components):
        raise ValueError("Held-out test paths are forbidden during development")
    return path


def implementation_paths():
    return {"src/arggnn/"+name:Path(__file__).with_name(name) for name in CORE} | {
        "scripts/bind_argmix_development.py":Path(__file__).resolve().parents[2]/"scripts/bind_argmix_development.py"}


def implementation_hashes():
    return {name:sha(path) for name,path in implementation_paths().items()}


def signature(path):
    stat=Path(path).stat()
    return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)


@dataclass
class IntegritySnapshot:
    """Exact initial bytes for controls, full hashes plus stat guards for inputs."""
    controls: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    source_signatures: dict = field(default_factory=dict)
    implementation_sha256: dict = field(default_factory=implementation_hashes)

    def control(self,path,expected):
        path=development_path(path);valid_sha(expected)
        content=path.read_bytes()
        if hashlib.sha256(content).hexdigest()!=expected:
            raise RuntimeError("Control SHA256 mismatch: "+str(path))
        if str(path) in self.controls and self.controls[str(path)]!=(expected,content):
            raise RuntimeError("Conflicting frozen control bindings")
        self.controls[str(path)]=(expected,content)
        return json.loads(content)

    def add_source(self,path,expected):
        path=development_path(path);valid_sha(expected)
        if str(path) in self.sources and self.sources[str(path)]!=expected:
            raise RuntimeError("Conflicting frozen source bindings")
        self.sources[str(path)]=expected
        self.source_signatures[str(path)]=signature(path)

    def check(self,full_sources=False):
        try:
            if implementation_hashes()!=self.implementation_sha256:
                raise RuntimeError("Frozen implementation/dependency drift")
            for path,(expected,_) in self.controls.items():
                if sha(path)!=expected:raise RuntimeError("Frozen control/config/manifest drift: "+path)
            for path,expected in self.sources.items():
                if signature(path)!=self.source_signatures[path]:
                    raise RuntimeError("Frozen dataset source identity drift: "+path)
                if full_sources:
                    if sha(path)!=expected:raise RuntimeError("Frozen dataset source SHA256 drift: "+path)
                    if signature(path)!=self.source_signatures[path]:
                        raise RuntimeError("Frozen dataset source changed during hashing: "+path)
        except OSError as error:
            raise RuntimeError("A frozen artifact is unavailable") from error

    def bytes_for(self,path):
        return self.controls[str(development_path(path))][1]


def grid():
    return [{"id":f"argmix_h{width}_do{int(dropout*10)}_lr{int(lr*1e6):04d}",
        "hidden_dim":width,"dropout":dropout,"learning_rate":lr}
        for width,dropout,lr in itertools.product((32,64),(0.,.1),(.0003,.001))]


def verify_grid(configurations):
    expected=grid()
    if isinstance(configurations,list):
        for configuration in configurations:
            if not isinstance(configuration,dict) or set(configuration)!={"id","hidden_dim","dropout","learning_rate"}:
                raise RuntimeError("Exact ARGMix configuration fields required")
            safe_id(configuration["id"])
            if type(configuration["hidden_dim"]) is not int or isinstance(configuration["dropout"],bool) or isinstance(configuration["learning_rate"],bool):
                raise RuntimeError("Boolean/nontyped grid values are forbidden")
    if configurations!=expected:
        if isinstance(configurations,list):
            for configuration in configurations:
                if isinstance(configuration,dict):safe_id(configuration.get("id"))
        raise RuntimeError("Exact ordered eight-configuration ARGMix grid required")
    for configuration in configurations:safe_id(configuration["id"])


def check_plan(plan):
    fixed={"schema_version":1,"kind":"argmix_fresh_development_search_plan_v1",
        "status":"proposal_only_not_executable_until_completed_preparation_is_bound",
        "family":"argmix_full_tree","scientific_status":STATUS,"splits":SPLITS,
        "training_seeds":SEEDS,"input_dim":48,"base_feature_count":8,"num_classes":4,
        "num_queries":64,"num_layers":2,"num_heads":4,"ffn_multiplier":4,
        "max_rel_distance":1000,"windows":8,"epochs":48,"patience":8,
        "weight_decay":.0001,"gradient_clip_norm":1.,"family_tie_tolerance_nats":.0001,
        "reference_adapter":ADAPTER,"selection":SELECTION,"source_paths":PATHS,
        "completion_receipt":COMPLETION,"preparation_freeze":PREPARATION_FREEZE,
        "confirmatory_access":False,"deterministic_execution_policy":DETERMINISTIC_POLICY}
    if any(plan.get(key)!=value for key,value in fixed.items()):
        raise RuntimeError("Frozen ARGMix plan dimensions/budget/source/selection changed")
    for key in ("input_dim","base_feature_count","num_classes","num_queries","num_layers","num_heads",
                "ffn_multiplier","max_rel_distance","windows","epochs","patience"):
        if type(plan[key]) is not int:raise RuntimeError("Frozen plan dimensions must be integers")
    verify_grid(plan.get("configurations"))
    if plan.get("reviewed_implementation_sha256")!=implementation_hashes():
        raise RuntimeError("Reviewed ARGMix implementation/dependency SHA256 binding failed")


def require_parent_records(manifest):
    expected={(s,i) for s in ("train","val") for i in range(SPLITS[s])}
    ids=[]
    for record in manifest["replicates"]:
        if type(record.get("replicate")) is not int or record.get("replicate",-1)<0:
            raise RuntimeError("Integer fresh-development parent IDs required")
        safe_id(str(record["replicate"]))
        ids.append((record["split"],record["replicate"]))
    if len(ids)!=80 or len(set(ids))!=80 or set(ids)!=expected:
        raise RuntimeError("Exactly64TRAIN/16VAL/TEST0 distinct parents required")


def verify_completed_development(completion_path,expected_completion_sha256,snapshot=None):
    """Verify sealed preparation facts; only inferred inputs become model sources."""
    snapshot=snapshot or IntegritySnapshot()
    completion_path=development_path(completion_path)
    if completion_path!=development_path(COMPLETION):raise RuntimeError("Unexpected completion receipt path")
    completion=snapshot.control(completion_path,expected_completion_sha256)
    if (completion.get("all_stages_complete") is not True or completion.get("train_parents")!=64
        or completion.get("validation_parents")!=16 or completion.get("test_parents")!=0
        or completion.get("inferred_boundary_and_age_tiles")!=640
        or completion.get("training_or_confirmatory_selection_frozen") is not False):
        raise RuntimeError("Successful64/16/TEST0 completed preparation required")
    facts=completion.get("stage_manifests",{})
    if set(facts)!=set(PATHS):raise RuntimeError("All four sealed preparation manifests required")
    manifests={}
    for stage,relative in PATHS.items():
        fact=facts[stage]
        if development_path(fact["path"])!=development_path(relative):
            raise RuntimeError("Preparation stage path differs from fresh-development namespace")
        manifests[stage]=snapshot.control(fact["path"],fact["sha256"])
        for key in ("protocol_sha256","preparation_sha256","source_manifest_sha256",
                    "boundary_manifest_sha256","boundary_preparation_sha256","verification_sha256"):
            if key in fact and manifests[stage].get(key)!=fact[key]:
                raise RuntimeError("Completion factual hashes differ from stage manifest")
    freeze_sha=valid_sha(completion.get("development_execution_freeze_file_sha256"))
    freeze=snapshot.control(PREPARATION_FREEZE,freeze_sha)
    if (freeze.get("kind")!="fresh_development_preparation_execution_freeze"
        or freeze.get("confirmatory_access") is not False or freeze.get("not_confirmatory_test_freeze") is not True):
        raise RuntimeError("Fresh-development preparation freeze required")
    if (completion.get("parent_registry_sha256")!=digest(freeze["parent_registry"])
        or completion.get("preparation_source_code_sha256")!=freeze["source_code_sha256"]):
        raise RuntimeError("Completion registry/source-code facts differ from execution freeze")
    config_files=freeze.get("config_files",{})
    if set(config_files)!={"configs/fresh_development_preparation_v1.json","configs/ancient_development64_v1.json"}:
        raise RuntimeError("Frozen preparation/oracle config files required")
    prep_plan=snapshot.control("configs/fresh_development_preparation_v1.json",config_files["configs/fresh_development_preparation_v1.json"])
    oracle_protocol=snapshot.control("configs/ancient_development64_v1.json",config_files["configs/ancient_development64_v1.json"])
    if prep_plan!=freeze["plan"] or oracle_protocol!=freeze["simulation_protocol"]:
        raise RuntimeError("Original preparation config bytes differ from execution freeze")
    registry=freeze["parent_registry"]
    require_parent_records({"replicates":registry})
    planned={(r["split"],r["replicate"]):r for r in registry}
    inferred=manifests["inferred"];oracle=manifests["oracle"]
    if oracle["protocol"]!=oracle_protocol or oracle["protocol_sha256"]!=config_files["configs/ancient_development64_v1.json"]:
        raise RuntimeError("Oracle protocol differs from frozen original config")
    for stage in ("oracle","inferred"):
        protocol=manifests[stage]["protocol"]
        require_parent_records(manifests[stage])
        if (protocol.get("splits")!=SPLITS or protocol.get("master_seed")!=20261101
            or protocol.get("reference_adapter")!=ADAPTER or protocol.get("windows")!=8
            or protocol["simulation"].get("query_diploids")!=32
            or protocol["simulation"].get("reference_diploids_per_source")!=3
            or protocol["simulation"].get("grid_points")!=128
            or protocol["simulation"].get("sequence_length")!=1000000):
            raise RuntimeError("Fresh observed-parent protocol/schema drift")
        for record in manifests[stage]["replicates"]:
            entry=planned[(record["split"],record["replicate"])]
            if record.get("seeds")!=entry["seeds"]:
                raise RuntimeError("Source ancestry/mutation seeds differ from frozen registry")
            if stage=="inferred" and record.get("inference_seed")!=entry["inference_seed"]:
                raise RuntimeError("Source inference seed differs from frozen registry")
    if (inferred["protocol"].get("benchmark")!="ancient-europe-stdpopsim-0.3.0-relate-inferred"
        or inferred["protocol"]["inference"].get("master_seed")!=20261102
        or inferred["protocol"]["inference"].get("method")!="Relate"
        or inferred["protocol"]["inference"].get("version")!="1.2.4"):
        raise RuntimeError("Pinned fresh inferred observation protocol required")
    protocol_path=development_path(PATHS["inferred"]).parent/"protocol.json"
    if development_path(facts["inferred"]["protocol_path"])!=protocol_path:
        raise RuntimeError("Inferred protocol file path differs from completion fact")
    protocol=snapshot.control(protocol_path,inferred["protocol_sha256"])
    if protocol!=inferred["protocol"]:raise RuntimeError("Inferred manifest/protocol bytes do not agree")
    source_parent=inferred["source_parent"]
    if (development_path(source_parent["config"])!=development_path("configs/ancient_development64_v1.json")
        or source_parent["config_sha256"]!=oracle["protocol_sha256"]
        or development_path(source_parent["manifest"])!=development_path(PATHS["oracle"])):
        raise RuntimeError("Inferred source parent is not bound to original oracle config/manifest")
    for key in ("master_seed","simulation","splits","reference_adapter","windows"):
        if inferred["protocol"][key]!=oracle_protocol[key]:raise RuntimeError("Inferred observation/source protocol field drift")
    boundary=manifests["boundary"];age=manifests["age"]
    if (inferred["source_parent"]!=inferred["protocol"]["source_parent"]
        or inferred["development_freeze_file_sha256"]!=freeze_sha
        or inferred["generation_code_sha256"]!=freeze["source_code_sha256"]
        or oracle["development_freeze_file_sha256"]!=freeze_sha
        or oracle["generation_code_sha256"]!=freeze["source_code_sha256"]
        or inferred["source_parent"]["manifest_sha256"]!=facts["oracle"]["sha256"]
        or boundary["source"]["manifest_sha256"]!=facts["inferred"]["sha256"]
        or age["source_manifest_sha256"]!=facts["inferred"]["sha256"]
        or age["boundary_manifest_sha256"]!=facts["boundary"]["sha256"]
        or age["boundary_preparation_sha256"]!=boundary["preparation_sha256"]
        or boundary.get("status")!="complete" or age.get("status")!="complete"):
        raise RuntimeError("Sealed oracle/inferred/boundary/age preparation source chain drift")
    for record in inferred["replicates"]:
        directory=development_path(record["directory"])
        required=development_path(PATHS["inferred"]).parent/record["split"]/f"rep{record['replicate']:03d}"/"inference/dataset"
        if directory!=required:raise RuntimeError("Inferred raw parent directory differs from frozen identity")
        for field,hash_field in (("metadata","metadata_sha256"),("tensor_file","tensor_sha256"),
                                 ("prepared_tensor_file","prepared_tensor_sha256")):
            path=development_path(record[field])
            if path!=directory/{"metadata":"metadata.json","tensor_file":"tensors.pt.gz","prepared_tensor_file":"prepared.pt"}[field]:
                raise RuntimeError("Raw parent source path/directory correspondence failed")
            snapshot.add_source(path,record[hash_field])
    snapshot.check(full_sources=True)
    return completion,inferred,snapshot


def bound_settings(plan,plan_path,plan_sha,completion,completion_path,completion_sha):
    facts=completion["stage_manifests"]
    return {key:plan[key] for key in ("schema_version","family","scientific_status","splits",
        "training_seeds","input_dim","base_feature_count","num_classes","num_queries","num_layers",
        "num_heads","ffn_multiplier","max_rel_distance","windows","epochs","patience","weight_decay",
        "gradient_clip_norm","family_tie_tolerance_nats","reference_adapter","selection","configurations","deterministic_execution_policy")} | {
        "stage":"fresh_development_whole_tree_argmix_equation_adaptation_v1",
        "development_namespace":"development-v1","search_plan":str(development_path(plan_path)),
        "search_plan_sha256":plan_sha,"completion_receipt":str(development_path(completion_path)),
        "completion_receipt_sha256":completion_sha,
        "dataset_manifest":str(development_path(facts["inferred"]["path"])),
        "dataset_manifest_sha256":facts["inferred"]["sha256"],
        "source_dataset_manifest_sha256":facts["inferred"]["sha256"],
        "boundary_dataset_manifest_sha256":facts["boundary"]["sha256"],
        "boundary_preparation_sha256":facts["boundary"]["preparation_sha256"],
        "age_dataset_manifest_sha256":facts["age"]["sha256"],
        "age_preparation_sha256":facts["age"]["preparation_sha256"],
        "preparation_execution_freeze_sha256":completion["development_execution_freeze_file_sha256"],
        "nll_policy":"float64 stable log-softmax of uncalibrated raw logits",
        "dataset_protocol":str(development_path(facts["inferred"]["protocol_path"])),
        "dataset_protocol_sha256":facts["inferred"]["protocol_sha256"],
        "preparation_freeze_sha256":completion["development_execution_freeze_file_sha256"],
        "bound_implementation_sha256":dict(plan["reviewed_implementation_sha256"]),
        "interpretation":"Fresh development validation only. Independent ARGMix paper-equation whole-tree adaptation, not exact native reproduction or independent confirmation.",
        "input_scope":"all raw observed marginal-tree nodes/topology and F48, original F8 prefix with verified full-parent reference adapter; no canonicalization or truth features",
        "cross_tree_aggregation":"none","external_pretraining":False,"confirmatory_access":False}


def bind_completed_development(completion_path,expected_completion_sha256,plan_path=PLAN,
                               expected_plan_sha256=None,output_path=None):
    """Root-facing binder. No executable config until every factual binding passes."""
    valid_sha(expected_plan_sha256)
    snapshot=IntegritySnapshot()
    plan=snapshot.control(plan_path,expected_plan_sha256);check_plan(plan)
    completion,manifest,snapshot=verify_completed_development(completion_path,expected_completion_sha256,snapshot)
    # Verify original prefix, fresh full-parent adapter, external labels and joins.
    from .train_argmix_development import load_verified_parents
    parents=load_verified_parents(manifest,snapshot)
    del parents
    settings=bound_settings(plan,plan_path,expected_plan_sha256,completion,completion_path,expected_completion_sha256)
    snapshot.check(full_sources=True)
    content=(json.dumps(settings,indent=2,allow_nan=False)+"\n").encode()
    if output_path is not None:
        output=development_path(output_path)
        if output.exists():raise RuntimeError("Never overwrite an existing bound config")
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open("xb") as handle:handle.write(content)
        snapshot.check(full_sources=True)
    return settings,content,hashlib.sha256(content).hexdigest()


def capture_training_snapshot(config_path,expected_config_sha256):
    snapshot=IntegritySnapshot()
    settings=snapshot.control(config_path,expected_config_sha256)
    plan=snapshot.control(settings["search_plan"],settings["search_plan_sha256"]);check_plan(plan)
    completion,manifest,snapshot=verify_completed_development(settings["completion_receipt"],settings["completion_receipt_sha256"],snapshot)
    expected=bound_settings(plan,settings["search_plan"],settings["search_plan_sha256"],completion,
        settings["completion_receipt"],settings["completion_receipt_sha256"])
    if settings!=expected:raise RuntimeError("Bound training config differs from verified plan/completion facts")
    snapshot.check(full_sources=True)
    return settings,manifest,snapshot
