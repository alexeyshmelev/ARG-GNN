"""Fresh-development completion binding and exact family grids, no generation."""
from __future__ import annotations
import hashlib
from copy import deepcopy
import json
import re
from pathlib import Path

from .resources import ROOT, project_path

SPLITS = {"train":64,"val":16,"test":0}
SEEDS = [101,202,303]
SELECTION = "minimum mean parent-validation uncalibrated logit NLL"
CORE = ("development_training_bindings.py","train_anchor_temporal_development.py",
    "train_anchor_temporal.py","anchor_temporal_baselines.py","anchor_kernel.py",
    "boundary_memory.py","boundary_dataset.py","kron_interface.py","models.py",
    "data.py","reference_encoder.py","genealogy_baselines.py","simulate.py",
    "train_study.py","train_boundary.py","train_anchor.py","train_anchor_development.py","age_dataset.py","age_interface.py",
    "train.py","metrics.py","resources.py",
    "confirmatory_metrics.py")
SHA = re.compile(r"[0-9a-f]{64}\Z")
SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def valid_sha(value):
    return isinstance(value,str) and SHA.fullmatch(value) is not None and value!="0"*64


def read_bound(path, expected):
    path=no_test_path(path)
    if not valid_sha(expected):raise RuntimeError("Real completed SHA256 required")
    try:content=path.read_bytes()
    except OSError as error:raise RuntimeError("Bound artifact unavailable: "+str(path)) from error
    if hashlib.sha256(content).hexdigest()!=expected:raise RuntimeError("Bound file SHA256 mismatch: "+str(path))
    return json.loads(content)


def snapshot(extra=()):
    return {name:sha(ROOT/"src/arggnn"/name) for name in (*CORE,*extra)}


def implementation_snapshot(settings):
    pins=settings.get("bound_implementation_sha256")
    if not isinstance(pins,dict) or not set(CORE).issubset(pins):
        raise RuntimeError("Complete common implementation locks required")
    if any(not isinstance(name,str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*[.]py",name) is None
           or not valid_sha(value) for name,value in pins.items()):
        raise RuntimeError("Safe .py dependency names and exact hashes required")
    return {name:sha(ROOT/"src/arggnn"/name) for name in pins}


def no_test_path(path):
    p=project_path(path)
    if any(token in str(p.relative_to(ROOT)).lower() for token in ("confirmatory","final_test","final-test")):
        raise RuntimeError("Final held-out paths are unavailable in development")
    return p


def verify_completion(path, expected_sha, *, freeze_path=None, with_snapshot=False):
    """Read completed fresh cohort/cache bindings only, no native jobs."""
    from .boundary_dataset import load_boundary_dataset
    from .age_dataset import load_age_caches
    path=no_test_path(path)
    completion,source_lock=capture_completion_sources(path,expected_sha,freeze_path)
    if (completion.get("all_stages_complete") is not True
        or completion.get("train_parents")!=64 or completion.get("validation_parents")!=16
        or completion.get("test_parents")!=0 or completion.get("inferred_boundary_and_age_tiles")!=640
        or completion.get("training_or_confirmatory_selection_frozen") is not False):
        raise RuntimeError("Successful64/16/TEST0 preparation completion required")
    facts=completion.get("stage_manifests",{})
    if set(facts)!={"oracle","inferred","boundary","age"}:raise RuntimeError("Incomplete stage manifest facts")
    manifests={stage:read_bound(no_test_path(facts[stage]["path"]),facts[stage]["sha256"]) for stage in facts}
    for stage,payload in manifests.items():
        for key in ("protocol_sha256","preparation_sha256","source_manifest_sha256",
                    "boundary_manifest_sha256","boundary_preparation_sha256"):
            if key in facts[stage] and facts[stage][key]!=payload.get(key):
                raise RuntimeError("Completion facts differ from sealed manifest: "+stage+":"+key)
    if not valid_sha(completion.get("development_execution_freeze_file_sha256")):
        raise RuntimeError("Real development preparation execution freeze required")
    for stage in ("oracle","inferred"):
        p=manifests[stage]["protocol"]
        if p["splits"]!=SPLITS or p["master_seed"]!=20261101:raise RuntimeError("Fresh parent protocol drift")
        expected={(s,i) for s in ("train","val") for i in range(SPLITS[s])}
        ids=[(r["split"],r["replicate"]) for r in manifests[stage]["replicates"]]
        if len(ids)!=80 or len(set(ids))!=80 or set(ids)!=expected:raise RuntimeError("Parent identities changed")
    inferred=manifests["inferred"]; boundary=manifests["boundary"]; age=manifests["age"]
    if inferred["protocol"]["inference"]["master_seed"]!=20261102:raise RuntimeError("Fresh inference seed drift")
    if (inferred["source_parent"]["manifest_sha256"]!=facts["oracle"]["sha256"]
        or boundary["source"]["manifest_sha256"]!=facts["inferred"]["sha256"]
        or age["boundary_manifest_sha256"]!=facts["boundary"]["sha256"]
        or age["source_manifest_sha256"]!=facts["inferred"]["sha256"]
        or age["boundary_preparation_sha256"]!=boundary["preparation_sha256"]):
        raise RuntimeError("Preparation source/cache chain changed")
    for stage in ("boundary","age"):
        if manifests[stage].get("status")!="complete":raise RuntimeError("Incomplete cache manifest")
    dataset=load_boundary_dataset(no_test_path(facts["boundary"]["path"]).parent,
        expected_manifest_sha256=facts["boundary"]["sha256"],
        expected_preparation_sha256=boundary["preparation_sha256"],
        expected_source_manifest_sha256=facts["inferred"]["sha256"])
    for split in ("train","val"):
        cases=dataset.cases(split)
        if (len(cases)!=SPLITS[split] or any(c.metadata["split"]!=split
            or c.metadata["master_seed"]!=20261101 or c.metadata["num_classes"]!=4
            or c.query_node_ids.numel()!=64 for c in cases)):
            raise RuntimeError("Verified development case counts/roles drift")
    if len(dataset.records)!=640:raise RuntimeError("Require every verified development tile")
    age_caches=load_age_caches(no_test_path(facts["age"]["path"]).parent,dataset,
        expected_manifest_sha256=facts["age"]["sha256"],expected_preparation_sha256=age["preparation_sha256"])
    if len(age_caches)!=640:raise RuntimeError("Age caches do not cover development tiles")
    # Recheck every file after full source/cache verification.
    read_bound(path,expected_sha)
    for stage,fact in facts.items():read_bound(fact["path"],fact["sha256"])
    source_lock.check(full=True)
    return (completion,dataset,age_caches,source_lock) if with_snapshot else (completion,dataset,age_caches)


def make_configs(completion,completion_path,completion_sha):
    facts=completion["stage_manifests"]
    common={"schema_version":1,"stage":"fresh_development_same_source_training_v1",
        "development_namespace":"development-v1","splits":dict(SPLITS),"training_seeds":list(SEEDS),
        "num_layers":2,"latent_dim":8,"num_anchors":32,"num_scales":3,"num_classes":4,
        "epochs":48,"patience":8,"weight_decay":.0001,"source_rate_weight_decay":0.,
        "initial_coupling":[5.,20.,80.],"max_attention_score_bytes":268435456,
        "source_dataset_manifest_sha256":facts["inferred"]["sha256"],
        "boundary_dataset_manifest_sha256":facts["boundary"]["sha256"],
        "boundary_preparation_sha256":facts["boundary"]["preparation_sha256"],
        "age_dataset_manifest_sha256":facts["age"]["sha256"],
        "age_preparation_sha256":facts["age"]["preparation_sha256"],
        "boundary_data":str(no_test_path(facts["boundary"]["path"]).parent),
        "age_data":str(no_test_path(facts["age"]["path"]).parent),
        "completion_receipt":str(no_test_path(completion_path)),
        "completion_receipt_sha256":completion_sha,
        "preparation_execution_freeze_sha256":completion["development_execution_freeze_file_sha256"],
        "preparation_execution_freeze":str(no_test_path(completion.get("development_execution_freeze_path",DEFAULT_PREPARATION_FREEZE))),
        "selection":SELECTION,"family_selection":"Mean three fixed-seed selected-checkpoint equal-parent VAL NLL. No test access.",
        "tie_policy":"Within0.0001nats:parameter count then lexicographic frozen model ID",
        "interpretation":"Fresh development only. No independent confirmation, final test or superiority claim.",
        "nll_policy":"float64 stable log-softmax of uncalibrated raw logits",
        "checkpoint_storage":"ignored checkpoints directory only",
        "comparison_note":"Same47observed fields/plain sourceGNN and32fixed physical anchors, independent readouts. Temporal capacities differ. Smoother uses only observed reference role/Cqr subset and fits no parameters."}
    configs={}
    for family in ("gru","transformer"):
        settings={**deepcopy(common),"family":family,"models":[],"model_specs":{},"model_learning_rates":{},
                  "bound_implementation_sha256":snapshot()}
        for width in (32,64):
            for dropout in (0.,.1):
                for lr in (.0003,.001):
                    name=f"{family}_h{width}_do{int(dropout*10):01d}_lr{int(lr*1e6):04d}"
                    settings["models"].append(name)
                    settings["model_specs"][name]={"family":family,"hidden_dim":width,
                        "temporal_depth":2,"dropout":dropout,"transformer_heads":4}
                    settings["model_learning_rates"][name]=lr
        configs[family]=settings
    smoother={**deepcopy(common),"family":"reference_smoother","models":[],"model_specs":{},
              "model_learning_rates":{},"bound_implementation_sha256":snapshot(),
              "fitting":"None. Hyperparameters selected on VAL; three seed runs are identical deterministic checks."}
    for bw in (10000,30000,100000,300000):
        for pc in (.001,.01):
            name=f"reference_bw{bw}_pc{int(pc*1e6):05d}"
            smoother["models"].append(name)
            smoother["model_specs"][name]={"family":"reference_smoother","bandwidth_bp":bw,"pseudocount":pc}
    configs["reference_smoother"]=smoother
    candidate={**deepcopy(common),"family":"candidate","hidden_dim":32,"rate_weight_decay":0.,
        "initial_damping":1.,"selection":"minimum mean parent-validation NLL",
        "models":[],"model_specs":{},"model_learning_rates":{},
        "bound_implementation_sha256":snapshot()}
    for geometry in ("plain","age"):
        for depth in (1,2):
            for lr in (.0003,.001):
                name=f"candidate_{geometry}_d{depth}_lr{int(lr*1e6):04d}"
                candidate["models"].append(name)
                candidate["model_specs"][name]={"geometry":geometry,"coupling":"full",
                    "kernel_depth":depth,"rate_parameterization":"log"}
                candidate["model_learning_rates"][name]=lr
    configs["candidate"]=candidate
    return configs


def checked_protocol(path,expected_sha):
    settings=read_bound(path,expected_sha)
    if (settings.get("development_namespace")!="development-v1" or settings.get("splits")!=SPLITS
        or settings.get("training_seeds")!=SEEDS or settings.get("selection")!=SELECTION):
        raise RuntimeError("Require fixed fresh-development namespace/counts/seeds/rawNLL selection")
    fixed={"num_layers":2,"latent_dim":8,"num_anchors":32,"num_scales":3,"num_classes":4,
           "epochs":48,"patience":8,"weight_decay":.0001,"source_rate_weight_decay":0.,
           "max_attention_score_bytes":268435456}
    if any(settings.get(k)!=v or isinstance(settings.get(k),bool) for k,v in fixed.items()):
        raise RuntimeError("Fixed source dimensions/optimization budget changed")
    if settings.get("initial_coupling")!=[5.,20.,80.]:raise RuntimeError("Fixed source coupling changed")
    if settings.get("nll_policy")!="float64 stable log-softmax of uncalibrated raw logits":
        raise RuntimeError("Uncalibrated raw-logit NLL policy required")
    for key in ("completion_receipt_sha256","source_dataset_manifest_sha256","boundary_dataset_manifest_sha256",
                "boundary_preparation_sha256","age_dataset_manifest_sha256","age_preparation_sha256",
                "preparation_execution_freeze_sha256"):
        if not valid_sha(settings.get(key)):raise RuntimeError("Completed source/cache/receipt hashes required")
    family=settings.get("family"); names=settings.get("models"); specs=settings.get("model_specs")
    if family not in ("gru","transformer","reference_smoother"):raise RuntimeError("Unsupported development baseline family")
    if not isinstance(names,list) or len(names)!=8 or len(set(names))!=8 or any(
        not isinstance(n,str) or SAFE.fullmatch(n) is None for n in names):
        raise RuntimeError("Eight safe distinct model IDs required")
    if not isinstance(specs,dict) or set(names)!=set(specs):raise RuntimeError("Exact model specifications required")
    rates=settings.get("model_learning_rates")
    actual=[]
    for name in names:
        spec=specs[name]
        if family=="reference_smoother":
            if set(spec)!={"family","bandwidth_bp","pseudocount"} or spec["family"]!=family:
                raise RuntimeError("Smoother grid schema changed")
            actual.append((spec["bandwidth_bp"],spec["pseudocount"]))
        else:
            if (set(spec)!={"family","hidden_dim","temporal_depth","dropout","transformer_heads"}
                or spec["family"]!=family or type(spec["hidden_dim"]) is not int
                or spec["temporal_depth"]!=2 or spec["transformer_heads"]!=4
                or isinstance(spec["dropout"],bool)):
                raise RuntimeError("Neural temporal source/width/depth/head grid changed")
            if not isinstance(rates,dict) or set(rates)!=set(names):raise RuntimeError("Exact per-model learning rates required")
            actual.append((spec["hidden_dim"],spec["dropout"],rates[name]))
    expected=({(bw,pc) for bw in (10000,30000,100000,300000) for pc in (.001,.01)}
        if family=="reference_smoother" else
        {(w,do,lr) for w in (32,64) for do in (0.,.1) for lr in (.0003,.001)})
    if len(set(actual))!=8 or set(actual)!=expected:raise RuntimeError("Require exact eight-configuration family grid")
    if family=="reference_smoother" and rates!={}:raise RuntimeError("No optimizer learning rates for no-fit smoother")
    for key in ("boundary_data","age_data","completion_receipt","preparation_execution_freeze"):no_test_path(settings[key])
    return settings


def check_completion_settings(settings,completion):
    facts=completion["stage_manifests"]
    expected={
        "source_dataset_manifest_sha256":facts["inferred"]["sha256"],
        "boundary_dataset_manifest_sha256":facts["boundary"]["sha256"],
        "boundary_preparation_sha256":facts["boundary"]["preparation_sha256"],
        "age_dataset_manifest_sha256":facts["age"]["sha256"],
        "age_preparation_sha256":facts["age"]["preparation_sha256"],
        "preparation_execution_freeze_sha256":completion["development_execution_freeze_file_sha256"]}
    if any(settings.get(k)!=v for k,v in expected.items()):
        raise RuntimeError("Training config differs from its completed preparation receipt")
    actual_freeze=no_test_path(completion.get("development_execution_freeze_path",DEFAULT_PREPARATION_FREEZE))
    if no_test_path(settings["preparation_execution_freeze"])!=actual_freeze:
        raise RuntimeError("Training freeze path differs from completed preparation receipt")
    for name,stage in (("boundary_data","boundary"),("age_data","age")):
        if no_test_path(settings[name])!=no_test_path(facts[stage]["path"]).parent:
            raise RuntimeError("Training cache path differs from its completion receipt")


def check_live(settings,config_path,config_sha,*,source_snapshot=None,full_sources=True):
    read_bound(config_path,config_sha)
    pins=settings["bound_implementation_sha256"]
    if pins!=implementation_snapshot(settings):raise RuntimeError("Frozen baseline implementation drift")
    if source_snapshot is None:
        completion,source_snapshot=capture_completion_sources(settings["completion_receipt"],settings["completion_receipt_sha256"],settings.get("preparation_execution_freeze"))
    else:
        completion=read_bound(settings["completion_receipt"],settings["completion_receipt_sha256"])
    check_completion_settings(settings,completion)
    source_snapshot.check(full=full_sources)
    for key,directory in (("boundary_dataset_manifest_sha256","boundary_data"),
                          ("age_dataset_manifest_sha256","age_data")):
        read_bound(no_test_path(settings[directory])/"manifest.json",settings[key])
    return pins


# AD-03: bind all current on-disk inputs without repeatedly loading tensors.
DEFAULT_PREPARATION_FREEZE = ROOT/"configs/fresh_development_execution_freeze_v1.json"


def artifact_signature(path):
    value=Path(path).stat()
    return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)


class SourceSnapshot:
    def __init__(self, hashes):
        self.hashes=dict(hashes)
        self.signatures={p:artifact_signature(p) for p in self.hashes}
        self.check(full=True)

    def check(self,full=False):
        try:
            for path,expected in self.hashes.items():
                no_test_path(path)
                if artifact_signature(path)!=self.signatures[path]:
                    raise RuntimeError("Bound input artifact changed: "+path)
                if full and sha(path)!=expected:
                    raise RuntimeError("Bound input artifact SHA256 changed: "+path)
                if artifact_signature(path)!=self.signatures[path]:
                    raise RuntimeError("Bound input artifact changed during hashing: "+path)
        except OSError as error:
            raise RuntimeError("Bound input artifact unavailable") from error

    def manifest(self):
        return {"artifact_sha256":self.hashes,
                "artifact_signatures":{p:list(v) for p,v in self.signatures.items()},
                "epoch_check":"stat identity only; no tensor reload",
                "seal_check":"full SHA256 plus before/after stat identity"}


def expected_parent_registry():
    from .ancient_simulate import ancient_replicate_seeds
    from .prepare_relate_study import inference_seed
    return [{"split":s,"replicate":i,"seeds":ancient_replicate_seeds(20261101,s,i),
             "inference_seed":inference_seed(20261102,s,i)}
            for s in ("train","val") for i in range(SPLITS[s])]


def capture_completion_sources(path,expected_sha,freeze_path=None):
    """Verify real freeze/registry, source chain, and every declared input byte."""
    hashes={}
    def add(value,expected):
        p=no_test_path(value)
        if not valid_sha(expected):raise RuntimeError("Missing actual artifact hash: "+str(p))
        key=str(p)
        if key in hashes and hashes[key]!=expected:raise RuntimeError("Conflicting artifact hashes")
        hashes[key]=expected
        return p
    def json_file(value,expected):
        return read_bound(add(value,expected),expected)
    path=add(path,expected_sha);completion=json_file(path,expected_sha)
    if (completion.get("all_stages_complete") is not True
        or completion.get("train_parents")!=64 or completion.get("validation_parents")!=16
        or completion.get("test_parents")!=0 or completion.get("inferred_boundary_and_age_tiles")!=640
        or completion.get("training_or_confirmatory_selection_frozen") is not False):
        raise RuntimeError("Successful64/16/TEST0 preparation completion required")
    facts=completion.get("stage_manifests",{})
    if set(facts)!={"oracle","inferred","boundary","age"}:raise RuntimeError("Incomplete stage manifest facts")
    # Guard every stage path before any stage file is opened.
    for fact in facts.values():no_test_path(fact["path"])
    freeze_path=no_test_path(freeze_path or completion.get(
        "development_execution_freeze_path",DEFAULT_PREPARATION_FREEZE))
    frozen=json_file(freeze_path,completion["development_execution_freeze_file_sha256"])
    if (frozen.get("kind")!="fresh_development_preparation_execution_freeze"
        or frozen.get("confirmatory_access") is not False
        or frozen.get("not_confirmatory_test_freeze") is not True
        or frozen.get("plan",{}).get("expected_splits")!=SPLITS
        or frozen.get("plan",{}).get("expected_simulation_master_seed")!=20261101
        or frozen.get("plan",{}).get("relate_master_seed")!=20261102
        or frozen.get("parent_registry")!=expected_parent_registry()):
        raise RuntimeError("Actual preparation freeze/parent registry differs from fresh development")
    for name,value in frozen["config_files"].items():json_file(name,value)
    for name,value in frozen["source_code_sha256"].items():add(name,value)
    # Enforce exact frozen stage locations BEFORE opening any stage manifest.
    for stage,fact in facts.items():
        if no_test_path(fact["path"]).parent!=no_test_path(frozen["plan"]["paths"][stage]):
            raise RuntimeError("Stage directory differs from actual preparation freeze")
    manifests={s:json_file(facts[s]["path"],facts[s]["sha256"]) for s in facts}
    registry={(r["split"],r["replicate"]):r for r in frozen["parent_registry"]}
    sources={}
    for stage,payload in manifests.items():
        for k in ("protocol_sha256","preparation_sha256","source_manifest_sha256",
                  "boundary_manifest_sha256","boundary_preparation_sha256"):
            if k in facts[stage] and facts[stage][k]!=payload.get(k):
                raise RuntimeError("Completion facts differ from sealed manifest: "+stage+":"+k)
        expected_directory=no_test_path(frozen["plan"]["paths"][stage])
        if no_test_path(facts[stage]["path"]).parent!=expected_directory:
            raise RuntimeError("Stage directory differs from actual preparation freeze")
    for stage in ("oracle","inferred"):
        m=manifests[stage];p=m["protocol"]
        if p["splits"]!=SPLITS or p["master_seed"]!=20261101:raise RuntimeError("Fresh parent protocol drift")
        records=m["replicates"];ids=[(r["split"],r["replicate"]) for r in records]
        if len(ids)!=80 or len(set(ids))!=80 or set(ids)!=set(registry):raise RuntimeError("Parent identities changed")
        if stage=="oracle":
            protocol_path=frozen["plan"]["oracle_config"]
        else:
            protocol_path=facts[stage].get("protocol_path",str(Path(facts[stage]["path"]).parent/"protocol.json"))
        if json_file(protocol_path,m["protocol_sha256"])!=p:raise RuntimeError("Current source protocol drift")
        sources[stage]={(r["split"],r["replicate"]):r for r in records}
        for record in records:
            identity=(record["split"],record["replicate"])
            declared=no_test_path(record["directory"])
            expected_dir=no_test_path(facts[stage]["path"]).parent/identity[0]/f"rep{identity[1]:03d}"
            if stage=="inferred":expected_dir=expected_dir/"inference/dataset"
            if declared!=expected_dir:
                raise RuntimeError("Parent directory differs from frozen stage identity")
            for field,filename in (("metadata","metadata.json"),("tensor_file","tensors.pt.gz"),
                                   ("prepared_tensor_file","prepared.pt")):
                if no_test_path(record[field])!=declared/filename:
                    raise RuntimeError("Parent artifact path differs from declared directory")
            if record["seeds"]!=registry[identity]["seeds"]:raise RuntimeError("Parent simulation seed registry mismatch")
            for field,h in (("metadata","metadata_sha256"),("tensor_file","tensor_sha256"),
                            ("prepared_tensor_file","prepared_tensor_sha256")):
                add(record[field],record[h])
            metadata=json_file(record["metadata"],record["metadata_sha256"])
            if (metadata["split"],metadata["replicate"],metadata["seeds"],metadata["master_seed"])!=(
                    *identity,registry[identity]["seeds"],20261101):
                raise RuntimeError("Current source metadata/registry mismatch")
            if record.get("observed_tree_file"):add(record["observed_tree_file"],record["observed_tree_sha256"])
            if stage=="inferred":
                if record["inference_seed"]!=registry[identity]["inference_seed"]:
                    raise RuntimeError("Inference seed registry mismatch")
                original=sources["oracle"][identity];binding=record["source_parent"]
                if no_test_path(binding["directory"])!=no_test_path(original["directory"]):
                    raise RuntimeError("Original oracle directory mismatch")
                for _,h in (("metadata","metadata_sha256"),("tensor_file","tensor_sha256"),
                            ("prepared_tensor_file","prepared_tensor_sha256")):
                    if binding[h]!=original[h]:raise RuntimeError("Original oracle file binding mismatch")
                oracle_dir=no_test_path(original["directory"])
                add(oracle_dir/"truth.trees.gz",binding["truth_archive_sha256"])
                add(oracle_dir/"observed.trees.gz",binding["clean_observed_archive_sha256"])
                raw=no_test_path(record["directory"]);native=raw.parent
                add(native/"inferred.trees",record["native_inferred_tree_sha256"])
                add(native/"execution.json",record["execution_sha256"])
                prep_path=native.parent/"inputs/preparation.json"
                prep=json_file(prep_path,record["preparation_sha256"])
                for key,value in prep["files_sha256"].items():add(prep["paths"][key],value)
    inferred=manifests["inferred"];boundary=manifests["boundary"];age=manifests["age"]
    if (inferred["protocol"]["inference"]["master_seed"]!=20261102
        or inferred["source_parent"]["manifest_sha256"]!=facts["oracle"]["sha256"]
        or boundary["source"]["manifest_sha256"]!=facts["inferred"]["sha256"]
        or age["boundary_manifest_sha256"]!=facts["boundary"]["sha256"]
        or age["source_manifest_sha256"]!=facts["inferred"]["sha256"]
        or age["boundary_preparation_sha256"]!=boundary["preparation_sha256"]):
        raise RuntimeError("Preparation source/cache chain changed")
    for stage in ("boundary","age"):
        if manifests[stage].get("status")!="complete":raise RuntimeError("Incomplete cache manifest")
    bdir=no_test_path(facts["boundary"]["path"]).parent
    adir=no_test_path(facts["age"]["path"]).parent
    for record in boundary["parents"]:
        add(bdir/record["supervision_file"],record["supervision_sha256"])
    for record in boundary["records"]:add(bdir/record["bank_file"],record["bank_sha256"])
    for record in age["records"]:add(adir/record["operator_file"],record["operator_sha256"])
    for stage in ("oracle","inferred"):
        m=manifests[stage]
        if m.get("verification_file"):add(m["verification_file"],m["verification_sha256"])
    return completion,SourceSnapshot(hashes)


def admit_run(settings,config_path,config_sha,*,imported_pins,provided_data=None,provided_age=None):
    """Mandatory production admission used by BOTH candidate and baseline."""
    for path in (config_path,settings["completion_receipt"],settings["boundary_data"],settings["age_data"]):
        no_test_path(path)
    if provided_data is not None and no_test_path(provided_data)!=no_test_path(settings["boundary_data"]):
        raise RuntimeError("Supplied data path differs from bound completion cache")
    if provided_age is not None and no_test_path(provided_age)!=no_test_path(settings["age_data"]):
        raise RuntimeError("Supplied age path differs from bound completion cache")
    read_bound(config_path,config_sha)
    live=implementation_snapshot(settings)
    if settings.get("bound_implementation_sha256")!=live or imported_pins!=live:
        raise RuntimeError("Bound/imported implementation dependency lock mismatch")
    completion,dataset,ages,lock=verify_completion(settings["completion_receipt"],
        settings["completion_receipt_sha256"],freeze_path=settings["preparation_execution_freeze"],
        with_snapshot=True)
    check_completion_settings(settings,completion)
    lock.hashes[str(no_test_path(config_path))]=config_sha
    lock.signatures[str(no_test_path(config_path))]=artifact_signature(config_path)
    check_live(settings,config_path,config_sha,source_snapshot=lock,full_sources=True)
    return completion,dataset,ages,lock
