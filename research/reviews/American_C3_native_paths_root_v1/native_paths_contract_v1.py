"""Pure frozen contracts for one C3 TRAIN native-path preparation."""
from pathlib import Path
import ast,hashlib,importlib.util,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/American_C3_native_paths_root_v1"
SOURCE=REL+"/native_paths_contract_v1.py"
DRIVER=REL+"/root_native_paths_driver.py"
WRAPPER="src/arggnn/american_c3_native_paths_consumer_v1.py"
PATTERN_REF=dict(path="research/reviews/American_C3_observation_supervisor_repair_source_v1/root_observation_driver.py",
 sha256="8e282edad3253a0e1d32807a94104129a7593fab3010b8a00b9f5526f8a8fe93")
if hashlib.sha256((ROOT/PATTERN_REF["path"]).read_bytes()).hexdigest()!=PATTERN_REF["sha256"]:
    raise ValueError("Frozen repaired C3 supervisor source changed")
_ms=importlib.util.spec_from_file_location("_C3_native_paths_original_supervisor",ROOT/PATTERN_REF["path"])
previous=importlib.util.module_from_spec(_ms);_ms.loader.exec_module(previous)
require,ref,read,save,local=(previous.require,previous.ref,previous.read,previous.save,previous.local)
GUARD_REF=dict(path="research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py",sha256="ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef")
POLICY_REF=dict(path="research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json",sha256="a0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85")
REPAIR_REF=dict(path="research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json",sha256="f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db")
MEMO_REF=dict(path="research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py",sha256="c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364")
V4_REF=dict(path="src/arggnn/project_budget_v4.py",sha256="a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47")
PATHS_REF=dict(path="src/arggnn/american_c3_paths_v1.py",sha256="3b32f16e0a81160852e5b93fe9033abf9bc2eeb56c3b15ca552d73f4aacf374c")
COMMON_REF=dict(path="src/arggnn/american_c3_common_v1.py",sha256="615d3892f1df7e9de39e57991eab883b2eca1af9d13e424661541aac4b2d2abf")
AUTHOR_REF=dict(path=".runtime/author_sources/argformer_f7545be7bca0/src/data/prepare_data_pretrain.py",sha256="e7e66a029694b21a2ba1f9c9a6c4fbc193d81b8831282580ba9639a9e224726c")
COUNTS_REF=dict(path=REL+"/root_counts.json",sha256="00db9317aa168ef87b441e6f41eaa30a4d2fd686719a57322f77097e0732dbcd")
RUNTIME_REF=dict(path="research/reviews/ARGformer_American_C3_implementation_20261015_v1/shared_cpu_runtime_metadata.json",sha256="e0944a1c2c4b2a26d2edfe68fad11db691b4b6853ebeddd8e3b79692b43fa602")
OBSERVATION_REF=dict(path="experiments/results/American_C3_observed_full30mb_supervisor_repair_20261015_v1/observed/manifest.json",sha256="c3d58a8d10d901a162f406e2e218a10e17e5f1617b3c626503d72bcbd1b4ccee")
COMPLETION_REF=dict(path="experiments/results/American_C3_observed_full30mb_supervisor_repair_20261015_v1/completed.json",sha256="3e2fe2aa49dc92b9ae634f2803d8dd1a36bb8bec31a4c4ffc7008b5818c620c0")
CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e"
BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
OUTPUT="experiments/results/American_C3_native_paths_train000_full30mb_seed92001_v1"
CAP=11500000000
LOG_CAP=16777216
METADATA_ALLOWANCE=1048576
DEADLINE_SECONDS=7200
RSS_CAP=512000000000
NATIVE_CLOSURE=[COMMON_REF,PATHS_REF,AUTHOR_REF]

def configuration_v1(active=False):
    require(ref(PATHS_REF["path"])==PATHS_REF,"Unchanged native paths source required")
    tree=ast.parse((ROOT/PATHS_REF["path"]).read_text())
    default=next(n.value for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=="DEFAULTS" for t in n.targets))
    cfg={kw.arg:ast.literal_eval(kw.value) for kw in default.keywords}
    cfg.update(execution_enabled=active,observation_ref=OBSERVATION_REF,
        max_path_occurrences=122980800,max_vocabulary_nodes=106769,output_cap_bytes=CAP)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="storage_bound")
    ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),PATHS_REF["path"]+":pure_storage_bound","exec"),ns)
    bound=ns["storage_bound"](cfg)
    require(bound==11171171968 and bound+LOG_CAP+METADATA_ALLOWANCE<=CAP,
            "Exact unchanged native storage bound plus root overhead must fit11.5GB; no fallback")
    return cfg,bound

def validate_counts_v1():
    counts=read(COUNTS_REF)
    require(counts["root_authorizer"]=="root_agent" and counts["truth_opened"] is False
        and counts["source_observation_ref"]==OBSERVATION_REF and counts["num_samples"]==1600
        and counts["num_nodes"]==106769 and counts["num_trees"]==76863
        and counts["num_samples"]*counts["num_trees"]==counts["full_path_occurrence_upper_count"]==122980800
        and counts["file_bytes"]==34756124 and counts["sequence_length"]==30000000.,
        "Exact root-only SHA-verified full30Mb observation counts")
    return counts

def original_guard_v1():
    # Future root mutation process only. No memo install in this module.
    m=previous.overlay_guard_v5(POLICY_REF,dict(overlay=GUARD_REF,repair_authority=REPAIR_REF))
    require(all(getattr(m.budget.v4,n).__module__==m.budget.v4.__name__ for n in ("canonical","file_bytes","mutex")),
            "Root reserve/close operations require unmemoized originalv4")
    return m

def validate_close_receipt_v1(record):
    value=read(record)
    require(value.get("kind")=="concurrent_completed_scope_close_receipt_v1"
        and value.get("key")=="H041/American-C3-observe-full30mb-supervisor-repair-20261015-v1"
        and value.get("scope_measurements")==1 and value.get("artifacts_deleted")==0
        and value.get("old_journal_mutated") is False
        and value["value"]["request"]["completed_ref"]==COMPLETION_REF
        and value["value"]["future_writes_and_relaunch_revoked"] is True,
        "Root must supply actual exact completed full30Mb credit receipt before admission")
    return value

def closure_v1():
    return [ref(p) for p in (SOURCE,DRIVER,WRAPPER,PATTERN_REF["path"],MEMO_REF["path"],V4_REF["path"],
        GUARD_REF["path"],*tuple(r["path"] for r in NATIVE_CLOSURE))]
