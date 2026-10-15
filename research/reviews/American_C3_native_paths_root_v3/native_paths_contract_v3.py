"""Frozen contracts for complete original C3 corpus with narrower write budgets."""
from pathlib import Path
import ast,hashlib,importlib.util,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/American_C3_native_paths_root_v3"
SOURCE=REL+"/native_paths_contract_v3.py"
DRIVER=REL+"/root_native_paths_driver.py"
WRAPPER="src/arggnn/american_c3_native_paths_consumer_v3.py"
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
ORIGINAL_PATHS_REF=dict(path="src/arggnn/american_c3_paths_v1.py",sha256="3b32f16e0a81160852e5b93fe9033abf9bc2eeb56c3b15ca552d73f4aacf374c")
PATHS_REF=dict(path="src/arggnn/american_c3_native_paths_consumer_v3.py",sha256="3957212a54e852575173fcdbc12d4e4fabf6bb08e44f2fe4008634d6263fb4e6")
COMMON_REF=dict(path="src/arggnn/american_c3_common_v1.py",sha256="615d3892f1df7e9de39e57991eab883b2eca1af9d13e424661541aac4b2d2abf")
AUTHOR_REF=dict(path=".runtime/author_sources/argformer_f7545be7bca0/src/data/prepare_data_pretrain.py",sha256="e7e66a029694b21a2ba1f9c9a6c4fbc193d81b8831282580ba9639a9e224726c")
COUNTS_REF=dict(path="research/reviews/American_C3_native_paths_root_v1/root_counts.json",sha256="00db9317aa168ef87b441e6f41eaa30a4d2fd686719a57322f77097e0732dbcd")
RUNTIME_REF=dict(path="research/reviews/ARGformer_American_C3_implementation_20261015_v1/shared_cpu_runtime_metadata.json",sha256="e0944a1c2c4b2a26d2edfe68fad11db691b4b6853ebeddd8e3b79692b43fa602")
OBSERVATION_REF=dict(path="experiments/results/American_C3_observed_full30mb_supervisor_repair_20261015_v1/observed/manifest.json",sha256="c3d58a8d10d901a162f406e2e218a10e17e5f1617b3c626503d72bcbd1b4ccee")
COMPLETION_REF=dict(path="experiments/results/American_C3_observed_full30mb_supervisor_repair_20261015_v1/completed.json",sha256="3e2fe2aa49dc92b9ae634f2803d8dd1a36bb8bec31a4c4ffc7008b5818c620c0")
EXACT_COUNT_REF=dict(path="research/reviews/American_C3_native_paths_capacity_source_v2/actual_counts_v1/completed.json",
 sha256="d743c880b5c7773221aaa87da23a26307b0b7fbed6ce4b03217be3c256d2c86a")
COUNT_WORKER_REF=dict(path="research/reviews/American_C3_native_paths_capacity_source_v2/exact_path_count_worker_v1.py",
 sha256="0f3150ee97ac2841e4c0c9f168ddca71e741068159e6bfb6fd800cdbb011e517")
CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e"
BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
OUTPUT="experiments/results/American_C3_native_paths_train000_full30mb_seed92001_v3"
CAP=6000000000
LOG_CAP=16777216
METADATA_ALLOWANCE=1048576
DEADLINE_SECONDS=7200
RSS_CAP=512000000000
PER_SAMPLE_REF=dict(path="research/reviews/American_C3_native_paths_capacity_source_v2/actual_counts_v1/per_sample_counts.json",
 sha256="74505550902e3bc0c990f956868bfe8149bfa68173b05e1217cab2b9d099254f")
SERIALIZER_CONTRACT_REF={'path': 'research/reviews/American_C3_native_paths_root_v3/serializer_contract.json', 'sha256': '97935084b0298c00b835a630306afb94b4afeb43854d4f8d6b55aca0f78b5000'}
SERIALIZER_RUNTIME_SOURCE_REF={'path': '.runtime/env/lib/python3.11/pickle.py', 'sha256': '227b9550b1491f5ed3bf7b444a97f91113d47a3bb0e82fb49da24cbc255d407f'}
REFERENCE_PATHS=9583977
REFERENCE_TOKENS=116080787
NATIVE_BOUND=5924203578
NATIVE_CLOSURE=[COMMON_REF,PATHS_REF,AUTHOR_REF,ORIGINAL_PATHS_REF]



def configuration_v1(active=False):
    require(ref(PATHS_REF["path"])==PATHS_REF and ref(ORIGINAL_PATHS_REF["path"])==ORIGINAL_PATHS_REF,
        "Frozen budget successor and unchanged original science required")
    counts=validate_counts_v1();exact=validate_exact_count_v3(counts)
    validate_serializer_contract_v3()
    tree=ast.parse((ROOT/PATHS_REF["path"]).read_text())
    default=next(n.value for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=="DEFAULTS" for t in n.targets))
    cfg={kw.arg:ast.literal_eval(kw.value) for kw in default.keywords}
    cfg.update(execution_enabled=active,observation_ref=OBSERVATION_REF,
        max_unique_paths=exact["unique_paths"],max_path_tokens=exact["path_tokens"],
        max_path_occurrences=exact["path_occurrences"],max_vocabulary_nodes=exact["vocabulary_nodes"],
        exact_native_path_count_ref=EXACT_COUNT_REF,exact_per_sample_counts_ref=PER_SAMPLE_REF,
        reference_unique_paths=REFERENCE_PATHS,reference_path_tokens=REFERENCE_TOKENS,
        exact_max_raw_path_nodes=29,output_cap_bytes=CAP)
    p=exact["unique_paths"];t=exact["path_tokens"];v=exact["vocabulary_nodes"];f=10000
    n=p+REFERENCE_PATHS+f
    bound=5*(2*t+REFERENCE_TOKENS+29*f)+8*n+8*p+404779200+448*v+24*n+8*REFERENCE_PATHS+5*REFERENCE_PATHS+16*16777216
    require(bound==NATIVE_BOUND and bound+LOG_CAP+METADATA_ALLOWANCE==5942029370
        and CAP-bound-LOG_CAP-METADATA_ALLOWANCE==57970630,
        "Exact complete serialization bound plus original root overhead must fit6GB; no fallback")
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
    paths=(SOURCE,DRIVER,WRAPPER,PATTERN_REF["path"],MEMO_REF["path"],V4_REF["path"],
        GUARD_REF["path"],COUNT_WORKER_REF["path"],*tuple(r["path"] for r in NATIVE_CLOSURE))
    sources=[ref(p) for p in dict.fromkeys(paths)]
    return sources+[EXACT_COUNT_REF,PER_SAMPLE_REF,SERIALIZER_CONTRACT_REF]


def validate_exact_count_v3(counts):
    """Compact original full-count metadata only; no corpus or tree read here."""
    exact=read(EXACT_COUNT_REF)
    require(exact.get("schema")=="american_c3_exact_native_path_count_result_v1"
        and exact.get("status")=="completed_native_C3_exact_path_count"
        and exact.get("parent_id")=="american_c3_train_000" and exact.get("split")=="train"
        and exact.get("observation_ref")==OBSERVATION_REF and exact.get("tree_ref")==counts["tree_ref"]
        and exact.get("root_counts_ref")==COUNTS_REF and exact.get("native_paths_source_ref")==ORIGINAL_PATHS_REF
        and exact.get("source_author_function_ref")==AUTHOR_REF and exact.get("source_ref")==COUNT_WORKER_REF
        and ref(COUNT_WORKER_REF["path"])==COUNT_WORKER_REF,
        "Exact completed full30Mb native count and unchanged original path/author identities required")
    true_flags=("full_corpus_scanned","metadata_only_output","exact_get_path_key_tuples",
        "per_sample_dedup_equals_original_global_dedup","native_split_not_called_count_only",
        "original_split_seed42_preserved_for_future_full_prep","original_raw_path_cap_1024_sufficient")
    false_flags=("corpus_written","path_keys_or_vocab_written","labels_or_roster_opened",
        "query_ancestry_truth_opened","scientific_subsampling","author_Table1_exact_reproduction")
    require(all(exact.get(k) is True for k in true_flags)
        and all(exact.get(k) is False for k in false_flags),
        "Complete exact non-subset count with no corpus/truth output and unchanged future seed42 split")
    expected=dict(max_unique_paths=25574587,max_path_tokens=316672003,
        max_path_occurrences=122980800,max_vocabulary_nodes=106769)
    require(exact.get("native_storage_cap_fields")==expected
        and [exact.get(k) for k in ("unique_paths","path_tokens","path_occurrences","vocabulary_nodes")]
            ==[25574587,316672003,122980800,106769]
        and exact.get("path_occurrences")==counts["full_path_occurrence_upper_count"]
        and exact.get("vocabulary_nodes")<=counts["num_nodes"]
        and exact.get("max_raw_path_nodes")==29 and exact["max_raw_path_nodes"]<=1024
        and exact.get("native_storage_bound_bytes")==12228346285,
        "Actual complete native caps and unchanged1024 full-path-length guard; no fallback")
    validate_per_sample_counts_v3(exact)
    return exact


def validate_per_sample_counts_v3(exact):
    """Only the original count-only JSON; never a corpus/roster/tree payload."""
    require(exact.get("per_sample_counts_ref")==PER_SAMPLE_REF,"Exact original per-sample count evidence required")
    p=local(PER_SAMPLE_REF["path"])
    with p.open("rb") as stream:raw=stream.read(1048577)
    require(len(raw)<=1048576 and hashlib.sha256(raw).hexdigest()==PER_SAMPLE_REF["sha256"],
        "Bounded unchanged original per-sample count JSON required")
    detail=json.loads(raw);rows=detail["per_sample_counts"]
    require(detail.get("schema")=="american_c3_exact_native_path_per_sample_counts_v1"
        and detail.get("root_counts_ref")==COUNTS_REF and detail.get("parent_id")=="american_c3_train_000"
        and detail.get("sample_count")==1600 and detail.get("full_corpus_scanned") is True
        and detail.get("query_ancestry_truth_opened") is False and detail.get("corpus_written") is False
        and type(rows) is list and len(rows)==1600,"Exact label-blind complete per-sample count evidence required")
    fields={"sample_index","sample_node","unique_paths","path_tokens","path_occurrences","max_raw_path_nodes"}
    for i,row in enumerate(rows):
        require(type(row) is dict and set(row)==fields and all(type(row[k]) is int for k in fields)
            and row["sample_index"]==row["sample_node"]==i and 0<row["unique_paths"]<=76863
            and row["unique_paths"]<=row["path_tokens"]<=29*row["unique_paths"]
            and row["path_occurrences"]==76863 and 0<row["max_raw_path_nodes"]<=29,
            "Original1600 ordered full-sample counts required; no fallback")
    require([sum(row[k] for row in rows) for k in ("unique_paths","path_tokens","path_occurrences")]
        ==[25574587,316672003,122980800]
        and [sum(row[k] for row in rows[:600]) for k in ("unique_paths","path_tokens")]
        ==[REFERENCE_PATHS,REFERENCE_TOKENS]
        and max(row["max_raw_path_nodes"] for row in rows)==29,
        "Exact original full/reference aggregates for role-checked future producer required")
    return detail


def validate_serializer_contract_v3():
    value=read(SERIALIZER_CONTRACT_REF)
    require(value.get("schema")=="American_C3_native_serialization_contract_v3"
        and value.get("python_implementation")=="cpython" and value.get("python_version")=="3.11.17"
        and value.get("pickle_highest_protocol")==5
        and value.get("runtime_pickle_source_ref")==SERIALIZER_RUNTIME_SOURCE_REF
        and ref(SERIALIZER_RUNTIME_SOURCE_REF["path"])==SERIALIZER_RUNTIME_SOURCE_REF,
        "Exact read-only qualified serializer source/version contract required")
    return value
