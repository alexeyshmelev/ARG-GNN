"""Completion-bound canonical-tree development protocol, with one source catalog."""
from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path

from . import argmix_development_bindings as shared
from .development_execution import DETERMINISTIC_POLICY

PLAN="configs/canonical_development_search_plan_v1.json"
FAMILY="canonical_gru"
NLL_POLICY="float64 stable log-softmax of uncalibrated raw logits"
INPUT_SCOPE="all full canonical observed-tree nodes and F48 after original F8 canonicalization and fresh full-parent reference adaptation; every retained canonical record inside each scored physical tile enters the GRU"
AGGREGATION="two-layer bidirectional per-query GRU over all chronological canonical records within each scored physical tile; original readouts gather containing records"
CANONICALIZATION={"suppress_anonymous_unary":True,"merge_identical_adjacent":True,
                  "fresh_reference_adapter_after_canonicalization":True}
SOURCE_KEYS=("completion_receipt_sha256","source_dataset_manifest_sha256","boundary_dataset_manifest_sha256",
    "boundary_preparation_sha256","age_dataset_manifest_sha256","age_preparation_sha256",
    "preparation_execution_freeze_sha256")


def canonical_path(value):
    """Reject held-out names before resolving aliases or opening any bytes."""
    path=Path(value)
    if any(part.lower()=="test" or any(token in part.lower() for token in
        ("confirmatory","confirmation","final_test","final-test")) for part in path.parts):
        raise ValueError("Held-out canonical development paths are forbidden before reads")
    resolved=shared.development_path(value)
    if any("confirmation" in part.lower() for part in resolved.relative_to(shared.ROOT.resolve()).parts):
        raise ValueError("Held-out canonical development alias destination is forbidden")
    return resolved


def admit_consumed_control_paths(value):
    """Preflight only path fields the shared verifier will actually consume."""
    if not isinstance(value,dict):return
    for key in ("search_plan","completion_receipt","dataset_manifest","dataset_protocol"):
        if isinstance(value.get(key),str):canonical_path(value[key])
    facts=value.get("stage_manifests")
    if isinstance(facts,dict):
        for fact in facts.values():
            if isinstance(fact,dict):
                for key in ("path","protocol_path"):
                    if isinstance(fact.get(key),str):canonical_path(fact[key])
    records=value.get("replicates")
    if isinstance(records,list):
        for record in records:
            if isinstance(record,dict):
                for key in ("directory","metadata","tensor_file","prepared_tensor_file"):
                    if isinstance(record.get(key),str):canonical_path(record[key])
    source=value.get("source_parent")
    if isinstance(source,dict):
        for key in ("config","manifest"):
            if isinstance(source.get(key),str):canonical_path(source[key])


def extra_implementation_paths():
    names=("canonical_development_bindings.py","train_canonical_development.py",
           "canonical_gru.py","tree_canonicalization.py","development_execution.py")
    return {"src/arggnn/"+name:Path(__file__).with_name(name) for name in names} | {
        "scripts/bind_canonical_development.py":Path(__file__).resolve().parents[2]/"scripts/bind_canonical_development.py"}


def implementation_hashes():
    """Extend the shared source admission catalog, never copy its live file list."""
    return shared.implementation_hashes() | {name:shared.sha(path) for name,path in extra_implementation_paths().items()}


class CanonicalIntegritySnapshot:
    """Compose shared control/raw-source checks with additional canonical pins."""
    def __init__(self):
        self.shared=shared.IntegritySnapshot()
        self.implementation_sha256=self.shared.implementation_sha256 | {
            name:shared.sha(path) for name,path in extra_implementation_paths().items()}

    @property
    def controls(self):return self.shared.controls
    @property
    def sources(self):return self.shared.sources
    @property
    def source_signatures(self):return self.shared.source_signatures
    def control(self,path,expected):
        value=self.shared.control(canonical_path(path),expected)
        admit_consumed_control_paths(value)
        return value
    def add_source(self,path,expected):return self.shared.add_source(canonical_path(path),expected)
    def bytes_for(self,path):return self.shared.bytes_for(canonical_path(path))

    def check(self,full_sources=False):
        self.shared.check(full_sources=full_sources)
        try:
            current=implementation_hashes()
        except OSError as error:
            raise RuntimeError("A frozen canonical implementation artifact is unavailable") from error
        if current!=self.implementation_sha256:
            raise RuntimeError("Frozen canonical implementation/dependency drift")


def grid():
    return [{"id":f"canonical_h{width}_do{int(dropout*10)}_lr{int(lr*1e6):04d}",
        "hidden_dim":width,"dropout":dropout,"learning_rate":lr}
        for width,dropout,lr in itertools.product((32,64),(0.,.1),(.0003,.001))]


def check_plan(plan):
    fixed={"schema_version":1,"kind":"canonical_fresh_development_search_plan_v1",
        "status":"proposal_only_not_executable_until_completed_preparation_is_bound",
        "family":FAMILY,"scientific_status":"ordinary_canonicalization_GNN_GRU_comparator_no_novelty_or_predictive_claim",
        "splits":shared.SPLITS,"training_seeds":shared.SEEDS,"input_dim":48,"base_feature_count":8,
        "num_classes":4,"num_queries":64,"num_layers":2,"temporal_layers":2,"windows":8,
        "epochs":48,"patience":8,"weight_decay":.0001,"gradient_clip_norm":1.,
        "family_tie_tolerance_nats":.0001,"reference_adapter":shared.ADAPTER,
        "selection":shared.SELECTION,"nll_policy":NLL_POLICY,"canonicalization":CANONICALIZATION,
        "deterministic_policy":DETERMINISTIC_POLICY,
        "source_paths":shared.PATHS,"completion_receipt":shared.COMPLETION,
        "preparation_freeze":shared.PREPARATION_FREEZE,"input_scope":INPUT_SCOPE,
        "cross_tree_aggregation":AGGREGATION,"external_pretraining":False,"confirmatory_access":False}
    if any(plan.get(key)!=value for key,value in fixed.items()):
        raise RuntimeError("Frozen canonical development source/scope/budget/selection changed")
    for key in ("input_dim","base_feature_count","num_classes","num_queries","num_layers","temporal_layers",
                "windows","epochs","patience"):
        if type(plan[key]) is not int:raise RuntimeError("Typed integer canonical dimensions required")
    configurations=plan.get("configurations")
    if isinstance(configurations,list):
        for row in configurations:
            if not isinstance(row,dict) or set(row)!={"id","hidden_dim","dropout","learning_rate"}:
                raise RuntimeError("Exact canonical configuration fields required")
            shared.safe_id(row["id"])
            if type(row["hidden_dim"]) is not int or isinstance(row["dropout"],bool) or isinstance(row["learning_rate"],bool):
                raise RuntimeError("Typed nonboolean canonical grid values required")
    if configurations!=grid():raise RuntimeError("Exact ordered eight-configuration canonical grid required")
    if plan.get("reviewed_implementation_sha256")!=implementation_hashes():
        raise RuntimeError("Reviewed canonical/shared dependency hash binding failed")


def bound_settings(plan,plan_path,plan_sha,completion,completion_path,completion_sha,sources):
    facts=completion["stage_manifests"]
    keys=("schema_version","family","scientific_status","splits","training_seeds","input_dim","base_feature_count",
        "num_classes","num_queries","num_layers","temporal_layers","windows","epochs","patience","weight_decay",
        "gradient_clip_norm","family_tie_tolerance_nats","reference_adapter","selection","nll_policy","canonicalization","deterministic_policy",
        "configurations","input_scope","cross_tree_aggregation","external_pretraining","confirmatory_access")
    return {key:plan[key] for key in keys} | {
        "stage":"fresh_development_canonical_tree_GNN_GRU_v1","development_namespace":"development-v1",
        "search_plan":str(shared.development_path(plan_path)),"search_plan_sha256":plan_sha,
        "completion_receipt":str(shared.development_path(completion_path)),"completion_receipt_sha256":completion_sha,
        "dataset_manifest":str(shared.development_path(facts["inferred"]["path"])),
        "dataset_manifest_sha256":facts["inferred"]["sha256"],"source_dataset_manifest_sha256":facts["inferred"]["sha256"],
        "dataset_protocol":str(shared.development_path(facts["inferred"]["protocol_path"])),
        "dataset_protocol_sha256":facts["inferred"]["protocol_sha256"],
        "boundary_dataset_manifest_sha256":facts["boundary"]["sha256"],
        "boundary_preparation_sha256":facts["boundary"]["preparation_sha256"],
        "age_dataset_manifest_sha256":facts["age"]["sha256"],"age_preparation_sha256":facts["age"]["preparation_sha256"],
        "preparation_execution_freeze_sha256":completion["development_execution_freeze_file_sha256"],
        "source_artifact_sha256":dict(sources),"bound_implementation_sha256":dict(plan["reviewed_implementation_sha256"]),
        "interpretation":"Fresh64TRAIN/16VAL/TEST0 development only. Standard canonicalization/GNN/GRU comparator, no independent confirmation, novelty, mechanism or superiority claim."}


def bind_completed_canonical_development(completion_path,expected_completion_sha256,plan_path=PLAN,
                                        expected_plan_sha256=None,output_path=None):
    """Root-facing admission and exclusive config write, never generation/fitting."""
    shared.valid_sha(expected_plan_sha256)
    snapshot=CanonicalIntegritySnapshot()
    plan=snapshot.control(plan_path,expected_plan_sha256);check_plan(plan)
    completion,manifest,snapshot=shared.verify_completed_development(canonical_path(completion_path),expected_completion_sha256,snapshot)
    from .train_canonical_development import prepare_verified_parents
    parents=prepare_verified_parents(manifest,snapshot)
    del parents
    settings=bound_settings(plan,plan_path,expected_plan_sha256,completion,completion_path,expected_completion_sha256,snapshot.sources)
    snapshot.check(full_sources=True)
    content=(json.dumps(settings,indent=2,allow_nan=False)+"\n").encode()
    if output_path is not None:
        output=canonical_path(output_path)
        if output.exists():raise RuntimeError("Never overwrite an existing bound canonical config")
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open("xb") as handle:handle.write(content)
        snapshot.check(full_sources=True)
    return settings,content,hashlib.sha256(content).hexdigest()


def capture_training_snapshot(config_path,expected_config_sha256):
    """Admit actual receipt/freeze/sources before input/model/device admission."""
    snapshot=CanonicalIntegritySnapshot()
    settings=snapshot.control(config_path,expected_config_sha256)
    plan=snapshot.control(settings["search_plan"],settings["search_plan_sha256"]);check_plan(plan)
    completion,manifest,snapshot=shared.verify_completed_development(canonical_path(settings["completion_receipt"]),settings["completion_receipt_sha256"],snapshot)
    expected=bound_settings(plan,settings["search_plan"],settings["search_plan_sha256"],completion,
        settings["completion_receipt"],settings["completion_receipt_sha256"],snapshot.sources)
    if settings!=expected:raise RuntimeError("Canonical training config differs from captured plan/completion/source facts")
    snapshot.check(full_sources=True)
    return settings,manifest,snapshot


def settings_from_original_bytes(content,expected_sha256):
    shared.valid_sha(expected_sha256)
    if not isinstance(content,bytes) or hashlib.sha256(content).hexdigest()!=expected_sha256:
        raise ValueError("Original canonical config bytes do not match expected SHA256")
    return json.loads(content)
