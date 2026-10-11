"""Exact driver/config delta and hashed pretraining failure metadata only."""
import ast
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
root=Path.cwd();out=root/"research/reviews/full_reservation_storage_v1_independent"
sha=lambda p:hashlib.sha256((root/p).read_bytes()).hexdigest()
ref=lambda p:{"path":p,"sha256":sha(p)}
packet_path="research/reviews/full_reservation_consumers_v1_author/differential_receipt.json"
assert sha(packet_path)=="467bc5f4955c2744526613a593e87c80b399b7a0def7b269d484433ad3758357"
packet=json.loads((root/packet_path).read_text())
assert all(sha(r["path"])==r["sha256"] for r in packet["prior_source_refs_preserved"])
adapter="src/arggnn/project_storage_full_reservation_v1.py"
assert sha(adapter)=="c0fdace58e26020c8227606e9b77325bc7308e7ec6691fe3a7e7c18cd4846105"
checks={}
for c in packet["consumers"]:
    name=c["hypothesis_id"]
    for key in ("predecessor_driver","successor_driver","predecessor_config","successor_config","source_diff","config_diff"):
        r=c[key];assert sha(r["path"])==r["sha256"]
    old=(root/c["predecessor_driver"]["path"]).read_text()
    expected=old.replace(c["predecessor_driver"]["path"],c["successor_driver"]["path"]).replace(c["predecessor_config"]["path"],c["successor_config"]["path"]).replace("from .project_budget_v9_consumer_retry_v3 import ","from .project_storage_full_reservation_v1 import ")
    actual=(root/c["successor_driver"]["path"]).read_text()
    assert actual==expected;ast.parse(actual)
    old_cfg=json.loads((root/c["predecessor_config"]["path"]).read_text())
    cfg=json.loads((root/c["successor_config"]["path"]).read_text())
    normalized=deepcopy(cfg)
    assert cfg["helper_refs"]["budget_V9_consumer"]==ref(adapter)
    normalized["helper_refs"]["budget_V9_consumer"]=old_cfg["helper_refs"]["budget_V9_consumer"]
    for key in ("budget_guard","resource_boundaries"):normalized[key]=old_cfg[key]
    accounting=normalized.pop("storage_accounting")
    assert accounting["accounting_mode"]=="full_reserved_no_closed_scope_credit" and accounting["human_hard_limit_bytes"]==1000000000000 and accounting["global_proc_inspection_in_training_consumer"] is False
    if name=="H032":
        failure=normalized["runtime_recovery"].pop("additional_v3_pretraining_failure")
        assert normalized["runtime_recovery"].pop("current_storage_correction")==accounting
        for key in tuple(normalized["runtime_recovery"]["predecessor_refs"]):
            if key.startswith("retry_v3_"):normalized["runtime_recovery"]["predecessor_refs"].pop(key)
        for key,r in failure["predecessor_refs"].items():
            assert cfg["runtime_recovery"]["predecessor_refs"]["retry_v3_"+key]==r
    else:
        failure=normalized.pop("pretraining_storage_failure")
        for key in tuple(normalized["helper_refs"]):
            if key.startswith("pretraining_v3_"):normalized["helper_refs"].pop(key)
        for key,r in failure["predecessor_refs"].items():assert cfg["helper_refs"]["pretraining_v3_"+key]==r
    assert normalized==old_cfg
    assert failure["new_optimizer_updates"]==0 and failure["science_changes"] is False
    prev=failure["predecessor_refs"]
    assert prev==c["v3_failure_lineage"] and all(sha(r["path"])==r["sha256"] for r in prev.values())
    failed=json.loads((root/prev["failed"]["path"]).read_text())
    original_input=json.loads((root/prev["input"]["path"]).read_text())
    assert failed["status"]=="failed_partial_preserved" and failed["error"]==failure["error"] and failed["models"]=={}
    assert "first_training_tile" not in failed and "first_recovery_training_tiles" not in failed and "recovery_lineage" not in failed
    assert failed["input_ref"]==prev["input"] and failed["config_ref"]==prev["config"]
    assert original_input["refs"]["config"]==prev["config"] and original_input["source_sha256"][prev["driver"]["path"]]==prev["driver"]["sha256"]
    report=json.loads((out/f"{name}_source_report.json").read_text())
    assert report["implementation"]==c["successor_driver"] and report["config"]==c["successor_config"]
    assert report["status"].startswith("source_only") and report["steps_per_arm"]==160 and report["seed"]==101 and report["dtype"]=="float32"
    assert report["module"]["sha256"]==sha(report["module"]["path"])
    checks[name]={"verdict":"PASS","exact_source_substitution_only":True,"normalized_config_matches_predecessor":True,"operative_adapter_and_accounting_mode":True,"immutable_v3_failure_input_config_driver_join":True,"new_v3_optimizer_updates":0,"prior_files_and_scientific_trajectory_preserved":True,"source_only_CLI_exit0":True}
(out/"consumer_checks.json").write_text(json.dumps(checks,indent=2)+"\n")
print(json.dumps(checks,indent=2))
