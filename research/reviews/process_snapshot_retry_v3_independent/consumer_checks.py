"""Exact successor source/config comparisons and immutable failure metadata only."""
import ast
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
root=Path.cwd();out=root/"research/reviews/process_snapshot_retry_v3_independent"
sha=lambda p:hashlib.sha256((root/p).read_bytes()).hexdigest()
ref=lambda p:{"path":p,"sha256":sha(p)}
packet_path="research/reviews/process_snapshot_retry_v3_consumers_author/differential_receipt.json"
assert sha(packet_path)=="fcdbda28f329d85210fb7fdcd2ae9761bde594dd395a81e6fc116a347ce1bc32"
packet=json.loads((root/packet_path).read_text())
adapter="src/arggnn/project_budget_v9_consumer_retry_v3.py"
assert sha(adapter)=="dc735ee914dcddca9cd6078948225fa9652b10c4decef6f42df508798649de2b"
assert all(sha(r["path"])==r["sha256"] for r in packet["prior_source_refs_preserved"])
checks={}
for consumer in packet["consumers"]:
    name=consumer["hypothesis_id"]
    for key in ("predecessor_driver","successor_driver","predecessor_config","successor_config","source_diff","config_diff"):
        r=consumer[key];assert sha(r["path"])==r["sha256"]
    old_text=(root/consumer["predecessor_driver"]["path"]).read_text()
    expected=old_text.replace(consumer["predecessor_driver"]["path"],consumer["successor_driver"]["path"]).replace(consumer["predecessor_config"]["path"],consumer["successor_config"]["path"]).replace("from .project_budget_v9_consumer_retry_v2 import ","from .project_budget_v9_consumer_retry_v3 import ")
    actual=(root/consumer["successor_driver"]["path"]).read_text()
    assert expected==actual
    ast.parse(actual)
    old_cfg=json.loads((root/consumer["predecessor_config"]["path"]).read_text())
    cfg=json.loads((root/consumer["successor_config"]["path"]).read_text())
    normalized=deepcopy(cfg)
    assert cfg["helper_refs"]["budget_V9_consumer"]==ref(adapter)
    normalized["helper_refs"]["budget_V9_consumer"]=old_cfg["helper_refs"]["budget_V9_consumer"]
    extra=None
    if name=="H032":
        extra=normalized["runtime_recovery"].pop("additional_pretraining_failure")
        for key in tuple(normalized["runtime_recovery"]["predecessor_refs"]):
            if key.startswith("retry_v2_"):normalized["runtime_recovery"]["predecessor_refs"].pop(key)
        assert extra["original_training_updates"]==0 and extra["stage"]=="prepare_resources_guard" and extra["status"]=="failed_before_any_training_update"
        assert all(sha(r["path"])==r["sha256"] for r in extra["predecessor_refs"].values())
        prev=extra["predecessor_refs"]
        failed=json.loads((root/prev["failed"]["path"]).read_text())
        old_input=json.loads((root/prev["input"]["path"]).read_text())
        assert failed["status"]=="failed_partial_preserved" and failed["error"]==extra["outer_error"] and failed["error_type"]=="ValueError"
        assert failed["models"]=={} and "first_training_tile" not in failed
        assert failed["input_ref"]==prev["input"] and failed["config_ref"]==prev["config"]
        assert old_input["refs"]["config"]==prev["config"] and old_input["source_sha256"][prev["driver"]["path"]]==prev["driver"]["sha256"]
        trace=(root/prev["causal_trace"]["path"]).read_text()
        assert "PermissionError: [Errno 13] Permission denied: '/proc/1631866/fd'" in trace
        assert "line 418, in run_v1" in trace and "targets = prepare_v1(" in trace and "line 192, in prepare_v1" in trace
        assert "first_training_tile_arm" not in trace
        for k,r in prev.items():
            assert cfg["runtime_recovery"]["predecessor_refs"]["retry_v2_"+k]==r
    assert normalized==old_cfg
    report=json.loads((out/f"{name}_source_report.json").read_text())
    assert report["implementation"]==consumer["successor_driver"] and report["config"]==consumer["successor_config"]
    assert report["status"].startswith("source_only") and report["seed"]==101 and report["steps_per_arm"]==160 and report["dtype"]=="float32"
    assert report["module"]["sha256"]==sha(report["module"]["path"])
    checks[name]={"exact_text_substitution_only":True,"normalized_config_exact_predecessor":True,"prior_source_refs_preserved":True,"source_only_CLI_exit0":True,"additional_failure_lineage_verified":name=="H032","original_training_updates":0 if name=="H032" else None,"model_seed_loss_scorer_selection_schedule_unchanged":True}
(out/"consumer_checks.json").write_text(json.dumps(checks,indent=2)+"\n")
print(json.dumps(checks,indent=2))
