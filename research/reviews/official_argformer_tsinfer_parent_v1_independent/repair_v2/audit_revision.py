"""Exact repaired source, closure and immutable initial-review evidence audit."""
import ast
import copy
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
BASE=ROOT/"research/reviews/official_argformer_tsinfer_parent_v1_independent"
OWN=BASE/"repair_v2"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
expected={
 "scripts/prepare_official_argformer_tsinfer_parent_v1.py":"9ac0ca01b214e86841808fee951fb55e74bf57bac7227656d6ee7410d9c8a8ed",
 "configs/official_argformer_tsinfer_parent_v1.json":"9e5212c6de07bb6c28e04f644b336faade04f1e06d6d488f5af49739e2529c55",
 "docs/official_argformer_tsinfer_parent_v1.md":"6e294b3e59344fdbae30b8d7cc26847f58df4ee1cadafb2c14ae913304fcf90e",
 "tests/test_official_argformer_tsinfer_parent_v1.py":"be7069800eec82d170b70ef7715b727d86c26e2ee73f48b8ae94cda6a1c13569",
 "research/reviews/official_argformer_tsinfer_parent_v1_author/repair_v2/review_receipt.json":"7dc1f2aca5344b67c71b250f959f748db68ae9112e6259f726fd4a769c311465",
 "research/reviews/official_argformer_tsinfer_biological_v2_independent/review_receipt_final.json":"77ef95968b48c7561071144b753820de402c155a023174dcda39d94942fc760d",
}
actual={n:sha(ROOT/n) for n in expected};assert actual==expected
assert sha(BASE/"review_receipt.json")=="e3df087b46b4616c6aa5f088dfaa32ab17367313a727b1d4b7379e3dcd2d11af"
initial=json.loads((BASE/"review_receipt.json").read_text())
assert all(sha(ROOT/n)==h for n,h in initial["artifact_sha256"].items())
assert all(sha(ROOT/n)==h for n,h in initial["initial_source_snapshot_sha256"].items())
initial_pins=json.loads((BASE/"source_pins.json").read_text())
assert all(sha(ROOT/n)==h for n,h in initial_pins["frozen_files_sha256"].items())
original=ast.parse((BASE/"initial_source_snapshot/dispatcher.py").read_text())
revised=ast.parse((ROOT/"scripts/prepare_official_argformer_tsinfer_parent_v1.py").read_text())
def strip_budget(module):
 return ast.Module(body=[copy.deepcopy(n) for n in module.body if not isinstance(n,ast.FunctionDef) or n.name!="budget"],type_ignores=[])
assert ast.dump(strip_budget(original),include_attributes=False)==ast.dump(strip_budget(revised),include_attributes=False)
old_budget=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=="budget")
new_budget=copy.deepcopy(next(n for n in revised.body if isinstance(n,ast.FunctionDef) and n.name=="budget"))
guard=ast.parse('if pure*query_needed>bc["retrieval"]["max_exhaustive_pairs_per_parent"]:\n    raise ValueError("full-corpus cost needs fresh explicit admission; never subsample baseline")').body[0]
guard_dump=ast.dump(guard,include_attributes=False)
indexes=[i for i,n in enumerate(new_budget.body) if ast.dump(n,include_attributes=False)==guard_dump]
assert len(indexes)==1
guard_index=indexes[0]
feasible_index=next(i for i,n in enumerate(new_budget.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="feasible")
assert guard_index<feasible_index
new_budget.body.pop(guard_index)
assert ast.dump(new_budget,include_attributes=False)==ast.dump(old_budget,include_attributes=False)
assert (ROOT/"tests/test_official_argformer_tsinfer_parent_v1.py").read_bytes().startswith((BASE/"initial_source_snapshot/tests.py").read_bytes())
assert sha(OWN/"test_author_replay.py")==expected["tests/test_official_argformer_tsinfer_parent_v1.py"]
spec=importlib.util.spec_from_file_location("independent_repaired_dispatcher",ROOT/"scripts/prepare_official_argformer_tsinfer_parent_v1.py")
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
c=h.policy();h.verify_source(c);bridge,bc=h.load_bridge(c)
assert c==json.loads((BASE/"initial_source_snapshot/config.json").read_text())
report=h.source_report(c,bridge,bc)
assert report["native20_function_ASTs"]==initial["native20_function_ASTs"]
assert report["native_preparation_tail_AST_sha256"]==initial["native_preparation_tail_AST_sha256"]
assert report["original60_author_tree"]==initial["original60_author_tree"]
for prefix,name in [(ROOT,"scripts/prepare_official_argformer_tsinfer_parent_v1.py"),(ROOT,"configs/official_argformer_tsinfer_parent_v1.json"),(ROOT,"docs/official_argformer_tsinfer_parent_v1.md"),(ROOT,"tests/test_official_argformer_tsinfer_parent_v1.py"),(ROOT,"research/reviews/official_argformer_tsinfer_parent_v1_author/repair_v2/review_receipt.json")]:
 output=OWN/"source_snapshot"/Path(name).name
 output.parent.mkdir(parents=True,exist_ok=True)
 with output.open("xb") as stream:stream.write((prefix/name).read_bytes())
(OWN/"source_report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
audit={
 "kind":"official_argformer_tsinfer_parent_v1_independent_repair_audit",
 "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "reviewed_handoff_sha256":actual,
 "initial_BLOCKED_review_unchanged":True,"initial_source_snapshots_and_all_review_artifacts_unchanged":True,
 "only_budget_AST_changed":True,"exact_new_guard_precedes_feasibility_branch":True,
 "remove_new_guard_restores_original_budget_AST":True,"config_unchanged":True,
 "native20_function_ASTs_unchanged":True,"native_preparation_tail_AST_unchanged":True,
 "author60_tree_unchanged":True,"all_frozen_sources_unchanged":True,
 "original40_author_tests_retained_byte_for_byte_prefix":True,"copied_revised42_author_tests_exact":True,
 "actual_arrays_native_runtime_ledger_or_permits_accessed":False,
 "source_snapshot_sha256":{str(p.relative_to(ROOT)):sha(p) for p in sorted((OWN/"source_snapshot").iterdir())},
}
(OWN/"source_audit.json").write_text(json.dumps(audit,sort_keys=True,indent=2)+"\n")
print(json.dumps(audit,sort_keys=True))
