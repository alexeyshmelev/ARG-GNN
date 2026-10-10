"""Metadata/source bytes and AST closure only. Never imports native libraries."""
import ast
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OWN=ROOT/"research/reviews/official_argformer_tsinfer_parent_v1_independent"
expected={
 "scripts/prepare_official_argformer_tsinfer_parent_v1.py":"ecd25b2a85b3e5778c5861222a7287887c41f5c42c1adeb42c8d4970a45ef427",
 "configs/official_argformer_tsinfer_parent_v1.json":"9e5212c6de07bb6c28e04f644b336faade04f1e06d6d488f5af49739e2529c55",
 "docs/official_argformer_tsinfer_parent_v1.md":"0a2bfd20c104cdc82d1ebff9eb9032d4f507ac3eedc2e13a23e986774cb13864",
 "tests/test_official_argformer_tsinfer_parent_v1.py":"f6db10112d89665364cb12d2aec3cc9d7785b9e87e67b1cc57840e89061d5407",
 "research/reviews/official_argformer_tsinfer_parent_v1_author/review_receipt.json":"e8dd535c7aa76b537d088844897583c02529c1bfc51bac2ae1a9574c601dcc79",
 "research/reviews/official_argformer_tsinfer_biological_v2_independent/review_receipt_final.json":"77ef95968b48c7561071144b753820de402c155a023174dcda39d94942fc760d",
}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
actual={n:sha(ROOT/n) for n in expected}
assert actual==expected
spec=importlib.util.spec_from_file_location("independent_source_dispatcher",ROOT/"scripts/prepare_official_argformer_tsinfer_parent_v1.py")
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
c=h.policy();h.verify_source(c);bridge,bc=h.load_bridge(c)
report=h.source_report(c,bridge,bc)
sealed=json.loads((ROOT/"research/reviews/official_argformer_tsinfer_biological_v2_independent/review_receipt_final.json").read_text())
assert report["native20_function_ASTs"]==sealed["native20_function_ASTs"]
assert report["native_preparation_tail_AST_sha256"]==sealed["preparation_tail_AST_sha256"]
assert report["original60_author_tree"]["tree_sha256"]==sealed["author60_tree_sha256"]
assert sha(OWN/"test_author_replay.py")==expected["tests/test_official_argformer_tsinfer_parent_v1.py"]
functions={n.name:n for n in ast.parse((ROOT/"scripts/prepare_official_argformer_tsinfer_parent_v1.py").read_text()).body if isinstance(n,ast.FunctionDef)}
roles={"train":{"pure_source_reference":4095,"mixed_reference_only":57,"query":2171291},"val":{"pure_source_reference":13000,"mixed_reference_only":7,"query":228709}}
gap=h.budget(bridge,bc,roles,2400000)
assert gap["exhaustive_query_corpus_pairs"]==41028000000
assert not gap["native_MLM512_contrastive4096_feasible"]
source_pins={"created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"reviewed_handoff_sha256":actual,"frozen_files_sha256":{n:sha(ROOT/n) for n in sorted(set(c["frozen_files_sha256"])|set(bc["frozen_files_sha256"]))},"handmade_only":True,"actual_arrays_or_live_runtime_ledger_access":False}
(OWN/"source_pins.json").write_text(json.dumps(source_pins,sort_keys=True,indent=2)+"\n")
(OWN/"source_report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
(OWN/"budget_cost_gap.json").write_text(json.dumps({"handmade_arithmetic_only":True,"counts":roles,"query_needed":2400000,"limit":bc["retrieval"]["max_exhaustive_pairs_per_parent"],"returned":gap,"implication":"Cost validation is skipped in the source-budget infeasible branch. No fit or retrieval is admitted."},sort_keys=True,indent=2)+"\n")
print(json.dumps({"source_pins_match":True,"native20_ASTs_match_sealed":True,"native_tail_AST_matches_sealed":True,"author60_tree_matches_sealed":True,"copied_author_tests_exact":True,"source_only_actual_arrays_opened":report["actual_arrays_opened"],"cost_gap_reproduced":True},sort_keys=True))
