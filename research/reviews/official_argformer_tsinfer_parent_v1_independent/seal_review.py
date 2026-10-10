"""Seal an independent source/handmade review; emits no execution authority."""
import datetime
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OWN=ROOT/"research/reviews/official_argformer_tsinfer_parent_v1_independent"
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ref(path):return {"path":str(path.relative_to(ROOT)),"sha256":sha(path)}
c=json.loads((ROOT/"configs/official_argformer_tsinfer_parent_v1.json").read_text())
source=json.loads((OWN/"source_report.json").read_text())
pins=json.loads((OWN/"source_pins.json").read_text())
assert all(sha(ROOT/n)==h for n,h in pins["reviewed_handoff_sha256"].items())
assert all(sha(ROOT/n)==h for n,h in pins["frozen_files_sha256"].items())
root=ET.parse(OWN/"pytest_final.xml").getroot();cases=root.findall(".//testcase")
failures=[case.attrib["name"] for case in cases if case.find("failure") is not None]
assert len(cases)==83 and len(failures)==1 and "[4095]" in failures[0]
concerns=[{
 "id":"P1","severity":"medium","blocking_this_dispatcher_source_review":True,
 "title":"The infeasible source-budget branch bypasses the frozen exhaustive-pair cost rule",
 "source":"scripts/prepare_official_argformer_tsinfer_parent_v1.py","lines":[362,370],
 "evidence":"Independent handmade counts preserve exact floor(90%) MLM split totals,64 mixed-reference paths and at most20000*120 scored query IDs. pure_reference_train=4095, pure_reference_val=13000, query_needed=2400000 returns exhaustive_query_corpus_pairs=41028000000 (>40000000000) without the existing cost rejection. The4096 comparison raises the expected cost error.",
 "impact":"The condition still stops at the native source4096 infeasibility (exit2); this does not admit fit or retrieval. However, the required excessive-cost failure and fresh-explicit-admission requirement are skipped in the infeasible branch, contradicting the documented unconditional retrieval-cost rule.",
 "required_change":"Check measured full pure-reference corpus times genuine query-needed count against unchanged frozen max_exhaustive_pairs_per_parent before branching on source-budget feasibility. Preserve under-cost infeasible preparation with exit2 and over-cost stopped partial evidence. Do not alter frozen native functions, preparation tail, scientific configuration or corpus.",
 "tests":["test_pair_cost_guard_required_even_if_source_budget_infeasible[4095]","test_pair_cost_guard_required_even_if_source_budget_infeasible[4096]"],
 "artifacts":["budget_cost_gap.json","pytest_final.log","pytest_final.xml"]
}]
(OWN/"concerns.json").write_text(json.dumps(concerns,sort_keys=True,indent=2)+"\n")
review="""Independent source and handmade CPU review: BLOCKED for dispatcher ecd25b2a85b3e5778c5861222a7287887c41f5c42c1adeb42c8d4970a45ef427.

40 author tests replayed unchanged in the independently owned project test directory.43 added independent tests tested typed single-parent admission, exact original-roster joins, import gating, true O_RDWR existing mutex opens with unchanged lock/ledger bytes,450GB warning projection and disk headroom failures, outer-log inclusion in the2GB cap, write-once confinement,64KiB failure evidence, all-role full occurrence partitions, seeded membership and genuine author truncation/token-row IDs. Mocked execute stages demonstrate reachability with in-memory root-shaped active admission and unchanged execution_enabled=false, and exact tree-format failure preservation before native preparation. No active permit was created.

Final guarded suite:83 total,82 passed,1 failed. The failure exposes the unconditional pair-cost guard gap described in concerns.json. In the infeasible4095 branch,41,028,000,000 measured exhaustive pairs return a source-budget negative rather than required cost rejection. This does not permit fitting or retrieval. A local unconditional check against the same frozen40,000,000,000 threshold should precede the feasibility branch.

Exact reviewed hashes, all20 sealed native-function ASTs, original preparation-tail AST c14c253190b51c3f8ad0576647257f4ca4e7fadb46b80b620293fa885bc67617 and60-author-file tree2295d86ace359ac1a7d0522c1001b6f2822b30cf9b16200517ea3473f234ca5b remain unchanged. This dispatcher avoids the inactive capsule's unguarded writer/loader and derives actual counts from full split/roster/joins.

Scope and factual limitations: source and handmade fixtures only. No actual observation/tree/grid/model/target array was decoded, no native preparation was run, no native CPU runtime was inspected, no live ledger/controller/mutex state was accessed, no GPU was queried, no active project permit was issued, no package was installed, and Git was not used. Writer tskit1.0.3 -> reader0.6.4 compatibility, actual280 sample identities/ages/site counts, actual full corpus/support/counts/token fingerprints/cost and actual storage capacity remain root-canary obligations. The2GB cap is sampled at stage/write boundaries and includes declared outer launch log plus caches/TMP; it is not an OS quota or transient-peak certificate. Global projection is historical audited baseline plus current reservations/external reserve, not a fresh whole-project byte scan.

BLOCKED is bound to exact source/config/bridge and is not approval for actual use, fitting, retrieval, remaining-parent sequence, or AmericanAdmixture paper reproduction. Only own review artifacts were written.
"""
(OWN/"review.txt").write_text(review)
receipt={
 "kind":"official_argformer_tsinfer_parent_v1_independent_source_review",
 "review_role":"independent_reviewer","verdict":"BLOCKED",
 "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "review_scope":"Exact dispatcher/config/bridge/frozen source byte and AST closure plus independent guarded handmade CPU fixtures only",
 "implementation":ref(ROOT/"scripts/prepare_official_argformer_tsinfer_parent_v1.py"),
 "config":ref(ROOT/"configs/official_argformer_tsinfer_parent_v1.json"),"bridge":c["bridge"],
 "reviewed_author_receipt":ref(ROOT/"research/reviews/official_argformer_tsinfer_parent_v1_author/review_receipt.json"),
 "sealed_bridge_independent_review":ref(ROOT/"research/reviews/official_argformer_tsinfer_biological_v2_independent/review_receipt_final.json"),
 "condition":c["condition"],"combined_observed_manifest":c["combined_observed_manifest"],
 "tests":{"authored_replayed":40,"independent":43,"total":83,"passed":82,"failures":1,"errors":0,"skipped":0,"failing_tests":failures,"fake_only_and_write_read_guarded":True},
 "concerns":concerns,
 "native20_function_ASTs":source["native20_function_ASTs"],
 "native_preparation_tail_AST_sha256":source["native_preparation_tail_AST_sha256"],
 "original60_author_tree":source["original60_author_tree"],
 "frozen_sources_unchanged":True,"default_source_mode_arrays_runtime_ledger_access":False,
 "one_parent_root_authority_before_native_imports_or_writes_source_verified":True,
 "unchanged_inactive_bridge_no_longer_permanently_blocks_separate_dispatcher":True,
 "production_execute_or_actual_native_preparation_run":False,
 "handmade_active_memory_gate_to_execute_stage_reachable":True,
 "original_tree_format_failure_preservation_mock_verified":True,
 "original_tree_format_actual_compatibility_certified":False,
 "genuine_author_tokenizer_handmade_ID_join_verified":True,
 "no_query_truth_including_TRAIN_source_and_handmade_verified":True,
 "actual_corpus_counts_tokens_occurrences_optimizer_updates_or_predictions_certified":False,
 "true_RDWR_existing_mutex_flags_and_unchanged_bytes_handmade_verified":True,
 "storage_cap":"2GB sampled stage/write boundaries plus declared outer launch log; not OS quota or continuous transient peak bound",
 "global_storage_projection":"Historical audited baseline plus current reservations/controllers130GB external reserve; not a fresh project-wide byte scan",
 "limitations":["Source and handmade CPU fixture verdict only; no actual use approval.","Fixed native CPU Python/NumPy/tskit/Torch environment and package origins were not inspected.","Actual writer1.0.3 tree readability in fixed tskit0.6.4 remains untested and must preserve failure without conversion or upgrade.","Actual280 identities/ages/sites, complete full-path corpus/counts/tokens/joins,4096 feasibility and exhaustive cost remain unmeasured.","No actual model, fit, embeddings, retrieval, metrics or original-paper reproduction certified."],
 "mutations":{"authored_sources":False,"frozen_author_source":False,"observation_or_native_preparations":False,"live_runtime_ledger_GPU_controller":False,"active_project_permits":False,"installs_Git":False,"owned_review_artifacts_only":True},
 "approval":{"actual_use":False,"native_fit_or_retrieval":False,"root_resource_and_registration_admission":False},
 "reviewed_handoff_sha256":pins["reviewed_handoff_sha256"],
 "artifact_sha256":{}
}
for p in sorted(OWN.iterdir()):
 if p.is_file() and p.name not in {"review_receipt.json","review_receipt.sha256"}:
  receipt["artifact_sha256"][str(p.relative_to(ROOT))]=sha(p)
receipt["initial_source_snapshot_sha256"]={str(p.relative_to(ROOT)):sha(p) for p in sorted((OWN/"initial_source_snapshot").iterdir())}
(OWN/"review_receipt.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
(OWN/"review_receipt.sha256").write_text(sha(OWN/"review_receipt.json")+"  research/reviews/official_argformer_tsinfer_parent_v1_independent/review_receipt.json\n")
print(json.dumps({"verdict":receipt["verdict"],"tests":receipt["tests"],"receipt":ref(OWN/"review_receipt.json")},sort_keys=True))
