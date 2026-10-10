"""Seal repaired exact-source PASS; actual execution remains root-owned and unapproved."""
import datetime
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
BASE=ROOT/"research/reviews/official_argformer_tsinfer_parent_v1_independent"
OWN=BASE/"repair_v2"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {"path":str(p.relative_to(ROOT)),"sha256":sha(p)}
initial_path=BASE/"review_receipt.json"
assert sha(initial_path)=="e3df087b46b4616c6aa5f088dfaa32ab17367313a727b1d4b7379e3dcd2d11af"
initial=json.loads(initial_path.read_text())
audit=json.loads((OWN/"source_audit.json").read_text())
assert all(sha(ROOT/n)==h for n,h in audit["reviewed_handoff_sha256"].items())
assert all(sha(ROOT/n)==h for n,h in initial["artifact_sha256"].items())
assert all(sha(ROOT/n)==h for n,h in initial["initial_source_snapshot_sha256"].items())
cases=ET.parse(OWN/"pytest.xml").getroot().findall(".//testcase")
assert len(cases)==90 and all(n.find("failure") is None and n.find("error") is None and n.find("skipped") is None for n in cases)
review="""Repaired independent source and handmade CPU review: PASS for exact dispatcher9ac0ca01b214e86841808fee951fb55e74bf57bac7227656d6ee7410d9c8a8ed and unchanged config9e5212c6de07bb6c28e04f644b336faade04f1e06d6d488f5af49739e2529c55.

The initial ecd25... review remains immutable and BLOCKED. The revised dispatcher adds precisely one unconditional full-corpus cost guard before source-batch feasibility. Removing this guard restores the original budget() AST. Every other dispatcher AST is identical. Config, bridge, all frozen sources, native20 ASTs, original preparation-tail AST, genuine tokenizer methods and60-author-file source tree remain unchanged.

Final guarded suite:90 passed (42 updated author tests and48 independent tests),0 failures/errors/skips. The original43 independent checks replay unchanged. Added tests verify exact40bn equality, one-over failure with4095 and4096 pure training paths, and mocked execute preservation of completed preparation plus failed_partial_preserved exit1 on excessive cost. The previously failing exact90/10/64-mixed/2.4m-query fixture now rejects41.028bn pairs. Under-cost source4096 infeasibility continues to return its preserved negative exit2 without reducing budgets/corpus. Initial40 author checks remain an exact byte prefix of the updated author test file.

Other reviewed guarantees retain the earlier evidence: typed exact single-parent root scope before native imports/writes, source/config/review/runtime/input/output/registration bindings, separated executable dispatcher with unchanged inactive bridge, true O_RDWR existing mutex descriptor with no create/truncate/byte write, full ledger/controller/external reserve accounting, shared2GB output/cache/TMP/outer-log sampled cap, write-once output confinement, bounded failure evidence, exact original observed-roster mapping, full paths before truncation, full96 pure-reference train-plus-VAL corpus, genuine author token rows/raw leaf IDs, and complete all-role half-open occurrence/readout partitions. Mocked execute can reach the separate admitted preparation stage, and simulated incompatible tree format stops before native preparation while preserving the error.

Scope and limitations: source and handmade CPU fixtures only. No actual observation/tree/grid/model/target array was decoded, no native preparation was run, no fixed native CPU runtime/package origin was inspected, no live ledger/controller/mutex was accessed, no GPU query/job/context was created, no active project permit was issued, no install was performed, and Git was not used. Actual tskit1.0.3-written trees read by fixed native0.6.4 remain untested; first actual canary must retain that exact file load and preserve failure without conversion/upgrade/fallback. Actual280 identities/ages/sites, corpus/counts/tokens/joins/4096 feasibility/pair cost and current resource state remain root obligations.

The2GB boundary is sampled at stage/write boundaries including declared launch log, not an OS quota or continuous transient bound. Global projection uses historical audited baseline plus current reservations/controllers and130GB external reserve, not a fresh whole-project byte scan.

PASS is exact-source/handmade only. It is not actual-use, fit, retrieval, remaining-parent, metric or original AmericanAdmixture paper-reproduction approval. Root alone owns actual runtime/resource/experiment admission and any canary.
"""
with (OWN/"review.txt").open("x") as stream:stream.write(review)
receipt=dict(initial)
receipt.update({
 "verdict":"PASS","created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "review_scope":"Repaired exact dispatcher/config/bridge/frozen byte/AST closure plus guarded author replay and independent handmade CPU fixtures only",
 "implementation":ref(ROOT/"scripts/prepare_official_argformer_tsinfer_parent_v1.py"),
 "config":ref(ROOT/"configs/official_argformer_tsinfer_parent_v1.json"),
 "reviewed_author_receipt":ref(ROOT/"research/reviews/official_argformer_tsinfer_parent_v1_author/repair_v2/review_receipt.json"),
 "tests":{"authored_replayed":42,"independent":48,"total":90,"passed":90,"failures":0,"errors":0,"skipped":0,"failing_tests":[],"fake_only_and_write_read_guarded":True},
 "concerns":[],
 "resolved_concerns":[{"id":"P1","initial_source_review":ref(initial_path),"repair":"One unconditional full-corpus cost check against unchanged frozen40bn limit precedes feasibility; under-cost negative semantics unchanged.","independent_tests":"Original over-cost regression,40bn equality and one-over across4095/4096, mocked execute partial-preparation cost-failure preservation all pass."}],
 "supersedes_historical_source_review":ref(initial_path),
 "initial_BLOCKED_source_review_and_all_artifacts_preserved":True,
 "reviewed_handoff_sha256":audit["reviewed_handoff_sha256"],
 "repaired_only_budget_AST":True,
 "native_and_preparation_source_ASTs_unchanged":True,
 "conditional_budget_feasibility_cannot_bypass_unconditional_cost_guard":True,
 "exact40bn_boundary_admitted_for_counting_and_above_rejected":True,
 "over_cost_partial_preparation_and_failure_receipt_mock_verified":True,
 "actual_use_approved":False,
 "artifact_sha256":{},
 "source_snapshot_sha256":audit["source_snapshot_sha256"],
 "independent_guard":ref(BASE/"conftest.py"),
})
# Historical snapshots remain referenced through the immutable initial receipt.
receipt.pop("initial_source_snapshot_sha256",None)
for p in sorted(OWN.iterdir()):
 if p.is_file() and p.name not in {"review_receipt.json","review_receipt.sha256"}:
  receipt["artifact_sha256"][str(p.relative_to(ROOT))]=sha(p)
with (OWN/"review_receipt.json").open("x") as stream:stream.write(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
with (OWN/"review_receipt.sha256").open("x") as stream:stream.write(sha(OWN/"review_receipt.json")+"  research/reviews/official_argformer_tsinfer_parent_v1_independent/repair_v2/review_receipt.json\n")
print(json.dumps({"verdict":receipt["verdict"],"implementation":receipt["implementation"],"config":receipt["config"],"receipt":ref(OWN/"review_receipt.json"),"tests":receipt["tests"]},sort_keys=True))
