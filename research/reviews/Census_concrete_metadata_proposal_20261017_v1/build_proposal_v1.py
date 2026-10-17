"""Source-only concrete Census metadata proposal. No guard or operational imports."""
from pathlib import Path
import json,hashlib,stat
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=ROOT/"research/reviews/Census_concrete_metadata_proposal_20261017_v1"
SOURCE=ROOT/"research/reviews/Census_inspect_inline_ROOT_orchestration_source_20261017_v1"
RESERVE="research/reviews/C3_remaining_cells_GPU0_ROOT_predict_val_000_root01_oct17/reserve_receipt.json"
RESERVE_SHA="66e40f49c8045ca0e6cf492cb0375e5f7a5b69a36743d3263ac50bdd83a59b20"
CAPTURE="research/reviews/Census_observation_ROOT_actual_inspect_retention_20261017_v1/fresh_capture.root_v1.json"
CAPTURE_SHA="a203f211e5927c68d85026f424d26734d653dc966c7c2cca5785d38cbb15974c"
def ref(path):
 p=ROOT/path
 assert p.resolve()==p and stat.S_ISREG(p.lstat().st_mode) and p.lstat().st_nlink==1
 return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(path,sha=None):
 r=ref(path)
 assert sha is None or r["sha256"]==sha,(path,r["sha256"])
 return json.loads((ROOT/path).read_text())
def put(name,value):
 data=(json.dumps(value,sort_keys=True,indent=2)+"\n").encode()
 assert len(data)<=65536,(name,len(data))
 with (OUT/name).open("xb") as f:f.write(data)
 return ref(str((OUT/name).relative_to(ROOT)))
reserve=read(RESERVE,RESERVE_SHA)
capture=read(CAPTURE,CAPTURE_SHA)
prefix=reserve["initial_ledger_pins"]
roles=capture["guard_roles"]
assert len(roles)==47 and prefix["completed_overlay_prefix"]==dict(events=6,tail_sha256="1aaf8cc65c7f28b2fa8bea43cfa04d12d8aed963c22cd156297255dee8859baf")
assert reserve["projected_bytes"]==949489576510
cb={r["path"]:r["sha256"] for r in reserve["value"]["consumer_bindings"]}
assert all(cb.get(r["path"])==r["sha256"] for r in roles.values())
templates={}
for a,b in [
 ("ROOT_instruction.INACTIVE.template.v1.json","ROOT_instruction.PROPOSED_INACTIVE.v1.json"),
 ("ROOT_actual_operation_intent.INACTIVE.template.v1.json","ROOT_actual_operation_intent.PROPOSED_INACTIVE.v1.json"),
 ("whole_join_review.INACTIVE.template.v1.json","whole_join_review.PROPOSED_INACTIVE.v1.json")]:
 o=json.loads((SOURCE/a).read_text())
 o["expected_before"]=prefix
 o["expected_guard_roles"]=roles
 if "active" in o:assert o["active"] is False
 if "verdict" in o:
  assert o["independent"] is False and o["verdict"]=="UNREVIEWED_INACTIVE"
  o["blockers"]=[
   "Genuine independent complete source and concrete whole-join review remains required.",
   "Prefix6 is the last known recorded after-reserve prefix. ROOT must finish unresolved mutations and explicitly bind the exact latest actual receipt prefix without discovery or fallback.",
   "ROOT finite instruction and intent are inactive and have no operation authorization."
  ]
 templates[b]=o
instruction=templates["ROOT_instruction.PROPOSED_INACTIVE.v1.json"]
review=templates["whole_join_review.PROPOSED_INACTIVE.v1.json"]
intent=templates["ROOT_actual_operation_intent.PROPOSED_INACTIVE.v1.json"]
assert instruction["expected_before"]==review["expected_before"]==intent["expected_before"]
assert instruction["expected_guard_roles"]==review["expected_guard_roles"]==intent["expected_guard_roles"]
assert instruction["created_at_utc"] is instruction["expires_at_utc"] is None
assert intent["created_at_utc"] is intent["expires_at_utc"] is None
sourcepass="research/reviews/Census_inspect_inline_ROOT_orchestration_independent_20261017_v1/review.v1.json"
sp=read(sourcepass,"26669c29d426423c0083be64847616c76b70783b5f0d4b48285d2759410f4758")
assert sp["independent"] is True and sp["verdict"]=="SOURCE_PASS" and sp["blockers"]==[]
manifest_source="research/reviews/Census_inspect_inline_ROOT_orchestration_source_20261017_v1/integration_manifest.INACTIVE.template.v1.json"
integration=read(manifest_source,"3f98e6a2f7da5d6e3ddcd4dbf0ab90074f19e0d48d344609a5fc70a654682986")
assert len(integration["source_sha256"])==217
for future,original in [
 (instruction["callback_source_ref"],"research/reviews/Census_inspect_inline_ROOT_orchestration_source_20261017_v1/ROOT_authorize_inline_retirement_v1.py"),
 (intent["orchestration_source_ref"],"research/reviews/Census_inspect_inline_ROOT_orchestration_source_20261017_v1/orchestrate_v1.py"),
 (instruction["source_manifest_ref"],manifest_source)]:
 assert ref(original)["sha256"]==future["sha256"]
selected=[
 "research/reviews/Census_inspect_inline_ROOT_orchestration_source_20261017_v1/ROOT_RECIPE.v1.json",
 "research/reviews/Census_inspect_inline_ROOT_orchestration_source_20261017_v1/staging_layout.v1.json",
 "research/reviews/Census_inspect_inline_ROOT_orchestration_source_20261017_v1/source_manifest.v1.json",
 "research/reviews/Census_inspect_inline_ROOT_orchestration_source_20261017_v1/source_handoff.v1.json",
 manifest_source,sourcepass,RESERVE,CAPTURE,
 "research/reviews/Census_inspect_inline_ROOT_orchestration_source_20261017_v1/ROOT_authorize_inline_retirement_v1.py",
 "research/reviews/Census_inspect_inline_ROOT_orchestration_source_20261017_v1/orchestrate_v1.py"]
selected.extend(instruction[k]["path"] for k in ["guard_ref","helper_ref","memo_ref","memo_manifest_ref","memo_review_ref","bridge_ref","policy_ref","selector_ref","permanent_revocation_ref"])
selected.extend([
 "research/reviews/Census_retention_inline_independent_20261017_v1/review.v1.json",
 "research/reviews/Census_retention_inline_source_20261017_v1/source_manifest.v1.json",
 "research/reviews/Census_inspect_inline_ROOT_orchestration_source_20261017_v1/preservation.v1.json",
 "research/reviews/research_execution_ROOT_20261017_v27/Census_inspect_actual_outer1_before_append.root_v1.json"])
dependency_refs=[ref(p) for p in dict.fromkeys(selected)]
for r in dependency_refs:
 if r["path"] in integration["source_sha256"]:assert r["sha256"]==integration["source_sha256"][r["path"]],r
for k in ["guard_ref","helper_ref","memo_ref","memo_manifest_ref","memo_review_ref","bridge_ref","policy_ref","selector_ref","permanent_revocation_ref"]:
 assert ref(instruction[k]["path"])==instruction[k]
existing=list(roles.values())+[
 {"path":intent["instruction_path"],"sha256":None},
 instruction["source_manifest_ref"],instruction["whole_join_review_ref"],instruction["memo_review_ref"],
 instruction["helper_ref"],instruction["memo_ref"],instruction["bridge_ref"],
 instruction["selector_ref"],instruction["policy_ref"],instruction["callback_source_ref"],instruction["permanent_revocation_ref"]]
union={}
for r in existing:
 if r["path"] in union:assert union[r["path"]]==r["sha256"]
 union[r["path"]]=r["sha256"]
assert len(union)==50 and all(p not in union for p in instruction["fresh_metadata_paths"].values())
assert len(union)+len(instruction["fresh_metadata_paths"])==53<=64
outputs=[]
for name,value in templates.items():outputs.append(put(name,value))
outputs.append(put("expected_prefix.LAST_KNOWN_RECORDED.v1.json",{
 "schema":"Census_concrete_last_known_recorded_prefix_proposal_v1",
 "current_state_asserted":False,"fresh_capture":False,"source_reserve_receipt_ref":ref(RESERVE),
 "expected_before":prefix,"recorded_projected_bytes":reserve["projected_bytes"],
 "must_rebind_after_unresolved_ROOT_mutations":True,
 "rebind_rule":"ROOT supplies the exact latest actual receipt and copies its initial_ledger_pins into instruction, intent and independent whole review. Source remains unchanged. No discovery, extension or fallback.",
 "all_ancestor_prefixes_preserved":True}))
outputs.append(put("expected_guard_roles.HISTORICAL_CAPTURE.v1.json",{
 "schema":"Census_concrete_historical47roles_proposal_v1",
 "fresh_capture":False,"source_capture_ref":ref(CAPTURE),"source_capture_utc":capture["utc"],
 "expected_guard_roles":roles,"role_count":47,"unique_role_paths":40,
 "every_role_ref_matches_exact_C3v3_reserve_consumer":True,"current_state_asserted":False}))
outputs.append(put("exact_dependencies.v1.json",{
 "schema":"Census_concrete_metadata_exact_dependencies_v1","source_only":True,
 "selected_direct_ref_authentication":dependency_refs,
 "integration_source_map_entries":217,"integration_map_inherited_under_existing_SOURCE_PASS":True,
 "all217entries_rehashed_by_this_author":False,
 "future_aliases_created":False,"future_aliases":[instruction["callback_source_ref"],intent["orchestration_source_ref"],instruction["source_manifest_ref"]],
 "whole_join_source_review":ref(sourcepass),
 "current_child_source":instruction["guard_ref"],"current_selector":instruction["selector_ref"],"current_policy":instruction["policy_ref"]}))
checks={
 "schema":"Census_concrete_metadata_source_only_checks_v1","source_only":True,
 "guard_or_helper_import":False,"guard_state_accounting_mutex_process_measurement_operations":False,
 "actual_execution_directory_inspected_or_staged":False,"finite_ROOT_capability_activated":False,
 "prefix_receipt_sha_authenticated":True,"recorded_child_events":6,
 "recorded_child_tail_sha256":prefix["completed_overlay_prefix"]["tail_sha256"],
 "prefix_current_asserted":False,"ROOT_rebind_after_pending_mutations_required":True,
 "roles":47,"role_unique_paths":40,"roles_subset_exact_recorded_reserve_consumers":True,
 "existing_full_consumer_union_paths":50,"plus_three_future_paths":53,"original_consumer_ref_cap":64,
 "union_result_is_static_metadata_only_not_actual_admission":True,
 "inactive_instruction_intent_review_prefix_role_join_equal":True,
 "instruction_active":False,"intent_active":False,"review_independent":False,
 "review_verdict":"UNREVIEWED_INACTIVE","clocks":None,"actual_credit_accepted_bytes":0,
 "historical_failures_unknown_waits_and_both20GB_charges_preserved":True,
 "source_unchanged":True,"no_Git_GPU_jobs_install":True}
outputs.append(put("checks.v1.json",checks))
actions=[
 "Finish and serialize unresolved actual ROOT journal mutations. Prefix6 is explicitly last-known historical. Select an exact latest genuine reserve/close/retire receipt without calling state inside orchestration or fabricating a prefix. Rebind all three active records and independent review to that receipt's complete pins.",
 "Have an independent reviewer inspect unchanged accepted source, actual complete metadata DAG, exact all47roles and explicit latest prefix. Keep this proposal's UNREVIEWED_INACTIVE record distinct from genuine SOURCE_PASS.",
 "ROOT checks canonical names, fresh absence, own administrative accounting and immutable fixed staging only at execution preparation. No source test claims fit or absence.",
 "Stage exact byte-identical callback/orchestrator/integration manifest in original staging_layout names. Do not use alternate namespaces, old instructions or old observation/authority.",
 "Produce independent active whole review with SOURCE_PASS/independent:true/blockers:[] and exact final prefix/roles. Its digest becomes common whole_join_review_ref.",
 "Choose one new finite created_at_utc and expires_at_utc as explicit ROOT inputs, preserve them unchanged. Create active intent with exact instruction_path and no instruction hash, then hash it. Create exact original-schema active instruction with the intent ref in uncached_current_refs and that independent review ref.",
 "Invoke exact recipe CLI once only after explicit ROOT authorization, using fresh nonoptimized single-thread /usr/bin/python3 -B -s and both control hashes. Keep300s prelock and120s proof unchanged.",
 "Consume genuine terminal. On any uncertain append/persistence/wait, audit original journal, preserve all artifacts, never auto retry or infer no append. Accept per-key credit only after genuine receipt, outer0 and final original-journal acceptance.",
 "Numerical20GB Census key remains fully charged and is not retired by inspect-only operation. No native launch, science completion, deletion or benchmark truth follows."]
outputs.append(put("ROOT_actions.PROPOSED_INACTIVE.v1.json",{
 "schema":"Census_concrete_metadata_ROOT_actions_inactive_v1","operation_authorized":False,
 "ROOT_owns_actual_execution":True,"actions":actions,
 "future_metadata_directory":intent["metadata_directory"],
 "instruction_required_ROOT_fields":["active:true","root_authorizer:root_agent","created_at_utc:explicit finite ROOT input","expires_at_utc:explicit finite ROOT input","whole_join_review_ref.sha256:genuine independent review hash","uncached_current_refs[0].sha256:exact active intent hash","expected_before:complete exact latest recorded pins"],
 "intent_required_ROOT_fields":["active:true","root_authorizer:root_agent","ROOT_authorizes_one_actual_original_Census_retirement:true","created_at_utc:same explicit ROOT input","expires_at_utc:same explicit ROOT input","whole_join_review_ref.sha256:same genuine review hash","expected_before:same complete exact latest recorded pins"],
 "no_operational_source_change_required_for_prefix_rebinding":True,
 "preservation_ref":ref("research/reviews/Census_inspect_inline_ROOT_orchestration_source_20261017_v1/preservation.v1.json")}))
report="""# Inactive Census concrete metadata proposal

This package fills the three inactive original templates with the complete recorded prefix and all 47 guard roles. It grants no capability and creates no actual execution staging.

The last known after-reserve input is C3v3 reserve_receipt.json, SHA256 66e40f49c8045ca0e6cf492cb0375e5f7a5b69a36743d3263ac50bdd83a59b20. Its child prefix is 6 events with tail 1aaf8cc65c7f28b2fa8bea43cfa04d12d8aed963c22cd156297255dee8859baf and recorded projection 949489576510 bytes. This is not a fresh or current-state assertion. ROOT has unresolved mutations, so the exact latest genuine receipt must replace this prefix in all three records and the independent review after serialization. The unchanged operational sources need no modification.

All 47 roles come from the historical 2026-10-17T10:26:08.642667+00:00 capture. Each role ref exactly appears in the actual C3v3 reservation consumers. They contain 40 unique paths. The unchanged helper's complete metadata union has 50 existing paths plus its three future names, fitting the original 64-path cap as a static join. This does not establish current resource fit.

The accepted generic helper pass and callback/orchestration source pass remain distinct from the mandatory final independent concrete whole review. The proposed review remains UNREVIEWED_INACTIVE. Instruction and intent are inactive, authorizer and finite clocks are null, review and active intent digests are unfilled. These fields are genuine ROOT choices or derive from future immutable active bytes, so this author has not invented them.

No guard import, state call, accounting operation, target observation, disk measurement, authorization activation, actual staging, GPU work, installation or Git action occurred. Selected direct source pins were authenticated. The inherited217-entry map remains bound through its accepted source pass and exact manifest bytes, rather than repeating that campaign.

Original failed29922outer1, both20GB charges, original files and clocks, genuine32/32 inspection/directworkerwait0 and unknown supervisor/historical descendant/drain status are preserved. Inspect-only retirement cannot authorize numerical retirement or a native job.
"""
with (OUT/"REPORT.md").open("x") as f:f.write(report)
outputs.append(ref(str((OUT/"REPORT.md").relative_to(ROOT))))
outputs.append(ref(str((OUT/"build_proposal_v1.py").relative_to(ROOT))))
man=put("MANIFEST.v1.json",{"schema":"Census_concrete_metadata_proposal_manifest_v1","status":"INACTIVE_REQUIRES_ROOT_LATEST_PREFIX_AND_INDEPENDENT_WHOLE_JOIN_REVIEW","source_only":True,"actual_operation_authorized":False,"files":outputs,"dependency_refs":dependency_refs})
handoff=put("HANDOFF.v1.json",{"schema":"Census_concrete_metadata_proposal_handoff_v1","status":"INACTIVE_METADATA_READY_LAST_KNOWN_PREFIX_NOT_CURRENT","manifest_ref":man,"checks_ref":ref(str((OUT/"checks.v1.json").relative_to(ROOT))),"ROOT_actions_ref":ref(str((OUT/"ROOT_actions.PROPOSED_INACTIVE.v1.json").relative_to(ROOT))),"actual_staging_created":False,"source_only":True,"operation_authorized":False,"original_scopes_charges_failures_unknowns_preserved":True,"ROOT_latest_prefix_after_unresolved_mutations_required":True,"independent_whole_join_review_required":True,"finite_ROOT_clocks_and_hash_DAG_required":True})
print(json.dumps({"handoff_ref":handoff,"manifest_ref":man,"files":len(outputs)+2,"union_paths_with_future":53,"current_prefix_asserted":False,"operations":0},indent=2))
