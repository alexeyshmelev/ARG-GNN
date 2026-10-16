"""Seal independent source-only review after rechecking permitted immutable pins."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
R=Path("/workspace-SR003.nfs2/ARG-GNN")
C="research/reviews/European_outer8_current_policy_successor_source_20261016_v1"
OLD="research/reviews/European_outer8_preparation_source_20261016_v3"
B="research/reviews/European_outer8_current_policy_successor_independent_review_20261016_v1"
def ref(path):return dict(path=path,sha256=hashlib.sha256((R/path).read_bytes()).hexdigest())
def read(path):return json.loads((R/path).read_text())
def write(name,value):
 p=R/B/name
 with p.open("x")as f:json.dump(value,f,sort_keys=True,indent=2);f.write("\n")
 return ref(B+"/"+name)
handoff=ref(C+"/source_handoff.v1.json");manifest=ref(C+"/source_manifest.v1.json")
assert handoff["sha256"]=="6bfd01d40fb0e53946143e8e11634fe987fbb0fb408e1c875da767a5484174c4"
assert manifest["sha256"]=="48bbdc455db2a5047db21f63dcdc0ad636ad32fb3f1c171f9e16f66690d2fbb3"
pins=read(B+"/hash_verification.v1.json")
assert len(pins["verified"])==247 and len(pins["deferred"])==18
for row in pins["verified"]:
 assert not row["path"].startswith(("data/","experiments/results/"))
 assert ref(row["path"])==dict(path=row["path"],sha256=row["sha256"]),row["path"]
old=read(OLD+"/source_manifest.v1.json");new=read(C+"/source_manifest.v1.json")
assert len(old["refs"])==226 and len(new["refs"])==241
assert all(rr in new["refs"]for rr in old["refs"])
assert old["expected_original_metadata_and_assets"]==new["expected_original_metadata_and_assets"]
assert len(old["expected_original_metadata_and_assets"])==6
prior=ref("research/reviews/European_outer8_preparation_independent_review_20261016_v3/review.v3.json")
assert prior["sha256"]=="cc7acabdbf24548584f328b7b0115b4f7d7f5e886bc545a5a72c90bb2f43d37a"
recheck=write("final_pin_recheck.v1.json",dict(schema="European_outer8_successor_final_independent_pin_recheck_v1",status="PASS_SOURCE_ONLY",
 verified_count=247,deferred_count=18,all_verified_refs_unchanged=True,all226_original_refs_preserved_in241=True,all6_original_expected_assets_preserved_unopened=True,
 candidate_handoff_ref=handoff,candidate_manifest_ref=manifest,original_independent_review_ref=prior,
 no_deferred_data_model_result_or_original_asset_path_opened=True,no_actual_guard_state_probe_accounting_admission_or_jobs=True))
checks=read(B+"/independent_static_checks.v1.json")
selector=read(B+"/independent_selector_memo.v1.json")
consumer=read(B+"/independent_consumer_gate.v1.json")
generation=read(B+"/independent_generation_gate.v1.json")
assert all(x["status"]=="PASS_SOURCE_ONLY"for x in (checks,selector,consumer,generation))
assert consumer["complete_flows"]==4 and consumer["negative_case_count"]==60
assert generation["complete_flows"]==2 and generation["negative_case_count"]==26
evidence={}
for key,name in (("pin_verification","hash_verification.v1.json"),("final_pin_recheck","final_pin_recheck.v1.json"),
 ("author50_reproduction","author_contract_reproduction.v1.json"),("independent_static_checks","independent_static_checks.v1.json"),
 ("independent_selector_memo","independent_selector_memo.v1.json"),("independent_consumer_gate","independent_consumer_gate.v1.json"),
 ("independent_generation_gate","independent_generation_gate.v1.json"),("independent_static_source","independent_static_checks_v1.py"),
 ("independent_selector_memo_source","independent_selector_memo_v1.py"),("independent_consumer_gate_source","independent_consumer_gate_v1.py"),
 ("independent_generation_gate_source","independent_generation_gate_v1.py")):evidence[key]=ref(B+"/"+name)
created=datetime.now(timezone.utc).isoformat()
source=ref(C+"/prepare_outer000_v1.py");consumer_source=ref(C+"/check_prediction_barrier_v1.py")
scientific_manifest=ref(OLD+"/source_manifest.v1.json")
assert source["sha256"]=="5bc07b0021b1b35b4ddee9ae6dce8680b61773413593ab829869125a7b7a3c3d"
assert consumer_source["sha256"]=="f16d0461fd0bfbde5d320847d5f50f6492995b9a915dc96fdf5ce7dd40ba4f8f"
prerequisites=[
 dict(id="O1",title="Genuine prospective freeze and independent-confirmation registration",
 required="Before first outer generation, ROOT supplies the original actual immutable five-method/protocol/registry freeze and genuine H038/H050-linked independent-confirmation decision and ledger snapshots. Registration must bind the exact freeze/protocol/registry, state the repeat reason, forbid outer tuning and precede the new request. Synthetic tests and inactive templates provide no registration."),
 dict(id="O2",title="Actual current policy, direct roles and capacity",
 required="ROOT supplies the exact immutable selector for the authority actually permitted to mutate. A selected native7/H050 child requires its actual reviewed activation and frozen lineage; revoked native14 ancestors cannot be used for new mutations. ROOT alone checks current capacity and serialized reservation authority. Recompute the union from every actually returned guard plus the full operation roles; all must remain direct and at most64. This review makes no actual current-role fit, guard state, activation or headroom claim."),
 dict(id="O3",title="Actual generation and CPU consumer runtime qualifications",
 required="Supply and qualify the real original eight-CPU generation runtime, original config/producer and actual versions/CPU identities; qualify each observed-validation or scoring CPU runtime against both scientific and operational manifests and the actual entry source. Supply fresh ROOT intent, source reviews, actual existing reservation and separate scoring authority. Source contracts do not qualify any runtime or authorize execution."),
 dict(id="O4",title="Real native drivers, prior completions and full952 ARGformer fits",
 required="Each native preparation/fit/prediction needs a separately reviewed guarded wait driver, immutable writer/context, genuine prior observed-bank/cache/cut/prep completion and wait/drain/ROOT-terminal evidence, actual runtime and ROOT stage admission. Qualify full952 observed-only tsinfer/tsdate/shared-ancestor corpus and each same-parent ARGformer phase pair with unchanged batches/epochs/precision, expected/live/saved optimizer counters, actual checkpoints and directwait0/drain; do not transfer across ARG vocabularies or fabricate missing lineage. This review does not assert a ready full952 runtime or native-driver role union."),
 dict(id="O5",title="Complete actual products and separate scoring",
 required="All8 biological parents, all40 fixed primary cells and all8 independent prespecified rawtop5 products with full support and genuine native execution/checkpoint lineage are required before any target boundary. Missing or failed cells block the complete comparison. Separate ROOT scoring must use the unchanged validator/scorer and provenance, equal-parent Present182 query-bp C4 accuracy/NLL, sample SD/units, paired differences, all686/ancient strata and coverage/failure reports. No actual validation, score or scientific result follows from this source-only verdict.")
]
report=dict(schema="European_outer8_current_policy_successor_independent_source_review_v1",created_at_utc=created,
 reviewer="fresh_independent_native_v9_source_reviewer",verdict="SOURCE_PASS",
 scope="Narrow operational successor to the sealed original European outer8 v3. Authenticated permitted source/compact metadata; exact-code tests used synthetic modules, authority documents and virtual filesystems only. Deferred data/model/result/original-asset refs were compared through the preserved inherited manifests and left unopened.",
 source_ref=source,consumer_source_ref=consumer_source,source_manifest_ref=manifest,operational_manifest_ref=manifest,source_handoff_ref=handoff,
 inherited_scientific_manifest_ref=scientific_manifest,inherited_independent_review_ref=prior,
 fidelity_delta_ref=ref(C+"/fidelity_delta.v1.diff"),recipe_ref=ref(C+"/ROOT_RECIPE.txt"),
 findings=[],source_blockers=[],completion_and_execution_blockers=prerequisites,
 component_verdicts=dict(narrow_operational_delta="SOURCE_PASS",first_parent_generation="SOURCE_PASS_DEFAULT_INACTIVE",
 explicit_selector_before_memo="SOURCE_PASS_SYNTHETIC_EXACT_CODE",CPU_consumer_entry="SOURCE_PASS_DEFAULT_INACTIVE",
 original_scientific_sources="UNCHANGED_INHERITED_V3_SOURCE_PASS",complete_outer8_benchmark="INCOMPLETE_NOT_SCIENTIFICALLY_READY"),
 verification=dict(authenticated_source_compact_ref_count=247,deferred_unopened_ref_count=18,original_manifest_ref_count=226,new_manifest_ref_count=241,
 original_expected_assets_unchanged_count=6,author_contracts_reproduced_exactly=50,independent_static_groups=checks["check_count"],
 exact_real_selector_memo_choices=selector["check_count"],complete_virtual_consumer_flows=4,consumer_negative_cases=60,
 complete_virtual_generation_flows=2,generation_negative_cases=26,inactive_default_preparation_exit=0,inactive_default_consumer_exit=2,
 exact64_binding_cases_pass=True,complete65_binding_cases_reject=True,missing_actually_returned_guard_rejects=True,final_immutable_pin_recheck="PASS_SOURCE_ONLY"),
 operational_conclusions=[
  "The published delta is exact and operational. Original configuration, producer, helper, registry, method identities, protocol, model selection, native adapters/algorithms, validator and scorer retain original paths and digests. Every original v3 manifest ref and expected asset pin is preserved.",
  "The exact original worker/lifecycle code changes only the selected guard argument. Original first-parent952sample/686query/266reference/full2Mb/Relate5size generation remains, with4GiB whole reservation,4000000000scientific bytes,128MiBAUX,8CPU,32GiBRSS polling stop,43200s wall,7200s native-step cap and30s cleanup. No scientific workload reduction or retry/reseed/resume is introduced.",
  "The new wrapper requires an explicit pinned ROOT current/native7-H050 selection. Exact real selector code configures the synthetic selected module once before the exact real consumer namespace memo. The wrapper then reuses its cached selection and guard after memo without selector reentry or reinstall. Different ref, policy, mode, ceiling or an initially wrapped namespace is rejected.",
  "The complete source-derived models retain every returned guard and every required operation ref directly. Parent37raw keys represent30unique refs; child45raw keys represent38unique refs. The resulting generation/validation/scoring unions are52/53/56 or59/60/63. Unique exact path/SHA records are deduplicated; raw keys are not silently dropped. Unexpected actual roles must fit64 or block; no actual current role fit is established here.",
  "Generation request/admission now requires supplied actual runtime qualification and prospective freeze/registration snapshots. It does not manufacture runtime qualification. In virtual flows there is one serialized reserve with the exact prior pins, original caps and complete direct role set; execution remains separate.",
  "The consumer separately binds the original scientific manifest and new operational manifest, exact new entry, actual ROOT intent, freeze/registration, qualified CPU runtime and source review. Under the synthetic wrapped mutex it obtains fresh=False returned state, reconstructs the complete guard+operation union, requires every ref directly in the existing reservation and applies storage/writer boundaries.",
  "After normalizing only absolute imports, run_v3 is AST-identical to the original. Original validator enforces all8parents and full5by8 Cartesian primary cells plus eight independent ARGformer rawtop5 products. Original scorer repeats complete40+8 matrix validation and inherited metric/provenance preflight before every target boundary. No partial score path or changed metric/model selection is introduced.",
  "No-argument preparation returns SOURCE_ONLY_INACTIVE exit0. No-argument consumer returns INACTIVE_SOURCE_ONLY exit2. All authority/registration/qualification templates remain inactive. The recipe explicitly disclaims ready full952ARGformer runtime and requires genuine later native drivers, prerequisites, reviews, capacities and ROOT admissions."
 ],
 direct_role_models=checks["checks"][3]["detail"],
 scientific_readiness=False,full952_ARGformer_runtime_ready=False,actual_runtime_native_driver_context_qualified=False,
 actual_policy_activation_or_current_guard_role_fit_established=False,current_capacity_established=False,
 actual_observed_matrix_validated=False,actual_scientific_results_produced=False,
 admission_authorized=False,execution_authorized=False,scoring_authorized=False,
 actual_guard_configure_probe_state_accounting_measurement_activation_retention=False,
 datasets_outer_paths_arrays_truth_predictions_models_checkpoints_opened=False,
 actual_resource_process_lease_GPU_probe_or_job=False,Git_or_install_or_candidate_source_modification=False,
 verification_limits="All guard/budget modules and state, affinity, registrations, runtime qualifications, policies and payloads in behavioral tests are synthetic. The current fixed policy identifier and original scientific manifest identifier are virtualized to synthetic metadata in the relevant VFS tests; real hash/source closure authentication is established separately by the247pin checks. These tests are not actual policy, reservation, runtime, data or scientific evidence. The original full scientific review is reused only for byte-identical scientific sources.",
 evidence_refs=evidence)
main=write("review.v1.json",report)
entry_report=dict(schema="European_outer8_current_policy_consumer_independent_source_review_v1",created_at_utc=created,reviewer=report["reviewer"],verdict="SOURCE_PASS",
 source_ref=consumer_source,preparation_source_ref=source,source_manifest_ref=manifest,operational_manifest_ref=manifest,
 inherited_scientific_manifest_ref=scientific_manifest,source_handoff_ref=handoff,comprehensive_review_ref=main,
 scope="Exact new CPU observed-validation/scoring entry and its selected-policy operational closure. See comprehensive review for source-derived checks, exact-code synthetic test evidence, unchanged scientific40+8pretruth barriers and all operational/scientific prerequisites.",
 findings=[],source_blockers=[],scientific_readiness=False,actual_runtime_qualified=False,actual_observed_matrix_validated=False,
 current_capacity_or_actual_guard_role_fit_established=False,admission_authorized=False,execution_authorized=False,scoring_authorized=False,
 actual_guard_probe_accounting_data_model_truth_job_or_Git=False)
entry=write("consumer_review.v1.json",entry_report)
artifact_names=["hash_verification.v1.json","final_pin_recheck.v1.json","author_contract_reproduction.v1.json",
 "independent_static_checks_v1.py","independent_static_checks.v1.json","independent_selector_memo_v1.py","independent_selector_memo.v1.json",
 "independent_consumer_gate_v1.py","independent_consumer_gate.v1.json","independent_generation_gate_v1.py","independent_generation_gate.v1.json",
 "seal_review_v1.py","review.v1.json","consumer_review.v1.json"]
inventory=write("review_handoff.v1.json",dict(schema="European_outer8_current_policy_successor_independent_review_handoff_v1",created_at_utc=created,status="SEALED_SOURCE_PASS_ONLY",
 source_handoff_ref=handoff,source_manifest_ref=manifest,review_ref=main,consumer_entry_review_ref=entry,
 no_source_defects_found=True,remaining_prerequisites=prerequisites,scientific_readiness=False,admission_authorized=False,
 execution_authorized=False,scoring_authorized=False,actual_guard_or_capacity_fit_claimed=False,artifacts=[ref(B+"/"+name)for name in artifact_names]))
print(json.dumps(dict(verdict="SOURCE_PASS",review_ref=main,consumer_review_ref=entry,review_handoff_ref=inventory,remaining_prerequisite_count=len(prerequisites)),sort_keys=True))
