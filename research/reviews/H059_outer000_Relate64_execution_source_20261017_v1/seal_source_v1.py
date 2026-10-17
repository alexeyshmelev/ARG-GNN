"""Compact source/metadata hash sealing only. No imports of project/runtime/guards."""
from pathlib import Path
import ast,hashlib,json,stat
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H059_outer000_Relate64_execution_source_20261017_v1"
OLD="research/reviews/European_outer8_current_policy_successor_source_20261016_v1"
H=ROOT/HERE
def rr(path):
 p=ROOT/path;s=p.lstat()
 if not(p.resolve()==p and p.is_relative_to(ROOT)and stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_size<=32*2**20 and p.suffix in{".py",".sh",".json",".md",".diff",".patch"}):raise ValueError("Compact source/metadata only: "+path)
 return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(path):return json.loads((ROOT/path).read_text())
def put(name,value):(H/name).write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")
b=read(HERE+"/source_bindings.v1.json");result=read(HERE+"/source_check.result.v1.json");base=read(OLD+"/source_manifest.v1.json");refs={}
def add(record):
 path,sha=record["path"],record["sha256"]
 if path in refs and refs[path]!=sha:raise ValueError("Conflicting preserved pin: "+path)
 refs[path]=sha
for r in base["refs"]:add(r)
for k,v in b.items():
 if isinstance(v,dict)and set(v)=={"path","sha256"}:add(v)
selector=read(b["actual_selector_ref"]["path"]);guardmanifest=read(selector["source_manifest_ref"]["path"])
for path,sha in guardmanifest["source_sha256"].items():add(dict(path=path,sha256=sha))
for r in guardmanifest["runtime_authenticated_historical_source_and_metadata_refs"]:add(r)
for k in("source_manifest_ref","source_review_ref","measurement_source_ref","repair_authority_ref","policy_ref"):add(selector[k])
add(read(selector["policy_ref"]["path"])["root_activation_ref"])
for r in read(b["guard_role_snapshot_ref"]["path"])["bindings"].values():add(r)
for p in H.iterdir():
 if p.is_file()and p.name not in("source_manifest.v1.json","source_handoff.v1.json","audit_manifest.v1.json"):add(rr(str(p.relative_to(ROOT))))
expected=base["expected_original_metadata_and_assets"]
for r in expected:
 if Path(r["path"]).suffix==".txt" and r["path"]in refs:raise ValueError("Numeric map/age payload unexpectedly included as source: "+r["path"])
for path,sha in refs.items():
 if rr(path)["sha256"]!=sha:raise ValueError("Preserved source/metadata SHA mismatch: "+path)
 if path.endswith(".py"):ast.parse((ROOT/path).read_text(),filename=path)
manifest=dict(schema="H059_outer000_Relate64_complete_operational_source_manifest_v1",source_only=True,sealed=True,execution_authorized=False,admission_authorized=False,independent_review_completed=False,
 refs=[dict(path=p,sha256=s)for p,s in sorted(refs.items())],expected_original_metadata_and_assets=expected,original_numeric_map_age_asset_refs_carried_without_reads_or_hashes_in_source_task=True,all8_original_parents_and_all40_primary_plus8_top5_preserved=True,model_weight_refs_never_opened=True,no_returned_current_guard_role_may_be_aggregated_or_omitted=True,historical_only_original_refs_authenticated_inside_sealed_manifest=True,predecessor_manifest_ref=b["predecessor_manifest_ref"],
 cycle_exclusions=[HERE+"/source_manifest.v1.json",HERE+"/source_handoff.v1.json",HERE+"/audit_manifest.v1.json"],source_sha256_count=len(refs),source_bindings_ref=rr(HERE+"/source_bindings.v1.json"),source_check_ref=rr(HERE+"/source_check.result.v1.json"))
put("source_manifest.v1.json",manifest)
handoff=dict(schema="H059_outer000_Relate64_inactive_complete_source_handoff_v1",status="SEALED_SOURCE_ONLY_INDEPENDENT_REVIEW_REQUIRED",source_only=True,source_ref=rr(HERE+"/prepare_outer000_v1.py"),source_manifest_ref=rr(HERE+"/source_manifest.v1.json"),source_map_ref=rr(HERE+"/source_map.v1.json"),source_bindings_ref=rr(HERE+"/source_bindings.v1.json"),source_checker_ref=rr(HERE+"/source_check_v1.py"),source_check_ref=rr(HERE+"/source_check.result.v1.json"),ROOT_recipe_ref=rr(HERE+"/ROOT_RECIPE.md"),report_ref=rr(HERE+"/REPORT.md"),caps_ref=rr(HERE+"/caps.INACTIVE.v1.json"),new_module=HERE.replace("/",".")+".prepare_outer000_v1",source_metadata_files_verified=len(refs),preserved_predecessor_refs=len(base["refs"]),original_expected_assets_declared=len(expected),numeric_map_age_assets_declared_without_reads=sum(Path(r["path"]).suffix==".txt"for r in expected),actual_selector_ref=b["actual_selector_ref"],whole_scope_bytes=4294967296,scientific_output_cap_bytes=4000000000,worker_aux_cap_bytes=134217728,packet_admin_cap_bytes=8388608,full_role_union_metadata_only=result["full_role_union_count"],historical_named_guard_roles=46,historical_distinct_guard_roles=39,role_ceiling=64,original_science_unchanged=True,frozen_methods=["H050_stable_leaf_memory","H050_h045_continued_training","AncestralPaths_native_Dense5","ARGMix_current_native","ARGformer_native_within_ARG"],H058_automatically_included=False,all8parents_all40primary_plus8rawtop5_before_truth=True,registration_ref=b["registration_ref"],frozen_method_freeze_ref=b["frozen_method_freeze_ref"],original_runtime_qualification_ref=b["inherited_runtime_qualification_ref"],actual_guard_state_resources_processes_capacity_key_absence_or_scope_fit_observed=False,actual_guard_runtime_science_import_payload_probe_admission_journal_mutex_GPU_install_or_Git=False,independent_review_completed=False,admission_or_execution_authorized=False,
 next_ROOT_requirements=["Genuine independent complete source review","Explicit permanent cooperative old preparation-route revocation","Genuine current-source runtime compatibility acceptance backed by original qualification and actual CPU metadata","Fresh unused-key/complete-target absence, source/current roles/resource/policy validation","Explicit ROOT finite authority; real admission0 before exact one-shot emitted setsid argv"],
 later_prediction_blockers=["Actual native method drivers/writers and genuine same-parent observed preparation/wait lineage","Full952 observed-only ARGformer runtime, native per-parent5epochMLM+1epochcontrastivefits and checkpoint pairs","All40primary+8rawtop5products plus independently reviewed actual matrix validator and separate ROOT scoring authority"],scientific_performance_fit_or_original_paper_reproduction_claim=False)
put("source_handoff.v1.json",handoff)
audit={str(p.relative_to(ROOT)):rr(str(p.relative_to(ROOT)))["sha256"]for p in sorted(H.iterdir())if p.is_file()and p.name!="audit_manifest.v1.json"}
put("audit_manifest.v1.json",dict(schema="H059_outer000_Relate64_source_audit_v1",source_only=True,independent=False,files=audit,files_count=len(audit),source_manifest_ref=rr(HERE+"/source_manifest.v1.json"),source_handoff_ref=rr(HERE+"/source_handoff.v1.json"),all_source_metadata_refs_verified=len(refs),original_numeric_public_assets_models_arrays_trees_checkpoints_not_read=True,admission_or_execution_authority=False))
print(json.dumps(dict(driver=rr(HERE+"/prepare_outer000_v1.py"),manifest=rr(HERE+"/source_manifest.v1.json"),handoff=rr(HERE+"/source_handoff.v1.json"),roles=result["full_role_union_count"],source_files=len(refs)),indent=2))
