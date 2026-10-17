"""Seal source/metadata only inside this package. Never import guards or entrypoints."""
from pathlib import Path
import ast,difflib,hashlib,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");HERE="research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v3";H=ROOT/HERE
def r(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(record):
 actual=r(record["path"]);assert actual==record,record;return json.loads((ROOT/record["path"]).read_text())
def write(name,value):(H/name).write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")
def function(path,name):
 source=Path(path).read_text();n=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name==name);return n,ast.get_source_segment(source,n)
def sha_ast(node):return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
def main():
 b=json.loads((H/"source_bindings.v1.json").read_text())
 checker=H/"metadata_checks_v1.py";scope={"__name__":"SOURCE_ONLY_SEAL_MEMORY_CHECKS","__file__":str(checker)}
 exec(compile(checker.read_text(),str(checker),"exec"),scope);result=scope["main"]()
 assert result["status"]=="PASS_SOURCE_ONLY"and result["independent"]is False
 write("metadata_checks.result.v1.json",result)
 write("handwritten_role_unions.INACTIVE.fixture.v1.json",dict(schema="FLARE_all_original_roles_handwritten_fixture_v2",source_only_fixture=True,current_guard_query=False,original64cap=64,metadata_origin_ref=b["actual_guard_metadata_ref"],ref_placeholders_explicit=True,roles=result["handwritten_roles"]))
 original=b["original_driver_ref"]["path"];fw="research/reviews/American_TreeLoom_matched_CPU_execution_source_20261017_v4/root_driver_v1.py"
 oldn,olds=function(ROOT/fw,"execute");newn,news=function(H/"root_driver_v1.py","owned_execute")
 delta="".join(difflib.unified_diff(olds.splitlines(True),news.splitlines(True),fromfile=fw+"#execute",tofile=HERE+"/root_driver_v1.py#owned_execute"))
 delta+="".join(difflib.unified_diff((ROOT/b["original_worker_ref"]["path"]).read_text().splitlines(True),(H/"worker_entry_v1.py").read_text().splitlines(True),fromfile=b["original_worker_ref"]["path"],tofile=HERE+"/worker_entry_v1.py"))
 (H/"source_delta.v1.patch").write_text(delta)
 write("executor_AST_delta.v1.json",dict(schema="FLARE_phase_actual_AST_delta_v2",independent=False,original_framework_ref=r(fw),original_driver_ref=b["original_driver_ref"],original_worker_ref=b["original_worker_ref"],common_driver_ref=r(H/"root_driver_v1.py"),worker_ref=r(H/"worker_entry_v1.py"),original_execute_AST_sha256=sha_ast(oldn),new_owned_execute_AST_sha256=sha_ast(newn),delta_ref=r(H/"source_delta.v1.patch"),original_cleanup_wait_drain_AST_identical=True,original_helpers_extracted_unchanged=["input_spec","worker_python","sample_scope","boundary"],explicit_deltas=["Common contract/driver identity; exact original contract/admit/completion functions rebound to fresh source identities and actual selector","Fixed7200 setup starts before metadata; genuine READY/release owns full original scientific/work windows; combined and authority caps","Initial original storage prelude plus one check_storage_locked_v1 transaction returning actual bindings; unchanged consumer memo/final checks","Owned bootstrap gate retained; second science grant and bounded authenticated READY/scientific-entry receipts added","Actual cancellation and distinct setup/work/authority/evidence timing; original cleanup30 retained","Scalar progress wrappers call exact original WaitWriter/Writer bodies and stage execute without numerical changes","Fresh namespace and exact old-failure acceptance joins; all original direct roles plus accepted failure retained under64"],scientific_source_sha256=result["scientific_source_sha256"]))
 # Sources already pinned by the original scientific/framework package and selector/child packages.
 actual=read(b["actual_selector_metadata_ref"]);manifest_refs=[b["original_manifest_ref"],b["selector_protocol_manifest_ref"],actual["source_manifest_ref"],b["predecessor_v2_source_manifest_ref"]]
 sources={};metadata={}
 def add(record):
  assert set(record)=={"path","sha256"};p=record["path"]
  assert not p.startswith(".runtime/storage/")and p not in(".runtime/gpup/storage_reservations.json",)
  assert r(p)==record,record
  metadata[p]=record
 def merge(path,sha):
  assert path not in sources or sources[path]==sha
  assert r(path)["sha256"]==sha,path;sources[path]=sha
 for record in manifest_refs:
  m=read(record);add(record)
  for path,sha in m.get("source_sha256",{}).items():merge(path,sha)
  for rr in m.get("metadata_refs",[]):add(rr)
  for rr in m.get("runtime_authenticated_historical_source_and_metadata_refs",[]):add(rr)
 for value in b.values():
  if isinstance(value,dict)and set(value)=={"path","sha256"}:add(value)
 for value in b["old_failure"].values():
  if isinstance(value,dict)and set(value)=={"path","sha256"}:add(value)
 for key in("selector_source_ref","source_ref","source_manifest_ref","source_review_ref","measurement_source_ref","repair_authority_ref","policy_ref"):
  add(actual[key])
 policy=read(actual["policy_ref"]);add(policy["root_activation_ref"])
 for rr in json.loads((ROOT/b["actual_guard_metadata_ref"]["path"]).read_text())["bindings"].values():add(rr)
 write("source_audit.v1.json",dict(schema="FLARE_phase_source_implementation_audit_v2",status="SOURCE_CANDIDATE_CHECKS_PASS",independent=False,execution_authorized=False,entrypoints_guards_state_processes_payloads_numerics_or_Git_executed=False,driver_ref=r(H/"root_driver_v1.py"),worker_ref=r(H/"worker_entry_v1.py"),checker_ref=r(checker),checks_ref=r(H/"metadata_checks.result.v1.json"),AST_delta_ref=r(H/"executor_AST_delta.v1.json"),handwritten_roles_ref=r(H/"handwritten_role_unions.INACTIVE.fixture.v1.json"),fixed_setup_seconds=7200,stage_original_science_seconds={"convert":3600,"native":14400,"seal":3600},combined_seconds={"convert":11400,"native":22200,"seal":11400},role_unions={k:v["union_count"]for k,v in result["handwritten_roles"].items()},rejected_metadata_cases=result["rejected_cases"],old_failure_and_clocks_preserved=True,historical_cancellation_and_sites_unknown=True,current_guard_capacity_and_setup_fit_proven=False,whole_independent_review_required=True))
 for p in H.iterdir():
  if p.is_file()and p.name not in("source_manifest.v1.json","source_handoff.v1.json"):merge(str(p.relative_to(ROOT)),r(p)["sha256"])
 write("source_manifest.v1.json",dict(schema="FLARE_remaining_CPU_source_manifest_v2",source_only=True,execution_authorized=False,independent_whole_review_required=True,all_writes_confined_to=HERE,source_sha256=dict(sorted(sources.items())),metadata_refs=[metadata[k]for k in sorted(metadata)],original_scientific_algorithms_settings_caps_and_guard_bodies_unchanged=True,fixed_setup_seconds=7200,no_current_guard_state_or_capacity_probe=True,no_added_memo=True,ordinary_close_credit_retry_or_truth_authorized=False))
 write("source_handoff.v1.json",dict(schema="FLARE_phase_CPU_source_handoff_v2",source_only=True,independent=False,source_ref=r(H/"root_driver_v1.py"),source_manifest_ref=r(H/"source_manifest.v1.json"),worker_ref=r(H/"worker_entry_v1.py"),audit_ref=r(H/"source_audit.v1.json"),check_result_ref=r(H/"metadata_checks.result.v1.json"),recipe_ref=r(H/"ROOT_RECIPE.md"),source_delta_ref=r(H/"source_delta.v1.patch"),initial_future_scope="Separately authorized fresh VAL001 convert only after whole independent review",phase_budget={"convert":dict(setup_seconds=7200,scientific_seconds=3600,wall_seconds=4200,combined_seconds=11400,evidence_reserve_seconds=300,authority_margin_seconds=600),"native":dict(setup_seconds=7200,scientific_seconds=14400,wall_seconds=15000,combined_seconds=22200,evidence_reserve_seconds=300,authority_margin_seconds=600),"seal":dict(setup_seconds=7200,scientific_seconds=3600,wall_seconds=4200,combined_seconds=11400,evidence_reserve_seconds=300,authority_margin_seconds=600)},role_unions={k:v["union_count"]for k,v in result["handwritten_roles"].items()},no_execution_admission_current_guard_state_process_payload_numerics_or_Git=True,old_failure_ref=b["diagnosis_ref"],old_charge_bytes=889641408,historical_cancel_signal="UNKNOWN",historical_exact_sites_written="UNKNOWN"))
 # Final seal verification reads only pinned static sources and metadata.
 m=read(r(H/"source_manifest.v1.json"))
 for path,sha in m["source_sha256"].items():assert r(path)["sha256"]==sha
 for record in m["metadata_refs"]:assert r(record["path"])==record
 return dict(driver_ref=r(H/"root_driver_v1.py"),worker_ref=r(H/"worker_entry_v1.py"),manifest_ref=r(H/"source_manifest.v1.json"),handoff_ref=r(H/"source_handoff.v1.json"),audit_ref=r(H/"source_audit.v1.json"),checks_ref=r(H/"metadata_checks.result.v1.json"),checker_ref=r(checker),source_count=len(m["source_sha256"]),metadata_count=len(m["metadata_refs"]),rejected_cases=result["rejected_cases"],role_unions={k:v["union_count"]for k,v in result["handwritten_roles"].items()})
if __name__=="__main__":print(json.dumps(main(),sort_keys=True,indent=2))
