"""Independent source/metadata review. All synthetic writes remain in this review folder."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import ast,copy,hashlib,importlib,json,os,shutil,sys,traceback
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v1"
REVIEW="research/reviews/FLARE_original_remaining_CPU_execution_independent_review_20261017_v1"
def main():
 assert sys.dont_write_bytecode and sys.flags.no_user_site
 sys.path.insert(0,str(ROOT/"src"))
 d=importlib.import_module(SOURCE.replace("/",".")+".root_driver_v1");c=d.c
 h=importlib.import_module(SOURCE.replace("/",".")+".prepare_context_v1")
 common=importlib.import_module("arggnn.american_c3_common_v1")
 p=ROOT/SOURCE;review=ROOT/REVIEW
 assert c._h._GUARD is None and c.SELECTOR_REF is None
 c.sources();c.scientific_review()
 fixture=review/"SYNTHETIC_METADATA_NOT_AUTHORITY";assert not fixture.exists();fixture.mkdir()
 prep=fixture/"prep";prep.mkdir()
 virtual="research/reviews/FLARE_remaining_ROOT_preparation_REVIEW_SYNTHETIC_NOT_AUTHORITY"
 aliases={virtual:prep}
 originals=dict(local=c.local,read=c.read,put=c.put,common_read=common.read,selector=c.SELECTOR_REF,chosen=c._CHOSEN)
 def mapped(value):
  text=str(value)
  if text.startswith(str(ROOT)+"/"):text=text[len(str(ROOT))+1:]
  for key in sorted(aliases,key=len,reverse=True):
   if text==key or text.startswith(key+"/"):return aliases[key]/text[len(key):].lstrip("/")
  return value
 def local(value):return originals["local"](mapped(value))
 def read(record):
  actual=mapped(record["path"])
  if str(actual)!=record["path"]:
   record=dict(record,path=str(Path(actual).relative_to(ROOT)))
  return originals["read"](record)
 def put(name,value,virtual_ref=False):
  path=prep/name;record=originals["put"](path,value)
  return dict(record,path=virtual+"/"+name)if virtual_ref else record
 def reject(f,label):
  try:f()
  except (ValueError,KeyError,TypeError):return label
  raise AssertionError("Invalid metadata accepted: "+label)
 results={};negatives=[]
 c.local=local;c.read=read;common.read=read
 try:
  now=datetime.now(timezone.utc);dates=dict(created_at_utc=now.isoformat(),expires_at_utc=(now+timedelta(hours=8)).isoformat())
  reviewref=put("SYNTHETIC_source_review.json",dict(independent=True,verdict="SOURCE_PASS",source_only=True,
   source_ref=c.ref(c.DRIVER),source_manifest_ref=c.ref(c.MANIFEST),source_handoff_ref=c.ref(c.HANDOFF),blockers=[]))
  binding=json.loads((p/"ROOT_context_binding.INACTIVE.template.v1.json").read_text())
  binding.update(active=True,root_authorizer="root_agent",remaining_VAL_truth_not_previously_opened=True,
   source_manifest_ref=c.ref(c.MANIFEST),source_review_ref=reviewref,preparation_directory=virtual,**dates)
  bound=h.bind_selected_context_v1(put("SYNTHETIC_binding.json",binding,True))
  assert bound["status"]=="BOUND_METADATA_ONLY_NOT_ADMITTED"and bound["admission_or_execution"]is False
  assert bound["conversion_output_bytes"]==881252800 and bound["conversion_whole_reservation_bytes"]==889641408
  ctx=read(bound["context_ref"]);assert ctx["targets"]["american_c3_val_004"]is None
  assert ctx["targets"]["american_c3_val_001"]["count_manifest_ref"]is not None
  core=c.completion();core._context(bound["context_ref"],parent_id="american_c3_val_001")
  negatives.append(reject(lambda:core._context(bound["context_ref"]),"all_four_readiness_with_VAL004_unbound"))
  source_map=c.sources()["source_sha256"]
  runtime=put("SYNTHETIC_runtime.json",dict(prefix=str(ROOT/".runtime/env"),CPU_threads=4,cpu_ids=[11,12,13,14],
   CUDA_VISIBLE_DEVICES="",package_report_ref=c.BASE_RUNTIME))
  selector=read(c.ref("research/reviews/TreeLoom_matched_CPU_ROOT_preparation_20261017_v4/ROOT_request.root_active_v1.json"))["storage_policy_selector_ref"]
  registration=dict(bound["registration_ref"],path=virtual+"/"+Path(bound["registration_ref"]["path"]).name)
  row=ctx["targets"]["american_c3_val_001"]
  converted=dict(status=c.STATUS["convert"],parent_id="american_c3_val_001",observation_ref=row["observation_ref"],
   focal_registry_ref=row["focal_registry_ref"],native_options=core.OPTIONS,native_source_commit=core.COMMIT,
   input_sites=row["input_sites"],query_ancestry_truth_opened=False,native_expected_markers=200000)
  conversion=put("SYNTHETIC_conversion.json",converted)
  native=put("SYNTHETIC_native.json",dict(status=c.STATUS["native"],parent_id="american_c3_val_001",
   conversion_ref=conversion,binary_provenance_ref=c.DEPS["BINARY_PROVENANCE"]))
  def evidence(stage,completion):
   reserve=dict(key="SYNTHETIC_ONLY_"+stage,reserved_bytes=1,output=str(prep),auxiliary_scopes=[],consumer_bindings=[])
   inp=put("SYNTHETIC_"+stage+"_input.json",dict(schema="FLARE_remaining_CPU_admitted_input_v1",
    parent_id="american_c3_val_001",stage=stage,root_driver_ref=c.ref(c.DRIVER),main_source_ref=c.ref(c.ENTRY),
    source_sha256=source_map,output_directory=str(prep.relative_to(ROOT)),storage=dict(reservation=reserve)))
   wait=put("SYNTHETIC_"+stage+"_wait.json",dict(actual_direct_worker_wait=True,returncode=0,
    all_owned_processes_and_groups_drained=True,operational_failure=None,input_ref=inp,completed_ref=completion))
   outer=put("SYNTHETIC_"+stage+"_outer.json",dict(actual_terminal_consumed=True,actual_exit_code=0,
    completed_ref=completion,direct_wait_ref=wait))
   close=put("SYNTHETIC_"+stage+"_close.json",dict(kind="concurrent_completed_scope_close_receipt_v1",
    key=reserve["key"],artifacts_deleted=0,value=dict(future_writes_and_relaunch_revoked=True,
     request=dict(completed_ref=completion,direct_wait_ref=wait),
     reservation_sha256=hashlib.sha256(json.dumps(reserve,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest())))
   return put("SYNTHETIC_"+stage+"_stage_evidence.json",dict(schema="FLARE_remaining_stage_ROOT_terminal_evidence_v1",
    root_authorizer="root_agent",stage=stage,parent_id="american_c3_val_001",completion_ref=completion,
    input_ref=inp,actual_wait_ref=wait,outer_terminal_ref=outer,ordinary_close_receipt_ref=close))
  previous={"native":evidence("convert",conversion),"seal":evidence("native",native)}
  for stage in c.STAGES:
   q=json.loads((p/"ROOT_request.INACTIVE.template.v1.json").read_text())
   out="data/FLARE_original_remaining_REVIEW_SYNTHETIC_"+stage
   packet="research/reviews/FLARE_remaining_ROOT_actual_REVIEW_SYNTHETIC_"+stage
   aliases[out]=fixture/("future_output_"+stage);aliases[packet]=fixture/("future_packet_"+stage)
   q.update(active=True,root_authorizer="root_agent",execution_authorized=True,source_manifest_ref=c.ref(c.MANIFEST),
    source_review_ref=reviewref,stage=stage,parent_id="american_c3_val_001",phase=stage+"_american_c3_val_001",
    preparation_directory=virtual,packet_directory=packet,output_directory=out,context_ref=bound["context_ref"],
    registration_ref=registration,runtime_ref=runtime,storage_policy_selector_ref=selector,cpu_ids=[11,12,13,14],
    conversion_ref=conversion if stage!="convert"else None,native_run_ref=native if stage=="seal"else None,
    previous_stage_evidence_ref=previous.get(stage),**dates)
   cfg=c.make_config(q);q["caps"]=c.stage_caps(stage,cfg)
   directive=dict(q,schema="FLARE_remaining_CPU_ROOT_directive_v1",
    directive="Execute one exact missing original FLARE stage after genuine fresh admission0")
   q["root_directive_ref"]=put("SYNTHETIC_"+stage+"_directive.json",directive)
   qr=put("SYNTHETIC_"+stage+"_request.json",q,True)
   actual,caps,rt=c.validate_request(qr)
   assert actual==q and cfg["external_truth_ref"]is None
   if stage=="seal":assert cfg["mode"]=="seal_native"and cfg["write_derived_marker_tracts"]is False
   else:assert cfg["native_options"]==core.OPTIONS
   results[stage]=dict(whole_reservation=caps["reservation_bytes"],output_bytes=cfg["output_cap_bytes"],
    original_source_config_generated_and_full_request_metadata_accepted=True)
   bad=copy.deepcopy(q);bad["remaining_VAL_truth_authorized"]=True
   br=put("SYNTHETIC_"+stage+"_bad_truth.json",bad,True)
   negatives.append(reject(lambda:c.validate_request(br),stage+"_truth_authority"))
   bad=copy.deepcopy(q);bad["parent_id"]="american_c3_val_000"
   br=put("SYNTHETIC_"+stage+"_bad_parent.json",bad,True)
   negatives.append(reject(lambda:c.validate_request(br),stage+"_VAL000"))
  assert c._h._GUARD is None
 finally:
  c.local=originals["local"];c.read=originals["read"];common.read=originals["common_read"]
  c.SELECTOR_REF=originals["selector"];c._CHOSEN=originals["chosen"]
  shutil.rmtree(fixture)
 # Repeat actual admit ordering with the author's narrowly stubbed checker relocated to this review folder.
 checker=(p/"metadata_checks_v1.py").read_text()
 checker=checker.replace('fixture=ROOT/HERE/"SOURCE_ONLY_metadata_lifecycle_fixture"',
  'fixture=ROOT/'+repr(REVIEW)+'/"INDEPENDENT_SYNTHETIC_ADMIT_FIXTURE"')
 env={"__name__":"_independent_relocated_admit_checker"};exec(compile(checker,"<relocated_source_only_admit_checker>","exec"),env)
 lifecycle=env["main"]()
 assert lifecycle["status"]=="PASS_SOURCE_ONLY"and [x["direct_guard_union"]for x in lifecycle["stages"]]==[59,62,63]
 assert c._h._GUARD is None and c.SELECTOR_REF is None
 assert not any(n in sys.modules for n in("numpy","torch","torch_geometric","scripts.project_budget_v9","arggnn.project_budget_v4"))
 return dict(schema="FLARE_independent_source_metadata_check_v1",status="PASS_SOURCE_AND_SYNTHETIC_METADATA_ONLY",
  actual_selected_context_count_registration_helper_exercised=True,full_request_and_exact_config_routes=results,
  selected_readiness_does_not_fake_all_four=True,negative_checks=negatives,
  actual_admit_ordering_and_original_scope_union=lifecycle,
  all_synthetic_writes_confined_to_review_and_removed=True,no_actual_guard_mutex_process_probe_payload_array_binary_or_job=True)
if __name__=="__main__":
 try:print(json.dumps(main(),indent=2,sort_keys=True))
 except BaseException:traceback.print_exc(file=sys.stdout);raise
