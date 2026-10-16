"""Pure AST-extracted source/metadata fixtures. Never imports/configures a real guard."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import ast,copy,hashlib,json,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/Relate_failed64GiB_retention_successor_source_20261017_v1"
BASE="research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/stopped_retention_v1.py"
def need(ok,message):
 if not ok:raise ValueError(message)
def fn(tree,name):return next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name==name)
def ref(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()if p.exists()else hashlib.sha256(("SYNTHETIC_UNSEALED_"+str(p)).encode()).hexdigest())
def read(record):
 p=ROOT/record["path"];need(ref(p)==record,"Exact metadata source ref");return json.loads(p.read_text())
def integer(value,name,minimum=0):
 need(type(value)is int and value>=minimum,name);return value
def signature(record):
 need(isinstance(record,dict)and set(record)=={"path","sha256"}and isinstance(record["path"],str)
  and isinstance(record["sha256"],str)and len(record["sha256"])==64,"Synthetic pure-union ref shape")
def main():
 source=(ROOT/HERE/"relate_failed64_retention_v1.py").read_text();tree=ast.parse(source);old=ast.parse((ROOT/BASE).read_text())
 compile(source,str(ROOT/HERE/"relate_failed64_retention_v1.py"),"exec")
 history=json.loads((ROOT/HERE/"historical_evidence.v1.json").read_text());t=history["target"];key=t["key"]
 for r in history["runtime_authenticated_historical_source_and_metadata_refs"]:need(ref(r["path"])==r,"Original historical pin")
 unmodified=("require","charge","warning_fields","check_prefix","assert_open","_proof","_absent","_measurement","_pins",
  "consumer_bindings_v1","check_storage_locked_v1","budget_check_v9","validate_storage_v9","_stamp","_validate_targets",
  "_observe_targets","_retained_measurement","_recorded_stopped_proof")
 for name in unmodified:
  need(ast.dump(fn(tree,name),include_attributes=False)==ast.dump(fn(old,name),include_attributes=False),"Preserved original "+name)
 env=dict(copy=copy,hashlib=hashlib,json=json,datetime=datetime,timezone=timezone,Path=Path,ROOT=ROOT,
  SOURCE=HERE+"/relate_failed64_retention_v1.py",MANIFEST=HERE+"/source_manifest.v1.json",
  TARGET_KEYS=(key,),PARENT_STOPPED_KEYS=("native32/native-VAL000-full227990-borrowed32-GPU0-v1","H050/H050-direct-readout-GPU0-TRAIN-v1"),
  MALFORMED_KEY="H039/directed-clade-exchange-train-stage-seed101-v1",
  FRESH_TARGET_SECONDS=120,CEILING_BYTES=980000000000,HUMAN_LIMIT_BYTES=1000000000000,WARNING_BYTES=950000000000,
  META_CHARGE=524288,BOOTSTRAP_CHARGE=2097152,_ref=ref,_raw=signature,_json=read,
  budget=types.SimpleNamespace(v4=types.SimpleNamespace(integer=integer,encoded=lambda v:json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()),
   digest=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()))
 env["_manifest"]=lambda:dict(targets={key:t})
 names=("require","charge","warning_fields","consumer_bindings_v1","_stamp","_validate_targets","_historical_stop","_target","_retained_measurement",
  "_bindings","stopped_revocation_draft_v1","stopped_observation_draft_v1","_revocation","_stopped_proof","_recorded_stopped_proof")
 exec(compile(ast.Module(body=[fn(tree,n)for n in names],type_ignores=[]),"<pure_retention_metadata_functions>","exec"),env)
 need(env["_historical_stop"](t)is True,"Actual original failed lineage authenticates")
 rejected=[]
 def reject(call,label):
  try:call()
  except (ValueError,KeyError,TypeError):rejected.append(label);return
  raise AssertionError("Bad fixture accepted: "+label)
 for other in env["PARENT_STOPPED_KEYS"]+("other/failed_scope",env["MALFORMED_KEY"],"H041/American-C3-native-MLM-full30mb-mlm-fit-v8"):
  reject(lambda other=other:env["_target"](other),"non_target_"+other)
 for record_ref,field,bad,label in [(t["direct_wait_ref"],"returncode",0,"failed_worker_cannot_be_success"),
  (t["direct_wait_ref"],"all_owned_processes_and_groups_drained",False,"missing_recorded_drain"),
  (t["outer_terminal_ref"],"actual_exit_code",0,"failed_outer_cannot_be_success"),
  (t["outer_terminal_ref"],"actual_terminal_consumed",False,"unconsumed_outer")]:
  original=env["_json"]
  def altered(r,record_ref=record_ref,field=field,bad=bad):
   value=copy.deepcopy(original(r))
   if r==record_ref:value[field]=bad
   return value
  env["_json"]=altered;reject(lambda:env["_historical_stop"](t),label);env["_json"]=original
 now=datetime.now(timezone.utc)
 before=(now-timedelta(seconds=30)).isoformat();at=(now-timedelta(seconds=10)).isoformat();end=now
 facts=dict(PID_observations=[[r["pid"],r["start_ticks"],True,at]for r in t["known_PID_start_rows"]],
  group_observations=[[g,True,at]for g in t["known_process_groups"]])
 env["_validate_targets"](facts,t,datetime.fromisoformat(before),end,fresh=True)
 missing=copy.deepcopy(facts);missing["PID_observations"].pop()
 reject(lambda:env["_validate_targets"](missing,t,datetime.fromisoformat(before),end,fresh=True),"omitted_known_PID")
 reused=copy.deepcopy(facts);reused["group_observations"][0][1]=False
 reject(lambda:env["_validate_targets"](reused,t,datetime.fromisoformat(before),end,fresh=True),"live_or_reused_group")
 stale=copy.deepcopy(facts)
 for rows in stale.values():
  for row in rows:row[-1]=(now-timedelta(seconds=121)).isoformat()
 reject(lambda:env["_validate_targets"](stale,t,now-timedelta(seconds=180),end,fresh=True),"stale_known_targets")
 scopes=[dict(path=p,entries=1,conservative_scope_bytes=4096,node_types=dict(directory=1,regular=0,symlink=0,socket=0))for p in t["scope_paths"]]
 measure=dict(schema="two_key_exact_stopped_retained_node_measurement_v1",conservative_scope_bytes=8192,entries=2,
  scope_paths=t["scope_paths"],stat_only=True,per_scope=scopes,inventory_sha256="a"*64,
  two_complete_passes_and_final_no_follow_revalidation=True,per_node_additional_metadata_charge_bytes=4096,
  regular_payloads_opened=False,symlink_targets_followed=False,unsafe_nodes_rejected=True,node_metadata_preserved=True,
  namespace_normalization_performed=False,cooperative_snapshot_not_atomic_quota=True)
 need(env["_retained_measurement"](measure,t["original_reservation"])==8192,"Complete synthetic2scope measure")
 for field,bad,label in [("scope_paths",t["scope_paths"][:1],"omitted_original_scope"),
  ("regular_payloads_opened",True,"payload_opening_not_stat_only"),("namespace_normalization_performed",True,"normalization_not_allowed")]:
  invalid=copy.deepcopy(measure);invalid[field]=bad
  reject(lambda invalid=invalid:env["_retained_measurement"](invalid,t["original_reservation"]),label)
 socket=copy.deepcopy(measure);socket["per_scope"][0]["node_types"]["socket"]=1
 reject(lambda:env["_retained_measurement"](socket,t["original_reservation"]),"any_retained_socket_blocks")
 # Actual predecessor role snapshot, no live guard/state/policy/resource call.
 snapshot=json.loads((ROOT/"research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v1/inert_guard_roles.snapshot.v1.json").read_text())
 basebindings=snapshot["bindings"];basevalues=list(basebindings.values())
 need(len(basevalues)==45 and len({(r["path"],r["sha256"])for r in basevalues})==38,"Exact historical fixture roles")
 capsule=dict(path=HERE+"/SYNTHETIC_selector_capsule_NOT_AUTHORITY.json",sha256="c"*64)
 env["_SELECTOR_REF"]=capsule;env["_selected_record"]=lambda r:need(r==capsule,"Exact synthetic capsule")
 bindings=env["_bindings"](basebindings,{})
 need(all(bindings[k]==v for k,v in basebindings.items())and len(bindings)==46,"All45 predecessor roles plus one capsule")
 unique=env["consumer_bindings_v1"](list(bindings.values()))
 need(len(unique)==39,"Only one unique direct guard addition")
 # Compile unchanged actual FLARE direct role builder with explicit metadata stubs.
 flare=(ROOT/"research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v1/root_contract_v1.py").read_text()
 flaredefs=ast.parse(flare);flarehist=ROOT/"research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v1"
 deps=json.loads((flarehist/"dependencies.v1.json").read_text())
 fake=lambda name:dict(path=HERE+"/SYNTHETIC_"+name+".json",sha256=hashlib.sha256(name.encode()).hexdigest())
 fenv=dict(ref=ref,MANIFEST=str(flarehist.relative_to(ROOT))+"/source_manifest.v1.json",
  SOURCE=str(flarehist.relative_to(ROOT))+"/root_contract_v1.py",DRIVER=str(flarehist.relative_to(ROOT))+"/root_driver_v1.py",
  ENTRY=str(flarehist.relative_to(ROOT))+"/worker_entry_v1.py",HANDOFF=str(flarehist.relative_to(ROOT))+"/source_handoff.v1.json",
  PRODUCERS=dict(convert="src/arggnn/american_c3_flare_convert_v1.py",native="src/arggnn/american_c3_flare_run_v2.py",seal="src/arggnn/american_c3_flare_evaluate_v1.py"),
  COMPLETION_REVIEW=deps["COMPLETION_REVIEW"],SELECTOR=ref(HERE+"/relate_failed64_retention_v1.py"),
  NAMESPACE=deps["NAMESPACE"],PIDFD=deps["PIDFD"],OWNED=deps["OWNED"],FRAMEWORK=deps["FRAMEWORK"])
 class Core:
  def _context(self,*a,**k):return dict(targets={"american_c3_val_001":dict(count_manifest_ref=fake("count"))})
 fenv["completion"]=lambda:Core();fenv["make_config"]=lambda q:dict(binary_provenance_ref=deps["BINARY_PROVENANCE"])
 fenv["read"]=lambda r:dict(binary_provenance_ref=deps["BINARY_PROVENANCE"])
 exec(compile(ast.Module(body=[fn(flaredefs,"direct_roles")],type_ignores=[]),"<unchanged_FLARE_direct_roles_metadata_only>","exec"),fenv)
 role_counts={}
 for stage in("convert","native","seal"):
  q=dict(stage=stage,parent_id="american_c3_val_001",source_review_ref=fake("wrapper_review"),runtime_ref=fake("runtime"),
   root_directive_ref=fake("directive"),context_ref=fake("context"),registration_ref=fake("registration"),
   storage_policy_selector_ref=capsule,conversion_ref=fake("conversion"),previous_stage_evidence_ref=fake("previous"),native_run_ref=fake("native"))
  direct=fenv["direct_roles"](fake("request"),q,fake("config"),fake("closure"))
  union=env["consumer_bindings_v1"](list(bindings.values()),direct)
  role_counts[stage]=len(union);need(all(r in union for r in basevalues),"No original guard omitted")
 need(role_counts==dict(convert=60,native=63,seal=64),"FLARE original ceiling preserved with same64 cap")
 reject(lambda:env["consumer_bindings_v1"](list(bindings.values()),direct,[fake("extra_overflow")]),"65roles_fail_without_trimming")
 # Source safety/order assertions only: no actual target observers or measurement are called.
 operation=fn(tree,"apply_retire_stopped_v1");text=ast.get_source_segment(source,operation)
 need("OPERATION_WALL_SECONDS" in text and "helper.require=bounded_require" in text and "socket_is_absent=lambda path:False"in text,
  "Finite deadline wraps existing exact measurement; no socket cleanup")
 need(text.index("proof=_stopped_proof(request,own,observe=True)")<text.index("measurement=helper.measure_retained_nodes_v1")
  <text.index('proof["post_measurement_targets"]')<text.index("_append("),"Pre/post real known-target checks before append")
 need(env["charge"](100,200,50,1)==100+200-50+2097152+524288,"Unchanged conservative accounting")
 reject(lambda:env["charge"](980000000000,0,0,0),"980GB_ceiling")
 return dict(schema="Relate64_source_pure_metadata_check_v1",status="PASS_SOURCE_AND_PURE_METADATA_ONLY",
  original_historical_refs_authenticated=len(history["runtime_authenticated_historical_source_and_metadata_refs"]),
  genuine_Relate41415_outer1_and_worker_minus9_preserved=True,original_known_PID_and_group_count=[4,4],
  original_guard_functions_AST_unchanged=list(unmodified),all45raw38unique_predecessor_roles_preserved=True,
  direct_new_guard_roles=1,unique_new_guard_roles=1,future_FLARE_unions=role_counts,unchanged64role_overflow_rejected=True,
  exact_original2scope_measurement_validator_used=True,actual_measurement_or_process_absence_calls=0,
  finite_operation_source_order_checked=True,negative_fixtures=rejected,planning_inventory_not_recomputed=True,
  guards_imported_configured_or_resources_probed=False,policy_activation_credit_or_Git=False)
if __name__=="__main__":print(json.dumps(main(),sort_keys=True,indent=2))
