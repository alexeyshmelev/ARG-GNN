"""AST and memory-only verification. No entrypoints, actual guards, state, processes or payloads."""
from pathlib import Path
from datetime import datetime,timezone
import ast,contextlib,copy,hashlib,io,json,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v2"
H=ROOT/HERE
def need(ok,msg):
 if not ok:raise ValueError(msg)
def fn(tree,name):return next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
def dump(n):return ast.dump(n,include_attributes=False)
def rr(p):
 p=Path(p);p=p if p.is_absolute()else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def raw(r):
 p=ROOT/r["path"];v=p.read_bytes();need(hashlib.sha256(v).hexdigest()==r["sha256"],"Exact fixture pin");return v
def nodes(source,names,scope):
 tree=ast.parse(source);body=[fn(tree,n)for n in names]
 exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),"<source-only-extracted>","exec"),scope)
def reject(call):
 try:call()
 except (ValueError,KeyError,TypeError,RuntimeError):return
 raise AssertionError("Malformed metadata unexpectedly accepted")
def main():
 b=json.loads((H/"source_bindings.v1.json").read_text());source=(H/"root_driver_v1.py").read_text();tree=ast.parse(source);wsource=(H/"worker_entry_v1.py").read_text();wtree=ast.parse(wsource)
 for p in H.glob("*.py"):compile(p.read_text(),str(p),"exec")
 for k,v in b.items():
  if k.endswith("_ref")and isinstance(v,dict):raw(v)
 original=ast.parse(raw(b["original_contract_ref"]));framework=ast.parse((ROOT/"research/reviews/American_TreeLoom_matched_CPU_execution_source_20261017_v4/root_driver_v1.py").read_text())
 child_source=raw(b["selected_child_source_ref"]);childtree=ast.parse(child_source)
 memo=raw(b["consumer_memo_ref"])
 assert hashlib.sha256(memo).hexdigest()=="17a1377320f61c3c0cc8d3816d0259f9dae3121ed91bf7206a7ee63fcab8eb7f"
 assert "TransactionMemo"not in source and "install_consumer"not in source and "ROOT_mutation_transaction"not in source and "guard(False)"not in wsource
 # Static scientific identity: whole sources, including exact converter/GzipVCF/Writer/WaitWriter.
 scientific={}
 for path,sha in json.loads(raw(b["original_manifest_ref"]))["source_sha256"].items():
  if path.startswith("src/arggnn/american_c3_flare_")or path in("src/arggnn/american_c3_common_v1.py","research/reviews/FLARE_consumer_namespace_root_20261015_v1/root_context_980_v1.py"):
   raw(dict(path=path,sha256=sha));scientific[path]=sha
 assert scientific["src/arggnn/american_c3_flare_convert_v1.py"]=="6e55e295226a183548a23258e7230855842bb876b4d2a90875d8f2ad53181ff5"
 # Every original initial validation predicate is compiled unchanged except its final storage call.
 pre=fn(childtree,"validate_storage_v9");assert len(pre.body)==3
 assert dump(pre.body[-1])==dump(ast.parse("return budget_check_v9(storage,first=True,boundary=True)").body[0])
 # Original framework helpers are extracted without edits by activate(); cleanup/wait/drain body is exact.
 new=fn(tree,"owned_execute");old=fn(framework,"execute")
 ntry=next(n for n in new.body if isinstance(n,ast.Try));otry=next(n for n in old.body if isinstance(n,ast.Try))
 def cleanup_part(seq):
  start=next(i for i,n in enumerate(seq)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="stop"for t in n.targets))
  return [dump(n)for n in seq[start:]]
 assert cleanup_part(ntry.finalbody)==cleanup_part(otry.finalbody)
 for name in("input_spec","worker_python","sample_scope","boundary"):assert name in ast.get_source_segment(source,fn(tree,"activate"))
 assert "initial_storage_transaction(s,q,g)"in source and "verify_actual_roles(s,q,g)"not in source
 assert "g.validate_storage_v9("not in source and "g.validate_storage_v9("not in wsource
 assert "g.budget_check_v9(s[\"storage\"],boundary=True,required_write_bytes=32768)"in source
 assert "boundary(g,s,32768,target=\"aux\")"in source and "boundary(g,s,len(tail)*6+8192,target=\"aux\")"in source
 assert "super().prepare(name,budget_bytes)"in wsource and "super().finish(p,planned)"in wsource
 assert "value=g.budget_check_v9(*args,**kwargs)"in wsource and "result=module.execute(cfg,writer)"in wsource
 assert wsource.index("runtime_report()==")<wsource.index("compact_status(ready_fd,ready)")<wsource.index("module=importlib.import_module(c.MODULES")
 assert wsource.index("view=d.initial_storage_transaction")<wsource.index("compact_status(ready_fd,ready)")
 assert wsource.index("compact_status(ready_fd,entry)")<wsource.index("module=importlib.import_module(c.MODULES")
 # Extract only pure functions/classes, with all external operations replaced by in-memory stubs.
 env=dict(Path=Path,datetime=datetime,timezone=timezone,ast=ast,copy=copy,json=json,ROOT=ROOT,HERE=HERE,SOURCE=HERE+"/root_driver_v1.py",ENTRY=HERE+"/worker_entry_v1.py",MANIFEST=HERE+"/source_manifest.v1.json",HANDOFF=HERE+"/source_handoff.v1.json",SETUP_SECONDS=7200,EVIDENCE_RESERVE=300,ORIGINAL_MARGIN=600,need=need,raw=raw,ref=rr,bindings=lambda:b)
 nodes(source,["phase_budget","PhaseStop","PhaseClock","initial_storage_transaction","validate_worker_frame","contract"],env)
 assert env["phase_budget"]("convert")==dict(setup_seconds=7200,scientific_seconds=3600,wall_seconds=4200,combined_seconds=11400,evidence_reserve_seconds=300,authority_margin_seconds=600)
 assert env["phase_budget"]("native")["combined_seconds"]==22200 and env["phase_budget"]("seal")["combined_seconds"]==11400
 reject(lambda:env["phase_budget"]("reduce"))
 # Handwritten current46-name/39-distinct guards plus complete convert/native/seal direct roles.
 guards=json.loads(raw(b["actual_guard_metadata_ref"]))["bindings"];unique=lambda seq:list({(r["path"],r["sha256"]):r for r in seq}.values())
 assert len(guards)==46 and len(unique(guards.values()))==39
 P=lambda name:dict(path="SOURCE_ONLY/"+name,sha256=hashlib.sha256(name.encode()).hexdigest())
 deps=json.loads((ROOT/"research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v1/dependencies.v1.json").read_text())
 binding_records={};request_record=P("request");accept_record=P("accept");directive_record=P("directive")
 selector=types.SimpleNamespace(SELECTOR_REF=None,_CHOSEN=None,SELECTOR=None,GUARD_REF=None,POLICY_REF=None)
 actual=json.loads(raw(b["actual_selector_metadata_ref"]))
 def selected(r):need(r==b["actual_selector_metadata_ref"],"Pinned selector");return copy.deepcopy(actual)
 selector.selected=selected
 def select(r):
  v=selected(r);selector.SELECTOR_REF=r;selector._CHOSEN=v;selector.SELECTOR=v["selector_source_ref"];selector.GUARD_REF=v["source_ref"];selector.POLICY_REF=v["policy_ref"];return v
 selector.bind_selector=select;selector.guard=lambda consumer:("UNCHANGED_CONSUMER_MEMO"if consumer else"ORIGINAL_UNMEMOIZED_MUTATOR")
 c=types.SimpleNamespace(require=need,DEPS=deps,CAPS={"reservation_bytes":889641408},read=lambda r:copy.deepcopy(binding_records[r["path"]]),raw=lambda r:r,local=lambda p:ROOT/p,ref=lambda p:P(str(p)),validate_request=lambda r:("ORIGINAL_VALIDATION_REACHED",r),SELECTOR_REF=None,_CHOSEN=None,GUARD_REF=None,POLICY_REF=None)
 c.direct_roles=lambda *a:[] # Replaced below by the exact original direct_roles body.
 cenv=c.__dict__;cenv.update(STATUS={},PRODUCERS=deps.get("PRODUCERS",{}))
 for key in("NAMESPACE","PIDFD","OWNED","FRAMEWORK"):cenv[key]=deps[key]
 cenv["SELECTOR"]=deps["SELECTOR"];cenv["COMPLETION_REVIEW"]=deps["COMPLETION_REVIEW"]
 cenv["PRODUCERS"]=dict(convert="src/arggnn/american_c3_flare_convert_v1.py",native="src/arggnn/american_c3_flare_run_v2.py",seal="src/arggnn/american_c3_flare_evaluate_v1.py")
 cenv["completion"]=lambda:types.SimpleNamespace(_context=lambda *a,**k:{"targets":{"american_c3_val_001":{"count_manifest_ref":P("count")}}})
 cenv["make_config"]=lambda q:dict(binary_provenance_ref=deps["BINARY_PROVENANCE"])
 nodes(raw(b["original_contract_ref"]),["direct_roles"],cenv)
 original_direct=c.direct_roles
 env["load"]=lambda r,name:c if r==b["original_contract_ref"]else selector
 c=env["contract"]();env["c"]=c;c.bind_selector(b["actual_selector_metadata_ref"])
 assert c.guard(True)=="UNCHANGED_CONSUMER_MEMO"and c.guard(False)=="ORIGINAL_UNMEMOIZED_MUTATOR"
 q=dict(stage="convert",parent_id="american_c3_val_001",preparation_directory="research/reviews/FLARE_remaining_ROOT_preparation_SOURCE_ONLY_FRESH",packet_directory="research/reviews/FLARE_remaining_ROOT_actual_SOURCE_ONLY_FRESH",output_directory="data/FLARE_original_remaining_SOURCE_ONLY_FRESH",phase_budget=env["phase_budget"]("convert"),prior_failure_acceptance_ref=accept_record,root_directive_ref=directive_record,created_at_utc="2030-01-01T00:00:00+00:00",expires_at_utc="2030-01-01T04:00:00+00:00",storage_policy_selector_ref=b["actual_selector_metadata_ref"])
 accepted=dict(schema="ROOT_FLARE_failed_unsealed_acceptance_v2",root_authorizer="root_agent",accepted=True,**b["old_failure"],old_launch_resume_write_authority_revoked=True,old_files_scopes_key_charge_clocks_and_failure_unchanged=True,original_conversion_unsealed=True)
 binding_records.update({request_record["path"]:q,accept_record["path"]:accepted,directive_record["path"]:dict(phase_budget=q["phase_budget"],prior_failure_acceptance_ref=accept_record)})
 assert c.validate_request(request_record)[0]=="ORIGINAL_VALIDATION_REACHED"
 negative=[]
 for obj,key,bad in[(q,"phase_budget",dict(q["phase_budget"],setup_seconds=7199)),(accepted,"accepted",False),(accepted,"actual_cancel_signal",15),(accepted,"exact_sites_written",227316),(accepted,"old_charge_bytes",0),(accepted,"outer_returncode",True),(accepted,"old_created_at_utc","NEW"),(accepted,"old_expires_at_utc","NEW"),(accepted,"old_reservation_key","NEW"),(accepted,"old_launch_resume_write_authority_revoked",False),(accepted,"old_files_scopes_key_charge_clocks_and_failure_unchanged",False),(accepted,"original_conversion_unsealed",False),(binding_records[directive_record["path"]],"phase_budget",{}),(q,"expires_at_utc","2030-01-01T03:20:00+00:00")]:
  prior=obj[key];obj[key]=bad;reject(lambda:c.validate_request(request_record));obj[key]=prior;negative.append(key)
 for freshkey in("preparation_directory","packet_directory","output_directory"):
  for oldpath in b["excluded_namespaces"].values():
   for bad in(oldpath,oldpath+"/child",str(Path(oldpath).parent)):
    prior=q[freshkey];q[freshkey]=bad;reject(lambda:c.validate_request(request_record));q[freshkey]=prior;negative.append("cross_namespace")
 roles={}
 for stage in("convert","native","seal"):
  z=dict(q,stage=stage,context_ref=P("context"),source_review_ref=P("review"),runtime_ref=P("runtime"),registration_ref=P("registration"),conversion_ref=P("conversion"),native_run_ref=P("native"),previous_stage_evidence_ref=P("previous"))
  binding_records[P("native")["path"]]=dict(binary_provenance_ref=deps["BINARY_PROVENANCE"])
  group=c.direct_roles(request_record,z,P("config"),P("closure"))
  # Independently hand-enumerated role list, no role truncation or count-based filtering.
  handwritten=[c.ref(c.MANIFEST),c.ref(c.SOURCE),c.ref(c.DRIVER),c.ref(c.ENTRY),c.ref(c.HANDOFF),c.ref(c.PRODUCERS[stage]),deps["COMPLETION_REVIEW"],P("review"),P("runtime"),P("config"),directive_record,request_record,P("context"),P("count"),P("registration"),P("closure"),b["selected_child_source_ref"],b["actual_selector_metadata_ref"],deps["NAMESPACE"],deps["PIDFD"],deps["OWNED"],deps["FRAMEWORK"]]
  if stage!="convert":handwritten += [P("conversion"),P("previous"),deps["BINARY_PROVENANCE"]]
  if stage=="seal":handwritten += [P("native")]
  handwritten += [deps["SELECTOR"],accept_record]
  assert unique(group)==unique(handwritten)
  union=unique([*guards.values(),*group]);roles[stage]=dict(guard_names=46,distinct_guards=39,direct_unique=len(unique(group)),union_count=len(union),all_roles=union)
 assert [roles[s]["union_count"]for s in("convert","native","seal")]==[60,63,64],roles
 # One synthetic original mutex, exact original prelude and one actual-view check; no real guard import.
 lock={"held":False,"enters":0,"exits":0,"raw":0,"checks":0,"final_drift":False}
 class Lock:
  def __enter__(self):assert not lock["held"];lock["held"]=True;lock["enters"]+=1
  def __exit__(self,*a):lock["held"]=False;lock["exits"]+=1;need(not lock["final_drift"],"Original memo final drift predicate")
 def fake_raw(r):assert lock["held"];lock["raw"]+=1;need(r.get("sha256")!="BAD","Changed bound source")
 union_env=dict(require=need,copy=copy,_raw=fake_raw);nodes(child_source,["consumer_bindings_v1"],union_env)
 g=types.SimpleNamespace(require=need,_raw=fake_raw,budget=types.SimpleNamespace(v4=types.SimpleNamespace(mutex=lambda root:Lock())))
 def storage(s,**kw):
  assert lock["held"]and kw==dict(first=True,boundary=True);lock["checks"]+=1;return dict(bindings={"actual":P("actual")})
 g.check_storage_locked_v1=storage;g.consumer_bindings_v1=union_env["consumer_bindings_v1"]
 c.ref=lambda p:P(str(p));c.direct_roles=lambda *a:[P("direct")]
 s=dict(job_id="J",output_directory="data/SOURCE_ONLY",refs=dict(config=P("config")),runtime_ref=P("runtime"),root_stage_request_ref=request_record,transitive_source_closure_ref=P("closure"))
 s["storage"]=dict(job_id="J",output_directory=s["output_directory"],reserved_new_bytes=889641408,reservation=dict(consumer_bindings=[P(c.DRIVER),P("config"),P("runtime"),P("actual"),P("direct")]))
 env["initial_storage_transaction"](s,q,g)
 assert lock["enters"]==lock["exits"]==lock["checks"]==1 and not lock["held"]
 for mutate in("missing_direct","missing_actual","wrong_job","wrong_output","oversize","bad_source","bad_runtime","cap65","final_drift"):
  original_s=copy.deepcopy(s);original_check=g.check_storage_locked_v1
  if mutate=="missing_direct":s["storage"]["reservation"]["consumer_bindings"].remove(P("direct"))
  if mutate=="missing_actual":s["storage"]["reservation"]["consumer_bindings"].remove(P("actual"))
  if mutate=="wrong_job":s["storage"]["job_id"]="OTHER"
  if mutate=="wrong_output":s["storage"]["output_directory"]="data/OTHER"
  if mutate=="oversize":s["storage"]["reserved_new_bytes"]=889641409
  if mutate=="bad_source":s["refs"]["config"]["sha256"]="BAD";s["storage"]["reservation"]["consumer_bindings"][1]=s["refs"]["config"]
  if mutate=="bad_runtime":s["runtime_ref"]["sha256"]="BAD";s["storage"]["reservation"]["consumer_bindings"][2]=s["runtime_ref"]
  if mutate=="cap65":g.check_storage_locked_v1=lambda *a,**k:dict(bindings={str(i):P("role"+str(i))for i in range(65)})
  if mutate=="final_drift":lock["final_drift"]=True
  reject(lambda:env["initial_storage_transaction"](s,q,g));negative.append(mutate)
  s=original_s;g.check_storage_locked_v1=original_check;lock["final_drift"]=False;assert not lock["held"]
 # Actual PhaseClock and status validators with fake monotonic/UTC/signal surfaces.
 now={"m":100.0,"epoch":2000000000.0}
 class DT:
  @staticmethod
  def now(tz=None):return datetime.fromtimestamp(now["epoch"],tz or timezone.utc)
  fromisoformat=staticmethod(datetime.fromisoformat)
  fromtimestamp=staticmethod(datetime.fromtimestamp)
 sig=types.SimpleNamespace(SIGINT=2,SIGTERM=15,SIGALRM=14,ITIMER_REAL=0,getsignal=lambda n:None,getitimer=lambda n:(0.0,0.0),signal=lambda *a:None,setitimer=lambda *a:None)
 env.update(time=types.SimpleNamespace(monotonic=lambda:now["m"],time=lambda:now["epoch"]),signal=sig,datetime=DT,ref=lambda p:P(str(p)))
 def current(v,margin=0):
  created=datetime.fromisoformat(v["created_at_utc"]);expires=datetime.fromisoformat(v["expires_at_utc"])
  need(created.tzinfo and expires.tzinfo and created.timestamp()<=now["epoch"]and now["epoch"]+margin<expires.timestamp(),"Current actual ROOT authority/margin")
 c.current=current
 results=[]
 with contextlib.redirect_stdout(io.StringIO()):
  for stage in("convert","native","seal"):
   now.update(m=100.0,epoch=2000000000.0);clock=env["PhaseClock"]()
   z=dict(stage=stage,created_at_utc=datetime.fromtimestamp(now["epoch"]-1,timezone.utc).isoformat(),expires_at_utc=datetime.fromtimestamp(now["epoch"]+env["phase_budget"](stage)["combined_seconds"]+1200,timezone.utc).isoformat())
   clock.configure(z);now["m"]+=3085;now["epoch"]+=3085
   frame=dict(schema="FLARE_worker_bootstrap_ready_v2",input_ref=request_record,pid=123,monotonic=now["m"],utc=DT.now(timezone.utc).isoformat(),initial_metadata_storage_and_all_roles_validated=True,actual_role_count=roles[stage]["union_count"],worker_source_ref=P(env["ENTRY"]))
   env["validate_worker_frame"](frame,request_record,123,1,clock);grant=clock.release(z,frame)
   assert grant["scientific_deadline_monotonic"]-now["m"]==env["phase_budget"](stage)["scientific_seconds"]
   assert clock.work_deadline-now["m"]==env["phase_budget"](stage)["wall_seconds"]-300
   entry=dict(schema="FLARE_worker_scientific_entry_v2",input_ref=request_record,pid=123,monotonic=now["m"],utc=frame["utc"],phase_origin_monotonic=clock.phase_origin,scientific_deadline_monotonic=clock.scientific_deadline)
   env["validate_worker_frame"](entry,request_record,123,2,clock)
   for obj,key,bad in[(frame,"pid",124),(frame,"input_ref",P("wrong")),(frame,"worker_source_ref",P("wrong")),(frame,"initial_metadata_storage_and_all_roles_validated",False),(frame,"actual_role_count",65),(frame,"utc","2099-01-01T00:00:00+00:00"),(frame,"monotonic",float("nan")),(entry,"phase_origin_monotonic",0),(entry,"scientific_deadline_monotonic",0)]:
    prior=obj[key];obj[key]=bad;reject(lambda:env["validate_worker_frame"](obj,request_record,123,1 if obj is frame else 2,clock));obj[key]=prior;negative.append("status_"+key)
   results.append(dict(stage=stage,setup_elapsed=3085,full_scientific_seconds=grant["scientific_deadline_monotonic"]-clock.phase_origin,work_seconds=clock.work_deadline-clock.phase_origin,combined_seconds=clock.combined_deadline-clock.started))
   clock.close()
  now.update(m=100.,epoch=2000000000.);clock=env["PhaseClock"]();clock.configure(z);now["m"]=7300.;now["epoch"]+=7200
  reject(lambda:clock.release(z,frame));assert clock.phase_origin is None
  try:clock.cancelled(15,None)
  except RuntimeError:pass
  assert clock.cancel[0]==15 and clock.failure_kind=="CANCELLED"
  for kind in("SETUP_CUTOFF","WORK_CUTOFF","AUTHORITY_CUTOFF","EVIDENCE_CUTOFF"):
   clock.alarm_kind=kind
   reject(lambda:clock.alarm(14,None));assert clock.failure_kind==kind;negative.append(kind)
  reject(lambda:clock.arm(now["m"]-1,"WORK_CUTOFF"));assert clock.failure_kind=="WORK_CUTOFF";negative.append("expired_alarm")
  clock.closing=True;clock.cancelled(2,None);assert clock.cancel[0]==2;clock.close()
 for name in("ROOT_request.INACTIVE.template.v1.json","ROOT_directive.INACTIVE.template.v1.json"):
  v=json.loads((H/name).read_text());assert v["active"]is False and v["phase_budget"]==env["phase_budget"]("convert")and v["prior_failure_acceptance_ref"]is None
 assert json.loads((H/"prior_failure_acceptance.INACTIVE.template.v1.json").read_text())["accepted"]is False
 return dict(schema="FLARE_phase_AST_memory_metadata_checks_v2",status="PASS_SOURCE_ONLY",independent=False,no_entrypoint_guard_state_resource_process_payload_data_numerics_or_Git=True,source_syntax_compiled_in_memory=True,scientific_source_sha256=scientific,original_cleanup_wait_drain_AST_identical=True,initial_prelude_AST_exact=True,initial_storage_one_synthetic_original_mutex_one_actual_view=True,original_consumer_memo_sha256=b["consumer_memo_ref"]["sha256"],no_added_memo=True,handwritten_roles=roles,phase_cases=results,rejected_cases=len(negative),rejected_labels=negative)
if __name__=="__main__":print(json.dumps(main(),sort_keys=True,indent=2))
