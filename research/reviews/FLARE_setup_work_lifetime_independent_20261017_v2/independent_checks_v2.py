"""Independent FLARE whole integration: source/AST and handwritten memory only."""
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import ast,collections,contextlib,copy,hashlib,io,json,runpy,sys
BASE=Path("research/reviews/FLARE_setup_work_lifetime_independent_20261017_v2")
SRC=Path("research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v2")
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
def need(ok,msg):
 if not ok:raise ValueError(msg)
def ref(p):
 p=Path(p);p=p if p.is_absolute()else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def raw(r):
 need(Path(r["path"]).suffix in (".json",".py",".md",".patch",".sh"),"Source/compact metadata suffix only")
 data=(ROOT/r["path"]).read_bytes();need(hashlib.sha256(data).hexdigest()==r["sha256"],"Pinned "+r["path"]);return data
def fn(text,name):
 return next(n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
def asts(nodes):return [ast.dump(x,include_attributes=False)for x in nodes]
def isolate(text,names,env):
 nodes=[copy.deepcopy(fn(text,n))for n in names]
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),"<independent-isolated-source>","exec"),env)
def reject(label,call,out,allowed=(ValueError,KeyError,TypeError,RuntimeError)):
 try:call()
 except allowed as error:out.append(dict(case=label,error=str(error)));return
 raise AssertionError("Unexpected acceptance "+label)
def main():
 expected={"root_driver_v1.py":"c1d6a5bb835800f30d0c1e487c885ede13db253c18286b20bc115d8d70a7447f","worker_entry_v1.py":"13f5d7a124fd188e6c48a17bab4c0c6389f98abed9ec0fcfff6caa1d8228e9b6","source_manifest.v1.json":"89c542adbee9e360465c560b0fc343b8912816746221846e19ffd346f44e9141","source_handoff.v1.json":"94fab1267423d1992c5b63108202530881aaad4c47b9a719d5883239ba234fae","source_audit.v1.json":"9f0da9d764d0123173133f803ac9d3b06ae62b64a4455bd263d799be55dbe164"}
 for n,h in expected.items():need(ref(SRC/n)["sha256"]==h,"Exact candidate "+n)
 manifest=json.loads((SRC/"source_manifest.v1.json").read_text())
 def pin(path):return dict(path=path,sha256=manifest["source_sha256"][path])if path in manifest["source_sha256"]else next(r for r in manifest["metadata_refs"]if r["path"]==path)
 for path,h in manifest["source_sha256"].items():raw(dict(path=path,sha256=h))
 for r in manifest["metadata_refs"]:raw(r)
 handoff=json.loads((SRC/"source_handoff.v1.json").read_text());audit=json.loads(raw(handoff["audit_ref"]))
 for key in ("AST_delta_ref","checker_ref","checks_ref","driver_ref","handwritten_roles_ref","worker_ref"):raw(audit[key])
 b=json.loads((SRC/"source_bindings.v1.json").read_text());source=(SRC/"root_driver_v1.py").read_text();worker=(SRC/"worker_entry_v1.py").read_text()
 original_contract=raw(b["original_contract_ref"]).decode();original_driver=raw(b["original_driver_ref"]).decode();original_worker=raw(b["original_worker_ref"]).decode()
 framework=raw(pin("research/reviews/American_TreeLoom_matched_CPU_execution_source_20261017_v4/root_driver_v1.py")).decode()
 child=raw(b["selected_child_source_ref"]).decode()
 for p in SRC.glob("*.py"):compile(p.read_text(),str(p),"exec")
 # All science and original API bodies remain the pinned original sources.
 old_manifest=json.loads(raw(b["original_manifest_ref"]))
 science={p:h for p,h in old_manifest["source_sha256"].items()if p.startswith("src/arggnn/american_c3_flare_")or p in("src/arggnn/american_c3_common_v1.py","research/reviews/FLARE_consumer_namespace_root_20261015_v1/root_context_980_v1.py")}
 for p,h in science.items():need(manifest["source_sha256"].get(p)==h,"Exact whole scientific source "+p)
 need(science["src/arggnn/american_c3_flare_convert_v1.py"]=="6e55e295226a183548a23258e7230855842bb876b4d2a90875d8f2ad53181ff5","Exact full converter")
 for token in ("super().prepare(name,budget_bytes)","super().finish(p,planned)","value=g.budget_check_v9(*args,**kwargs)","result=module.execute(cfg,writer)"):
  need(token in worker,"Original callable/boundary "+token)
 need(worker.index("runtime_report()==")<worker.index("compact_status(ready_fd,ready)")<worker.index("compact_status(ready_fd,entry)")<worker.index("module=importlib.import_module(c.MODULES"),"Actual setup/ready/release/science order")
 need("return d.admit(record)" in ast.get_source_segment(source,fn(source,"admit")),"Exact original admission call")
 need("guard(False)"in ast.get_source_segment(original_driver,fn(original_driver,"admit")),"Original unmodified mutation route")
 need("TransactionMemo"not in source and "ROOT_mutation_transaction"not in source and "install_consumer"not in source,"No added/stacked memo")
 helper=raw(pin("research/reviews/European_outer8_current_policy_successor_source_20261016_v1/prepare_outer000_v1.py")).decode()
 need("if consumer and not hasattr(g,\"_outer_memo\")"in ast.get_source_segment(helper,fn(helper,"guard")),"Original consumer-only namespace memo route")
 ntry=next(n for n in fn(source,"owned_execute").body if isinstance(n,ast.Try));otry=next(n for n in fn(framework,"execute").body if isinstance(n,ast.Try))
 def cleanup(seq):
  i=next(i for i,n in enumerate(seq)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="stop"for t in n.targets))
  return asts(seq[i:])
 need(cleanup(ntry.finalbody)==cleanup(otry.finalbody),"Entire original finite cleanup/wait/drain AST")
 for text in ('g.budget_check_v9(s["storage"],boundary=True,required_write_bytes=32768)','boundary(g,s,32768,target="aux")','boundary(g,s,len(tail)*6+8192,target="aux")'):
  need(text in source,"Every original AUX growth boundary")
 need("clock=PhaseClock()"in ast.get_source_segment(source,fn(source,"execute")).split("activate()")[0],"Setup timer precedes metadata activation")
 need('if rc!=0 or any(x["returncode"]!=0 for x in adopted)'in source and "if not drained" in source,"Genuine nonzero/unknown waits fail")
 need("no_completion_or_automatic_retry_claim=True"in source and "outer_tool_terminal_consumed=False"in source,"No fabricated completion or outer acceptance")
 # Inspect and rerun only the author's already-read AST/memory checker.
 author_result=runpy.run_path(str(SRC/"metadata_checks_v1.py"))["main"]()
 sealed_author=json.loads(raw(audit["checks_ref"]))
 need(author_result==sealed_author,"Independent rerun exactly matches sealed AST/memory result")
 # Independent PhaseClock/status tests with exact original current() and fake clock/signals.
 bad=[];clock_now={"m":100.0,"epoch":2000000000.0};signal_calls=[]
 class DT:
  @staticmethod
  def now(tz=None):return datetime.fromtimestamp(clock_now["epoch"],tz or timezone.utc)
  fromisoformat=staticmethod(datetime.fromisoformat);fromtimestamp=staticmethod(datetime.fromtimestamp)
 signal=SimpleNamespace(SIGINT=2,SIGTERM=15,SIGALRM=14,ITIMER_REAL=0,getsignal=lambda n:None,getitimer=lambda n:(0.,0.),
  signal=lambda *a:signal_calls.append(("handler",a[0])),setitimer=lambda *a:signal_calls.append(("timer",*a)))
 env=dict(Path=Path,ROOT=ROOT,HERE=str(SRC),SOURCE=str(SRC/"root_driver_v1.py"),ENTRY=str(SRC/"worker_entry_v1.py"),MANIFEST=str(SRC/"source_manifest.v1.json"),HANDOFF=str(SRC/"source_handoff.v1.json"),
  SETUP_SECONDS=7200,datetime=DT,timezone=timezone,signal=signal,time=SimpleNamespace(monotonic=lambda:clock_now["m"],time=lambda:clock_now["epoch"]),need=need,json=json,copy=copy,ast=ast,ref=ref,raw=raw,bindings=lambda:b)
 isolate(helper,["current"],env);c=SimpleNamespace(current=env["current"]);env["c"]=c
 isolate(source,["phase_budget","PhaseStop","PhaseClock","validate_worker_frame"],env)
 phase_cases=[]
 with contextlib.redirect_stdout(io.StringIO()):
  for stage in ("convert","native","seal"):
   for elapsed in (0,3085,7199.5):
    clock_now.update(m=100.,epoch=2000000000.);clock=env["PhaseClock"]();budget=env["phase_budget"](stage)
    q=dict(stage=stage,created_at_utc=DT.now(timezone.utc).isoformat(),expires_at_utc=datetime.fromtimestamp(clock_now["epoch"]+budget["combined_seconds"]+1200,timezone.utc).isoformat())
    clock.configure(q);need(signal_calls[-1][-1]==1.0,"Alarm repeats during setup unwinding")
    clock_now["m"]+=elapsed;clock_now["epoch"]+=elapsed
    frame=dict(schema="FLARE_worker_bootstrap_ready_v2",input_ref={"fixture":"input"},pid=123,monotonic=clock_now["m"],utc=DT.now(timezone.utc).isoformat(),worker_source_ref=ref(SRC/"worker_entry_v1.py"),initial_metadata_storage_and_all_roles_validated=True,actual_role_count=60)
    env["validate_worker_frame"](frame,{"fixture":"input"},123,1,clock);grant=clock.release(q,frame)
    need(clock.scientific_deadline-clock.phase_origin==budget["scientific_seconds"],"Full unchanged science after actual ready")
    need(clock.work_deadline-clock.phase_origin==budget["wall_seconds"]-300,"Full original supervisor work reserve")
    need(clock.phase_origin+budget["wall_seconds"]<=clock.combined_deadline,"Entire work/evidence fits combined lifetime")
    entry=dict(schema="FLARE_worker_scientific_entry_v2",input_ref={"fixture":"input"},pid=123,monotonic=clock_now["m"],utc=frame["utc"],phase_origin_monotonic=clock.phase_origin,scientific_deadline_monotonic=clock.scientific_deadline)
    env["validate_worker_frame"](entry,{"fixture":"input"},123,2,clock)
    for obj,key,value in ((frame,"actual_role_count",65),(frame,"actual_role_count",True),(frame,"pid",124),(frame,"worker_source_ref",{}),(frame,"monotonic",float("nan")),(entry,"phase_origin_monotonic",0)):
     original=obj[key];obj[key]=value;reject(stage+":"+str(elapsed)+":"+key,lambda:env["validate_worker_frame"](obj,{"fixture":"input"},123,1 if obj is frame else 2,clock),bad);obj[key]=original
    clock.stop_work_alarm();clock_now["m"]=clock.work_deadline+30;clock_now["epoch"]+=budget["wall_seconds"]-270
    clock.evidence_alarm();need(clock.alarm_kind=="EVIDENCE_CUTOFF"and signal_calls[-1][-1]==1.0,"Finite post-drain evidence cap")
    need(signal_calls[-1][-2]<=270.00001,"Only unchanged evidence balance after cleanup")
    phase_cases.append(dict(stage=stage,setup_elapsed=elapsed,scientific_seconds=budget["scientific_seconds"],work_seconds=budget["wall_seconds"]-300,combined_seconds=clock.combined_deadline-clock.started))
    clock.close()
  clock_now.update(m=100.,epoch=2000000000.);clock=env["PhaseClock"]();q=dict(stage="convert",created_at_utc=DT.now(timezone.utc).isoformat(),expires_at_utc=datetime.fromtimestamp(clock_now["epoch"]+12600,timezone.utc).isoformat());clock.configure(q)
  clock_now["m"]+=7200;clock_now["epoch"]+=7200;reject("setup_at_exact_cap",lambda:clock.release(q,{}),bad);need(clock.phase_origin is None,"No release after setup cap")
  clock_now.update(m=100.,epoch=2000000000.);clock=env["PhaseClock"]();clock.configure(q);clock_now["epoch"]+=7800
  reject("ready_without_full_authority_margin",lambda:clock.release(q,{}),bad);need(clock.phase_origin is None,"No shortened work window")
  clock_now.update(m=100.,epoch=2000000000.)
  short=dict(q,expires_at_utc=datetime.fromtimestamp(clock_now["epoch"]+12000,timezone.utc).isoformat())
  reject("entry_without_full_combined_margin",lambda:clock.configure(short),bad)
  for kind in ("SETUP_CUTOFF","WORK_CUTOFF","AUTHORITY_CUTOFF","EVIDENCE_CUTOFF"):
   clock.alarm_kind=kind;reject(kind,lambda:clock.alarm(14,None),bad);need(clock.failure_kind==kind,"Distinct phase failure")
  reject("actual_cancel15",lambda:clock.cancelled(15,None),bad);need(clock.cancel[0]==15 and clock.failure_kind=="CANCELLED","Actual signal retained")
  clock.stop_work_alarm();clock.cancelled(2,None);need(clock.cancel[0]==2,"Cleanup cancellation retained without interruption");clock.close()
 # Independent wrapper roles and failure metadata, using original direct_roles body.
 deps=json.loads((ROOT/"research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v1/dependencies.v1.json").read_text())
 guards=json.loads(raw(b["actual_guard_metadata_ref"]))["bindings"];selected=json.loads(raw(b["actual_selector_metadata_ref"]))
 P=lambda n:dict(path="HANDWRITTEN/"+n,sha256=hashlib.sha256(n.encode()).hexdigest())
 records={};sel=SimpleNamespace(SELECTOR_REF=None,_CHOSEN=None,SELECTOR=None,GUARD_REF=None,POLICY_REF=None)
 def select(r):
  need(r==b["actual_selector_metadata_ref"],"Pinned selector");return copy.deepcopy(selected)
 def bind(r):
  v=select(r);sel.SELECTOR_REF=r;sel._CHOSEN=v;sel.SELECTOR=v["selector_source_ref"];sel.GUARD_REF=v["source_ref"];sel.POLICY_REF=v["policy_ref"];return v
 sel.selected=select;sel.bind_selector=bind;sel.guard=lambda consumer:("old_consumer_memo"if consumer else"raw_original_mutator")
 c=SimpleNamespace(require=need,DEPS=deps,ROOT=ROOT,CAPS={"reservation_bytes":889641408},read=lambda r:copy.deepcopy(records[r["path"]]),raw=lambda r:r,local=lambda p:ROOT/p,ref=ref,validate_request=lambda r:("original_validate_reached",r),
  SELECTOR_REF=None,_CHOSEN=None,SELECTOR=None,GUARD_REF=None,POLICY_REF=None)
 c.__dict__.update(NAMESPACE=deps["NAMESPACE"],PIDFD=deps["PIDFD"],OWNED=deps["OWNED"],FRAMEWORK=deps["FRAMEWORK"],COMPLETION_REVIEW=deps["COMPLETION_REVIEW"],
  PRODUCERS=dict(convert="src/arggnn/american_c3_flare_convert_v1.py",native="src/arggnn/american_c3_flare_run_v2.py",seal="src/arggnn/american_c3_flare_evaluate_v1.py"),
  completion=lambda:SimpleNamespace(_context=lambda *a,**k:{"targets":{"american_c3_val_001":{"count_manifest_ref":P("count")}}}),make_config=lambda q:{"binary_provenance_ref":deps["BINARY_PROVENANCE"]})
 isolate(original_contract,["direct_roles"],c.__dict__)
 env.update(c=c,load=lambda r,name:c if r==b["original_contract_ref"]else sel,datetime=datetime)
 isolate(source,["contract","initial_storage_transaction"],env);c=env["contract"]();env["c"]=c;c.bind_selector(b["actual_selector_metadata_ref"])
 need(c.guard(True)=="old_consumer_memo"and c.guard(False)=="raw_original_mutator","Original distinct consumer/mutator route")
 q=dict(stage="convert",parent_id="american_c3_val_001",context_ref=P("context"),source_review_ref=P("review"),runtime_ref=P("runtime"),root_directive_ref=P("directive"),registration_ref=P("registration"),storage_policy_selector_ref=b["actual_selector_metadata_ref"],
  preparation_directory="research/reviews/FLARE_remaining_ROOT_preparation_HANDWRITTEN",packet_directory="research/reviews/FLARE_remaining_ROOT_actual_HANDWRITTEN",output_directory="data/FLARE_original_remaining_HANDWRITTEN",
  phase_budget=env["phase_budget"]("convert"),prior_failure_acceptance_ref=P("acceptance"),created_at_utc="2030-01-01T00:00:00+00:00",expires_at_utc="2030-01-01T04:00:00+00:00")
 accepted=dict(b["old_failure"],schema="ROOT_FLARE_failed_unsealed_acceptance_v2",root_authorizer="root_agent",accepted=True,old_launch_resume_write_authority_revoked=True,old_files_scopes_key_charge_clocks_and_failure_unchanged=True,original_conversion_unsealed=True)
 records.update({P("request")["path"]:q,P("acceptance")["path"]:accepted,P("directive")["path"]:dict(phase_budget=q["phase_budget"],prior_failure_acceptance_ref=P("acceptance")),P("native")["path"]:dict(binary_provenance_ref=deps["BINARY_PROVENANCE"])})
 need(c.validate_request(P("request"))[0]=="original_validate_reached","All original request validation still called")
 for field in b["old_failure"]:
  old=accepted[field];accepted[field]=True if old is None else {"changed":True}
  reject("failure_acceptance:"+field,lambda:c.validate_request(P("request")),bad);accepted[field]=old
 old=q["phase_budget"];q["phase_budget"]={};reject("wrong_phase_budget",lambda:c.validate_request(P("request")),bad);q["phase_budget"]=old
 role_cases={}
 unique=lambda seq:list({(r["path"],r["sha256"]):r for r in seq}.values())
 for stage in ("convert","native","seal"):
  z=dict(q,stage=stage,conversion_ref=P("conversion"),native_run_ref=P("native"),previous_stage_evidence_ref=P("previous"))
  direct=c.direct_roles(P("request"),z,P("config"),P("closure"))
  manual=[ref(SRC/"source_manifest.v1.json"),ref(SRC/"root_driver_v1.py"),ref(SRC/"worker_entry_v1.py"),ref(SRC/"source_handoff.v1.json"),ref(c.PRODUCERS[stage]),deps["COMPLETION_REVIEW"],P("review"),P("runtime"),P("config"),P("directive"),P("request"),P("context"),P("count"),P("registration"),P("closure"),b["selected_child_source_ref"],b["actual_selector_metadata_ref"],deps["NAMESPACE"],deps["PIDFD"],deps["OWNED"],deps["FRAMEWORK"],deps["SELECTOR"],P("acceptance")]
  if stage!="convert":manual += [P("conversion"),P("previous"),deps["BINARY_PROVENANCE"]]
  if stage=="seal":manual += [P("native")]
  need(set((r["path"],r["sha256"])for r in direct)==set((r["path"],r["sha256"])for r in manual),"Complete handwritten role equality")
  union=unique([*guards.values(),*direct]);role_cases[stage]=dict(guard_names=len(guards),guard_distinct=len(unique(guards.values())),total=len(union),all_roles_preserved=True)
 need([role_cases[x]["total"]for x in ("convert","native","seal")]==[60,63,64],"Unraised original ceiling")
 # Actual check_storage_locked_v1 + original prelude + union, all over a memory state.
 events=[];held={"value":False,"drift":False};impl=next(r for r in guards.values()if r["path"]=="scripts/project_budget_v9.py");launcher=next(r for r in guards.values()if r["path"]=="scripts/launch_reserved_gpu_v6.py")
 class Lock:
  def __enter__(self):need(not held["value"],"Original mutex only");held["value"]=True;events.append("enter")
  def __exit__(self,*args):
   events.append("uncached_final_validation");held["value"]=False;events.append("unlock");need(not held["drift"],"Metadata drift at original memo exit")
 def memraw(r):
  need(held["value"],"Bound reads inside original transaction");need(r.get("sha256")!="BAD","Changed exact metadata");return b"handwritten"
 policy=dict(implementation_ref=impl,launcher_ref=launcher,source_ref=b["selected_child_source_ref"],base_pins={"fixture":"base"})
 own=dict(key="HANDWRITTEN_KEY",reserved_bytes=889641408,output=str(ROOT/q["output_directory"]),auxiliary_scopes=[str(ROOT/q["packet_directory"])])
 direct=c.direct_roles(P("request"),q,P("config"),P("closure"));own["consumer_bindings"]=unique([*guards.values(),*direct])
 storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode="Relate_failed64GiB_retention_v1",job_id="J",output_directory=q["output_directory"],reserved_new_bytes=889641408,reservation=own,
  completed_overlay_policy_ref=selected["policy_ref"],budget_components=dict(implementation=impl,launcher=launcher,consumer_adapter=b["selected_child_source_ref"]),initial_ledger_pins=dict(base_pins=policy["base_pins"],completed_overlay_prefix={"fixture":"prefix"}))
 state=({own["key"]:own},set(),{"bindings":guards},guards,[],[],{own["key"]:1})
 def state_locked(*args):need(held["value"],"State under original mutex");events.append("state_replay");return state
 def measure(*args):need(held["value"],"Scope measured under original mutex");events.append("fresh_scope_measure");return {"conservative_scope_bytes":0}
 budget=SimpleNamespace(v4=SimpleNamespace(mutex=lambda root:Lock(),integer=lambda n,label:need(type(n)is int and n>=0,"Original integer"),canonical=lambda root,p:root/p,measure_scopes=measure))
 g=SimpleNamespace(require=need,copy=copy,ROOT=ROOT,MODE="Relate_failed64GiB_retention_v1",_POLICY_REF=selected["policy_ref"],_raw=memraw,
  _expire=lambda s:events.append("fresh_expiry"),_policy=lambda:policy,_state_locked=state_locked,assert_open=lambda key,r,cl:need(key in r and key not in cl,"Actual open reservation"),check_prefix=lambda *a,**k:events.append("fresh_prefix"),_paths=lambda own:[own["output"]],budget=budget)
 isolate(child,["check_storage_locked_v1","consumer_bindings_v1"],g.__dict__)
 s=dict(job_id="J",output_directory=q["output_directory"],refs=dict(config=P("config")),runtime_ref=P("runtime"),storage=storage,root_stage_request_ref=P("request"),transitive_source_closure_ref=P("closure"))
 original_cap=c.CAPS;c.CAPS={"reservation_bytes":889641408}
 view=env["initial_storage_transaction"](s,q,g)
 need(view["bindings"]==guards and events.count("enter")==events.count("unlock")==events.count("state_replay")==events.count("fresh_scope_measure")==1,"One actual-state read-only storage transaction")
 need(events[-2:]==["uncached_final_validation","unlock"],"Original memo validates before unlock")
 baseline=copy.deepcopy(s)
 for case in ("missing_direct","wrong_job","wrong_output","oversized","inactive","bad_source","memo_final_drift","role65"):
  s=copy.deepcopy(baseline);prior_roles=c.direct_roles
  if case=="missing_direct":s["storage"]["reservation"]["consumer_bindings"].remove(P("config"))
  if case=="wrong_job":s["storage"]["job_id"]="other"
  if case=="wrong_output":s["storage"]["output_directory"]="data/other"
  if case=="oversized":s["storage"]["reserved_new_bytes"]=889641409
  if case=="inactive":s["storage"]["active"]=False
  if case=="bad_source":s["refs"]["config"]["sha256"]="BAD";s["storage"]["reservation"]["consumer_bindings"].append(s["refs"]["config"])
  if case=="memo_final_drift":held["drift"]=True
  if case=="role65":c.direct_roles=lambda *a:[P("extra"+str(i))for i in range(65)]
  reject("actual_storage:"+case,lambda:env["initial_storage_transaction"](s,q,g),bad);held["drift"]=False;c.direct_roles=prior_roles;need(not held["value"],"No transaction leak")
 # Exact original review-envelope predicate against its current identities.
 review_env=dict(independent=True,verdict="SOURCE_PASS",source_ref=ref(SRC/"root_driver_v1.py"),source_manifest_ref=ref(SRC/"source_manifest.v1.json"),source_handoff_ref=ref(SRC/"source_handoff.v1.json"),blockers=[])
 review_current={"value":review_env}
 scope=dict(read=lambda r:review_current["value"],require=need,ref=ref,DRIVER=str(SRC/"root_driver_v1.py"),MANIFEST=str(SRC/"source_manifest.v1.json"),HANDOFF=str(SRC/"source_handoff.v1.json"))
 isolate(original_contract,["reviewed"],scope);scope["reviewed"]({"fixture":"review"})
 for field in review_env:
  val=copy.deepcopy(review_env);val.pop(field);review_current["value"]=val;reject("actual_reviewed_missing:"+field,lambda:scope["reviewed"]({}),bad)
 # Exact terminal-evidence catch demonstrates its stale snapshot on late failures.
 owned=fn(source,"owned_execute");terminal=next(n for n in reversed(owned.body)if isinstance(n,ast.Try));handler=terminal.handlers[0]
 catch=ast.FunctionDef(name="actual_terminal_handler",args=ast.arguments(posonlyargs=[],args=[ast.arg(arg="observation"),ast.arg(arg="exc"),ast.arg(arg="clock")],kwonlyargs=[],kw_defaults=[],defaults=[]),body=copy.deepcopy(handler.body),decorator_list=[])
 catch_env=dict(json=json);exec(compile(ast.fix_missing_locations(ast.Module(body=[catch],type_ignores=[])),"<exact-original-terminal-handler>","exec"),catch_env)
 gaps=[]
 for cause,number in (("CANCELLED",15),("EVIDENCE_CUTOFF",None)):
  before=dict(phase_clock=dict(cancellation_signal=None,failure_kind=None),operational_failure=None,actual_direct_worker_wait=True,returncode=0,all_owned_processes_and_groups_drained=True)
  actual_clock=SimpleNamespace(evidence=lambda:dict(cancellation_signal=number,failure_kind=cause))
  captured=io.StringIO()
  with contextlib.redirect_stdout(captured):result=catch_env["actual_terminal_handler"](before,env["PhaseStop"](cause),actual_clock)
  printed=json.loads(captured.getvalue())
  need(result==dict(status="FAILED_EVIDENCE",returncode=1),"Original handler remains fail-closed")
  need(printed["actual_wait_observation"]["phase_clock"]==dict(cancellation_signal=None,failure_kind=None),"Reproduce stale actual clock metadata")
  need(printed["evidence_failure"]==cause,"Reason string survives")
  gaps.append(dict(handwritten_memory_only=True,cause=cause,current_clock=actual_clock.evidence(),printed=printed,return_value=result))
 need(not any(n in sys.modules for n in ("torch","numpy","torch_geometric","tskit")),"No numeric/scientific imports")
 return dict(schema="independent_FLARE_phase_whole_integration_checks_v2",verdict="SOURCE_BLOCKED",source_only=True,
  source_ref=ref(SRC/"root_driver_v1.py"),source_manifest_ref=ref(SRC/"source_manifest.v1.json"),source_handoff_ref=ref(SRC/"source_handoff.v1.json"),
  authenticated_source_files=len(manifest["source_sha256"]),authenticated_metadata_refs=len(manifest["metadata_refs"]),scientific_source_sha256=science,
  author_AST_memory_check_exact_rerun=True,author_negative_cases=author_result["rejected_cases"],
  original_cleanup_wait_drain_AST_exact=True,all_scientific_caps_callables_and_write_boundaries_preserved=True,
  original_consumer_memo_and_raw_mutation_route_preserved=True,
  phase_cases=phase_cases,handwritten_roles=role_cases,actual_original_storage_predicates_exercised=True,
  initial_storage_one_mutex_one_replay_one_scope_measure=True,actual_complete_original_review_predicate_pass=True,
  negative_cases=bad,late_terminal_reporting_gap=gaps,actual_entrypoints_guards_state_resources_processes_arrays_payloads_numerics_admissions_jobs_or_Git=False,
  note="Capacity, setup fit, real elapsed improvement and future ROOT authority remain unproven; no live operation is authorized.")

if __name__=="__main__":print(json.dumps(main(),indent=2,sort_keys=True))
