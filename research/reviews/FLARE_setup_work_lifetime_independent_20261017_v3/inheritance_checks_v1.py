"""Independent v3 review by exact v2 inheritance and terminal-handler memory fixtures."""
from pathlib import Path
from types import SimpleNamespace
import ast,contextlib,copy,hashlib,io,json,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OLD=Path("research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v2")
NEW=Path("research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v3")
PRIOR=Path("research/reviews/FLARE_setup_work_lifetime_independent_20261017_v2")
def ref(p):
 p=Path(p);p=p if p.is_absolute()else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def raw(r):
 assert Path(r["path"]).suffix in (".py",".json",".md",".patch",".sh")
 p=ROOT/r["path"];data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==r["sha256"],r["path"];return data
def fn(tree,name):return next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
def dump(n):return ast.dump(n,include_attributes=False)
def main():
 pins={"root_driver_v1.py":"33224366b726f048c792f484a59d31722700789f41a3c70ef71a3bcaaa2bc882","source_manifest.v1.json":"217e436d656236345b6a8233cbc68b450629dc4df8f6d770a5463e07950b6283","source_handoff.v1.json":"6606ffca9dd85c920931063d0ed4cf98f9213d58f0da591a30ad8b854b6daa33","source_audit.v1.json":"94c6f22ff101bd5e16347693a5f988089dcd2c35eecdae70bdc5333e881de6b5","metadata_checks.result.v1.json":"0cec2ce2be7535d7b49d3ca0bddef058f3ea4d8917489ce942a287548d31c678","worker_entry_v1.py":"13f5d7a124fd188e6c48a17bab4c0c6389f98abed9ec0fcfff6caa1d8228e9b6"}
 for n,h in pins.items():assert ref(NEW/n)["sha256"]==h,n
 assert ref(PRIOR/"review.v2.json")["sha256"]=="4cbf25c67f9cb12ca08595330fb3366e80616383cf83e2666026660a394792f6"
 assert ref(PRIOR/"review_manifest.v2.json")["sha256"]=="6a8ab0025189c0aa0ec4dd9a18613740257a4895e23682741df1674e5c2b1333"
 assert ref(PRIOR/"independent_checks.v2.json")["sha256"]=="cd822733956577658cc77538a9f6da4380b25f7bc605e1c58d18b00a371494a5"
 prior=json.loads((PRIOR/"review.v2.json").read_text());assert [b["id"]for b in prior["blockers"]]==["FLARE_V2_LATE_EVIDENCE_STALE_CLOCK"]
 m2=json.loads((OLD/"source_manifest.v1.json").read_text());m3=json.loads((NEW/"source_manifest.v1.json").read_text())
 for path,h in m3["source_sha256"].items():raw(dict(path=path,sha256=h))
 for r in m3["metadata_refs"]:raw(r)
 assert all(m3["source_sha256"].get(p)==h for p,h in m2["source_sha256"].items()),"Full unchanged v2 source closure carried forward"
 handoff=json.loads((NEW/"source_handoff.v1.json").read_text());audit=json.loads(raw(handoff["audit_ref"]))
 assert handoff["source_ref"]==audit["driver_ref"]==ref(NEW/"root_driver_v1.py")
 assert handoff["source_manifest_ref"]==ref(NEW/"source_manifest.v1.json")and handoff["worker_ref"]==audit["worker_ref"]==ref(NEW/"worker_entry_v1.py")
 assert handoff["check_result_ref"]==audit["checks_ref"]==ref(NEW/"metadata_checks.result.v1.json")
 for k in("AST_delta_ref","checker_ref","checks_ref","driver_ref","handwritten_roles_ref","worker_ref"):raw(audit[k])
 for k in("recipe_ref","source_delta_ref"):raw(handoff[k])
 inheritance=json.loads((NEW/"v2_to_v3_inheritance.v1.json").read_text())
 for k in("driver_ref","predecessor_driver_ref","source_delta_ref"):raw(inheritance[k])
 b2=json.loads((OLD/"source_bindings.v1.json").read_text());b3=json.loads((NEW/"source_bindings.v1.json").read_text())
 for k,v in b2.items():assert b3.get(k)==v,"Unchanged original binding "+k
 extras={k:v for k,v in b3.items()if k not in b2}
 assert extras and all("predecessor_v2"in k for k in extras),extras
 for k,v in extras.items():raw(v)
 oldtext=(OLD/"root_driver_v1.py").read_text();newtext=(NEW/"root_driver_v1.py").read_text()
 oldtree=ast.parse(oldtext);newtree=ast.parse(newtext.replace(str(NEW),str(OLD)))
 oldowned=fn(oldtree,"owned_execute");newowned=fn(newtree,"owned_execute")
 oldhandler=next(n for n in reversed(oldowned.body)if isinstance(n,ast.Try)).handlers[0]
 newhandler=next(n for n in reversed(newowned.body)if isinstance(n,ast.Try)).handlers[0]
 assert len(oldhandler.body)==2 and len(newhandler.body)==4
 assert [dump(n)for n in newhandler.body[-2:]]==[dump(n)for n in oldhandler.body],"Original print and failure1 return preserved"
 addition=ast.parse("clock.failure_kind=exc.kind if isinstance(exc,PhaseStop)else('AUTHORITY_CUTOFF'if'Current actual ROOT authority/margin'in str(exc)else clock.failure_kind or'EXCEPTION')\nobservation.update(phase_clock=clock.evidence(),prior_operational_failure=observation.get('operational_failure'),operational_failure=type(exc).__name__+': '+str(exc)[:2000])").body
 assert [dump(n)for n in newhandler.body[:2]]==[dump(n)for n in addition],"Only exact reviewed metadata refresh added"
 saved_handler=copy.deepcopy(newhandler)
 newhandler.body=copy.deepcopy(oldhandler.body)
 assert dump(newtree)==dump(oldtree),"Whole driver AST identical except HERE and one terminal handler"
 assert (NEW/"worker_entry_v1.py").read_bytes()==(OLD/"worker_entry_v1.py").read_bytes(),"Worker bytes identical"
 for n in("__init__.py","metadata_checks_v1.py","prepare_context_v1.py"):
  assert (NEW/n).read_text().replace(str(NEW),str(OLD))==(OLD/n).read_text(),"Source identity only "+n
 oldseal=(OLD/"seal_source_v1.py").read_text();newseal=(NEW/"seal_source_v1.py").read_text().replace(str(NEW),str(OLD))
 assert newseal.replace(',b["predecessor_v2_source_manifest_ref"]','')==oldseal,"Source-only sealer adds only exact prior closure join"
 for p in NEW.glob("*.py"):compile(p.read_text(),str(p),"exec")
 author3=json.loads((NEW/"metadata_checks.result.v1.json").read_text());author2=json.loads((OLD/"metadata_checks.result.v1.json").read_text())
 assert author3["rejected_cases"]==author2["rejected_cases"]==82
 assert author3["scientific_source_sha256"]==author2["scientific_source_sha256"]
 assert author3["phase_cases"]==author2["phase_cases"]
 assert {k:v["union_count"]for k,v in author3["handwritten_roles"].items()}=={"convert":60,"native":63,"seal":64}
 assert handoff["phase_budget"]==json.loads((OLD/"source_handoff.v1.json").read_text())["phase_budget"]
 # Only isolated exception class + exact current terminal catch, no module entry.
 scope=dict(json=json)
 cls=copy.deepcopy(fn(ast.parse(newtext),"PhaseStop"))
 args=ast.arguments(posonlyargs=[],args=[ast.arg(arg="observation"),ast.arg(arg="exc"),ast.arg(arg="clock")],kwonlyargs=[],kw_defaults=[],defaults=[])
 catch=ast.FunctionDef(name="actual_terminal_handler",args=args,body=copy.deepcopy(saved_handler.body),decorator_list=[])
 exec(compile(ast.fix_missing_locations(ast.Module(body=[cls,catch],type_ignores=[])),"<exact-v3-terminal-handler-only>","exec"),scope)
 cases=[]
 for kind,number,error in(("CANCELLED",15,scope["PhaseStop"]("CANCELLED")),("CANCELLED",2,scope["PhaseStop"]("CANCELLED")),("EVIDENCE_CUTOFF",None,scope["PhaseStop"]("EVIDENCE_CUTOFF")),("AUTHORITY_CUTOFF",None,ValueError("Current actual ROOT authority/margin")),("EXCEPTION",None,OSError("handwritten write failure"))):
  for prior_failure in(None,"Prior genuine worker failure"):
   observation=dict(phase_clock=dict(cancellation_signal=None,failure_kind=None),operational_failure=prior_failure,
    actual_direct_worker_wait=True,actual_OS_wait_status=0,returncode=0,all_owned_processes_and_groups_drained=True,adopted_waits=[dict(pid=321,status=0,returncode=0)],completed_ref=None)
   before=copy.deepcopy(observation);clock=SimpleNamespace(failure_kind=None)
   clock.evidence=lambda:dict(cancellation_signal=number,failure_kind=clock.failure_kind)
   captured=io.StringIO()
   with contextlib.redirect_stdout(captured):result=scope["actual_terminal_handler"](observation,error,clock)
   printed=json.loads(captured.getvalue());out=printed["actual_wait_observation"]
   assert out["phase_clock"]==dict(cancellation_signal=number,failure_kind=kind)
   assert out["prior_operational_failure"]==prior_failure and out["operational_failure"]==type(error).__name__+": "+str(error)
   assert printed["evidence_failure"]==str(error)
   assert result==dict(status="FAILED_EVIDENCE",returncode=1)
   assert all(out[k]==before[k]for k in("actual_direct_worker_wait","actual_OS_wait_status","returncode","all_owned_processes_and_groups_drained","adopted_waits","completed_ref"))
   cases.append(dict(handwritten_memory_only=True,kind=kind,signal=number,prior_failure=prior_failure,printed_clock=out["phase_clock"],operational_failure=out["operational_failure"],all_wait_drain_and_product_facts_unchanged=True,return_value=result))
 assert not any(n in sys.modules for n in("torch","numpy","torch_geometric","tskit"))
 return dict(schema="independent_FLARE_v3_exact_inheritance_checks_v1",verdict="SOURCE_PASS",blockers=[],source_only=True,
  source_ref=ref(NEW/"root_driver_v1.py"),source_manifest_ref=ref(NEW/"source_manifest.v1.json"),source_handoff_ref=ref(NEW/"source_handoff.v1.json"),
  predecessor_whole_review_ref=ref(PRIOR/"review.v2.json"),predecessor_checks_ref=ref(PRIOR/"independent_checks.v2.json"),
  authenticated_source_files=len(m3["source_sha256"]),authenticated_metadata_refs=len(m3["metadata_refs"]),
  complete_v2_source_closure_preserved=True,whole_driver_AST_exact_except_namespace_and_two_added_metadata_statements=True,
  worker_byte_identical=True,other_execution_and_context_helpers_namespace_only=True,no_added_bridge=True,
  original_v2_science_guard_memo_caps_roles_timers_waits_drain_barriers_preserved=True,
  late_reporting_fixtures=cases,original_wait_drain_return_and_prior_failure_preserved=True,
  author82_cases_and_role_metadata_match_inherited_source=True,original_independent93_rejections_and9phase_scenarios_inherited_without_rerun=True,
  actual_entrypoints_guards_state_mutex_resources_processes_payloads_arrays_numerics_admissions_jobs_or_Git=False)
if __name__=="__main__":print(json.dumps(main(),indent=2,sort_keys=True))
