"""Focused Relate source review. No imports of project modules or actual payload reads."""
import argparse,ast,contextlib,copy,hashlib,io,json,pathlib,re,sys,traceback
from datetime import datetime,timezone
HERE="research/reviews/Relate_exactALL_native_only_source_20261017_v1"
def need(ok,msg):
 if not ok:raise AssertionError(msg)
def main():
 p=pathlib.Path(HERE);sources={name:(p/name).read_bytes()for name in("root_contract_v1.py","root_driver_v1.py","worker_entry_v1.py")}
 trees={name:ast.parse(data)for name,data in sources.items()}
 for name,t in trees.items():compile(t,str(p/name),"exec")
 b=json.loads((p/"scientific_binding.v1.json").read_bytes())
 deps=json.loads((p/"dependencies.v1.json").read_bytes())
 caps=json.loads((p/"caps.inactive.v1.json").read_bytes())
 commands=json.loads(pathlib.Path(b["source_evidence"]["original_commands_ref"]["path"]).read_bytes())
 need(b["exact_argv"]==commands["commands"][2]["argv"]and commands["commands"][2]["stage"]=="infer_all","Exact original author ALL argv")
 need(b["native_threads"]==caps["CPU_threads"]==1 and b["support"]==dict(diploids=1611,haplotypes=3222,observed_SNPs=311599,span_bp=35178458),"Full original support/threads")
 need(caps["reservation_bytes"]==32*2**30 and caps["native_bytes"]==31*2**30 and caps["all_aux_bytes"]==2**30
  and caps["aux_bytes"]+caps["root_packet_bytes"]==caps["all_aux_bytes"]
  and caps["rss_bytes"]==256*2**30 and caps["scientific_seconds"]==43200,"Exact operational caps")
 functions={name:{n.name:n for n in t.body if isinstance(n,ast.FunctionDef)}for name,t in trees.items()}
 # Execute only the actual metadata predicates with a handwritten read table.
 selected_names={"authenticate_closure","selected","bind_selector"}
 fns=[n for n in trees["root_contract_v1.py"].body if isinstance(n,ast.FunctionDef)and n.name in selected_names]
 table={};reads=[]
 def ref(name):return dict(path=str(name),sha256="a"*64)
 def read(record):reads.append(record);return copy.deepcopy(table[record["path"]])
 def raw(record):reads.append(record);return b"handwritten metadata only"
 scope=dict(HERE=HERE,require=need,read=read,raw=raw,binding=lambda:copy.deepcopy(b),ref=ref,
  re=re,datetime=datetime,timezone=timezone,copy=copy,SELECTOR_REF=None,POLICY_REF=None,_CHOSEN=None,**deps)
 # Preserve real predicate rejection semantics.
 scope["require"]=lambda ok,msg:None if ok else (_ for _ in()).throw(ValueError(msg))
 exec(compile(ast.Module(body=fns,type_ignores=[]),"actual_Relate_metadata_predicates","exec"),scope)
 closure=json.loads((p/"ROOT_five_input_authentication.INACTIVE.template.v1.json").read_bytes())
 closure.update(root_authorizer="root_agent",status="authenticated_readonly",scientific_binding_ref=ref(HERE+"/scientific_binding.v1.json"),
  original_inputs_unchanged=True,no_payload_rewrite_copy_or_normalization=True,authenticated_at_utc="2026-10-16T00:00:00+00:00")
 for i,row in enumerate(closure["files"]):row.update(sha256=("%064x"%(i+1)),bytes=i+1)
 cr=ref("handwritten/closure_not_created.json");table[cr["path"]]=closure
 need(scope["authenticate_closure"](cr)==closure,"Handwritten closure positive")
 checks=["handwritten_closure_accepted_by_actual_predicate"]
 def rejects(label,value):
  table[cr["path"]]=value
  try:scope["authenticate_closure"](cr)
  except(ValueError,TypeError):checks.append(label+"_rejected")
  else:raise AssertionError(label)
 q=copy.deepcopy(closure);q["status"]="INACTIVE_NOT_AUTHENTICATED";rejects("inactive_ROOT_authentication",q)
 q=copy.deepcopy(closure);q["files"]=q["files"][:-1];rejects("missing_original_input",q)
 q=copy.deepcopy(closure);q["files"][0]["sha256"]=None;rejects("unfilled_SHA",q)
 q=copy.deepcopy(closure);q["files"][0]["bytes"]=True;rejects("boolean_byte_count",q)
 q=copy.deepcopy(closure);q["support"]["diploids"]=1;rejects("reduced_support",q)
 # No actual current selector is read; source/profile identity predicates see handbuilt metadata.
 selector=dict(schema="explicit_ROOT_storage_policy_selector_v1",active=True,root_authorizer="root_agent",
  selector_source_ref=deps["SELECTOR"],source_ref=deps["GUARD_REF"],source_manifest_ref=deps["GUARD_MANIFEST"],
  source_review_ref=deps["GUARD_REVIEW"],accounting_mode="Relate_failed64GiB_retention_v1",scheduling_ceiling_bytes=980000000000,
  human_limit_bytes=1000000000000,warning_threshold_bytes=950000000000,no_fallback_discovery_upgrade_or_in_run_swap=True,
  future_ROOT_adapters_only=True,policy_ref=ref("handwritten/policy_not_created.json"),
  repair_authority_ref=ref("handwritten/repair_not_created.json"),measurement_source_ref=ref("handwritten/measurement_not_created.json"))
 sr=ref("handwritten/selector_not_created.json");table[sr["path"]]=selector
 need(scope["bind_selector"](sr)==selector,"Only legal child source interface")
 for label,change in[("revoked_native7_profile",dict(accounting_mode="native7_H050_stopped_retention_v1")),("inactive_selector",dict(active=False))]:
  table[sr["path"]]=dict(selector,**change)
  try:scope["selected"](sr)
  except ValueError:checks.append(label+"_rejected")
  else:raise AssertionError(label)
 table[sr["path"]]=selector
 swap=ref("handwritten/second_selector_not_created.json");table[swap["path"]]=selector
 try:scope["bind_selector"](swap)
 except ValueError:checks.append("lifetime_selector_swap_rejected")
 else:raise AssertionError("selector swap")
 # Default CLI branches only; no module imports or project callbacks.
 defaults={}
 for name in("root_driver_v1.py","worker_entry_v1.py"):
  ns=dict(argparse=argparse,json=json);node=functions[name]["main"]
  exec(compile(ast.Module(body=[node],type_ignores=[]),"actual_inert_default_"+name,"exec"),ns)
  out=io.StringIO()
  with contextlib.redirect_stdout(out):need(ns["main"]([])==0,"Inert0 "+name)
  value=json.loads(out.getvalue());need(value["status"]=="SOURCE_ONLY_INACTIVE"and value["execution"]is False,"Inert result")
  defaults[name]=value
 # Successful non-default worker has no Python return0 path: only exact fdexec.
 worker_fn=functions["worker_entry_v1.py"]["worker"]
 need(not any(isinstance(n,ast.Return)for n in ast.walk(worker_fn)),"No worker Python success return")
 need(isinstance(worker_fn.body[-1],ast.Raise)and isinstance(worker_fn.body[-2],ast.Expr)
  and isinstance(worker_fn.body[-2].value,ast.Call)and ast.unparse(worker_fn.body[-2].value.func)=="os.execve"
  and ast.unparse(worker_fn.body[-2].value.args[0])=="native_fd","Exact binary-fd exec or fail")
 # Actual completion gate expression, with no imports or postwait payload work.
 completion=functions["root_driver_v1.py"]["native_completion"]
 gate=next(n for n in completion.body if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=="need").value.args[0]
 evaluate=lambda v:eval(compile(ast.Expression(gate),"actual_native_completion_gate","eval"),{},dict(observation=v))
 good=dict(returncode=0,actual_direct_native_wait=True,all_owned_processes_and_groups_drained=True,operational_failure=None)
 need(evaluate(good),"Positive native wait0/drain gate")
 for field,bad in[("returncode",1),("actual_direct_native_wait",False),("all_owned_processes_and_groups_drained",False),("operational_failure","failed")]:
  need(not evaluate(dict(good,**{field:bad})),"Reject completion "+field)
  checks.append("completion_"+field+"_required")
 # Verify inherited finite cleanup exactly except aggregate byte counting.
 predecessor=pathlib.Path(deps["SUPERVISOR"]["path"]).read_bytes()
 need(hashlib.sha256(predecessor).hexdigest()==deps["SUPERVISOR"]["sha256"],"Pinned supervisor")
 def final(t):return next(n for n in ast.walk(next(x for x in t.body if isinstance(x,ast.FunctionDef)and x.name=="execute"))if isinstance(n,ast.Try)and n.finalbody).finalbody
 old=ast.Module(body=final(ast.parse(predecessor)),type_ignores=[])
 new=ast.Module(body=copy.deepcopy(final(trees["root_driver_v1.py"])),type_ignores=[])
 class StripLogCounter(ast.NodeTransformer):
  def visit_AugAssign(self,node):return None if isinstance(node.target,ast.Name)and node.target.id=="log_total"else self.generic_visit(node)
 need(ast.dump(old,include_attributes=False)==ast.dump(StripLogCounter().visit(new),include_attributes=False),"Exact inherited finite cleanup")
 # Key boundary ordering from actual source AST.
 def calls(node):return sorted((n.lineno,n.col_offset,ast.unparse(n.func))for n in ast.walk(node)if isinstance(n,ast.Call))
 admit=[n for _,__,n in calls(functions["root_driver_v1.py"]["admit"])]
 need(admit.index("c.validate_request")<admit.index("guard")<admit.index("g.budget.v4.scope_spec")<admit.index("g.operation_draft_v1")<admit.index("g.apply_reserve_v1"),"Validation/allroles preflight before reserve mutation")
 request=[n for _,__,n in calls(functions["root_contract_v1.py"]["validate_request"])]
 need("authenticate_closure"in request,"ROOT closure required before request returns")
 worker=[n for _,__,n in calls(functions["worker_entry_v1.py"]["worker"])]
 need(worker.index("os.read")<worker.index("payload")<worker.index("c.local(q['output_directory']).mkdir")<worker.index("os.execve"),"Owned gate/authentication/fresh directory/exact fdexec order")
 need("c.verify_actual_roles"in worker and"c.verify_actual_roles"in[n for _,__,n in calls(functions["root_driver_v1.py"]["execute"])],"Actual returned role revalidation")
 return dict(schema="Relate_exactALL_independent_focused_source_checks_v1",status="SOURCE_SEAMS_PASS",
  source_sha256={k:hashlib.sha256(v).hexdigest()for k,v in sources.items()},checks=checks,inert_defaults=defaults,
  exact_author_ALL_argv=True,full_original_support=True,operational_caps_exact=True,
  worker_success_only_via_exact_authenticated_fdexec=True,
  original_binary_not_read=True,actual_five_input_bytes_not_read=True,actual_current_ROOT_authority_or_selector_not_read=True,
  actual_metadata_predicates_with_handwritten_records=True,finite_finally_AST_matches_reviewed_predecessor=True,
  all_role_preflight_before_reserve_authority=True,gate_then_auth_then_fresh_cwd_then_fdexec=True,
  no_guard_resource_payload_binary_model_job_package_or_Git_operations=True,
  fixture_limits="Read/ref/raw seams use handwritten metadata only; no authenticity/current-capacity/runtime/fit claim.")
if __name__=="__main__":
 try:print(json.dumps(main(),sort_keys=True,indent=2))
 except Exception:traceback.print_exc(file=sys.stdout);raise SystemExit(1)
