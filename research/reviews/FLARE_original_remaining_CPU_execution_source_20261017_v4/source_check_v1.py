"""Static AST and handwritten metadata only; never imports project/guard/runtime modules."""
from pathlib import Path
from datetime import datetime
import ast,copy,hashlib,json,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v4"
OLD="research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v3"
H=ROOT/HERE
checks=[];negative=[]
def need(ok,message):
    if not ok:raise ValueError(message)
def fn(tree,name):return next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
def dump(node):return ast.dump(node,include_attributes=False)
def rr(path):
    p=Path(path);p=p if p.is_absolute()else ROOT/p
    need(p.resolve()==p and p.is_relative_to(ROOT)and p.is_file()and p.stat().st_size<=32*2**20,"Compact contained source/metadata")
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(path):return json.loads((ROOT/path).read_text())
def raw(record):
    need(rr(record["path"])==dict(path=record["path"],sha256=record["sha256"]),"Exact source/metadata SHA")
    return (ROOT/record["path"]).read_bytes()
def nodes(tree,names,scope):
    body=[copy.deepcopy(fn(tree,n))for n in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),"<stdlib-source-metadata-only>","exec"),scope)
def record(label):checks.append(label)
def reject(call,label):
    try:call()
    except(ValueError,KeyError,TypeError):negative.append(label);return
    raise AssertionError("Malformed metadata accepted: "+label)

b=read(HERE+"/source_bindings.v1.json")
tree=ast.parse((H/"root_driver_v1.py").read_text())
oldtree=ast.parse(raw(b["predecessor_v3_driver_ref"]))
wtree=ast.parse((H/"worker_entry_v1.py").read_text())
oldwtree=ast.parse(raw(b["predecessor_v3_worker_ref"]))
original=ast.parse(raw(b["original_contract_ref"]))
for p in H.glob("*.py"):ast.parse(p.read_text())
for p in("prepare_context_v1.py","__init__.py"):need((H/p).read_bytes()==(ROOT/OLD/p).read_bytes(),"Context/init retained")
record("all_new_source_AST_valid_original_context_and_initializer_retained")

before={n.name:n for n in oldtree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
after={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
need(set(before)==set(after),"No extra executor/memo framework")
for name in before.keys()-{"phase_budget","contract"}:need(dump(before[name])==dump(after[name]),"Unchanged lifecycle/admission/helper "+name)
record("original_owned_lifecycle_clock_gate_cleanup_wait_drain_admission_and_guard_transactions_AST_unchanged")

newworker=copy.deepcopy(fn(wtree,"worker"));oldworker=fn(oldwtree,"worker")
newadapter=next(n for n in ast.walk(newworker)if isinstance(n,ast.ClassDef)and n.name=="Adapter")
oldadapter=next(n for n in ast.walk(oldworker)if isinstance(n,ast.ClassDef)and n.name=="Adapter")
newmethod=fn(newadapter,"budget_check_v9");oldmethod=fn(oldadapter,"budget_check_v9")
need(sum(isinstance(n,ast.Call)and ast.unparse(n.func)=="g.budget_check_v9"for n in ast.walk(newmethod))==1,"Exactly one unchanged original budget call")
need(sum(isinstance(n,ast.Call)and ast.unparse(n.func)=="time.process_time"for n in ast.walk(newmethod))==3,"CPU clock start and mutually exclusive success/failure end")
need(sum(isinstance(n,ast.Call)and ast.unparse(n.func)=="time.monotonic"for n in ast.walk(newmethod))==3,"Wall clock start and mutually exclusive success/failure end")
need(any(isinstance(n,ast.keyword)and n.arg=="boundary_wall_seconds"for n in ast.walk(newmethod))and any(isinstance(n,ast.keyword)and n.arg=="boundary_process_cpu_seconds"for n in ast.walk(newmethod)),"Both success deltas logged")
newmethod.body=copy.deepcopy(oldmethod.body)
need(dump(newworker)==dump(oldworker),"Whole worker AST differs only boundary timing/logging")
for name in("compact_status","grant_from_parent","main"):need(dump(fn(wtree,name))==dump(fn(oldwtree,name)),"Other worker helpers retained")
record("whole_worker_unchanged_except_lightweight_wall_process_CPU_success_failure_deltas_original_budget_call_once")

env=dict(need=need,SETUP_SECONDS=7200)
nodes(tree,["phase_budget"],env)
phase=env["phase_budget"]
expected=dict(setup_seconds=7200,scientific_seconds=10800,wall_seconds=11400,combined_seconds=18600,evidence_reserve_seconds=300,authority_margin_seconds=600)
need(phase("convert")==expected,"Exact new3h envelope")
for stage in("native","seal"):
    oldenv=dict(need=need,SETUP_SECONDS=7200);nodes(oldtree,["phase_budget"],oldenv)
    need(phase(stage)==oldenv["phase_budget"](stage),"Native/seal budget unchanged")
need(expected["setup_seconds"]+expected["wall_seconds"]==expected["combined_seconds"]and expected["wall_seconds"]-expected["scientific_seconds"]==600,"Original overhead counted once")
record("7200setup10800science600overhead18600combined_native_and_seal_budgets_unchanged")

P=lambda name:dict(path="SOURCE_ONLY/"+name,sha256=hashlib.sha256(name.encode()).hexdigest())
deps=read("research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v1/dependencies.v1.json")
records={}
actual=read(b["actual_selector_metadata_ref"]["path"])
selector=types.SimpleNamespace(SELECTOR_REF=None,_CHOSEN=None,SELECTOR=None,GUARD_REF=None,POLICY_REF=None)
def selected(record):
    need(record==b["actual_selector_metadata_ref"],"Pinned selector");return copy.deepcopy(actual)
def select(record):
    value=selected(record);selector.SELECTOR_REF=record;selector._CHOSEN=value;selector.SELECTOR=value["selector_source_ref"];selector.GUARD_REF=value["source_ref"];selector.POLICY_REF=value["policy_ref"];return value
selector.selected=selected;selector.bind_selector=select;selector.guard=lambda consumer:"UNCHANGED_CONSUMER_MEMO"if consumer else"ORIGINAL_UNMEMOIZED_MUTATOR"
input_case={};c=types.SimpleNamespace(input_spec=lambda record:input_case["result"],require=need,DEPS=deps,CAPS={},read=lambda r:copy.deepcopy(records[r["path"]]),raw=lambda r:r,local=lambda p:ROOT/p,ref=lambda p:P(str(p)),validate_request=lambda r:("ORIGINAL_VALIDATION_REACHED",r),SELECTOR_REF=None,_CHOSEN=None,GUARD_REF=None,POLICY_REF=None,STAGES=("convert","native","seal"),VAL=("american_c3_val_001",))
cenv=c.__dict__
for key in("NAMESPACE","PIDFD","OWNED","FRAMEWORK","SELECTOR","COMPLETION_REVIEW"):cenv[key]=deps[key]
cenv["PRODUCERS"]=dict(convert="src/arggnn/american_c3_flare_convert_v1.py",native="src/arggnn/american_c3_flare_run_v2.py",seal="src/arggnn/american_c3_flare_evaluate_v1.py")
cenv["completion"]=lambda:types.SimpleNamespace(_context=lambda *a,**k:{"targets":{"american_c3_val_001":{"count_manifest_ref":P("count")}}})
cenv["make_config"]=lambda q:dict(binary_provenance_ref=deps["BINARY_PROVENANCE"])
nodes(original,["stage_caps","direct_roles"],cenv)
env.update(Path=Path,datetime=datetime,copy=copy,HERE=HERE,SOURCE=HERE+"/root_driver_v1.py",ENTRY=HERE+"/worker_entry_v1.py",MANIFEST=HERE+"/source_manifest.v1.json",HANDOFF=HERE+"/source_handoff.v1.json",bindings=lambda:b,load=lambda record,name:c if record==b["original_contract_ref"]else selector)
nodes(tree,["contract"],env)
c=env["contract"]();c.bind_selector(b["actual_selector_metadata_ref"])
need(c.STAGES==("convert",)and c.VAL==("american_c3_val_001",),"Only new VAL001 convert")
need(c.guard(True)=="UNCHANGED_CONSUMER_MEMO"and c.guard(False)=="ORIGINAL_UNMEMOIZED_MUTATOR","No extra memo")
cfg=dict(output_cap_bytes=881252800)
caps=c.stage_caps("convert",cfg)
need(caps==read(HERE+"/conversion_caps.INACTIVE.v4.json"),"Actual source derives new exact caps")
need({k:v for k,v in caps.items()if k not in("scientific_seconds","wall_seconds")}=={k:v for k,v in b["predecessor_convert_caps"].items()if k not in("scientific_seconds","wall_seconds")},"All resource caps retained")
reject(lambda:c.stage_caps("native",cfg),"native_not_authorized")
reject(lambda:c.stage_caps("convert",dict(output_cap_bytes=881252801)),"new_output_cap_rejected")
record("actual_private_contract_caps_only_new_conversion_time_all_resource_limits_retained")

request=P("request");accept=P("accept");directive=P("directive");registration=P("registration")
q=dict(stage="convert",parent_id="american_c3_val_001",phase_budget=expected,prior_failure_acceptance_ref=accept,root_directive_ref=directive,registration_ref=registration,source_review_ref=P("review"),source_manifest_ref=P(HERE+"/source_manifest.v1.json"),storage_policy_selector_ref=b["actual_selector_metadata_ref"],created_at_utc="2030-01-01T00:00:00+00:00",expires_at_utc="2030-01-01T06:00:00+00:00",proposed_reservation_key=b["fresh_reservation_key"],**b["fresh_namespaces"],**b["fresh_condition"])
accepted=dict(schema="ROOT_FLARE_failed_unsealed_acceptance_v2",root_authorizer="root_agent",accepted=True,**b["old_failure"],old_launch_resume_write_authority_revoked=True,old_files_scopes_key_charge_clocks_and_failure_unchanged=True,original_conversion_unsealed=True)
latest=dict(schema="ROOT_FLARE_setupv3_failed_unsealed_acceptance_v1",root_authorizer="root_agent",accepted=True,**b["setupv3_failure"],old_launch_resume_write_authority_revoked=True,old_files_scopes_key_charge_clocks_and_failure_unchanged=True,original_conversion_unsealed=True)
accepted["latest_setupv3_failure_acceptance"]=latest
registered=dict(schema="FLARE_fresh3h_VAL001_completion_condition_registration_v4",root_authorizer="root_agent",registered_before_request=True,source_manifest_ref=P(HERE+"/source_manifest.v1.json"),source_review_ref=P("review"),proposed_reservation_key=b["fresh_reservation_key"],phase_budget=expected,original_failed_input_refs=[b["old_failure"]["original_input_ref"],b["setupv3_failure"]["original_input_ref"]],decision_snapshot_ref=P("decision"),ledger_snapshot_ref=P("ledger"),registered_utc="2029-12-31T23:00:00+00:00",**b["fresh_condition"])
directed=dict(phase_budget=expected,prior_failure_acceptance_ref=accept,proposed_reservation_key=b["fresh_reservation_key"],**b["fresh_condition"])
records.update({request["path"]:q,accept["path"]:accepted,directive["path"]:directed,registration["path"]:dict(fresh_completion_condition_registration=registered)})
need(c.validate_request(request)[0]=="ORIGINAL_VALIDATION_REACHED","New wrapper leads to unchanged full original validation")
admitted=dict(storage=dict(reservation=dict(key=b["fresh_reservation_key"])),job_id=b["fresh_reservation_key"].split("/",1)[1])
input_case["result"]=(admitted,q,cfg)
need(c.input_spec(P("admitted"))==input_case["result"],"Actual storage/job exact fresh key")
for obj,key in((admitted["storage"]["reservation"],"key"),(admitted,"job_id")):
    prior=obj[key];obj[key]="OLD";reject(lambda:c.input_spec(P("admitted")),"actual_admitted_"+key);obj[key]=prior

mutations=[
(q,"stage","native"),(q,"stage","seal"),(q,"parent_id","american_c3_val_002"),(q,"proposed_reservation_key","OLD"),
(q,"phase_budget",dict(expected,scientific_seconds=3600)),(q,"phase_budget",dict(expected,scientific_seconds=10801)),
(q,"expires_at_utc","2030-01-01T05:20:00+00:00"),(q,"storage_policy_selector_ref",P("oldselector")),
(accepted,"accepted",False),(accepted,"old_charge_bytes",0),(accepted,"old_launch_resume_write_authority_revoked",False),
(latest,"accepted",False),(latest,"direct_worker_returncode",0),(latest,"owned_drain",False),(latest,"outer_returncode",True),
(latest,"actual_wait_ref",P("oldwait")),(latest,"old_expires_at_utc","NEW"),(latest,"exact_sites_written",227316),
(latest,"old_launch_resume_write_authority_revoked",False),(latest,"old_files_scopes_key_charge_clocks_and_failure_unchanged",False),
(directed,"phase_budget",{}),(directed,"proposed_reservation_key","OTHER"),
(registered,"registered_before_request",False),(registered,"source_manifest_ref",P("oldmanifest")),(registered,"source_review_ref",P("oldreview")),
(registered,"registered_utc","2030-01-01T00:00:01+00:00"),(registered,"registered_utc","2029-12-31T23:00:00"),
(registered,"original_failed_input_refs",[]),(registered,"phase_budget",{}),(registered,"proposed_reservation_key","OTHER")]
for target in(q,directed,registered):
    for key,value in b["fresh_condition"].items():mutations.append((target,key,not value))
for key in b["fresh_namespaces"]:
    mutations.append((q,key,q[key]+"_OTHER"))
    for oldset in b["all_failed_namespaces"]:
        for value in oldset.values():mutations.append((q,key,value))
for obj,key,bad in mutations:
    prior=obj[key];obj[key]=bad
    reject(lambda:c.validate_request(request),key)
    obj[key]=prior
record("actual_new_request_wrapper_rejects_wrong_stage_parent_scope_key_budget_authority_flags_and_registration")

for failure in(b["old_failure"],b["setupv3_failure"]):
    for key in("original_input_ref","original_request_ref","actual_wait_ref","outer_wait_ref"):raw(failure[key])
need(b["old_failure"]["outer_tool"]==12977 and b["setupv3_failure"]["outer_tool"]==97739 and b["setupv3_failure"]["direct_worker_returncode"]==1 and b["setupv3_failure"]["completed_ref"]is None,"Exact distinct failure lineages")
record("both_genuine12977_and97739_failure_refs_caps_clocks_and_no_completion_retained")

guards=read(b["actual_guard_metadata_ref"]["path"])["bindings"]
unique=lambda seq:list({(r["path"],r["sha256"]):r for r in seq}.values())
need(len(guards)==46 and len(unique(guards.values()))==39,"All historical guard roles")
roleq=dict(q,context_ref=P("context"),runtime_ref=P("runtime"),registration_ref=registration)
group=c.direct_roles(request,roleq,P("config"),P("closure"))
handwritten=[c.ref(c.MANIFEST),c.ref(c.SOURCE),c.ref(c.DRIVER),c.ref(c.ENTRY),c.ref(c.HANDOFF),c.ref(c.PRODUCERS["convert"]),deps["COMPLETION_REVIEW"],P("review"),P("runtime"),P("config"),directive,request,P("context"),P("count"),registration,P("closure"),b["selected_child_source_ref"],b["actual_selector_metadata_ref"],deps["NAMESPACE"],deps["PIDFD"],deps["OWNED"],deps["FRAMEWORK"],deps["SELECTOR"],accept]
need(unique(group)==unique(handwritten),"Complete original direct roles no omission")
union=unique([*guards.values(),*group])
need(len(union)==60 and len(union)<=64,"Full fresh pure metadata union60/64")
fixture=dict(schema="FLARE_fresh3h_VAL001_pure_metadata_role_fixture_v4",source_only=True,current_guard_state_or_capacity_observed=False,guard_names=46,distinct_guards=39,direct_unique=len(unique(group)),union_count=len(union),original64cap=64,all_roles=union,placeholders_are_not_authority=True)
(H/"role_union_fixture.v1.json").write_text(json.dumps(fixture,indent=2,sort_keys=True)+"\n")
record("all46named39distinct_historical_guards_full_fresh_metadata_union60_below64")

for name in("ROOT_request.INACTIVE.template.v1.json","ROOT_directive.INACTIVE.template.v1.json"):
    v=read(HERE+"/"+name)
    need(v["active"]is False and v["root_authorizer"]is None and v["created_at_utc"]is None and v["expires_at_utc"]is None,"Inactive ROOT templates")
    need(v["phase_budget"]==expected and v["caps"]==caps and all(v[k]==value for k,value in b["fresh_namespaces"].items()),"Concrete exact new scopes/budget")
review=read(HERE+"/independent_review.INACTIVE.template.v1.json")
need(review["independent"]is False and review["verdict"]=="UNREVIEWED_INACTIVE"and review["blockers"],"No fabricated review")
need(read(HERE+"/registration.INACTIVE.template.v1.json")["fresh_completion_condition_registration"]["registered_before_request"]is False,"No fabricated registration")
record("all_templates_inactive_no_fabricated_review_registration_runtime_authority")

baseline=read(OLD+"/source_manifest.v1.json")
for path,digest in baseline["source_sha256"].items():raw(dict(path=path,sha256=digest))
for r0 in baseline["metadata_refs"]:raw(r0)
conv="src/arggnn/american_c3_flare_convert_v1.py"
need(rr(conv)["sha256"]==baseline["source_sha256"][conv],"Original author converter exact")
source=ast.parse((ROOT/conv).read_text())
periodic=[n for n in ast.walk(fn(source,"execute"))if isinstance(n,ast.If)and ast.unparse(n.test)=="site % 4096 == 0"]
need(len(periodic)==1 and ast.unparse(periodic[0].body[0])=="w.adapter.budget_check_v9(w.storage, boundary=True)","Every original4096site budget check")
record("complete_predecessor176source154metadata_closure_and_original_converter_authenticate")

need(not any(isinstance(n,(ast.Import,ast.ImportFrom))and((isinstance(n,ast.ImportFrom)and(n.module or"").startswith(("arggnn","research","scripts")))or(isinstance(n,ast.Import)and any(a.name.startswith(("arggnn","research","scripts"))for a in n.names)))for n in ast.walk(ast.parse(Path(__file__).read_text()))),"No project imports in checker")
record("stdlib_AST_and_handwritten_metadata_only_no_runtime_guard_resources_payloads_or_numerics")

result=dict(schema="FLARE_fresh3h_VAL001_source_only_checks_v4",status="PASS_SOURCE_ONLY_INACTIVE_UNREVIEWED",independent=False,checks=checks,checks_count=len(checks),handwritten_metadata_rejected_cases=len(negative),rejected_labels=negative,full_role_union=60,role_ceiling=64,fresh_conversion_budget=expected,whole_source_independent_review_required=True,guard_runtime_resource_process_payload_conversion_numerical_or_Git_operations=False)
(H/"source_check.result.v1.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print(json.dumps({k:v for k,v in result.items()if k!="rejected_labels"},sort_keys=True))
