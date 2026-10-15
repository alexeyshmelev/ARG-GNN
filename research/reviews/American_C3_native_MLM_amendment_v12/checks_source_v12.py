"""Focused source and pure mocked v12 checks. No guard/state/GPU/process/socket actions."""
from pathlib import Path
from types import SimpleNamespace, ModuleType
from datetime import datetime,timezone
from unittest.mock import patch
import ast,copy,contextlib,difflib,hashlib,io,json,os,stat,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
P=ROOT/"research/reviews/American_C3_native_MLM_amendment_v12"
OLD=Path(str(P).replace("v12","v11"))
checks=[]
def check(name,condition):
    if not condition:raise AssertionError(name)
    checks.append(dict(name=name,status="PASS"))
def rejects(name,call):
    try:call()
    except ValueError:check(name,True)
    else:raise AssertionError(name)
def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    return dict(path=str(p.relative_to(ROOT)),sha256=digest(p))
def function(name,file,ns):
    tree=ast.parse((P/file).read_text())
    n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[n],type_ignores=[]),str(P/file)+":mock","exec"),ns)
    return ns[name]
files=sorted(f for f in P.glob("*.py") if f.name!="checks_source_v12.py")
check("exact17_execution_files",len(files)==17)
for f in files:compile(f.read_text(),str(f),"exec")
check("all17_compile_without_bytecode",True)
closure=read(P/"execution_source_closure.json")
check("closure64_and17actual",len(closure["source_sha256"])==64 and closure["actual_new_execution_source_count"]==17)
check("every_closure_pin_byte_exact",all(digest(ROOT/p)==s for p,s in closure["source_sha256"].items()))
oldclosure=read(OLD/"execution_source_closure.json")
check("all47_v11_closure_pins_preserved",len(oldclosure["source_sha256"])==47 and all(closure["source_sha256"].get(p)==s for p,s in oldclosure["source_sha256"].items()))
preserved=read(P/"preserved_v11_source_refs.json")
check("all16_v11_execution_sources_unchanged",len(preserved)==16 and all(ref(r["path"])==r for r in preserved))
check("qualified_ESRCH_helper_exact",digest(P/"owned_lifecycle_v2.py")=="93d71acc40f1037d0b37acb2ca0859300ea4bd9741409e9bd00ca21bf100d633")
check("qualified_abstract_IPC_helper_exact",digest(P/"abstract_unix_bootstrap_v1.py")=="4550ead44f36eeb0fda8a817a2002082c3d067cf1a84fd817301532577d8b014")
a=SimpleNamespace()
an={"__name__":"_v12_checks_amendment"}
exec(compile((P/"amendment_v12.py").read_text(),str(P/"amendment_v12.py"),"exec"),an)
a.__dict__.update(an)
oldns={"__name__":"_v11_static"}
exec(compile((OLD/"amendment_v11.py").read_text(),str(OLD/"amendment_v11.py"),"exec"),oldns)
check("original_allocation_identity_cap_expiry_refs_unchanged",all(an[k]==oldns[k] for k in ["KEY","JOB","CAP","CREATED","EXPIRES","FIXED_REFS","PARENT_OUTPUT","AUX","PARENT_TMP"]))
check("fresh_children_within_original_parents",Path(an["OUTPUT"]).parent==Path(an["PARENT_OUTPUT"]) and an["TMP"].parent==an["PARENT_TMP"] and an["ATTEMPT"].parent==an["AUX"] and all(an[k]!=oldns[k] for k in ["OUTPUT","TMP","ATTEMPT","LOCK"]))
check("all17_actual_execution_sources_required",set(an["required_execution_sources"]())=={str(f.relative_to(ROOT)) for f in files})
templates={n:read(P/n) for n in ["execution_authority.inactive.json","root_directive.inactive.json","namespace_cleanup_authority.inactive.json"]}
check("three_authority_templates_inactive",all(t["active"] is False for t in templates.values()))
spec=read(P/"input.inactive_template.json")
check("unresolved_template_actual_GPU_and_UUID",spec["physical_gpu"] is None and spec["execution_gpu_uuid"] is None and templates["execution_authority.inactive.json"]["GPU_UUID"] is None)
cr=ref(P/"execution_source_closure.json")
check("all_template_closure_refs_exact",spec["transitive_source_closure_ref"]==cr and all(t["source_closure_ref"]==cr for n,t in templates.items() if "source_closure_ref" in t))
check("inactive_authority_input_ref_exact",templates["execution_authority.inactive.json"]["input_ref"]==ref(P/"input.inactive_template.json"))
original=read(ROOT/an["ORIGINAL_INPUT_REF"]["path"])
expected=copy.deepcopy(original)
expected.update(main_source_ref=ref(P/"entry_v12.py"),actual_launcher_ref=ref(P/"launcher_v12.py"),
 root_context_module_ref=ref(P/"context_factory_v1.py"),source_sha256=closure["source_sha256"],
 transitive_source_closure_ref=cr,input_binding_mode="actual_root_execution_amendment_v12",
 output_directory=an["OUTPUT"],physical_gpu=None,execution_gpu_uuid=None)
check("input_only_execution_bindings_differ",spec==expected)
unchanged=["consumer_contract_v1.py","corpus_auth_v2.py","entry_v12.py","native_runtime_v1.py","root_native_contract_v1.py","supervisor_v1.py","terminal_bounds_v1.py","terminal_capture_v12.py","terminal_namespace_v12.py"]
check("nine_execution_files_namespace_only",all((P/n).read_text()==(OLD/n.replace("v12","v11")).read_text().replace("v11","v12").replace("V11","V12") for n in unchanged))
child=ast.parse((P/"native_child_v1.py").read_text())
oldchild=ast.parse((OLD/"native_child_v1.py").read_text().replace("v11","v12").replace("V11","V12"))
check("all_native_science_helper_functions_unchanged",all(ast.dump(n,include_attributes=False)==ast.dump(next(o for o in oldchild.body if isinstance(o,ast.FunctionDef) and o.name==n.name),include_attributes=False) for n in child.body if isinstance(n,ast.FunctionDef) and n.name!="main"))
text=(P/"native_child_v1.py").read_text()
check("bootstrap_precedes_torch_preparation_author",text.index("ipc_bootstrap=bootstrap_native_execution_v1(")<text.index("prep,_,prepared_dir=preparation_v1(spec)")<text.index("    import torch")<text.index("trainer=module.main(config,return_trainer=True)"))
check("science_schedule_assertions_unchanged",all(s in text for s in ['observed_trainer["epoch"]==5','observed_trainer["batch"]==224775','observed_optimizer["unique_step_values"]==[224775]','trainer=module.main(config,return_trainer=True)']))
check("completion_namespace12",(P/"amendment_v12.py").read_text().count("execution_namespace_version=12")==1)
uuid="GPU-01234567-89ab-cdef-0123-456789abcdef"
for gpu in (4,5):
    check(f"root_choice_GPU{gpu}",an["execution_gpu_choice"](dict(physical_gpu=gpu,GPU_UUID=uuid))==(gpu,uuid))
for value in [None,True,3,6,"4"]:
    rejects(f"invalid_root_GPU_{value}",lambda v=value:an["execution_gpu_choice"](dict(physical_gpu=v,GPU_UUID=uuid)))
rejects("missing_actual_UUID",lambda:an["execution_gpu_choice"](dict(physical_gpu=4,GPU_UUID=None)))
# The eligibility function runs with metadata only; no launcher import or GPU probe.
for gpu in (4,5):
    ns=dict(amendment=SimpleNamespace(execution_gpu_choice=lambda:(gpu,uuid)))
    eligible=function("eligible_free","launcher_v12.py",ns)
    row=dict(index=gpu,uuid=uuid,memory_total_mib=60000,memory_used_mib=0)
    check(f"eligible_exact_GPU{gpu}_UUID",eligible(row))
    check(f"reject_wrong_GPU{gpu}_or_UUID_or_memory",not eligible(dict(row,index=9-gpu)) and not eligible(dict(row,uuid="GPU-other")) and not eligible(dict(row,memory_used_mib=10000)))
# Lease comparison uses an in-memory replacement for the imported lease module.
for gpu in (4,5):
    lease=dict(physical_gpu=gpu,visible_gpu_uuid=uuid)
    native=ModuleType("arggnn.native_argformer_exec")
    native.validate_lease=lambda:lease
    native.apply_and_validate_title=lambda:None
    package=ModuleType("arggnn");package.__path__=[]
    ns=dict(os=SimpleNamespace(environ={"ARGGNN_LOGICAL_JOB_NAME":an["JOB"]}),
      amendment=SimpleNamespace(execution_gpu_choice=lambda:(gpu,uuid)))
    call=function("_lease","context_factory_v1.py",ns)
    with patch.dict(sys.modules,{"arggnn":package,"arggnn.native_argformer_exec":native}):
        check(f"lease_exact_GPU{gpu}_UUID",call(a,dict(physical_gpu=gpu,execution_gpu_uuid=uuid,job_id=an["JOB"]),{},False)==lease)
        rejects(f"lease_GPU{gpu}_wrong_UUID",lambda:call(a,dict(physical_gpu=gpu,execution_gpu_uuid="GPU-other",job_id=an["JOB"]),{},False))
# Actual authority validator, with every metadata and namespace dependency mocked.
validator=an["validate_authority"]
baseline=templates["execution_authority.inactive.json"]
auth=copy.deepcopy(baseline)
auth.update(active=True,physical_gpu=4,GPU_UUID=uuid,root_directive_ref={"path":"directive","sha256":"d"},
 namespace_cleanup_receipt_ref={"path":"receipt","sha256":"r"})
for k in ["all_prior_launch_resume_write_capabilities_cooperatively_revoked","parent_completion_envelope_after_actual_wait0_authorized","checkpoint_latest_regular_copy_after_drain_authorized","current_attempt_dead_IPC_cleanup_after_genuine_owned_drain_authorized"]:auth[k]=True
rootref=dict(path="authority",sha256="authority")
actualspec=copy.deepcopy(spec);actualspec.update(physical_gpu=4,execution_gpu_uuid=uuid)
records={rootref["path"]:auth,auth["input_ref"]["path"]:actualspec,
 an["ORIGINAL_INPUT_REF"]["path"]:original,an["ORIGINAL_STORAGE_REF"]["path"]:read(ROOT/an["ORIGINAL_STORAGE_REF"]["path"]),
 auth["actual_old_outer_wait_ref"]["path"]:dict(actual_direct_outer_supervisor_wait=True,pid=2817234,returncode=1),
 cr["path"]:copy.deepcopy(closure)}
oldread,oldref=an["read"],an["ref"]
an["read"]=lambda r:copy.deepcopy(records[r["path"]])
an["ref"]=lambda p:dict(path=str(p),sha256=closure["source_sha256"].get(str(p),an["FIXED_REFS"].get("NONE",{}).get("sha256","")))
def mockedref(p):
    p=str(p)
    for row in an["FIXED_REFS"].values():
        if p==row["path"]:return row
    return dict(path=p,sha256=closure["source_sha256"][p])
an["ref"]=mockedref
nsmod=ModuleType("root_namespace_v12")
nsmod.validate_directive=lambda r:None;nsmod.validate_cleanup_receipt=lambda r,d:None
with patch.dict(sys.modules,{"root_namespace_v12":nsmod}):
    for gpu in (4,5):
        auth["physical_gpu"]=gpu;actualspec["physical_gpu"]=gpu
        check(f"full_authority_binding_GPU{gpu}",validator(rootref)[1]==actualspec)
    auth["active"]=False;rejects("inactive_execution_authority_rejected",lambda:validator(rootref));auth["active"]=True
    actualspec["phase"]="tampered";rejects("scientific_input_mutation_rejected",lambda:validator(rootref));actualspec["phase"]=spec["phase"]
    missing=records[cr["path"]]["source_sha256"].pop(str((P/"abstract_unix_bootstrap_v1.py").relative_to(ROOT)))
    rejects("incomplete_actual_source_closure_rejected",lambda:validator(rootref))
    records[cr["path"]]["source_sha256"][str((P/"abstract_unix_bootstrap_v1.py").relative_to(ROOT))]=missing
an["read"],an["ref"]=oldread,oldref
# Pure memory cleanup simulation: fake paths, guards, ESRCH, lstat, endpoints, unlink and writes.
rows=[dict(path=f".runtime/tmp/c3full8/f/v11a/mlm/pymp-{i:02d}/listener-{i:02d}",lstat=[1,100+i,stat.S_IFSOCK|0o600,1,0,1,1]) for i in range(16)]
helperref=dict(path="helper",sha256="helper")
for scenario in ["success","ESRCH","identity","type","endpoint","ancestor","expiry","unlink","native_guard"]:
    state=dict(events=[],saved={},order=[],attempt=False)
    class FakePath:
        def __init__(self,path):self.path=str(path)
        def __str__(self):return self.path
        def __truediv__(self,n):return FakePath(self.path+"/"+n)
        def exists(self):return state["attempt"] if self.path=="attempt" else self.path in {r["path"] for r in rows[len(state["events"]):]}
        def is_symlink(self):return False
        def mkdir(self):state["attempt"]=True;state["order"].append("mkdir")
        def lstat(self):
            index=next(i for i,r in enumerate(rows) if r["path"]==self.path)
            value=list(rows[index]["lstat"])
            state["order"].append(("lstat",index))
            if "before_cleanup.json" in state["saved"] and index==7 and scenario in ("identity","type"):value[1 if scenario=="identity" else 2]+=1
            return SimpleNamespace(**dict(zip(["st_dev","st_ino","st_mode","st_nlink","st_size","st_mtime_ns","st_ctime_ns"],value)))
        def unlink(self):
            index=next(i for i,r in enumerate(rows) if r["path"]==self.path)
            if index==7 and scenario=="unlink":raise ValueError("mock unlink failure")
            state["events"].append(index);state["order"].append(("unlink",index))
    cleanupauth=copy.deepcopy(templates["namespace_cleanup_authority.inactive.json"])
    cleanupauth.update(active=True,all_prior_capabilities_revoked=True,helper_ref=helperref,root_directive_ref={"path":"directive","sha256":"d"})
    cleanupclosure=dict(schema="native_MLM_fullfit_execution_source_closure_v12",original_storage_ref=an["ORIGINAL_STORAGE_REF"],source_sha256={"helper":"helper"})
    def fake_read(r):
        if r==rootref:return cleanupauth
        if r==cleanupauth["source_closure_ref"]:return cleanupclosure
        if r==an["ORIGINAL_STORAGE_REF"]:return {"kind":"native"}
        return {}
    def fake_save(path,value):
        name=Path(str(path)).name;state["saved"][name]=copy.deepcopy(value);state["order"].append(("save",name))
        return dict(path=name,sha256=name)
    def known():
        if "before_cleanup.json" in state["saved"] and len(state["events"])==7 and scenario=="ESRCH":raise ValueError("mock old PID remains")
        state["order"].append(("ESRCH",len(state["events"])))
        return dict(checks=[dict(ESRCH=True)]*8)
    def ancestors(path):
        if "before_cleanup.json" in state["saved"] and len(state["events"])==7 and scenario=="ancestor":raise ValueError("mock noncanonical ancestor")
        state["order"].append(("ancestors",str(path)))
    def endpoints(paths):
        state["order"].append(("endpoint",len(state["events"])))
        return ["mock listener"] if len(paths)==1 and len(state["events"])==7 and scenario=="endpoint" else []
    def expiry(*args):
        if len(state["events"])==7 and scenario=="expiry":raise ValueError("mock expiry")
    def guard(storage,**kwargs):
        state["order"].append(("guard",storage,kwargs))
        if storage=={"kind":"native"} and scenario=="native_guard":raise ValueError("mock native guard")
    fakea=SimpleNamespace(read=fake_read,require=an["require"],ref=lambda p:helperref,KEY=an["KEY"],CAP=an["CAP"],EXPIRES=an["EXPIRES"],
       ORIGINAL_STORAGE_REF=an["ORIGINAL_STORAGE_REF"],local=lambda p:FakePath(p),save=fake_save,check_expiry=expiry)
    ns=dict(a=fakea,SOURCE="helper",ADMIN_STORAGE_REF=cleanupauth["original_admin_storage_ref"],PRESERVED_REFS=cleanupauth["preserved_metadata_refs"],
      OLD_V11_OBSERVATION_REF=cleanupauth["old_v11_observation_ref"],V11_FRAMED_DIAGNOSIS_REF=cleanupauth["v11_framed_diagnosis_ref"],
      ATTEMPT=FakePath("attempt"),validate_directive=lambda r:None,original_socket_rows=lambda:copy.deepcopy(rows),socket_ancestors=ancestors,
      stamp=lambda s:[s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns],listeners=endpoints,
      known_roots_groups_absent=known,admin_guard=lambda:(SimpleNamespace(budget_check_v9=guard),{"kind":"admin"}),
      datetime=datetime,timezone=timezone,json=json)
    call=function("cleanup","root_namespace_v12.py",ns)
    with contextlib.redirect_stdout(io.StringIO()):
        if scenario=="success":call(rootref)
        else:rejects("mock_cleanup_"+scenario+"_fails_closed",lambda:call(rootref))
    receipt=state["saved"]["cleanup_receipt.json"]
    check("mock_cleanup_"+scenario+"_one_manifest_one_batch",set(state["saved"])=={"before_cleanup.json","cleanup_receipt.json"})
    check("mock_cleanup_"+scenario+"_honest_event_batch",[r["index"] for r in receipt["per_node_events"]]==state["events"] and receipt["active"] is (scenario=="success") and ((receipt["cleanup_failure"] is None)==(scenario=="success")))
    if scenario=="success":
        check("mock_cleanup_exact16_unlinks",state["events"]==list(range(16)))
        unlinkpositions=[i for i,v in enumerate(state["order"]) if isinstance(v,tuple) and v[0]=="unlink"]
        check("mock_cleanup_zero_accounting_or_writes_in_unlink_loop",not any(isinstance(v,tuple) and v[0] in ("save","guard") for v in state["order"][unlinkpositions[0]:unlinkpositions[-1]+1]))
        check("mock_cleanup_original_native_parent_boundary_preserved",any(isinstance(v,tuple) and v[0]=="guard" and v[1]=={"kind":"native"} for v in state["order"]))
    else:check("mock_cleanup_"+scenario+"_stops_exactly",len(state["events"])==(16 if scenario=="native_guard" else 7))
# Close is rejected before source/authority or accounting access.
ns=dict()
close=function("close","root_native_driver_v1.py",ns)
rejects("old_v3_close_is_unconditionally_revoked",lambda:close("mlm",{}))
check("driver_close_CLI_rejects_before_authority",(P/"root_native_driver_v1.py").read_text().index('if args.close:return close("mlm",None)')<(P/"root_native_driver_v1.py").read_text().index('c.require(args.authority and args.expected_authority_sha256'))
# CLI defaults are exercised with all operational entry functions replaced by traps.
for name,flag in [("root_native_driver_v1.py","actual_execution"),("root_namespace_v12.py","actual_cleanup")]:
    ns=dict(argparse=__import__("argparse"),json=json,__doc__="mock inert CLI")
    def trap(*args,**kwargs):raise AssertionError("operational action called by default CLI")
    ns.update(c=SimpleNamespace(require=trap),a=SimpleNamespace(require=trap),close=trap,self_probe_v1=trap,amendment=SimpleNamespace(activate=trap),cleanup=trap,launch=trap)
    main=function("main",name,ns);stream=io.StringIO()
    with contextlib.redirect_stdout(stream):result=main([])
    check("default_inert_"+name,result==0 and json.loads(stream.getvalue())[flag] is False)
# Freeze compact review diff; tests and metadata are outside the execution-source closure.
parts=[]
for f in files:
    prior=OLD/f.name.replace("v12","v11")
    if not prior.exists():parts.append("NEW accepted helper: "+f.name+"\n");continue
    parts.extend(difflib.unified_diff(prior.read_text().splitlines(keepends=True),f.read_text().splitlines(keepends=True),fromfile=str(prior.relative_to(ROOT)),tofile=str(f.relative_to(ROOT)),n=2))
(P/"namespace_source_diff.patch").write_text("".join(parts))
evidence=dict(schema="native_MLM_v12_focused_source_mock_checks_v1",checks=checks,total=len(checks),all_passed=True,
 actual_guard_or_accounting_state_access=False,actual_GPU_or_corpus_or_author_execution=False,actual_process_probe_or_signal=False,actual_unlink=False,
 actual_fork_toy_repeated=False,execution_source_closure_ref=ref(P/"execution_source_closure.json"),
 qualified_prior_helper_handoff_ref=closure["qualified_ESRCH_abstract_IPC_handoff_ref"],
 cleanup_simulation="In-memory fake paths, ESRCH, lstat, canonical ancestors, endpoints, unlink, guard and writes only")
(P/"source_checks.json").write_text(json.dumps(evidence,sort_keys=True,indent=2)+"\n")
print(json.dumps(dict(total=len(checks),all_passed=True,source_checks_ref=ref(P/"source_checks.json"),diff_ref=ref(P/"namespace_source_diff.patch"))))
