"""Actual unchanged v4 scope checks. No accounting API, launch or scientific payload read."""
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,importlib.util,json,subprocess,sys,os
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");HERE=Path(__file__).parent;OLD=HERE.with_name(HERE.name[:-1]+"2")
RESULTS=[]
def check(name,ok=True):
    if not ok:raise AssertionError(name)
    RESULTS.append(dict(name=name,status="PASS"))
def ref(p):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def load(p,name):
    sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def read(p):return json.loads(Path(p).read_bytes())
def put(p,v):
    p.write_text(json.dumps(v,sort_keys=True,indent=2)+"\n");return ref(p)
def reject(name,fn,message=None):
    try:fn()
    except ValueError as e:
        check(name,message is None or message in str(e));return
    raise AssertionError(name)
ct=load(HERE/"root_contract_v1.py","_VALv3_source_contract_test")
v4=load(ROOT/"src/arggnn/project_budget_v4.py","_VALv3_unchanged_real_v4_scope")
check("actual_unchanged_v4_sha",ref(ROOT/"src/arggnn/project_budget_v4.py")==ct.V4_REF)
oldauth=read(OLD/"admitted_val003_infer_v2/reserve_authority.json")
base=list(oldauth["bindings"].values())
check("exact_actual_locked_audit_base37",len(base)==37)
check("exact_actual_old_authority_SHA",ref(OLD/"admitted_val003_infer_v2/reserve_authority.json")["sha256"]=="7d84a1ec18c64d7c9b522f430a45ea5c6da952ab09506722f45813779bfcdb8e")
reject("original_exact_v2_union_reproduces_lock_rejection",lambda:v4.scope_spec(ROOT,oldauth["request"]),"consumer bindings must be source/runtime authority metadata")
ct.POLICY_REF=read(OLD/"request_val003_infer.root_active_v1.json")["policy_ref"]
manifest=read(HERE/"source_manifest.v1.json")
ct.validate_slot_resources(manifest["refs"])
check("exact_original_two_slot_inodes_and_hashes_authenticated")
check("full_current_transitive_closure_authenticated",len(ct.source_paths())==len(manifest["refs"]))
fixtures=HERE/"real_scope_metadata_fixtures";fixtures.mkdir(exist_ok=True)
plans={}
for parent,row in read(HERE/"readiness.v1.json")["parents"].items():
    for stage in ("infer","native_paths","plan"):
        ct.select(parent,stage);folder=fixtures/(parent+"_"+stage);folder.mkdir(exist_ok=True)
        refs={}
        for role in ("request","caps","config","closure","independent_review"):
            refs[role]=put(folder/(role+".json"),dict(schema="SOURCE_TEST_ROLE_FIXTURE_ONLY",role=role,parent=parent,stage=stage,actual_scientific_or_admission_evidence=False))
        if stage=="infer":
            prior={k:row[v] for k,v in (("completion_ref","simulation_completion_ref"),("wait_ref","simulation_wait_ref"),("outer_ref","simulation_outer_terminal_ref"),("close_ref","simulation_close_ref"),("observation_ref","simulation_observation_ref"))}
        else:
            prior={k:put(folder/("prior_"+k+".json"),dict(schema="SOURCE_TEST_FUTURE_ROLE_FIXTURE_ONLY",role=k,parent=parent,stage=stage,actual_scientific_or_admission_evidence=False)) for k in ("completion_ref","wait_ref","outer_ref","close_ref","observation_ref")}
        q=dict(caps_ref=refs["caps"],source_review_ref=refs["independent_review"],prior=prior)
        roles=ct.direct_roles(refs["request"],q,refs["config"],refs["closure"],ct.RUNTIME_REF)
        union=ct.unique([*base,*roles])
        check(parent+"_"+stage+"_complete_exact_actual_base_role_union62",len(union)==62)
        check(parent+"_"+stage+"_no_direct_resource_suffix",all(Path(r["path"]).suffix in (".py",".sh",".json") for r in union))
        request=dict(key="SOURCE_TEST_ONLY/"+parent+"_"+stage,output=str(HERE/"uncreated_products"/(parent+"_"+stage)),auxiliary_scopes=[str(HERE/"uncreated_aux"/(parent+"_"+stage))],reserved_bytes=ct.CAPS["reservation_bytes"],consumer_bindings=union)
        paths=v4.scope_spec(ROOT,request)
        check(parent+"_"+stage+"_actual_unchanged_scope_spec_accepts_complete_union",len(paths)==2)
        bad=copy.deepcopy(request);bad["consumer_bindings"].append(manifest["original_shared_slot_refs"][0])
        reject(parent+"_"+stage+"_real_scope_rejects_direct_lock",lambda:v4.scope_spec(ROOT,bad),"consumer bindings must be source/runtime authority metadata")
        bad=copy.deepcopy(request)
        for i in range(3):bad["consumer_bindings"].append(put(folder/("extra_"+str(i)+".json"),dict(schema="SOURCE_TEST_ROLE_FIXTURE_ONLY",extra=i)))
        reject(parent+"_"+stage+"_real_scope_rejects65",lambda:v4.scope_spec(ROOT,bad),"source/runtime consumer metadata bindings required")
        bad=copy.deepcopy(request);bad["consumer_bindings"][0]["sha256"]="0"*64
        reject(parent+"_"+stage+"_real_scope_rejects_changed_bound_source",lambda:v4.scope_spec(ROOT,bad),"bound source or authority bytes changed")
        plans[parent+"/"+stage]=dict(count=len(union),consumer_bindings=union,fixture_refs=[r for r in union if r["path"].startswith(str(fixtures.relative_to(ROOT)))],actual_base_roles_ref=ref(OLD/"admitted_val003_infer_v2/reserve_authority.json"),actual_scope_spec="src/arggnn/project_budget_v4.py",actual_scope_spec_accepted=True,future_roles_are_only_explicit_source_test_fixtures=True,actual_admission=False)
def functions(p):return {n.name:n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def normalized(node):
    text=ast.unparse(copy.deepcopy(node)).replace(OLD.name,HERE.name).replace("_native14v3_v2","_native14v3_v3").replace("-native14v3-v2","-native14v3-v3").replace("'_v2'","'_v3'")
    return ast.parse(text).body[0]
allowed={"root_contract_v1.py":{"source_paths","direct_roles","validate_request","bind"},"root_driver_v1.py":{"admit"},"worker_entry_v1.py":set(),"plan_worker_v1.py":set()}
changed={}
for name,permitted in allowed.items():
    before=functions(OLD/name);after=functions(HERE/name);delta=[]
    for k,n in before.items():
        if ast.dump(normalized(n))!=ast.dump(after[k]):
            check("only_permitted_operational_body_"+name+"_"+k,k in permitted);delta.append(k)
    changed[name]=delta
check("cpu_slot_runtime_AST_identical",ast.dump(functions(OLD/"root_contract_v1.py")["cpu_slot"])==ast.dump(functions(HERE/"root_contract_v1.py")["cpu_slot"]))
check("supervisor_execute_and_cleanup_AST_identical",ast.dump(functions(OLD/"root_driver_v1.py")["execute"])==ast.dump(functions(HERE/"root_driver_v1.py")["execute"]))
check("plan_execute_AST_identical",ast.dump(functions(OLD/"plan_worker_v1.py")["execute"])==ast.dump(functions(HERE/"plan_worker_v1.py")["execute"]))
for name in ("readiness.v1.json","scientific_basis.v1.json","caps.template.inactive.v1.json","reuse_provenance.v1.json","infer.scientific_config.template.inactive.v1.json","native_paths.scientific_config.template.inactive.v1.json","plan.scientific_config.template.inactive.v1.json"):
    check("original_science_and_caps_byte_identical_"+name,(HERE/name).read_bytes()==(OLD/name).read_bytes())
check("exact_original_expiry_constant",ct.ORIGINAL_EXPIRY_UTC=="2026-10-16T20:04:16.988073+00:00")
source=(HERE/"root_contract_v1.py").read_text()
check("bind_does_not_mint_new_wall_deadline","end=now+timedelta(seconds=CAPS" not in source and "end=datetime.fromisoformat(ORIGINAL_EXPIRY_UTC)" in source)
check("request_expiry_cannot_be_reset",'q["expires_at_utc"]==ORIGINAL_EXPIRY_UTC' in source)
check("same_original_slots_not_cloned",not any(HERE.glob("cpu_slot_*.lock")))
for p in HERE.glob("*.py"):compile(p.read_text(),str(p),"exec")
cli=subprocess.run([str(ROOT/".runtime/env/bin/python"),"-B","-m",str(HERE.relative_to(ROOT)).replace("/",".")+".root_driver_v1"],cwd=ROOT,env=dict(os.environ,PYTHONPATH=".:src:scripts",PYTHONDONTWRITEBYTECODE="1"),capture_output=True,text=True)
check("real_inert_default_CLI",cli.returncode==0 and json.loads(cli.stdout)["execution"] is False and cli.stderr=="")
# New operational checks use source-extracted functions and fake metadata writers only.
import stat,types
slot_value=read(HERE/"slot_resource_identity.v3.json")
fn=functions(HERE/"root_contract_v1.py")["validate_slot_resources"]
for field,value in (("st_ino",slot_value["slots"][0]["st_ino"]+1),("st_dev",slot_value["slots"][0]["st_dev"]+1),("st_size",1),("st_nlink",2)):
    altered=copy.deepcopy(slot_value);altered["slots"][0][field]=value
    ns=dict(ct.__dict__);ns["read"]=lambda r,v=altered:v
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(fn)],type_ignores=[])),"<slot-identity-negative>","exec"),ns)
    reject("slot_identity_rejects_"+field,lambda:ns["validate_slot_resources"](manifest["refs"]))
reject("slot_identity_requires_bound_JSON_receipt",lambda:ct.validate_slot_resources([r for r in manifest["refs"] if r!=ct.SLOT_IDENTITY_REF]))
real_datetime=__import__("datetime").datetime
fixed=real_datetime.fromisoformat(ct.ORIGINAL_EXPIRY_UTC)
class FrozenDateTime:
    @staticmethod
    def now(tz):return fixed-__import__("datetime").timedelta(seconds=1800)
    fromisoformat=staticmethod(real_datetime.fromisoformat)
saved={}
ns=dict(ct.__dict__,datetime=FrozenDateTime,bind_policy=lambda r:None,original_guard=lambda:None,
    select=lambda p,s:None,reviewed=lambda r:None,source_paths=lambda:(),actual_prior=lambda p,s:{"fixture":True},
    scientific_config=lambda *a:{"fixture":True},read=lambda r:{"renewed_expires_at_utc":"2026-10-20T00:00:00+00:00"},
    save=lambda folder,name,v:saved.setdefault(name,copy.deepcopy(v)) or {"fixture":True},
    local=lambda p:p,ref=lambda p:{"path":str(p),"sha256":"0"*64},worker_caps=lambda s:{"fixture":True},
    validate_request=lambda r:None)
exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(functions(HERE/"root_contract_v1.py")["bind"])],type_ignores=[])),"<bind-metadata-only>","exec"),ns)
ns["bind"]("american_c3_val_003","infer",{"fixture":True},{"fixture":True})
check("source_bind_fixed_original_expiry_not_fresh3h",saved["request_val003_infer.root_active_v1.json"]["expires_at_utc"]==ct.ORIGINAL_EXPIRY_UTC)
class ExpiredDateTime(FrozenDateTime):
    @staticmethod
    def now(tz):return fixed
ns["datetime"]=ExpiredDateTime;saved.clear()
reject("source_bind_after_original_expiry_rejected",lambda:ns["bind"]("american_c3_val_003","infer",{},{}),"Original failed-admission deadline")
check("source_bind_expired_writes_nothing",not saved)

value=dict(schema="VAL003004_real_unchanged_scope_contract_results_v3",status="PASS",created_utc=datetime.now(timezone.utc).isoformat(),checks=RESULTS,passed=len(RESULTS),phases=plans,changed_production_bodies=changed,original_expiry_utc=ct.ORIGINAL_EXPIRY_UTC,fixture_interpretation="All actual source/helper/runtime/guard refs and37 actual base bindings are real. Not-yet-existing stage authority/predecessor roles use bounded JSON SOURCE_TEST_ONLY fixtures. Real unchanged scope_spec hashes and validates each full62-ref union. No future scientific evidence is asserted.",actual_v4_source_ref=ct.V4_REF,complete_manifest_refs=len(manifest["refs"]),actual_accounting_or_admission_calls=0,actual_process_device_socket_probes=0,actual_scientific_execute_calls=0,scientific_payload_reads=0,default_cli_stdout=cli.stdout.strip())
put(HERE/"real_scope_contract_results.v3.json",value)
print(json.dumps(dict(status="PASS",checks=len(RESULTS),all_six_complete_unions=62,real_scope_spec=True,scientific_jobs=0)))
