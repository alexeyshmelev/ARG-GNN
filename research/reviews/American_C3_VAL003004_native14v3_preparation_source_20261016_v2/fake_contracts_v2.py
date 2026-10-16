"""Meaningful pure/fake-source contracts. No guard/accounting/process/device execution."""
import ast,copy,contextlib,hashlib,json,types
from pathlib import Path
from datetime import datetime,timezone,timedelta
HERE=Path(__file__).parent
CT=ast.parse((HERE/"root_contract_v1.py").read_text());DT=ast.parse((HERE/"root_driver_v1.py").read_text())
RESULTS=[]
def check(name,value=True):
    if not value:raise AssertionError(name)
    RESULTS.append(dict(name=name,status="PASS"))
def require(ok,message):
    if not ok:raise ValueError(message)
def rejected(name,fn):
    try:fn()
    except (ValueError,KeyError,TypeError):check(name);return
    raise AssertionError(name)
def node(t,name):return next(n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
def extract(t,names,ns):
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node(t,n)) for n in names],type_ignores=[])),"<fake-source-contract>","exec"),ns)
def readjson(p):return json.loads(Path(p).read_bytes())
basis=readjson(HERE/"scientific_basis.v1.json");ready=readjson(HERE/"readiness.v1.json")
originals={s:readjson(basis[s+"_config_ref"]["path"]) for s in ["infer","native_paths"]}
originals["plan"]=readjson(basis["plan_config_ref"]["path"])
for parent,row in ready["parents"].items():
    # Run the actual scientific-config rebinder through fake metadata APIs only.
    for stage in ["infer","native_paths","plan"]:
        refs=[]
        prior=dict(observation_ref=dict(path="FAKE/"+parent+"/"+stage+"/observed.json",sha256="1"*64),
            completion_ref=dict(path="FAKE/"+parent+"/native_completed.json",sha256="2"*64),
            focal_registry_ref=dict(path="FAKE/"+parent+"/focals.npz",sha256="3"*64))
        def fake_read(r):
            refs.append(r)
            path=r["path"] if isinstance(r,dict) else r
            if path.endswith("scientific_basis.v1.json"):return copy.deepcopy(basis)
            if path.endswith("readiness.v1.json"):return copy.deepcopy(ready)
            if ".scientific_config.template.inactive.v1.json" in path:return {}
            for k in ["infer","native_paths","plan"]:
                if path==basis[k+"_config_ref"]["path"]:return copy.deepcopy(originals[k])
            raise AssertionError(path)
        ns=dict(require=require,read=fake_read,ref=lambda p:dict(path=p,sha256="0"*64),
            BASIS=str(HERE/"scientific_basis.v1.json"),READINESS=str(HERE/"readiness.v1.json"),REL=str(HERE))
        extract(CT,["scientific_config"],ns)
        cfg=ns["scientific_config"](parent,stage,prior,True)
        allowed={"execution_enabled","observation_ref"}|({"path_sampling_seed"} if stage=="native_paths" else {"native_preparation_ref","focal_registry_ref"} if stage=="plan" else set())
        check(parent+"_"+stage+"_original_config_only_allowed_rebindings",
            set(cfg)==set(originals[stage]) and {k for k in cfg if cfg[k]!=originals[stage][k]}<=allowed)
        check(parent+"_"+stage+"_no_payload_ref_opened",all(not (r["path"] if isinstance(r,dict) else r).endswith((".npz",".trees",".pt",".npy",".pkl")) for r in refs))
        if stage=="native_paths":check(parent+"_original_focal_seed",cfg["path_sampling_seed"]==row["seed_row"]["path_sampling_seed"])
        if stage=="plan":
            check(parent+"_full1600_plan_all_original_block_bounds",
                cfg["expected_haplotypes"]==1600 and cfg["reference_counts"]==[200,200,200] and cfg["max_nodes_per_block"]==250000
                and cfg["max_trees_per_block"]==32 and cfg["max_sites_per_block"]==4096 and cfg["allow_missing"] is False)
# Actual observed-input validator rejects all scientific shortcuts/truth.
valid=dict(parent_id="american_c3_val_003",split="val",sequence_length_bp=30000000,sample_unit="haploid",
    class_order=["EUR","EAS","AFR"],counts_haploid=dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000),query_ancestry_truth_in_inputs=False)
current=copy.deepcopy(valid)
env=dict(require=require,read=lambda r:current)
extract(CT,["observation"],env)
check("full_original_observation_identity_accepted",env["observation"]({},"american_c3_val_003")==valid)
for key,value in [("sequence_length_bp",2000000),("parent_id","american_c3_val_004"),("query_ancestry_truth_in_inputs",True),
    ("counts_haploid",dict(AFR=200,EUR=200,ASIA=200,ADMIX=999)),("class_order",["AFR","EUR","EAS"]),("sample_unit","diploid")]:
    current=copy.deepcopy(valid);current[key]=value
    rejected("observed_input_reject_"+key,lambda:env["observation"]({},"american_c3_val_003"))
# Exact existing simulation completion/wait/terminal/close and seed evidence.
for parent,row in ready["parents"].items():
    records={r["path"]:readjson(r["path"]) for k,r in row.items() if isinstance(r,dict) and set(r)=={"path","sha256"}}
    ns=dict(require=require,read=lambda r:records[r["path"]],ref=lambda p:dict(path=p,sha256="0"*64),
        READINESS="FAKE/readiness",observation=lambda r,p:records[r["path"]])
    records["FAKE/readiness"]=ready
    extract(CT,["existing_simulation"],ns)
    check(parent+"_saved_success_wait_outer_close",ns["existing_simulation"](parent)==row)
    for label,key,field,value in [
        ("wait_unknown","simulation_wait_ref","returncode",None),("wait_failure","simulation_wait_ref","operational_failure","failure"),
        ("drain_unknown","simulation_wait_ref","owned_worker_group_ESRCH",False),("outer_unconsumed","simulation_outer_terminal_ref","actual_terminal_consumed",False),
        ("outer_failed","simulation_outer_terminal_ref","actual_exit_code",1),("wrong_seed","simulation_config_ref","ancestry_seed",1)]:
        target=records[row[key]["path"]];saved=target[field];target[field]=value
        rejected(parent+"_reject_"+label,lambda:ns["existing_simulation"](parent));target[field]=saved
# Actual downstream binding function through a wholly fake metadata namespace.
for parent,row in ready["parents"].items():
    for stage in ["native_paths","plan"]:
        previous="infer" if stage=="native_paths" else "native_paths"
        root=Path("/FAKE");rel="source"
        folder=root/rel/("admitted_val"+parent[-3:]+"_"+previous+"_v2")
        output="experiments/results/American_C3_val"+parent[-3:]+"_full30mb_"+previous+"_native14v3_v2"
        def freference(p):return dict(path=str(p.relative_to(root)) if isinstance(p,Path) and p.is_absolute() else str(p),sha256="a"*64)
        cp=freference(output+"/completed.json");wr=freference(folder/"worker_aux/actual_wait.json")
        tr=freference(folder/"outer_terminal.root_actual_v1.json");cr=freference(folder/"closed/close_receipt.json")
        obsref=dict(path="FAKE/sanitized.json",sha256="b"*64)
        observed=dict(valid,parent_id=parent,genealogy_kind="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG",
            sites_and_sample_identities_verified=True,source_observation_ref=row["simulation_observation_ref"])
        done=dict(status="completed_current_C3_inference" if previous=="infer" else "completed_native_C3_path_preparation",
            observation_ref=obsref,parent_id=parent,query_focal_count=10000,query_ancestry_truth_opened=False,
            native_split_seed=42,native_train_fraction=.9,class_order=["EUR","EAS","AFR"],
            focals_ref=dict(path="FAKE/focals.npz",sha256="c"*64))
        wait=dict(completed_ref=cp,actual_direct_worker_wait=True,returncode=0,parent_id=parent,phase=previous+"_val_"+parent[-3:],
            input_ref=freference(folder/"input.json"),all_owned_processes_and_groups_drained=True,operational_failure=None)
        terminal=dict(root_authorizer="root_agent",actual_terminal_consumed=True,actual_exit_code=0,completed_ref=cp,direct_wait_ref=wr)
        closed=dict(key="H041/American-C3-val"+parent[-3:]+"-full30mb-"+previous+"-native14v3-v2",
            value=dict(future_writes_and_relaunch_revoked=True,proof=dict(completed_ref=cp,direct_wait_ref=wr)))
        records={cp["path"]:done,wr["path"]:wait,tr["path"]:terminal,cr["path"]:closed,obsref["path"]:observed}
        ns=dict(require=require,ROOT=root,REL=rel,local=lambda p:root/p,ref=freference,
            existing_simulation=lambda p:row,read=lambda r:records[r["path"]])
        extract(CT,["observation","actual_prior"],ns)
        prior=ns["actual_prior"](parent,stage)
        check(parent+"_"+stage+"_genuine_future_prerequisite_bound",prior["completion_ref"]==cp)
        for label,record,field,value in [
            ("failed_wait",wait,"returncode",-15),("incomplete_drain",wait,"all_owned_processes_and_groups_drained",False),
            ("wrong_wait_parent",wait,"parent_id","wrong"),("unconsumed_outer",terminal,"actual_terminal_consumed",False),
            ("not_revoked",closed["value"],"future_writes_and_relaunch_revoked",False),
            ("wrong_closed_key",closed,"key","H041/other"),("missing_site_verification",observed,"sites_and_sample_identities_verified",False),
            ("truth_in_input",observed,"query_ancestry_truth_in_inputs",True)]:
            saved=record[field];record[field]=value
            rejected(parent+"_"+stage+"_reject_"+label,lambda:ns["actual_prior"](parent,stage));record[field]=saved
        if previous=="native_paths":
            for label,field,value in [("focal_subset","query_focal_count",9999),("query_truth","query_ancestry_truth_opened",True),
                                     ("wrong_split_seed","native_split_seed",41)]:
                saved=done[field];done[field]=value
                rejected(parent+"_plan_reject_"+label,lambda:ns["actual_prior"](parent,stage));done[field]=saved

# Fake guard/namespace API proves current configure_with_renewal binding and role split.
for consumer in [False,True]:
    calls=[];v4=types.SimpleNamespace(__name__="fake_original_v4")
    for name in ["canonical","file_bytes","mutex"]:
        f=lambda:None;f.__module__=v4.__name__;setattr(v4,name,f)
    g=types.SimpleNamespace(MODE="native14_stopped_retention_v3",budget=types.SimpleNamespace(v4=v4))
    g.configure_with_renewal_v1=lambda *a,**k:calls.append(("configure",a,k))
    constants={}
    for n in CT.body:
        if isinstance(n,ast.Assign):
            try:value=ast.literal_eval(n.value)
            except (ValueError,TypeError):continue
            for target in n.targets:
                if isinstance(target,ast.Name):constants[target.id]=value
    # dict(...) refs are copied from literal source expressions using fake constructor only.
    for n in CT.body:
        if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=="dict":
            for target in n.targets:
                if isinstance(target,ast.Name):constants[target.id]={k.arg:ast.literal_eval(k.value) for k in n.value.keywords}
    constants["POLICY_REF"]=dict(path=constants["POLICY_PATH"],sha256="0"*64)
    ns=dict(constants,require=require,ROOT=Path("/FAKE"),sys=types.SimpleNamespace(path=[]),raw=lambda r:None,bind_policy=lambda r:None)
    memo=types.SimpleNamespace(install_consumer_v1=lambda *a,**kw:calls.append(("memo",a,kw)) or "memo_handle")
    ns["load"]=lambda r,name:g if r==constants["GUARD_REF"] else memo
    extract(CT,["original_guard","consumer_guard"],ns)
    got=ns["consumer_guard"]()[0] if consumer else ns["original_guard"]()
    check("current_Census_configure_with_renewal_"+str(consumer),got is g and calls[0]==("configure",(constants["REPAIR_REF"],),dict(policy_ref=constants["POLICY_REF"])))
    check("original_mutation_or_readonly_namespace_memo_"+str(consumer),len(calls)==(2 if consumer else 1))
# A generic old SOURCE_PASS cannot activate a different new source manifest.
review_current=dict(verdict="SOURCE_PASS",blockers=[],source_manifest_ref=dict(path="new/manifest.json",sha256="d"*64))
ns=dict(require=require,read=lambda r:review_current,ref=lambda p:dict(path="new/manifest.json",sha256="d"*64),MANIFEST="new/manifest.json",REL="new")
extract(CT,["reviewed"],ns)
check("independent_review_binds_exact_new_manifest",ns["reviewed"]({})==review_current)
for label,change in [("old_manifest",lambda z:z.update(source_manifest_ref=dict(path="old/manifest.json",sha256="e"*64))),
                     ("request_changes",lambda z:z.update(verdict="REQUEST_CHANGES")),
                     ("blocker",lambda z:z.update(blockers=["P1"]))]:
    original_review=copy.deepcopy(review_current);change(review_current)
    rejected("reject_review_"+label,lambda:ns["reviewed"]({}));review_current=original_review

# Finite-cleanup body is unchanged from the existing owned CPU framework.
original=ast.parse(Path(basis["finite_supervisor_source_ref"]["path"]).read_text())
oldexec=node(original,"execute");newexec=node(DT,"execute")
oldt=next(n for n in oldexec.body if isinstance(n,ast.Try) and n.finalbody)
newt=next(n for n in newexec.body if isinstance(n,ast.Try) and n.finalbody)
check("finite_owned_cleanup_finally_AST_identical",ast.dump(ast.Module(body=oldt.finalbody,type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=newt.finalbody,type_ignores=[]),include_attributes=False))
# Exercise the actual complete cleanup block with fake OS/time/tree APIs.
for clean in [True,False]:
    signals=[];clock=[0.];waited=[False];closed=[]
    def monotonic():clock[0]+=.01;return clock[0]
    def waitpid(pid,options):
        if not clean:return (0,0)
        if pid==77 and not waited[0]:waited[0]=True;return (77,0)
        if pid==-1:raise ChildProcessError()
        return (0,0)
    def killpg(group,sig):
        if clean:raise ProcessLookupError()
    osfake=types.SimpleNamespace(close=lambda fd:closed.append(fd),waitpid=waitpid,WNOHANG=1,
        waitstatus_to_exitcode=lambda status:0,read=lambda fd,n:b"" if clean else (_ for _ in ()).throw(BlockingIOError()),
        getpid=lambda:7,killpg=killpg)
    row=dict(pid=77,group=77)
    owned=types.SimpleNamespace(signal_identity=lambda r,sig:signals.append((r,sig)),
        _children=lambda pid:[] if clean else [77])
    proc=types.SimpleNamespace(pid=77,returncode=None,stdout=types.SimpleNamespace(fileno=lambda:99,close=lambda:closed.append("pipe")))
    tree=types.SimpleNamespace(known={77:row},refresh=lambda:None,identities_absent=lambda:clean)
    ns=dict(os=osfake,time=types.SimpleNamespace(monotonic=monotonic,sleep=lambda sec:clock.__setitem__(0,clock[0]+sec)),
        signal=types.SimpleNamespace(SIGTERM=15,SIGKILL=9,signal=lambda *a:None),owned=owned,proc=proc,tree=tree,seed=row,
        need=require,CAPS=dict(cleanup_seconds=30,log_bytes=1024),gate_read=1,gate_write=2,status=None,failure=None,
        tail=bytearray(),eof=False,nonblocking=True,old={})
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(newt.finalbody),type_ignores=[])),"<actual-cleanup-fake-OS>","exec"),ns)
    check("bounded_cleanup_time_"+str(clean),clock[0]<=30.5)
    check("only_genuine_wait_retained_"+str(clean),(ns["status"]==0) if clean else ns["status"] is None)
    check("no_pre_drain_metadata_or_guard_call_"+str(clean),signals and ns["groups"] if clean else signals and ns["status"] is None)
# The full plan worker execute body is exactly the executed original implementation.
pt=ast.parse((HERE/"plan_worker_v1.py").read_text())
opt=ast.parse(Path(basis["original_plan_worker_ref"]["path"]).read_text())
check("original_full_plan_worker_execute_AST_unchanged",ast.dump(node(pt,"execute"),include_attributes=False)==ast.dump(node(opt,"execute"),include_attributes=False))
# Two CPU leases through fake OFDs, never actual flock/process probes.
busy=set();held=set();fds={};nextfd=[10]
class SlotPath:
    def __init__(self,text):self.text=text;self.index=int(text[-6])
    def lstat(self):return types.SimpleNamespace(st_ino=100+self.index)
def slotopen(path,flags):
    fd=nextfd[0];nextfd[0]+=1;fds[fd]=path.index;return fd
def slotclose(fd):
    index=fds.pop(fd)
    if fd in held:busy.discard(index);held.discard(fd)
def flock(fd,flags):
    index=fds[fd]
    if index in busy:raise BlockingIOError()
    busy.add(index);held.add(fd)
slotenv=dict(require=require,contextlib=contextlib,REL="FAKE",SLOT_REL="FAKE",local=lambda p:SlotPath(p),raw=lambda r:None,ref=lambda p:p,
    fcntl=types.SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=flock),
    os=types.SimpleNamespace(O_RDWR=2,O_NOFOLLOW=4,open=slotopen,close=slotclose,
        fstat=lambda fd:types.SimpleNamespace(st_ino=100+fds[fd],st_size=0)))
extract(CT,["cpu_slot"],slotenv)
first=slotenv["cpu_slot"]();a=first.__enter__()
second=slotenv["cpu_slot"]();b=second.__enter__()
check("two_cpu_slots_can_be_held",a!=b and len(busy)==2)
rejected("third_cpu_producer_rejected_before_launch",lambda:slotenv["cpu_slot"]().__enter__())
second.__exit__(None,None,None);first.__exit__(None,None,None)
check("CPU_slot_inherited_by_gated_worker","pass_fds=(gate_read,SLOT_FD)" in (HERE/"root_driver_v1.py").read_text())

# Actual new policy binder with wholly fake exact metadata. No actual future policy hash exists.
constants={}
for n in CT.body:
    if isinstance(n,ast.Assign):
        try:value=ast.literal_eval(n.value)
        except (ValueError,TypeError):
            if isinstance(n.value,ast.Call) and getattr(n.value.func,"id",None)=="dict":
                value={k.arg:ast.literal_eval(k.value) for k in n.value.keywords}
            else:continue
        for target in n.targets:
            if isinstance(target,ast.Name):constants[target.id]=value
fake=lambda path:dict(path=path,sha256="0"*64)
pr=fake(constants["POLICY_PATH"]);ar=fake("FAKE/native14_activation");vr=fake("FAKE/native14_revocation");rr=fake("FAKE/native14_review")
policy=dict(active=True,schema="native14_exact_stopped_retention_policy_v1",source_ref=constants["GUARD_REF"],
    source_manifest_ref=constants["NATIVE_MANIFEST_REF"],measurement_source_ref=constants["NATIVE_HELPER_REF"],
    journal=".runtime/storage/native14_stopped_retention_v3",root_authorizer="root_agent",
    root_activation_ref=ar,permanent_revocation_ref=vr,source_review_ref=rr,base_pins={"FAKE":0},base_projected_bytes=1)
activation=dict(active=True,root_authorizer="root_agent",source_ref=constants["GUARD_REF"],
    source_manifest_ref=constants["NATIVE_MANIFEST_REF"],measurement_source_ref=constants["NATIVE_HELPER_REF"],
    new_journal=policy["journal"],permanent_revocation_ref=vr,source_review_ref=rr,frozen_parent_pins=policy["base_pins"],
    frozen_parent_projected_bytes=1)
review=dict(verdict="SOURCE_PASS",source_ref=constants["GUARD_REF"],source_manifest_ref=constants["NATIVE_MANIFEST_REF"],
    measurement_source_ref=constants["NATIVE_HELPER_REF"])
revocation=dict(active=True)
records={pr["path"]:policy,ar["path"]:activation,vr["path"]:revocation,rr["path"]:review}
authenticated=[]
def fake_raw(r):
    require(isinstance(r,dict) and set(r)=={"path","sha256"} and r["path"] in records and r["sha256"]=="0"*64,"fake exact policy metadata")
    authenticated.append(r)
penv=dict(constants,require=require,raw=fake_raw,read=lambda r:records[r["path"]])
extract(CT,["bind_policy"],penv)
check("explicit_actual_v3_policy_exact_binding",penv["bind_policy"](pr)==policy and penv["POLICY_REF"]==pr)
check("policy_authenticates_activation_revocation_review",all(r in authenticated for r in [pr,ar,vr,rr]))
for label,record,field,value in [
    ("inactive_policy",policy,"active",False),("wrong_guard",policy,"source_ref",fake("FAKE/old_guard")),
    ("wrong_manifest",policy,"source_manifest_ref",fake("FAKE/old_manifest")),("wrong_journal",policy,"journal","old"),
    ("inactive_activation",activation,"active",False),("changed_frozen_projection",activation,"frozen_parent_projected_bytes",2),
    ("wrong_root_review_manifest",review,"source_manifest_ref",fake("FAKE/old_manifest")),
    ("unrevoked_original",revocation,"active",False)]:
    saved=record[field];record[field]=value;penv["POLICY_REF"]=None
    rejected("v3_policy_reject_"+label,lambda:penv["bind_policy"](pr));record[field]=saved
penv["POLICY_REF"]=None
rejected("v3_policy_reject_template_path",lambda:penv["bind_policy"](fake("FAKE/inactive_policy")))
rejected("v3_policy_reject_changed_hash",lambda:penv["bind_policy"](dict(pr,sha256="1"*64)))
penv["POLICY_REF"]=dict(pr,sha256="1"*64)
rejected("v3_policy_reject_in_process_policy_swap",lambda:penv["bind_policy"](pr))
# Actual manifest authenticator sees every transitive source and the consolidated baseline config.
manifest=readjson(HERE/"source_manifest.v1.json");raw_calls=[]
menv=dict(read=lambda r:manifest,ref=lambda p:fake(p),MANIFEST=str(HERE/"source_manifest.v1.json"),raw=lambda r:raw_calls.append(r))
extract(CT,["source_paths"],menv)
sourcepaths=menv["source_paths"]()
check("every_transitive_manifest_ref_raw_authenticated",raw_calls==manifest["refs"] and len(sourcepaths)==len(manifest["refs"]))
check("consolidated_baseline_config_still_explicitly_authenticated",constants["BASE_CONFIG_REF"] in raw_calls)
def rejecting_raw(r):
    if r==constants["BASE_CONFIG_REF"]:raise ValueError("changed fake consolidated config")
menv["raw"]=rejecting_raw
rejected("changed_consolidated_baseline_ref_blocks_source_validation",lambda:menv["source_paths"]())
# Exercise only authored source/compact-metadata ref helpers; no guard or resource operation.
sourceenv=dict(require=require,Path=Path,ROOT=Path.cwd(),hashlib=hashlib)
extract(CT,["local","ref","raw"],sourceenv)
for r in manifest["refs"]:sourceenv["raw"](r)
check("all_manifest_source_metadata_suffixes_supported_by_actual_authenticator",True)
# Full actual direct_roles + unchanged native state ancestry shape, with future refs confined to this fixture.
def realref(p):
    p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
nativeplan=readjson(Path(constants["GUARD_REF"]["path"]).parent/"direct_role_plan.v3.json")
nativefuture=[pr,ar,vr,rr];allplans={}
for parent,row in ready["parents"].items():
    for stage in ["infer","native_paths","plan"]:
        packet=str(HERE)+"/admitted_val"+parent[-3:]+"_"+stage+"_v2"
        future=[fake(str(HERE)+"/request_val"+parent[-3:]+"_"+stage+".root_active_v1.json"),
            fake(str(HERE)+"/caps_val"+parent[-3:]+"_"+stage+".root_active_v1.json"),fake(packet+"/config.json"),
            fake(packet+"/source_and_input_closure.json"),fake("FAKE/VALv2_review")]
        if stage=="infer":
            prior={k:row[v] for k,v in [("completion_ref","simulation_completion_ref"),("wait_ref","simulation_wait_ref"),
                ("outer_ref","simulation_outer_terminal_ref"),("close_ref","simulation_close_ref"),("observation_ref","simulation_observation_ref")]}
        else:
            previous="infer" if stage=="native_paths" else "native_paths"
            prevpacket=str(HERE)+"/admitted_val"+parent[-3:]+"_"+previous+"_v2"
            prevoutput="experiments/results/American_C3_val"+parent[-3:]+"_full30mb_"+previous+"_native14v3_v2"
            prior=dict(completion_ref=fake(prevoutput+"/completed.json"),wait_ref=fake(prevpacket+"/worker_aux/actual_wait.json"),
                outer_ref=fake(prevpacket+"/outer_terminal.root_actual_v1.json"),close_ref=fake(prevpacket+"/closed/close_receipt.json"),
                observation_ref=fake("FAKE/"+parent+"/"+previous+"/actual_sanitized_observation"))
        q=dict(caps_ref=future[1],source_review_ref=future[4],prior=prior)
        env=dict(constants,POLICY_REF=pr,require=require,read=lambda r:basis,ref=realref,
            SOURCE=str(HERE/"root_contract_v1.py"),DRIVER=str(HERE/"root_driver_v1.py"),WORKER=str(HERE/"worker_entry_v1.py"),
            REL=str(HERE),READINESS=str(HERE/"readiness.v1.json"),BASIS=str(HERE/"scientific_basis.v1.json"),
            CAPFILE=str(HERE/"caps.template.inactive.v1.json"),MANIFEST=str(HERE/"source_manifest.v1.json"),STAGE=stage,
            raw=lambda r:None)
        extract(CT,["direct_roles","unique"],env)
        runtime=basis[{"infer":"inference_runtime_ref","native_paths":"native_runtime_ref","plan":"plan_runtime_ref"}[stage]]
        roles=env["direct_roles"](future[0],q,future[2],future[3],runtime)
        bound=env["unique"]([*nativeplan["concrete_union_refs"],*nativefuture,*roles])
        check(parent+"_"+stage+"_full_actual_role_union_exact64",len(bound)==64)
        check(parent+"_"+stage+"_only_baseline_config_consolidated",constants["BASE_CONFIG_REF"] not in roles)
        expected_helpers=[basis["common_producer_ref"],basis["inference_producer_ref"],basis["simulation_sanitize_helper_ref"],
            basis["native_producer_ref"],basis["author_path_helper_ref"],basis["plan_hook_ref"],basis["plan_dataset_ref"]]
        check(parent+"_"+stage+"_all_actual_science_helper_roles_still_direct",all(r in roles for r in expected_helpers))
        rejected(parent+"_"+stage+"_unchanged64_limit_rejects65",lambda:env["unique"]([*bound,fake("FAKE/extra65")]))
        known=[r for r in bound if r["sha256"]!="0"*64]
        unknown=[dict(path=r["path"] if not r["path"].startswith("FAKE/") else None,role=r["path"][5:] if r["path"].startswith("FAKE/") else r["path"],actual_exact_ref_required_at_bind_or_admit=True) for r in bound if r["sha256"]=="0"*64]
        allplans[parent+"/"+stage]=dict(concrete_refs=known,concrete_count=len(known),future_actual_role_descriptors=unknown,
            future_actual_count=len(unknown),complete_upper_union=len(bound),fits_original64=True)
# Source/API role plan never assigns the fixtures' fake hashes to actual future authorities.
(HERE/"direct_role_plan.v2.json").write_text(json.dumps(dict(schema="American_C3_VAL003004_native14v3_complete_role_plan_v2",
    source_only=True,original_direct_ceiling=64,native14_state_plan_ref=realref(Path(constants["GUARD_REF"]["path"]).parent/"direct_role_plan.v3.json"),
    phases=allplans,single_consolidated_ref=constants["BASE_CONFIG_REF"],
    every_transitive_ref_explicitly_raw_authenticated=True,all_native30_roles_and_actual_science_roles_direct=True,
    actual_future_refs_not_assigned_fake_hashes=True,actual_guard_or_accounting_not_called=True),sort_keys=True,indent=2)+"\n")
result=dict(schema="American_C3_VAL003004_source_fake_contract_results_v2",status="PASS",checks=RESULTS,passed=len(RESULTS),
    actual_guard_or_accounting_calls=False,actual_process_device_or_socket_probes=False,numerical_imports=False,scientific_payload_reads=False)
(HERE/"fake_contract_results.v2.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps(dict(status="PASS",checks=len(RESULTS)),sort_keys=True))
