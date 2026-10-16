"""Independent source-only review. Extracted functions use explicit in-memory fakes only."""
from pathlib import Path
from datetime import datetime,timezone
import ast,contextlib,copy,difflib,hashlib,json,stat,sys,types,traceback
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
D=ROOT/"research/reviews/American_C3_VAL003004_downward8GB_independent_review_20261016_v1"
NEW="research/reviews/American_C3_VAL003004_downward8GB_source_20261016_v1"
OLD="research/reviews/American_C3_VAL003004_native14v3_preparation_source_20261016_v3"
H="da140bb809942691d7d6f939a14043a1777bb8f4fab5c5dbeadbf4ebcace6da2"
M="99751fbc00a07694572fbd673d3c6abd0e149ec85fac0795e5c8f60b8de1f035"
authenticated=[];results=[]
def require(value,message):
    if not value:raise ValueError(message)
def check(name,value,detail=None):
    assert value,name
    results.append(dict(name=name,status="PASS",detail=detail))
def read_source(path):
    p=ROOT/path
    assert p.suffix in (".py",".json",".patch",".md") and p.stat().st_size<32*2**20
    return p.read_bytes()
def actual_ref(path):
    path=str(path.relative_to(ROOT)) if isinstance(path,Path) and path.is_absolute() else str(path)
    b=read_source(path)
    return dict(path=path,sha256=hashlib.sha256(b).hexdigest())
def auth(record):
    assert actual_ref(record["path"])==record,record
    authenticated.append(record)
    return read_source(record["path"])
def jfile(path):return json.loads(read_source(path))
def fixture_ref(label):
    return dict(path="INDEPENDENT_IN_MEMORY_FIXTURE/"+label,sha256=hashlib.sha256(("fixture:"+label).encode()).hexdigest())
def extract(path,names,ns,classes=()):
    tree=ast.parse(read_source(path).decode())
    nodes=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name in names) or (isinstance(n,ast.ClassDef) and n.name in classes)]
    assert len(nodes)==len(names)+len(classes)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"<independent_source_extraction_in_memory_fakes>","exec"),ns)
    return ns
def function_map(path):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(read_source(path).decode()).body if isinstance(n,ast.FunctionDef)}
def rejects(name,fn,errors=(ValueError,FileNotFoundError)):
    try:fn()
    except errors:results.append(dict(name=name,status="PASS_REJECTED"));return
    raise AssertionError("Invalid case accepted: "+name)
def run():
    seal=json.loads(auth(dict(path=NEW+"/source_handoff.v1.json",sha256=H)))
    manifest=json.loads(auth(dict(path=NEW+"/source_manifest.v1.json",sha256=M)))
    check("sealed_handoff_manifest_binding",seal["source_manifest_ref"]==actual_ref(NEW+"/source_manifest.v1.json") and seal["source_only"] and not seal["execution_or_admission_authorized"])
    for r in seal["packet_refs"]:
        assert r["path"].startswith(NEW+"/")
        auth(r)
    old_manifest=json.loads(auth(seal["predecessor_v3_manifest_ref"]))
    auth(seal["predecessor_v3_handoff_ref"])
    check("complete_frozen_predecessor_closure_retained",all(r in manifest["refs"] for r in old_manifest["refs"]),dict(predecessor_refs=len(old_manifest["refs"]),successor_refs=len(manifest["refs"])))
    parsed=[]
    for n in ("__init__.py","root_contract_v1.py","root_driver_v1.py","worker_entry_v1.py","plan_worker_v1.py","inert_downward_checks_v1.py"):
        ast.parse(read_source(NEW+"/"+n));parsed.append(n)
    check("all_successor_python_AST_parse",len(parsed)==6,parsed)
    diff=""
    for n in ("root_contract_v1.py","root_driver_v1.py","worker_entry_v1.py","plan_worker_v1.py","caps.template.inactive.v1.json"):
        diff+="".join(difflib.unified_diff(read_source(OLD+"/"+n).decode().splitlines(True),read_source(NEW+"/"+n).decode().splitlines(True),fromfile=OLD+"/"+n,tofile=NEW+"/"+n))
    check("independently_reproduced_patch_exact",diff.encode()==read_source(NEW+"/reservation_only_successor.v1.patch"),hashlib.sha256(diff.encode()).hexdigest())
    changed={}
    for n in ("root_contract_v1.py","root_driver_v1.py"):
        old=function_map(OLD+"/"+n);new=function_map(NEW+"/"+n)
        changed[n]=[name for name in new if old.get(name)!=new[name]]
    check("only_reviewed_contract_functions_changed",set(changed["root_contract_v1.py"])=={"select","phase_paths","predecessor_paths_v1","actual_prior","validate_request","bind","input_spec"},changed["root_contract_v1.py"])
    check("only_admit_driver_body_changed",changed["root_driver_v1.py"]==["admit"])
    for n in ("worker_entry_v1.py","plan_worker_v1.py"):
        expected=read_source(OLD+"/"+n).decode().replace(OLD,NEW).replace(OLD.replace("/","."),NEW.replace("/","."))
        check(n+"_namespace_only",read_source(NEW+"/"+n).decode()==expected)
    oldcaps=jfile(OLD+"/caps.template.inactive.v1.json");caps=jfile(NEW+"/caps.template.inactive.v1.json")
    expected=copy.deepcopy(oldcaps);expected["stages"]["native_paths"]["reservation_bytes"]=8000000000
    check("only_accounting_reservation_numerical_change",caps==expected)
    for n in ("native_paths.scientific_config.template.inactive.v1.json","plan.scientific_config.template.inactive.v1.json","scientific_basis.v1.json","readiness.v1.json","reuse_provenance.v1.json"):
        check(n+"_byte_identical",read_source(OLD+"/"+n)==read_source(NEW+"/"+n))
    basis=jfile(NEW+"/scientific_basis.v1.json");ready=jfile(NEW+"/readiness.v1.json")
    for key in ("common_producer_ref","native_producer_ref","author_path_helper_ref","plan_hook_ref","native_paths_config_ref","plan_config_ref"):
        auth(basis[key])
    # Execute only constant assignments, never imports or a live project module.
    ct=ast.parse(read_source(NEW+"/root_contract_v1.py").decode())
    ns=dict(Path=Path,ROOT=ROOT,require=require)
    wanted=("REL","SOURCE","DRIVER","WORKER","MANIFEST","SOURCE_PINS","READINESS","BASIS","CAPFILE","GUARD_REF","NATIVE_MANIFEST_REF","NATIVE_HELPER_REF","POLICY_PATH","POLICY_REF","ORIGINAL_EXPIRY_UTC","SLOT_IDENTITY_REF","SLOT_REL","REPAIR_REF","RENEWAL_REF","CERT_REF","MEMO_REF","NAMESPACE_REF","V4_REF","PIDFD","OWNED","PARENTS","STAGES","PARENT","CAPS")
    for node in ct.body:
        if isinstance(node,ast.Assign) and all(isinstance(t,ast.Name) for t in node.targets) and all(t.id in wanted or t.id in ("STAGE","PHASE") for t in node.targets):
            exec(compile(ast.Module(body=[node],type_ignores=[]),"<source_constants_only>","exec"),ns)
    check("future_stages_only",ns["STAGES"]==("native_paths","plan"))
    check("original_fixed_absolute_expiry",ns["ORIGINAL_EXPIRY_UTC"]=="2026-10-16T20:04:16.988073+00:00")
    slot=json.loads(auth(ns["SLOT_IDENTITY_REF"]))
    expected_slots=[ns["SLOT_REL"]+"/cpu_slot_"+str(i)+".lock" for i in range(2)]
    check("same_two_slot_receipt_without_resource_probe",slot["maximum_CPU_producers"]==2 and slot["same_original_pool"] and [r["ref"]["path"] for r in slot["slots"]]==expected_slots and all(r["st_size"]==0 and r["st_nlink"]==1 and r["file_kind"]=="regular" for r in slot["slots"]))
    check("slot_source_bodies_and_lifecycle_guards_unchanged",all(function_map(OLD+"/root_contract_v1.py")[n]==function_map(NEW+"/root_contract_v1.py")[n] for n in ("validate_slot_resources","source_paths","cpu_slot","original_guard","consumer_guard","bind_policy","direct_roles","unique","scientific_config","make_config")) and all(function_map(OLD+"/root_driver_v1.py")[n]==function_map(NEW+"/root_driver_v1.py")[n] for n in ("sample_scope","boundary","execute","close","validate_completion")))
    # Source metadata readers below are restricted to explicit frozen source JSON.
    source_docs={NEW+"/scientific_basis.v1.json":basis,NEW+"/readiness.v1.json":ready,NEW+"/caps.template.inactive.v1.json":caps,NEW+"/source_manifest.v1.json":manifest}
    for stage in ("native_paths","plan"):
        source_docs[NEW+"/"+stage+".scientific_config.template.inactive.v1.json"]=jfile(NEW+"/"+stage+".scientific_config.template.inactive.v1.json")
        source_docs[basis[stage+"_config_ref"]["path"]]=jfile(basis[stage+"_config_ref"]["path"])
    def sref(path):
        key=str(path.relative_to(ROOT)) if isinstance(path,Path) and path.is_absolute() else str(path)
        if key.startswith("INDEPENDENT_IN_MEMORY_FIXTURE/"):return fixture_ref(key.split("/",1)[1])
        # No endpoint can refer to a live slot or scientific payload here.
        assert not key.endswith(".lock")
        return actual_ref(key)
    def source_read(record):
        assert record["path"] in source_docs,record
        return copy.deepcopy(source_docs[record["path"]])
    ns.update(read=source_read,ref=sref,raw=lambda r:None,local=lambda path:ROOT/path)
    extract(NEW+"/root_contract_v1.py",{"select","predecessor_paths_v1","scientific_config","direct_roles","unique","actual_prior","observation"},ns)
    role_path="research/reviews/C3_saved600_artifact_acceptance_source_20261016_v1/source_manifest.v1.json"
    role_manifest_ref=actual_ref(role_path);roles=jfile(role_path)["current_guard_roles"]
    auth(ns["GUARD_REF"])
    check("complete_frozen_native14v3_role_count",len(roles)==37 and roles["native14_source"]==ns["GUARD_REF"] and roles["native14_manifest"]==ns["NATIVE_MANIFEST_REF"] and roles["native14_retained_measurement"]==ns["NATIVE_HELPER_REF"])
    ns["POLICY_REF"]=roles["native14_policy"]
    unions=[];mappings=[]
    for parent in ns["PARENTS"]:
        for stage in ns["STAGES"]:
            ns["select"](parent,stage)
            prior={k:fixture_ref(parent+"_"+stage+"_"+k) for k in ("completion_ref","wait_ref","outer_ref","close_ref","observation_ref")}
            q=dict(prior=prior,caps_ref=fixture_ref(parent+stage+"_caps"),source_review_ref=fixture_ref(parent+stage+"_review"))
            direct=ns["direct_roles"](fixture_ref(parent+stage+"_request"),q,fixture_ref(parent+stage+"_cfg"),fixture_ref(parent+stage+"_closure"),ns["RUNTIME_REF"])
            union=ns["unique"]([*roles.values(),*direct])
            assert len(union)==62 and all(r in union for r in [*roles.values(),*direct]) and not any(r["path"].endswith(".lock") for r in union)
            unions.append(dict(parent=parent,stage=stage,current_guard_roles=37,phase_direct_roles=len(direct),unique_union=62,limit=64,complete=True,bindings=union))
            previous,folder,output,key=ns["predecessor_paths_v1"](parent,stage)
            assert (stage=="native_paths" and folder==OLD+"/admitted_val"+parent[-3:]+"_infer_v3" and key.endswith("-infer-native14v3-v3")) or (stage=="plan" and folder==NEW+"/admitted_val"+parent[-3:]+"_native_paths_down8_v1" and key.endswith("-native_paths-native14v3-down8-v1"))
            mappings.append(dict(parent=parent,stage=stage,previous=previous,packet=folder,output=output,key=key))
    check("all_four_complete_role_unions_62_of64",len(unions)==4)
    rejects("role_hash_conflict_rejected",lambda:ns["unique"]([fixture_ref("same"),dict(path=fixture_ref("same")["path"],sha256="f"*64)]))
    rejects("role65_rejected",lambda:ns["unique"]([fixture_ref(str(i)) for i in range(65)]))
    rejects("infer_replay_route_rejected",lambda:ns["select"]("american_c3_val_003","infer"))
    rejects("other_parent_rejected",lambda:ns["predecessor_paths_v1"]("american_c3_val_002","native_paths"))
    cfg=ns["scientific_config"]("american_c3_val_003","native_paths",dict(observation_ref=fixture_ref("observation")),True)
    pn={};extract(basis["native_producer_ref"]["path"],{"storage_bound"},pn)
    analytic=pn["storage_bound"](cfg)
    check("full_native_scientific_counts_and16GB_preflight",cfg["output_cap_bytes"]==16000000000 and analytic==15218324396 and cfg["query_focal_count"]==10000 and cfg["max_unique_paths"]==31968234 and cfg["max_path_tokens"]==395840004 and cfg["max_path_occurrences"]==153726000 and cfg["max_vocabulary_nodes"]==133462 and cfg["max_raw_path_nodes"]==1024)
    check("all_full30Mb_R600_Q1000_registry_rows_unchanged",all(v["sequence_length_bp"]==30000000 and v["observed_haplotypes"]==1600 and v["reference_haplotypes"]==600 and v["query_haplotypes"]==1000 for v in ready["parents"].values()))
    # Genuine predecessor evidence joins: every document below is synthetic and never saved as a real prerequisite.
    gate_cases=[]
    for parent in ns["PARENTS"]:
        for stage in ns["STAGES"]:
            previous,folder,output,key=ns["predecessor_paths_v1"](parent,stage)
            def fref(path):
                p=str(path.relative_to(ROOT)) if isinstance(path,Path) and path.is_absolute() else str(path)
                return dict(path=p,sha256=hashlib.sha256(("in_memory_metadata:"+p).encode()).hexdigest())
            cp=fref(output+"/completed.json");wr=fref(folder+"/worker_aux/actual_wait.json");tr=fref(folder+"/outer_terminal.root_actual_v1.json");cr=fref(folder+"/closed/close_receipt.json")
            obs=fixture_ref(parent+stage+"_obs");simobs=fixture_ref(parent+"_simulation_obs");foc=fixture_ref(parent+"_new_down8_focal")
            complete=dict(status={"infer":"completed_current_C3_inference","native_paths":"completed_native_C3_path_preparation"}[previous],observation_ref=obs,parent_id=parent,query_focal_count=10000,query_ancestry_truth_opened=False,native_split_seed=42,native_train_fraction=.9,class_order=["EUR","EAS","AFR"],focals_ref=foc)
            wait=dict(completed_ref=cp,actual_direct_worker_wait=True,returncode=0,parent_id=parent,phase=previous+"_val_"+parent[-3:],input_ref=fref(folder+"/input.json"),all_owned_processes_and_groups_drained=True,operational_failure=None)
            outer=dict(root_authorizer="root_agent",actual_terminal_consumed=True,actual_exit_code=0,completed_ref=cp,direct_wait_ref=wr)
            close=dict(key=key,value=dict(future_writes_and_relaunch_revoked=True,proof=dict(completed_ref=cp,direct_wait_ref=wr)))
            observed=dict(parent_id=parent,split="val",sequence_length_bp=30000000,sample_unit="haploid",class_order=["EUR","EAS","AFR"],counts_haploid=dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000),query_ancestry_truth_in_inputs=False,genealogy_kind="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG",sites_and_sample_identities_verified=True,source_observation_ref=simobs)
            baseline={cp["path"]:complete,wr["path"]:wait,tr["path"]:outer,cr["path"]:close,obs["path"]:observed}
            docs=copy.deepcopy(baseline)
            def fake_read(record):
                if record["path"] not in docs:raise FileNotFoundError("Missing exact in-memory prerequisite")
                return copy.deepcopy(docs[record["path"]])
            gate_ns=dict(ns,ref=fref,read=fake_read,existing_simulation=lambda p:dict(simulation_observation_ref=simobs))
            extract(NEW+"/root_contract_v1.py",{"observation","actual_prior"},gate_ns)
            gate=gate_ns["actual_prior"]
            got=gate(parent,stage)
            assert got["completion_ref"]==cp and got["wait_ref"]==wr and got["outer_ref"]==tr and got["close_ref"]==cr
            if stage=="plan":
                built=ns["scientific_config"](parent,stage,got,True)
                assert got["focal_registry_ref"]==foc and built["native_preparation_ref"]==cp and built["focal_registry_ref"]==foc
            gate_cases.append(dict(parent=parent,stage=stage,baseline="PASS",completion_ref=cp,wait_ref=wr,outer_ref=tr,close_ref=cr,focal_ref=foc if stage=="plan" else None))
            mutants=[(wr,"actual_direct_worker_wait",False),(wr,"returncode",1),(wr,"completed_ref",fixture_ref("wrongcp")),(wr,"parent_id","wrong"),(wr,"phase","wrong"),(wr,"input_ref",fixture_ref("wronginput")),(wr,"all_owned_processes_and_groups_drained",False),(wr,"operational_failure","preserved_failure"),(tr,"root_authorizer","worker"),(tr,"actual_terminal_consumed",False),(tr,"actual_exit_code",1),(tr,"completed_ref",fixture_ref("wrongcp")),(tr,"direct_wait_ref",fixture_ref("wrongwait")),(cr,"key","old_v3_key"),(cp,"status","partial"),(obs,"parent_id","wrong"),(obs,"sequence_length_bp",29999999),(obs,"counts_haploid",dict(AFR=200,EUR=200,ASIA=200,ADMIX=999)),(obs,"query_ancestry_truth_in_inputs",True),(obs,"sites_and_sample_identities_verified",False),(obs,"source_observation_ref",fixture_ref("wrong_sim")),(obs,"genealogy_kind","simulated_true_ARG")]
            if stage=="plan":mutants += [(cp,"parent_id","wrong"),(cp,"query_focal_count",9999),(cp,"query_ancestry_truth_opened",True),(cp,"native_split_seed",43),(cp,"native_train_fraction",.8),(cp,"class_order",["AFR","EAS","EUR"])]
            for index,(record,field,bad) in enumerate(mutants):
                docs=copy.deepcopy(baseline);docs[record["path"]][field]=bad
                rejects(parent+"_"+stage+"_join_"+str(index)+"_"+field,lambda:gate(parent,stage))
            for field,bad in (("future_writes_and_relaunch_revoked",False),("proof",dict(completed_ref=fixture_ref("wrongcp"),direct_wait_ref=wr)),("proof",dict(completed_ref=cp,direct_wait_ref=fixture_ref("wrongwait")))):
                docs=copy.deepcopy(baseline);docs[cr["path"]]["value"][field]=bad
                rejects(parent+"_"+stage+"_close_"+field+"_"+str(len(results)),lambda:gate(parent,stage))
            docs={k.replace(NEW,OLD).replace("_down8_v1","_v3").replace("_native14v3_down8_v1","_native14v3_v3"):v for k,v in baseline.items()}
            if stage=="plan":rejects(parent+"_old_v3_only_namespace_no_fallback",lambda:gate(parent,stage))
    # Exact original native14 source guard with fake policy/state/measurement endpoints.
    R=[fixture_ref("guard_"+str(i)) for i in range(8)]
    own=dict(key="FAKE_KEY",output="FAKE_OUTPUT",auxiliary_scopes=["FAKE_AUX"],reserved_bytes=8000000000,consumer_bindings=R)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode="native14_stopped_retention_v3",expires_at_utc="FAKE_EXPIRY",job_id="FAKE_JOB",output_directory="FAKE_OUTPUT",reserved_new_bytes=8000000000,reservation=own,initial_ledger_pins=dict(base_pins={},completed_overlay_prefix={}),completed_overlay_policy_ref=R[3],budget_components=dict(implementation=R[0],launcher=R[1],consumer_adapter=R[2]))
    meter=dict(bytes=0,calls=[])
    policy=dict(implementation_ref=R[0],launcher_ref=R[1],source_ref=R[2],base_pins={})
    state=({own["key"]:own},set(),{},dict(required=R[7]),[],[],{own["key"]:1})
    v4=types.SimpleNamespace(integer=lambda v,n:require(type(v) is int and v>=0,n),canonical=lambda root,p:p,measure_scopes=lambda root,p:dict(conservative_scope_bytes=meter["bytes"]))
    gn=dict(copy=copy,require=require,MODE="native14_stopped_retention_v3",_expire=lambda x:None,_policy=lambda:policy,budget=types.SimpleNamespace(v4=v4),_POLICY_REF=R[3],_state_locked=lambda *a:state,assert_open=lambda key,r,c:require(key in r and key not in c,"open"),_raw=lambda x:None,check_prefix=lambda *a,**kw:None,_paths=lambda x:[x["output"],*x["auxiliary_scopes"]],ROOT="FAKE_ROOT")
    extract(ns["GUARD_REF"]["path"],{"check_storage_locked_v1","validate_storage_v9"},gn)
    def fake_budget(value,**kwargs):
        meter["calls"].append(kwargs)
        return gn["check_storage_locked_v1"](value,**kwargs)
    gn["budget_check_v9"]=fake_budget
    spec=dict(job_id="FAKE_JOB",output_directory="FAKE_OUTPUT",storage=storage)
    gn["validate_storage_v9"](spec,cfg,storage,source_ref=R[4],config_ref=R[5],runtime_ref=R[6])
    check("original_guard_accepts8GB_actual16GB_science",True)
    meter["bytes"]=7999999999
    rejects("actual_guard_prewrite8GB_overflow",lambda:fake_budget(storage,boundary=True,required_write_bytes=2))
    wrong=copy.deepcopy(storage);wrong["reserved_new_bytes"]=16000000000
    rejects("actual_guard_relabelled16GB_storage_rejected",lambda:fake_budget(wrong,boundary=True))
    rejects("actual_guard_write_without_boundary_rejected",lambda:fake_budget(storage,required_write_bytes=1))
    # Writer fake paths never touch disk. Blocking must occur before serialization/open.
    existing=set();events=[];written=[0]
    class FakePath:
        def __init__(self,key):self.key=key
        def __truediv__(self,name):return FakePath(self.key+"/"+name)
        @property
        def parent(self):return FakePath(self.key.rsplit("/",1)[0])
        def exists(self):return self.key in existing
        def mkdir(self,**kwargs):events.append("mkdir");existing.add(self.key)
        def stat(self):return types.SimpleNamespace(st_size=written[0])
        def open(self,*args,**kwargs):events.append("open");raise AssertionError("Blocked write reached serialization")
    wn=dict(require=require,local=lambda x:x if isinstance(x,FakePath) else FakePath(x),ref=lambda p:fixture_ref("finished"),json=json)
    extract(basis["common_producer_ref"]["path"],set(),wn,classes={"Writer"})
    Writer=wn["Writer"];adapter=types.SimpleNamespace(budget_check_v9=fake_budget)
    meter["bytes"]=8000000000-8191
    rejects("Writer_initial_directory_prewrite_cap",lambda:Writer(spec,adapter))
    assert not events
    writer=Writer.__new__(Writer);writer.storage=storage;writer.adapter=adapter;writer.path=FakePath("FAKE_OUTPUT")
    existing.add("FAKE_OUTPUT")
    rejects("Writer_nested_directory_prewrite_cap",lambda:writer.prepare("mlm/train/x.pkl",100))
    assert not events
    existing.add("FAKE_OUTPUT/mlm/train");meter["bytes"]=8000000000-8192
    fake_numpy=types.SimpleNamespace();saved_numpy=sys.modules.get("numpy")
    sys.modules["numpy"]=fake_numpy
    try:
        array=types.SimpleNamespace(dtype=types.SimpleNamespace(hasobject=False),nbytes=100)
        rejects("Writer_pickle_prewrite_cap",lambda:writer.pickle("mlm/train/x.pkl",{},1))
        rejects("Writer_json_prewrite_cap",lambda:writer.json("mlm/train/x.json",{}))
        rejects("Writer_npy_prewrite_cap",lambda:writer.npy("mlm/train/x.npy",array))
        rejects("Writer_npz_prewrite_cap",lambda:writer.npz("mlm/train/x.npz",a=array))
    finally:
        if saved_numpy is None:sys.modules.pop("numpy",None)
        else:sys.modules["numpy"]=saved_numpy
    assert not events
    written[0]=101
    rejects("Writer_serialization_bound_finish",lambda:writer.finish(FakePath("FAKE_OUTPUT/x.pkl"),100))
    written[0]=100;meter["bytes"]=8000000001
    rejects("Writer_whole_scope_finish",lambda:writer.finish(FakePath("FAKE_OUTPUT/x.pkl"),100))
    # Unchanged supervisor checks with fake walk/lstat. No real output/AUX reads.
    scope=dict(output=0,aux=0,scratch=0)
    class ScopePath:
        def __init__(self,key):self.key=key.key if isinstance(key,ScopePath) else str(key)
        def __truediv__(self,name):return ScopePath(self.key+"/"+name)
        def exists(self):return True
        def lstat(self):
            if self.key.endswith("data"):
                size=scope["scratch"] if "/tmp/" in self.key else (scope["output"] if self.key.startswith("OUT") else max(0,scope["aux"]-scope["scratch"]))
                return types.SimpleNamespace(st_mode=stat.S_IFREG,st_size=size,st_blocks=0,st_nlink=1)
            return types.SimpleNamespace(st_mode=stat.S_IFDIR,st_size=0,st_blocks=0,st_nlink=1)
    def fake_walk(p,followlinks=False):
        if p.key=="OUT":return iter([("OUT",[],["data"])])
        if p.key=="AUX":return iter([("AUX",["tmp"],["data"]),("AUX/tmp",[],["data"])])
        if p.key=="AUX/tmp":return iter([("AUX/tmp",[],["data"])])
        raise AssertionError(p.key)
    dn=dict(need=require,local=lambda x:ScopePath(x),Path=ScopePath,stat=stat,os=types.SimpleNamespace(walk=fake_walk),CAPS=caps["stages"]["native_paths"])
    extract(NEW+"/root_driver_v1.py",{"sample_scope"},dn)
    ss=dict(output_directory="OUT",worker_auxiliary_scope="AUX")
    scope.update(output=100,aux=100,scratch=50);dn["sample_scope"](ss)
    check("supervisor_bounded_scope_acceptance",True)
    scope.update(output=8000000000,aux=1,scratch=0)
    rejects("supervisor_whole8GB_cap",lambda:dn["sample_scope"](ss))
    scope.update(output=0,aux=167772161,scratch=0)
    rejects("supervisor_original_AUX_cap",lambda:dn["sample_scope"](ss))
    scope.update(output=0,aux=134217729,scratch=134217729)
    rejects("supervisor_original_private_work_cap",lambda:dn["sample_scope"](ss))
    # Admission prefix only: stop at fake mkdir so no authority/admission can occur.
    admit_node=next(n for n in ast.parse(read_source(NEW+"/root_driver_v1.py").decode()).body if isinstance(n,ast.FunctionDef) and n.name=="admit")
    prefix=copy.deepcopy(admit_node)
    idx=next(i for i,n in enumerate(prefix.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=="mkdir")
    prefix.body=prefix.body[:idx]
    prefix.body.append(ast.Return(value=ast.Constant(value="PREFIX_PASSED_NO_MUTATION")))
    ast.fix_missing_locations(prefix)
    ns["select"]("american_c3_val_003","native_paths")
    admission_state=[{},{"NEEDED_PREDECESSOR"},{},{},[],[],{}]
    fake_g=types.SimpleNamespace(budget=types.SimpleNamespace(SOURCE_REL="source",LAUNCHER_REL="launcher",v4=types.SimpleNamespace(mutex=lambda root:contextlib.nullcontext())),_state_locked=lambda *a,**kw:admission_state)
    class FreshPath:
        def __init__(self):self.present=False
        def exists(self):return self.present
    packet=FreshPath();out=FreshPath()
    fake_c=types.SimpleNamespace(PARENT="american_c3_val_003",STAGE="native_paths",PHASE="native_paths_val_003",validate_request=lambda r:(dict(prior=dict(closed_key="NEEDED_PREDECESSOR")),{},{}),phase_paths=lambda p:(packet,"FAKE_OUT"),original_guard=lambda:fake_g)
    an=dict(c=fake_c,need=require,local=lambda p:out,ref=lambda p:fixture_ref(p),ROOT="FAKE_ROOT")
    exec(compile(ast.Module(body=[prefix],type_ignores=[]),"<admission_prefix_only_no_mutation>","exec"),an)
    check("admission_fresh_closed_predecessor_prefix",an["admit"]({})=="PREFIX_PASSED_NO_MUTATION")
    admission_state[0]["H041/American-C3-val003-full30mb-native_paths-native14v3-v3"]={}
    rejects("already_reserved_or_closed_old_v3_stage_rejected",lambda:an["admit"]({}))
    admission_state[0]={};admission_state[1]=set()
    rejects("not_closed_predecessor_rejected",lambda:an["admit"]({}))
    admission_state[1]={"NEEDED_PREDECESSOR"};admission_state[0]["H041/American-C3-val003-full30mb-infer-native14v3-v2"]={}
    rejects("historical_partial_v2_key_rejected",lambda:an["admit"]({}))
    admission_state[0]={};packet.present=True
    rejects("packet_replay_rejected",lambda:an["admit"]({}))
    packet.present=False;out.present=True
    rejects("output_replay_rejected",lambda:an["admit"]({}))
    assert not any(k in sys.modules for k in ("torch","tskit","torch_geometric","numpy"))
    return dict(schema="VAL003004_downward8GB_independent_inert_verification_v1",status="PASS_SOURCE_ONLY",source_handoff_ref=actual_ref(NEW+"/source_handoff.v1.json"),source_manifest_ref=actual_ref(NEW+"/source_manifest.v1.json"),checks=results,check_count=len(results),authenticated_source_metadata_refs=authenticated,complete_role_union_evidence_source_ref=role_manifest_ref,role_unions=unions,stage_mappings=mappings,actual_prerequisites_checked=False,actual_slot_or_resource_probe=False,actual_payload_or_truth_opened=False,actual_guard_accounting_authority_admission_or_job=False,scientific_cap_bytes=16000000000,reservation_cap_bytes=8000000000,scientific_analytic_bound_bytes=analytic,fit_proven=False,finished_utc=datetime.now(timezone.utc).isoformat())
def save(name,value):
    with (D/name).open("x") as f:json.dump(value,f,sort_keys=True,indent=2);f.write("\n")
if __name__=="__main__":
    try:
        out=run();save("independent_verification.v1.json",out)
        print(json.dumps(dict(status=out["status"],checks=out["check_count"],role_unions=[dict(parent=r["parent"],stage=r["stage"],count=r["unique_union"]) for r in out["role_unions"]]),indent=2))
    except BaseException as error:
        save("independent_verification.attempt1.FAILURE.json",dict(error=repr(error),traceback=traceback.format_exc(),completed_checks=results))
        raise
