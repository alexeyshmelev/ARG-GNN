"""Narrow source-only downward-cap tests. Every resource endpoint is fake."""
from pathlib import Path
from unittest.mock import patch
import ast,copy,hashlib,importlib,json,stat,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OLD="research/reviews/American_C3_VAL003004_native14v3_preparation_source_20261016_v3"
HERE="research/reviews/American_C3_VAL003004_downward8GB_source_20261016_v1"
def require(value,message):
    if not value:raise ValueError(message)
def source_functions(path,names,namespace):
    tree=ast.parse((ROOT/path).read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"<unchanged_source_functions_fake_endpoints>","exec"),namespace)
    return namespace

def run():
    assert sys.dont_write_bytecode and sys.flags.no_user_site
    pkg=HERE.replace("/",".")
    c=importlib.import_module(pkg+".root_contract_v1")
    d=importlib.import_module(pkg+".root_driver_v1")
    w=importlib.import_module(pkg+".worker_entry_v1")
    pw=importlib.import_module(pkg+".plan_worker_v1")
    assert d.c is c and w.c is c and pw.c is c
    assert c.STAGES==("native_paths","plan")
    oldcaps=json.loads((ROOT/OLD/"caps.template.inactive.v1.json").read_text())
    caps=json.loads((ROOT/HERE/"caps.template.inactive.v1.json").read_text())
    expected=copy.deepcopy(oldcaps);expected["stages"]["native_paths"]["reservation_bytes"]=8000000000
    assert caps==expected
    for name in ("native_paths.scientific_config.template.inactive.v1.json","plan.scientific_config.template.inactive.v1.json",
        "scientific_basis.v1.json","readiness.v1.json","reuse_provenance.v1.json"):
        assert (ROOT/HERE/name).read_bytes()==(ROOT/OLD/name).read_bytes()
    assert c.SLOT_IDENTITY_REF["path"]==OLD+"/slot_resource_identity.v3.json"
    assert c.SLOT_REL=="research/reviews/American_C3_VAL003004_current_Census_preparation_source_20261016_v1"
    assert c.ORIGINAL_EXPIRY_UTC=="2026-10-16T20:04:16.988073+00:00"
    current_roles=json.loads((ROOT/"research/reviews/C3_saved600_artifact_acceptance_source_20261016_v1/source_manifest.v1.json").read_text())["current_guard_roles"]
    c.POLICY_REF=current_roles["native14_policy"]
    original_raw=c.raw
    def future(label):return dict(path="SYNTHETIC_FUTURE/"+label+".json",sha256="0"*64)
    def fake_future_raw(record):
        if record["path"].startswith("SYNTHETIC_FUTURE/"):return b"SYNTHETIC_SOURCE_ONLY"
        return original_raw(record)
    role_rows=[];mapping=[]
    for parent in c.PARENTS:
        for stage in c.STAGES:
            c.select(parent,stage)
            prior={key:future(parent+"_"+stage+"_"+key) for key in
                ("completion_ref","wait_ref","outer_ref","close_ref","observation_ref")}
            q=dict(caps_ref=future(parent+"_"+stage+"_caps"),source_review_ref=future(parent+"_"+stage+"_review"),prior=prior)
            direct=c.direct_roles(future(parent+"_"+stage+"_request"),q,future(parent+"_"+stage+"_config"),
                future(parent+"_"+stage+"_closure"),c.RUNTIME_REF)
            with patch.object(c,"raw",fake_future_raw):
                union=c.unique([*current_roles.values(),*direct])
            assert len(union)<=64 and all(r in union for r in [*current_roles.values(),*direct])
            assert not any(Path(r["path"]).suffix==".lock" for r in union)
            role_rows.append(dict(parent=parent,stage=stage,prospective_unique_union=len(union),
                all_declared_current_and_phase_roles_retained=True))
            previous,folder,output,key=c.predecessor_paths_v1(parent,stage)
            if stage=="native_paths":
                assert previous=="infer" and folder==OLD+"/admitted_val"+parent[-3:]+"_infer_v3"
                assert output.endswith("_infer_native14v3_v3") and key.endswith("-infer-native14v3-v3")
            else:
                assert previous=="native_paths" and folder==HERE+"/admitted_val"+parent[-3:]+"_native_paths_down8_v1"
                assert output.endswith("_native_paths_native14v3_down8_v1") and key.endswith("-native_paths-native14v3-down8-v1")
            mapping.append(dict(parent=parent,stage=stage,previous_stage=previous,packet=folder,output=output,closed_key=key))
    for parent,stage in (("american_c3_val_002","native_paths"),("american_c3_val_003","infer")):
        try:c.predecessor_paths_v1(parent,stage)
        except ValueError:pass
        else:raise AssertionError("Out-of-scope predecessor accepted")
    # Actual predecessor gate, new namespace only, with explicit fake documents.
    prerequisite_cases=[]
    for parent in c.PARENTS:
        previous,folder,output,key=c.predecessor_paths_v1(parent,"plan")
        def fake_ref(path):
            value=Path(path)
            rel=str(value.relative_to(ROOT)) if value.is_absolute() else str(value)
            return dict(path=rel,sha256=hashlib.sha256(("SYNTHETIC_ONLY:"+rel).encode()).hexdigest())
        cp=fake_ref(output+"/completed.json")
        wr=fake_ref(folder+"/worker_aux/actual_wait.json")
        tr=fake_ref(folder+"/outer_terminal.root_actual_v1.json")
        cr=fake_ref(folder+"/closed/close_receipt.json")
        obs=future(parent+"_NEW_DOWN8_INFERRED_OBSERVATION")
        focal=future(parent+"_NEW_DOWN8_FOCALS")
        original_observation=future(parent+"_ORIGINAL_SIMULATION_OBSERVATION")
        docs={
          cp["path"]:dict(status="completed_native_C3_path_preparation",observation_ref=obs,
            parent_id=parent,query_focal_count=10000,query_ancestry_truth_opened=False,
            native_split_seed=42,native_train_fraction=.9,class_order=["EUR","EAS","AFR"],focals_ref=focal),
          wr["path"]:dict(completed_ref=cp,actual_direct_worker_wait=True,returncode=0,parent_id=parent,
            phase="native_paths_val_"+parent[-3:],input_ref=fake_ref(folder+"/input.json"),
            all_owned_processes_and_groups_drained=True,operational_failure=None),
          tr["path"]:dict(root_authorizer="root_agent",actual_terminal_consumed=True,actual_exit_code=0,
            completed_ref=cp,direct_wait_ref=wr),
          cr["path"]:dict(key=key,value=dict(future_writes_and_relaunch_revoked=True,
            proof=dict(completed_ref=cp,direct_wait_ref=wr)))}
        def fake_read(record):
            if record["path"] not in docs:raise FileNotFoundError("Synthetic missing new prerequisite")
            return copy.deepcopy(docs[record["path"]])
        namespace=dict(c.actual_prior.__globals__,existing_simulation=lambda parent:dict(simulation_observation_ref=original_observation),
            local=lambda path:ROOT/path,ref=fake_ref,read=fake_read,
            observation=lambda record,parent:dict(genealogy_kind="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG",
                sites_and_sample_identities_verified=True,source_observation_ref=original_observation))
        actual_gate=types.FunctionType(c.actual_prior.__code__,namespace)
        prior=actual_gate(parent,"plan")
        assert prior["completion_ref"]==cp and prior["focal_registry_ref"]==focal and prior["closed_key"]==key
        plan_cfg=c.scientific_config(parent,"plan",prior,True)
        assert plan_cfg["native_preparation_ref"]==cp and plan_cfg["focal_registry_ref"]==focal
        prerequisite_cases.append(dict(parent=parent,name="actual_gate_new_down8_to_plan",status="PASS_INERT"))
        saved=docs[cr["path"]]["key"]
        docs[cr["path"]]["key"]="H041/American-C3-val"+parent[-3:]+"-full30mb-native_paths-native14v3-v3"
        try:actual_gate(parent,"plan")
        except ValueError:pass
        else:raise AssertionError("Obsolete v3 close accepted for new native output")
        docs[cr["path"]]["key"]=saved
        docs[tr["path"]]["actual_terminal_consumed"]=False
        try:actual_gate(parent,"plan")
        except ValueError:pass
        else:raise AssertionError("Unconsumed predecessor terminal accepted")
        docs[tr["path"]]["actual_terminal_consumed"]=True
        old_only={path.replace(HERE,OLD).replace("_down8_v1","_v3").replace("_native14v3_down8_v1","_native14v3_v3"):value
            for path,value in docs.items()}
        docs=old_only
        try:actual_gate(parent,"plan")
        except FileNotFoundError:pass
        else:raise AssertionError("Fallback to obsolete v3 native paths occurred")
        prerequisite_cases.append(dict(parent=parent,name="old_close_unconsumed_terminal_and_old_only_namespace_rejected",status="PASS_REJECTED_INERT"))
    # Actual unmodified scientific config builder retains all counts and16GB.
    c.select("american_c3_val_003","native_paths")
    cfg=c.scientific_config(c.PARENT,"native_paths",dict(observation_ref=future("same_parent_observation")),True)
    assert cfg["output_cap_bytes"]==16000000000 and c.CAPS["reservation_bytes"]==8000000000
    n=source_functions("src/arggnn/american_c3_paths_v1.py",{"storage_bound"},{})
    analytic=n["storage_bound"](cfg)
    assert analytic==15218324396 and analytic<=cfg["output_cap_bytes"] and analytic>8000000000
    map_planned=5*cfg["max_path_tokens"]+64*cfg["max_unique_paths"]+5*cfg["max_path_occurrences"]+16777216
    split_planned=5*cfg["max_path_tokens"]+32*cfg["max_unique_paths"]+64*cfg["max_vocabulary_nodes"]+16777216
    assert map_planned<8000000000 and split_planned<8000000000
    # Exact current-child bound/validation source bodies, synthetic state only.
    refs=[future(k) for k in ("implementation","launcher","guard","policy","producer","config","runtime","state_role")]
    own=dict(key="SYNTHETIC_KEY",output="SYNTHETIC_OUTPUT",auxiliary_scopes=["SYNTHETIC_AUX"],
        reserved_bytes=8000000000,consumer_bindings=refs)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
        accounting_mode="native14_stopped_retention_v3",expires_at_utc="SYNTHETIC_EXPIRY",
        job_id="SYNTHETIC_JOB",output_directory="SYNTHETIC_OUTPUT",reserved_new_bytes=8000000000,
        reservation=copy.deepcopy(own),initial_ledger_pins=dict(base_pins={"SYNTHETIC":True},
            completed_overlay_prefix=dict(events=1,tail_sha256="0"*64)),completed_overlay_policy_ref=refs[3],
        budget_components=dict(implementation=refs[0],launcher=refs[1],consumer_adapter=refs[2]))
    policy=dict(implementation_ref=refs[0],launcher_ref=refs[1],source_ref=refs[2],base_pins={"SYNTHETIC":True})
    meter=dict(charged=0,events=[])
    state=({own["key"]:own},set(),{},{"synthetic_required":refs[-1]},[],[],{own["key"]:1})
    def measure(root,paths):return dict(conservative_scope_bytes=meter["charged"])
    v4=types.SimpleNamespace(integer=lambda v,name:require(type(v) is int and v>=0,name),
        canonical=lambda root,path:path,measure_scopes=measure)
    ns=dict(copy=copy,require=require,MODE="native14_stopped_retention_v3",_expire=lambda value:None,
        _policy=lambda:policy,budget=types.SimpleNamespace(v4=v4),_POLICY_REF=refs[3],
        _state_locked=lambda *args:state,assert_open=lambda key,reservations,closed:require(key in reservations and key not in closed,"Synthetic open"),
        _raw=lambda record:None,check_prefix=lambda *args,**kwargs:None,_paths=lambda own:own["auxiliary_scopes"],ROOT="SYNTHETIC_ROOT")
    native_source="research/reviews/native14_stopped_retention_successor_source_20261016_v3/native14_stopped_retention_v1.py"
    source_functions(native_source,{"check_storage_locked_v1","validate_storage_v9"},ns)
    def budget_check(value,**kwargs):
        meter["events"].append(dict(endpoint="actual_guard_body_on_synthetic_state",**kwargs))
        return ns["check_storage_locked_v1"](value,**kwargs)
    ns["budget_check_v9"]=budget_check
    spec=dict(job_id=storage["job_id"],output_directory=storage["output_directory"],storage=storage)
    ns["validate_storage_v9"](spec,cfg,storage,source_ref=refs[4],config_ref=refs[5],runtime_ref=refs[6])
    wrong=copy.deepcopy(storage);wrong["reserved_new_bytes"]=16000000000
    try:budget_check(wrong,boundary=True)
    except ValueError:pass
    else:raise AssertionError("Mislabelled actual reservation accepted")
    # Actual unchanged Writer AST and methods. Fake paths never touch filesystem.
    class FakePath:
        def __init__(self,name):self.name=name
        def __truediv__(self,name):return FakePath(self.name+"/"+name)
        @property
        def parent(self):return FakePath(self.name.rsplit("/",1)[0])
        def exists(self):return self.name in existing
        def mkdir(self,**kwargs):meter["events"].append("FAKE_MKDIR");existing.add(self.name)
        def stat(self):return types.SimpleNamespace(st_size=fake_written[0])
        def open(self,*args,**kwargs):meter["events"].append("UNEXPECTED_FAKE_OPEN");raise AssertionError("No file writer should enter in blocked cases")
    source=ast.parse((ROOT/"src/arggnn/american_c3_common_v1.py").read_text())
    cls=next(node for node in source.body if isinstance(node,ast.ClassDef) and node.name=="Writer")
    wn=dict(require=require,local=lambda value:value if isinstance(value,FakePath) else FakePath(value),
        ref=lambda path:future("fake_finished"),json=json)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),"<unchanged_original_Writer_fake_paths>","exec"),wn)
    Writer=wn["Writer"];adapter=types.SimpleNamespace(budget_check_v9=budget_check)
    existing=set();fake_written=[0];meter["charged"]=8000000000-8191
    try:Writer(spec,adapter)
    except ValueError:pass
    else:raise AssertionError("Initial output creation exceeded8GB")
    assert "FAKE_MKDIR" not in meter["events"]
    existing={"SYNTHETIC_OUTPUT"};writer=Writer.__new__(Writer)
    writer.storage=storage;writer.adapter=adapter;writer.path=FakePath("SYNTHETIC_OUTPUT")
    meter["events"]=[];meter["charged"]=8000000000-8191
    try:writer.prepare("mlm/train/x.pkl",100)
    except ValueError:pass
    else:raise AssertionError("Parent creation exceeded8GB")
    assert "FAKE_MKDIR" not in meter["events"]
    existing.update(["SYNTHETIC_OUTPUT/mlm/train"])
    meter["events"]=[];meter["charged"]=8000000000-8192
    for kind in ("pickle","json","npy","npz"):
        fake_numpy=types.SimpleNamespace()
        array=types.SimpleNamespace(dtype=types.SimpleNamespace(hasobject=False),nbytes=100)
        with patch.dict(sys.modules,{"numpy":fake_numpy}):
            try:
                if kind=="pickle":writer.pickle("mlm/train/x.pkl",{},1)
                elif kind=="json":writer.json("mlm/train/x.json",{})
                elif kind=="npy":writer.npy("mlm/train/x.npy",array)
                else:writer.npz("mlm/train/x.npz",array=array)
            except ValueError:pass
            else:raise AssertionError("Boundary should block "+kind)
    assert "UNEXPECTED_FAKE_OPEN" not in meter["events"]
    fake_written[0]=101
    try:writer.finish(FakePath("SYNTHETIC_OUTPUT/x.pkl"),100)
    except ValueError:pass
    else:raise AssertionError("Serialization bound exceeded without rejection")
    # Exact supervisor full-scope/AUX/scratch checks against fake walk/lstat only.
    scope=dict(output=0,aux=0,scratch=0)
    class ScopePath:
        def __init__(self,path):self.path=str(path)
        def __truediv__(self,name):return ScopePath(self.path+"/"+name)
        def exists(self):return True
        def lstat(self):
            is_file=self.path.endswith("/data.bin")
            root=self.path.split("/")[0]
            size=scope["scratch"] if "/tmp/" in self.path else scope["output"] if root=="OUT" else scope["aux"]
            return types.SimpleNamespace(st_mode=stat.S_IFREG if is_file else stat.S_IFDIR,
                st_nlink=1,st_size=size if is_file else 0,st_blocks=0)
    def walk(path,followlinks=False):
        name=path.path
        return [(name,[],["data.bin"])]
    sample=types.FunctionType(d.sample_scope.__code__,dict(d.sample_scope.__globals__,
        local=lambda path:ScopePath(path),Path=ScopePath,os=types.SimpleNamespace(walk=walk),
        CAPS=caps["stages"]["native_paths"]))
    ss=dict(output_directory="OUT",worker_auxiliary_scope="AUX")
    scratch_cases=[]
    for name,values,allowed in [
        ("within_all_original_caps",dict(output=100,aux=100,scratch=100),True),
        ("whole8GB_overflow",dict(output=8000000000,aux=1,scratch=0),False),
        ("original_AUX_overflow",dict(output=0,aux=167772161,scratch=0),False),
        ("original_private_work_overflow",dict(output=0,aux=134217729,scratch=134217729),False)]:
        scope.update(values);accepted=False
        try:sample(ss);accepted=True
        except ValueError:pass
        assert accepted is allowed,name
        scratch_cases.append(dict(name=name,status="PASS" if accepted else "PASS_REJECTED"))
    oldtree=ast.parse((ROOT/OLD/"root_driver_v1.py").read_text());newtree=ast.parse((ROOT/HERE/"root_driver_v1.py").read_text())
    for name in ("sample_scope","boundary","execute","close","validate_completion","worker_python"):
        get=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(get(oldtree),include_attributes=False)==ast.dump(get(newtree),include_attributes=False),name
    for name in ("validate_slot_resources","cpu_slot","scientific_config","make_config","consumer_guard","original_guard","direct_roles"):
        old=next(n for n in ast.parse((ROOT/OLD/"root_contract_v1.py").read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        new=next(n for n in ast.parse((ROOT/HERE/"root_contract_v1.py").read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False),name
    assert not any(name in sys.modules for name in ("torch","numpy","tskit","torch_geometric"))
    return dict(schema="VAL003004_down8_reservation_only_inert_source_checks_v1",status="PASS_SOFTWARE_ONLY",
        role_unions=role_rows,next_stage_mapping=mapping,actual_prerequisite_gate_cases=prerequisite_cases,scientific_analytic_bound=analytic,
        unchanged_scientific_output_cap=16000000000,actual_accounting_cap=8000000000,
        fixed_count_maximum_single_mapping_planned_bytes=map_planned,
        fixed_count_maximum_single_sequence_pickle_planned_bytes=split_planned,
        actual_original_native14_guard_and_Writer_source_bodies_exercised_with_fakes=True,
        initial_directory_parent_directory_pickle_json_npy_npz_and_oversized_finish_rejections=True,
        original_full_scope_AUX_and_private_work_cases=scratch_cases,
        scientific_numerics_count_caps_slots_expiry_cleanup_and_guards_unchanged=True,
        actual_fit_or_transient_peak_measurement_claim=False,
        no_actual_payload_resource_probe_guard_accounting_authority_admission_or_job=True)
if __name__=="__main__":
    if sys.argv[1:]!=["--run"]:raise SystemExit("Explicit inert-only --run required")
    print(json.dumps(run(),sort_keys=True,indent=2))
