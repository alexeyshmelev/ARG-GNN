"""Whole actual FLARE metadata-validator path with RAM documents; no guards or operations."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import ast,copy,hashlib,json,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/FLARE_VAL001_native_seal_predecessor_bridge_source_20261017_v2"
OUT=ROOT/HERE
# Preserve and rerun the exact34 source/AST/RAM predecessor cases without its result writer.
old_check=(OUT/"source_check_v1.py").read_text()
prefix=old_check.split('result=dict(schema="FLARE_exact_v4_predecessor_bridge_source_check_v1"',1)[0]
assert prefix!=old_check
legacy={"__name__":"_RAM_predecessor_cases"}
exec(compile(prefix,"<same34_predecessor_cases>","exec"),legacy)
bridge_ns=legacy["ns"];bridge=types.SimpleNamespace(**bridge_ns)
manifest_raw=(OUT/"source_manifest.v2.json").read_bytes();manifest=json.loads(manifest_raw)
baseline=json.loads((ROOT/"research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v2/source_manifest.v1.json").read_text())
assert len(baseline["metadata_refs"])==148 and len(baseline["source_sha256"])==153
assert manifest["metadata_refs"]==baseline["metadata_refs"]
assert all(manifest["source_sha256"].get(p)==s for p,s in baseline["source_sha256"].items())
cases=list(legacy["cases"])
def passed(name):cases.append(dict(name=name,status="PASS"))
passed("Actual candidate manifest retains exact complete148metadata refs and153baseline source pins")
contract_text=(ROOT/bridge.CONTRACT_REF["path"]).read_text()
assert hashlib.sha256(contract_text.encode()).hexdigest()==bridge.CONTRACT_REF["sha256"]
engine_ref={"path":"research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v2/root_driver_v1.py","sha256":"c1d6a5bb835800f30d0c1e487c885ede13db253c18286b20bc115d8d70a7447f"}
engine_text=(ROOT/engine_ref["path"]).read_text()
assert hashlib.sha256(engine_text.encode()).hexdigest()==engine_ref["sha256"]
deps=json.loads((ROOT/"research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v1/dependencies.v1.json").read_text())
bindings=json.loads((OUT/"source_bindings.v1.json").read_text())
v4_manifest=json.loads((ROOT/bridge.V4_MANIFEST_REF["path"]).read_text())
basis=json.loads((ROOT/"research/reviews/FLARE_original_VAL001004_completion_source_20261017_v1/VAL000_basis.v1.json").read_text())
core_text=(ROOT/"research/reviews/FLARE_original_VAL001004_completion_source_20261017_v1/completion_v1.py").read_text()
class RamPath(PurePosixPath):
    def is_dir(self):return True
def fake_ref(label):
    return dict(path=label,sha256=hashlib.sha256(("RAM:"+label).encode()).hexdigest())
def key(r):return (r["path"],r["sha256"])
actual_refs={p:dict(path=p,sha256=s)for p,s in manifest["source_sha256"].items()}
actual_refs[HERE+"/source_manifest.v2.json"]=dict(path=HERE+"/source_manifest.v2.json",sha256=hashlib.sha256(manifest_raw).hexdigest())
for p in ("root_driver_v1.py","worker_entry_v1.py","source_handoff.v2.json"):
    q=OUT/p;actual_refs[HERE+"/"+p]=dict(path=HERE+"/"+p,sha256=hashlib.sha256(q.read_bytes()).hexdigest())
known={**manifest["source_sha256"],**{r["path"]:r["sha256"]for r in manifest["metadata_refs"]}}
for r in actual_refs.values():known[r["path"]]=r["sha256"]
for value in bindings["old_failure"].values():
    if isinstance(value,dict)and set(value)=={"path","sha256"}:known[value["path"]]=value["sha256"]
def function_nodes(text,names):
    tree=ast.parse(text)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names]
    assert {n.name for n in nodes}==set(names)
    return nodes
def fixture(stage="native"):
    records=legacy["base_records"]()
    records[key(bridge.INPUT_REF)]["source_sha256"]=v4_manifest["source_sha256"]
    raw_calls=[];current_calls=[];guard_calls=[]
    def read(r):return copy.deepcopy(records[key(r)])
    def raw(r):
        assert known.get(r["path"])==r["sha256"],("RAM source authentication failure",r)
        raw_calls.append(r["path"]);return b"RAM_authenticated_not_real_file_IO"
    def ref(p):
        if p in actual_refs:return copy.deepcopy(actual_refs[p])
        return fake_ref(p)
    local=lambda p:RamPath(str(ROOT))/p
    c=types.ModuleType("_RAM_exact_original_contract");c.__file__=str(ROOT/bridge.CONTRACT_REF["path"])
    c.__dict__.update(ROOT=ROOT,HERE=HERE,SOURCE=HERE+"/root_driver_v1.py",DRIVER=HERE+"/root_driver_v1.py",
        ENTRY=HERE+"/worker_entry_v1.py",MANIFEST=HERE+"/source_manifest.v2.json",HANDOFF=HERE+"/source_handoff.v2.json",
        require=bridge.require,read=read,raw=raw,ref=ref,local=local,current=lambda q,n=0:current_calls.append(n),
        datetime=datetime,timezone=timezone,copy=copy,hashlib=hashlib,json=json,
        DEPS=deps,STAGES=("convert","native","seal"),VAL=tuple("american_c3_val_%03d"%i for i in range(1,5)),
        STATUS={"convert":"completed_C3_FLARE_observation_conversion","native":"completed_C3_native_FLARE","seal":"sealed_C3_native_FLARE_calls_and_raw_posteriors"},
        PRODUCERS={"convert":"src/arggnn/american_c3_flare_convert_v1.py","native":"src/arggnn/american_c3_flare_run_v2.py","seal":"src/arggnn/american_c3_flare_evaluate_v1.py"},
        CAPS={},PHASE=None,PARENT=None,SELECTOR_REF=None,_CHOSEN=None,_ACTIVE=None)
    for n in ("SELECTOR","GUARD_REF","POLICY_REF","NAMESPACE","PIDFD","OWNED","BASE_RUNTIME","FRAMEWORK","WAIT_WRITER","COMPLETION","COMPLETION_REVIEW"):
        setattr(c,n,deps[n])
    names=("sources","reviewed","scientific_review","stage_caps","stage_evidence","make_config","validate_request","direct_roles","input_spec")
    exec(compile(ast.Module(body=function_nodes(contract_text,names),type_ignores=[]),str(ROOT/bridge.CONTRACT_REF["path"]),"exec"),c.__dict__)
    manifest_ref=ref(c.MANIFEST);records[key(manifest_ref)]=copy.deepcopy(manifest)
    records[key(bridge.V4_MANIFEST_REF)]=copy.deepcopy(v4_manifest)
    science_review=dict(status="SOURCE_PASS",source_ref=c.COMPLETION,source_manifest_ref=deps["COMPLETION_MANIFEST"],blocking_findings=[],execution_admission_or_truth_authority_granted=False)
    records[key(c.COMPLETION_REVIEW)]=science_review
    review_ref=fake_ref("RAM/source_review.json")
    records[key(review_ref)]=dict(independent=True,verdict="SOURCE_PASS",source_ref=ref(c.DRIVER),source_manifest_ref=manifest_ref,source_handoff_ref=ref(c.HANDOFF),blockers=[])
    context_ref=fake_ref("RAM/original_context.json");count_ref=fake_ref("RAM/count.json")
    target=json.loads((ROOT/"research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v2/ROOT_context_binding.INACTIVE.template.v1.json").read_text())["target_row"]
    target["count_manifest_ref"]=count_ref
    target["simulation_config_ref"]=deps["ORIGINAL_SIMULATION_CONFIGS"][bridge.PARENT]
    ctx={"targets":{bridge.PARENT:target}}
    records[key(count_ref)]=dict(schema="American_C3_full30Mb_observation_count_root_v1",truth_opened=False,source_observation_ref=target["observation_ref"],num_sites=227316,num_samples=1600,sequence_length=30000000,count_source_ref=target["plan_ref"])
    records[key(target["plan_ref"])]=dict(sites=227316,parent_id=bridge.PARENT,observation_ref=target["observation_ref"],focal_registry_ref=target["focal_registry_ref"],targets_opened=False)
    converted=dict(status="completed_C3_FLARE_observation_conversion",parent_id=bridge.PARENT,observation_ref=target["observation_ref"],focal_registry_ref=target["focal_registry_ref"],native_options=basis["configurations"]["native_config"]["native_options"],native_source_commit="ec0ad295aeebfacf10d8cbfb6486ab78981de1b1",input_sites=227316,query_ancestry_truth_opened=False,native_expected_markers=150000)
    records[key(bridge.COMPLETED_REF)]=converted
    core=types.ModuleType("_RAM_original_config_builder")
    core.__dict__.update(copy=copy,require=bridge.require,MISSING=("american_c3_val_001","american_c3_val_002","american_c3_val_003","american_c3_val_004"),
        OPTIONS=basis["configurations"]["native_config"]["native_options"],COMMIT="ec0ad295aeebfacf10d8cbfb6486ab78981de1b1",_sources=lambda:None,
        _context=lambda r,parent_id:copy.deepcopy(ctx),_local=lambda name:copy.deepcopy(basis),_tools=lambda:(read,None))
    exec(compile(ast.Module(body=function_nodes(core_text,("inactive_stage_configs_v1","_gzip_bound","_conversion_bound","_native_output_bound")),type_ignores=[]),"<actual_config_builder_RAM_only>","exec"),core.__dict__)
    c.completion=lambda:core
    protocol=types.ModuleType("_RAM_selector_protocol")
    chosen=dict(source_ref=bindings["selected_child_source_ref"],selector_source_ref=bindings["selected_child_source_ref"],policy_ref=legacy["ns"]["SELECTOR_REF"],accounting_mode="Relate_failed64GiB_retention_v1")
    # Correct actual policy identity is carried in the original static selector metadata.
    selector_value=json.loads((ROOT/bridge.SELECTOR_REF["path"]).read_text())
    chosen["policy_ref"]=selector_value["policy_ref"]
    def selected(record):
        bridge.require(record==bridge.SELECTOR_REF,"Exact RAM current selector");return copy.deepcopy(chosen)
    def bind_selector(record):
        value=selected(record)
        protocol.SELECTOR_REF=record;protocol._CHOSEN=value;protocol.SELECTOR=value["selector_source_ref"];protocol.GUARD_REF=value["source_ref"];protocol.POLICY_REF=value["policy_ref"]
        return value
    protocol.selected=selected;protocol.bind_selector=bind_selector
    protocol.guard=lambda *a,**kw:(_ for _ in ()).throw(AssertionError("No resource guard call in metadata tests"))
    d=types.ModuleType("_RAM_exact_v2_engine")
    d.__dict__.update(ROOT=ROOT,HERE=HERE,SOURCE=c.DRIVER,ENTRY=c.ENTRY,MANIFEST=c.MANIFEST,HANDOFF=c.HANDOFF,
        copy=copy,datetime=datetime,phase_budget=lambda s:dict(setup_seconds=7200,scientific_seconds=14400 if s=="native"else 3600,wall_seconds=15000 if s=="native"else 4200,combined_seconds=22200 if s=="native"else 11400,evidence_reserve_seconds=300,authority_margin_seconds=600),
        bindings=lambda:copy.deepcopy(bindings),load=lambda r,n:c if r==bindings["original_contract_ref"]else protocol)
    exec(compile(ast.Module(body=function_nodes(engine_text,("contract",)),type_ignores=[]),"<actual_v2_contract_factory_RAM_only>","exec"),d.__dict__)
    auth=types.SimpleNamespace()
    def authenticate(module,record,names):
        if module is auth:return {}
        assert module is c and record==bridge.CONTRACT_REF and names==("stage_evidence",)
        original=next(n for n in ast.parse(contract_text).body if isinstance(n,ast.FunctionDef)and n.name=="stage_evidence")
        return {"stage_evidence":(c.stage_evidence,original)}
    auth.functions_from_source=authenticate
    own_tree=ast.parse((OUT/"root_driver_v1.py").read_text())
    own_nodes=[n for n in own_tree.body if isinstance(n,ast.FunctionDef)and n.name=="engine"]
    own={"_ENGINE":None,"bridge":bridge,"sys":types.SimpleNamespace(dont_write_bytecode=True,flags=types.SimpleNamespace(no_user_site=True,optimize=0)),
        "HERE":HERE,"SOURCE":c.DRIVER,"ENTRY":c.ENTRY,"MANIFEST":c.MANIFEST,"HANDOFF":c.HANDOFF,
        "ENGINE_REF":engine_ref,"AUTH_REF":dict(path="research/reviews/ROOT_mutation_transaction_memo_source_20261017_v1/root_mutation_memo_v1.py",sha256="2923df9f5a48e76da0b59e6394f31d0b8f901c214a76cc62d65969d010655d98"),
        "_load_exact_source":lambda r,n:d if r==engine_ref else auth}
    exec(compile(ast.Module(body=own_nodes,type_ignores=[]),"<actual_new_engine_factory_RAM_only>","exec"),own)
    actual=own["engine"]().contract()
    prep="research/reviews/FLARE_remaining_ROOT_preparation_RAM_native_v2"
    packet="research/reviews/FLARE_remaining_ROOT_actual_RAM_native_v2"
    req_ref=fake_ref(prep+"/ROOT_request.json");registration_ref=fake_ref(prep+"/registration.json")
    runtime_ref=fake_ref(prep+"/runtime.json");directive_ref=fake_ref(prep+"/directive.json")
    failure_ref=fake_ref(prep+"/original_failure_acceptance.json")
    accepted=dict(bindings["old_failure"],schema="ROOT_FLARE_failed_unsealed_acceptance_v2",root_authorizer="root_agent",accepted=True,old_launch_resume_write_authority_revoked=True,old_files_scopes_key_charge_clocks_and_failure_unchanged=True,original_conversion_unsealed=True)
    records[key(failure_ref)]=accepted
    records[key(registration_ref)]=dict(schema="FLARE_root_target_registration_v1",root_authorizer="root_agent",parent_id=bridge.PARENT,observation_ref=target["observation_ref"],focal_registry_ref=target["focal_registry_ref"],count_manifest_ref=count_ref,scientific_role="registered_heldout_full30Mb_comparison",query_ancestry_truth_opened=False,score_authorized=False)
    records[key(runtime_ref)]=dict(prefix=str(ROOT/".runtime/env"),CPU_threads=4,cpu_ids=[0,1,2,3],CUDA_VISIBLE_DEVICES="",package_report_ref=deps["BASE_RUNTIME"])
    e_ref=legacy["E"]
    q=dict(schema="FLARE_remaining_CPU_ROOT_request_v1",active=True,root_authorizer="root_agent",execution_authorized=True,source_manifest_ref=manifest_ref,stage=stage,parent_id=bridge.PARENT,phase=stage+"_"+bridge.PARENT,GPU_authorized=False,CUDA_authorized=False,physical_gpu=None,remaining_VAL_truth_authorized=False,TEST_authorized=False,automatic_retry_resume_or_fallback=False,VAL000_rerun=False,source_review_ref=review_ref,storage_policy_selector_ref=bridge.SELECTOR_REF,preparation_directory=prep,packet_directory=packet,output_directory="data/FLARE_original_remaining_RAM_native_v2",cpu_ids=[0,1,2,3],registration_ref=registration_ref,context_ref=context_ref,conversion_ref=bridge.COMPLETED_REF,native_run_ref=None,previous_stage_evidence_ref=e_ref,runtime_ref=runtime_ref,root_directive_ref=directive_ref,created_at_utc="2026-10-17T00:00:00+00:00",expires_at_utc="2026-10-20T00:00:00+00:00",prior_failure_acceptance_ref=failure_ref)
    q["phase_budget"]=d.phase_budget(stage)
    if stage=="seal":
        native_ref=fake_ref("data/FLARE_original_remaining_RAM_completed_native/completed.json")
        native_e=fake_ref("RAM/native_stage_evidence.json");native_input=fake_ref("RAM/native_input.json");native_wait=fake_ref("RAM/native_wait.json");native_outer=fake_ref("RAM/native_outer.json");native_close=fake_ref("RAM/native_close.json")
        reserve=dict(key="RAM-native-key",output=str(ROOT/Path(native_ref["path"]).parent),reserved_bytes=123)
        records[key(native_ref)]=dict(parent_id=bridge.PARENT,focal_registry_ref=target["focal_registry_ref"])
        records[key(native_e)]=dict(schema="FLARE_remaining_stage_ROOT_terminal_evidence_v1",root_authorizer="root_agent",stage="native",parent_id=bridge.PARENT,completion_ref=native_ref,input_ref=native_input,actual_wait_ref=native_wait,outer_terminal_ref=native_outer,ordinary_close_receipt_ref=native_close)
        records[key(native_input)]=dict(schema="FLARE_remaining_CPU_admitted_input_v1",parent_id=bridge.PARENT,stage="native",root_driver_ref=ref(c.DRIVER),main_source_ref=ref(c.ENTRY),source_sha256=manifest["source_sha256"],output_directory=str(Path(native_ref["path"]).parent),storage={"reservation":reserve})
        records[key(native_wait)]=dict(actual_direct_worker_wait=True,returncode=0,all_owned_processes_and_groups_drained=True,operational_failure=None,input_ref=native_input,completed_ref=native_ref)
        records[key(native_outer)]=dict(actual_terminal_consumed=True,actual_exit_code=0,completed_ref=native_ref,direct_wait_ref=native_wait)
        records[key(native_close)]=dict(kind="concurrent_completed_scope_close_receipt_v1",key="RAM-native-key",artifacts_deleted=0,value=dict(future_writes_and_relaunch_revoked=True,request=dict(completed_ref=native_ref,direct_wait_ref=native_wait),reservation_sha256=hashlib.sha256(json.dumps(reserve,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()))
        q.update(native_run_ref=native_ref,previous_stage_evidence_ref=native_e)
    config=actual.make_config(q);q["caps"]=actual.stage_caps(stage,config)
    directive=dict(q,schema="FLARE_remaining_CPU_ROOT_directive_v1",directive="Execute one exact missing original FLARE stage after genuine fresh admission0")
    records[key(directive_ref)]=directive;records[key(req_ref)]=q
    return actual,records,req_ref,raw_calls,current_calls
c,records,req_ref,raw_calls,current_calls=fixture()
q,caps,_=c.validate_request(req_ref)
assert q["stage"]=="native"and len(raw_calls)>=len(manifest["source_sha256"])+148
passed("Actual new wrapper -> exact v2 contract -> original sources/reviewed/scientific_review/validate_request/stage_evidence/config builder native RAM path")
assert set(c.STAGES)=={"native","seal"}and c.SELECTOR_REF==bridge.SELECTOR_REF
passed("Actual wrapper current selector and native/seal-only factory gates run in RAM")
# Build genuine-shaped RAM admitted input and invoke actual input_spec and direct_roles.
sref=fake_ref(q["packet_directory"]+"/input.json");cfgref=fake_ref(q["preparation_directory"]+"/actual.config.json");sr=fake_ref(q["preparation_directory"]+"/storage.json");closure=fake_ref(q["preparation_directory"]+"/closure.json")
cfg=c.make_config(q);records[key(cfgref)]=cfg
roles=c.direct_roles(req_ref,q,cfgref,closure)
unique={key(r):r for r in roles};assert len(unique)<=64
storage=dict(reserved_new_bytes=caps["reservation_bytes"],reservation=dict(reserved_bytes=caps["reservation_bytes"],output=str(c.local(q["output_directory"])),auxiliary_scopes=[str(c.local(q["packet_directory"]))],consumer_bindings=list(unique.values())),output_directory=q["output_directory"],completed_overlay_policy_ref=c.POLICY_REF,storage_policy_selector_ref=c.SELECTOR_REF,job_id="RAM-job")
records[key(sr)]=storage;records[key(closure)]=dict(source_sha256=manifest["source_sha256"],root_stage_request_ref=req_ref)
s=dict(schema="FLARE_remaining_CPU_admitted_input_v1",active=True,root_authorizer="root_agent",execution_authorized=True,GPU_authorized=False,physical_gpu=None,root_stage_request_ref=req_ref,root_driver_ref=c.ref(c.DRIVER),main_source_ref=c.ref(c.ENTRY),worker_auxiliary_scope=q["packet_directory"]+"/worker_aux",storage=storage,source_sha256=manifest["source_sha256"],transitive_source_closure_ref=closure,refs=dict(config=cfgref,storage=sr),job_id="RAM-job")
for k in ("stage","phase","parent_id","output_directory","created_at_utc","expires_at_utc","runtime_ref"):s[k]=q[k]
records[key(sref)]=s
assert c.input_spec(sref)[0]==s
passed("Actual original input_spec/direct_roles/whole packet reservation and source closure metadata RAM path")
seal_c,seal_records,seal_req,_,_=fixture("seal")
assert seal_c.validate_request(seal_req)[0]["stage"]=="seal"
passed("Actual original seal request/native same-new-source predecessor RAM path")
for label,change in [
("missing metadata_refs",lambda c,r,q:r[key(c.ref(c.MANIFEST))].pop("metadata_refs")),
("wrong independent review manifest",lambda c,r,q:r[key(r[key(q)]["source_review_ref"])].update(source_manifest_ref=fake_ref("RAM/wrongmanifest"))),
("source authentication mismatch",lambda c,r,q:r[key(c.ref(c.MANIFEST))]["source_sha256"].update({"RAM/unregistered.py":"0"*64})),
("old selector",lambda c,r,q:r[key(q)].update(storage_policy_selector_ref=fake_ref("RAM/oldselector"))),
("reconversion",lambda c,r,q:r[key(q)].update(stage="convert")),
("different conversion",lambda c,r,q:r[key(q)].update(conversion_ref=fake_ref("RAM/otherconversion"))),
("missing close acceptance",lambda c,r,q:r[key(legacy["ACCEPT"])].update(accepted=False)),
("unconsumed close outer",lambda c,r,q:r[key(legacy["CLOSE_OUTER"])].update(actual_terminal_consumed=False)),
("unsupported budget",lambda c,r,q:r[key(q)]["phase_budget"].update(scientific_seconds=1)),
("missing full-site count",lambda c,r,q:r[key(fake_ref("RAM/count.json"))].update(num_sites=227315))]:
    actual,docs,req,_,_=fixture()
    change(actual,docs,req)
    try:actual.validate_request(req)
    except (ValueError,KeyError,AssertionError):passed("Whole metadata path rejects "+label)
    else:raise AssertionError("Whole metadata validator accepted "+label)
result=dict(schema="FLARE_predecessor_bridge_v2_actual_metadata_RAM_check",status="PASS_SOURCE_AST_AND_WHOLE_METADATA_RAM_ONLY",case_count=len(cases),cases=cases,actual_candidate_manifest_ref=actual_refs[HERE+"/source_manifest.v2.json"],preserved_original_metadata_refs=148,preserved_original_source_pins=153,actual_resource_guards_imported_or_executed=False,original_metadata_functions_executed_only_with_RAM_documents=True,source_loaders_guards_selector_runtime_filesystem_and_current_callbacks_replaced_by_explicit_RAM_doubles=True,real_wait_completion_checkpoint_truth_or_process_read=False,admission_launch_mutation_or_Git=False,independent_review=False)
p=OUT/"integration_check.result.v2.json"
if p.exists():raise ValueError("Immutable integration result exists")
p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print(json.dumps(dict(status=result["status"],case_count=len(cases),result_ref=dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))))
