"""Actual inert modules and metadata contracts, real sealed sources, explicit RAM future docs.
Never calls a guard, admission, accounting, slot, probe, launch or scientific producer.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import contextlib,copy,hashlib,importlib.util,io,json,runpy,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/frozen_TRAIN001_019_preparation_pipeline_source_20261017_v2"
CONTROL=dict(path="research/reviews/frozen_TRAIN002_019_batch_continuation_source_20261017_v3/ROOT_batch_admission_root01_oct17/batch_input.json",
 sha256="dfd6ced48bf7ccfe8ebf9d58e3ffb2ab5193e8a586882d873e3bd4eb9279c092")
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
checks=[]
def check(ok,label):
    if not ok:raise AssertionError(label)
    checks.append(label)
def forbidden(*a,**k):raise AssertionError("Forbidden actual operation/probe in metadata-only route")
def actual_ref(path):
    p=Path(path);p=p if p.is_absolute()else ROOT/p
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def actual_read(record):
    check(actual_ref(record["path"])==record,"Real immutable metadata "+record["path"])
    return json.loads((ROOT/record["path"]).read_text())
def load(path,name):
    p=ROOT/path;sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
with contextlib.redirect_stdout(io.StringIO()):
    harness=runpy.run_path(str(ROOT/REL/"author_RAM_checks_v1.py"),run_name="__RAM_helpers__")
fixture=harness["fixture"];ram=harness["ram"];rr=harness["rr"]
manifest=json.loads((ROOT/REL/"source_manifest.v1.json").read_text())
control=actual_read(CONTROL);simulation_authority=actual_read(control["authority_ref"])
# Existing original baseline identities are immutable metadata, not current capacity evidence.
original=json.loads((ROOT/"research/reviews/frozen_benchmark_training_ready_20261017_v1/train001_simulate_v5/ROOT_actual_TRAIN001_root02_oct17_v5/input.json").read_text())
baseline=original["storage"]["reservation"]["consumer_bindings"][:40]
for index,(parent,stage)in enumerate([("american_c3_train_002","infer"),("american_c3_train_002","native_paths"),("american_c3_train_001","native_paths")]):
    cr,cap=fixture(stage,parent)
    if parent.endswith("001"):
        cap["metadata"]["outer_ref"]=dict(root_authorizer="root_agent",actual_terminal_consumed=True,actual_exit_code=0,
            input_ref=cap["refs"]["input_ref"],completed_ref=cap["refs"]["completion_ref"],actual_worker_wait_ref=cap["refs"]["wait_ref"],
            all_future_writes_permanently_revoked=True,output_frozen=True)
        ram[cap["refs"]["outer_ref"]["path"]]=copy.deepcopy(cap["metadata"]["outer_ref"]);ram[cr["path"]]=cap
    w=load(REL+"/worker_entry_v1.py","_actual_prep_metadata_"+str(index))
    real_wread=w.read;real_wref=w.metadata_ref
    def is_future(key):return key.startswith("ram/")or"RAM_METADATA_ONLY"in key
    def read(record):
        if is_future(record["path"])and record["path"]in ram:return copy.deepcopy(ram[record["path"]])
        return real_wread(record)
    def mref(path):
        p=Path(path);key=str(p.relative_to(ROOT))if p.is_absolute()else str(p)
        return rr(key)if is_future(key)and key in ram else real_wref(path)
    w.read=read;w.metadata_ref=mref;w.guard=forbidden
    now=datetime.now(timezone.utc);end=min(now+timedelta(hours=40),datetime.fromisoformat(simulation_authority["expires_at_utc"]))
    ar=rr("ram/route_"+str(index)+"/authority.json");reviewref=rr("ram/route_"+str(index)+"/review.json")
    authority=json.loads((ROOT/REL/"ROOT_authority.template.INACTIVE.v1.json").read_text())
    authority.update(active=True,source_only=False,root_authorizer="root_agent",created_at_utc=now.isoformat(),expires_at_utc=end.isoformat(),
        source_manifest_ref=actual_ref(REL+"/source_manifest.v1.json"),source_review_ref=reviewref,simulation_batch_control_input_ref=CONTROL)
    ram[ar["path"]]=authority
    ram[reviewref["path"]]=dict(independent=True,verdict="SOURCE_PASS",blockers=[],source_manifest_ref=actual_ref(REL+"/source_manifest.v1.json"),
        reviewed_sources=[actual_ref(REL+"/"+f)for f in("worker_entry_v1.py","batch_ROOT_v1.py","close_unit_ROOT_v1.py")],
        RAM_FIXTURE_NOT_A_REAL_REVIEW=True)
    cfg=w.scientific_config(parent,stage,cap["refs"]["observation_ref"]);caps=manifest["condition"]["stage_caps"][stage]
    qr=rr("ram/route_"+str(index)+"/request.json")
    q=dict(schema="TreeLoom_TRAIN_preparation_ROOT_request_v1",root_authorizer="root_agent",active=True,execution_authorized=True,GPU_authorized=False,
        parent_id=parent,stage=stage,phase=stage+"_"+parent.removeprefix("american_c3_"),created_at_utc=now.isoformat(),expires_at_utc=end.isoformat(),
        fresh_unfinished_stage=True,automatic_retry=False,source_manifest_ref=actual_ref(REL+"/source_manifest.v1.json"),
        storage_policy_selector_ref=manifest["close_pins"]["selector"],query_truth_access_authorized=False,GNN_optimization_authorized=False,
        TEST_access_authorized=False,current_Census_original_reserve_execute_authorized=True,pipeline_authority_ref=ar,
        batch_control_input_ref=CONTROL,source_review_ref=reviewref,caps=caps,cpu_ids=list(range(8)),scope_id="RAM_METADATA_ONLY_"+str(index),
        runtime_ref=manifest["condition"]["stage_runtime_refs"][stage],scientific_config=cfg,native_MLM_dependency=False,prior_capsule_ref=cr,
        bindings=dict(parent_id=parent,stage=stage,scientific_config=cfg,prior_capsule_ref=cr),
        packet_directory=REL+"/ROOT_actual_"+parent.removeprefix("american_c3_").upper()+"_"+stage+"_RAM_METADATA_ONLY_"+str(index),
        output_directory="experiments/results/frozen_TreeLoom_full30Mb_"+parent.removeprefix("american_c3_").upper()+"_"+stage+"_RAM_METADATA_ONLY_"+str(index),
        source_sha256={r["path"]:r["sha256"]for r in manifest["refs"]},
        actual_capacity=dict(created_at_utc=now.isoformat(),expires_at_utc=end.isoformat(),root_authorizer="root_agent",
            actual_current_observation=True,source_only_estimate=False,parent_id=parent,cpu_ids=list(range(8)),
            admissible_new_bytes=caps["reservation_bytes"],available_owned_RSS_bytes=caps["rss_bytes"]),
        RAM_FIXTURE_NOT_REAL_AUTHORITY_OR_CAPACITY=True)
    ram[qr["path"]]=q
    check(w.select_parent_from_record(qr)==parent,"Actual selector "+parent+"/"+stage)
    c,d=w.initialize();real_cread=c.read;real_cref=c.ref
    def cread(record,*a,**k):return copy.deepcopy(ram[record["path"]])if is_future(record["path"])and record["path"]in ram else real_cread(record,*a,**k)
    def cref(path):
        p=Path(path);key=str(p.relative_to(ROOT))if p.is_absolute()else str(p)
        return rr(key)if is_future(key)and key in ram else real_cref(path)
    c.read=cread;c.ref=cref;d.read=cread;d.ref=cref
    check(c.PARENT==parent and c.STAGE==stage and d.c is c,"Actual initializer "+parent+"/"+stage)
    check(len(w.source_paths())==19,"Actual all19 source authentication "+parent+"/"+stage)
    got,gotcaps,runtime=w.validate_request(qr)
    check(got==q and gotcaps==caps,"Actual whole request validator "+parent+"/"+stage)
    check(w._guard is None and w._memo is None,"No actual guard/cache instantiated "+parent+"/"+stage)
    expected=ROOT/(".runtime/envs/biological_tsinfer_v1"if stage=="infer"else".runtime/env")/"bin/python"
    check(d.worker_python(q)==str(expected),"Actual original stage interpreter "+parent+"/"+stage)
    # The future admitted input exists only in RAM. No project packet is created.
    packet=q["packet_directory"];ir=rr(packet+"/input.json");cfgref=rr(packet+"/config.json");storeref=rr(packet+"/storage.json")
    aux=packet+"/worker_aux";ram[cfgref["path"]]=cfg
    roles=w.direct_roles(qr,q,cfgref,qr,q["runtime_ref"])
    bindings=[];seen=set()
    for r in baseline+roles:
        key=json.dumps(r,sort_keys=True)
        if key not in seen:seen.add(key);bindings.append(r)
    check(len(bindings)==64,"Actual complete64binding contract "+parent+"/"+stage)
    own=dict(key="H041/RAM_ONLY_"+str(index),reserved_bytes=caps["reservation_bytes"],consumer_bindings=bindings,
        output=str(ROOT/q["output_directory"]),auxiliary_scopes=[str(ROOT/aux)])
    storage=dict(reservation=own,reserved_new_bytes=caps["reservation_bytes"],
        accounting_mode="Census_observation_two_scope_retention_v1",completed_overlay_policy_ref=manifest["close_pins"]["policy"],
        budget_components=dict(implementation=manifest["close_pins"]["implementation"],launcher=manifest["close_pins"]["launcher"],
            consumer_adapter=manifest["close_pins"]["guard"]))
    ram[storeref["path"]]=storage
    s=dict(schema="TreeLoom_TRAIN_preparation_admitted_input_v1",root_authorizer="root_agent",execution_authorized=True,
        root_stage_request_ref=qr,root_driver_ref=actual_ref(REL+"/worker_entry_v1.py"),main_source_ref=actual_ref(REL+"/worker_entry_v1.py"),
        parent_id=parent,stage=stage,phase=q["phase"],GPU_authorized=False,query_truth_access_authorized=False,TEST_access_authorized=False,
        GNN_optimization_authorized=False,storage=storage,output_directory=q["output_directory"],worker_auxiliary_scope=aux,
        runtime_ref=q["runtime_ref"],created_at_utc=now.isoformat(),expires_at_utc=end.isoformat(),caps_ref=qr,caps_locator=["caps"],
        source_sha256=q["source_sha256"],transitive_source_closure_ref=qr,refs=dict(config=cfgref,storage=storeref,runtime=q["runtime_ref"]))
    ram[ir["path"]]=s
    ss,qq,cc=d.input_spec(ir)
    check(ss==s and qq==q and cc==cfg,"Actual originaldriver→patchedcontract input route "+parent+"/"+stage)
    obsref=rr(q["output_directory"]+"/observed/manifest.json")if stage=="infer"else cap["refs"]["observation_ref"]
    obs=copy.deepcopy(cap["metadata"]["observation_ref"])
    if stage=="infer":
        obs.update(genealogy_kind="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG",
            sites_and_sample_identities_verified=True,source_observation_ref=cap["refs"]["observation_ref"],
            migrations_and_internal_population_labels_removed=True)
    ram[obsref["path"]]=obs
    done=dict(status=w.stage_status(stage),observation_ref=obsref)
    if stage=="infer":done.update(unary_preprocessing_policy=manifest["condition"]["stage_templates"]["infer"]["unary_preprocessing_policy"],tsinfer_num_threads=8)
    else:done.update(parent_id=parent,query_focal_count=10000,native_split_seed=42,native_train_fraction=.9,query_ancestry_truth_opened=False,
        contrastive_labels_reference_only=True,focals_ref=rr("ram/focals"))
    d.validate_completion(done,q);check(True,"Actual completion metadata validator "+parent+"/"+stage)
    # Actual close metadata join with synthetic recorded OS identities only. No OS query.
    cp=rr(q["output_directory"]+"/completed.json");wr=rr(aux+"/actual_wait.json");lr=rr(aux+"/launch.json")
    sw=rr("ram/route_"+str(index)+"/supervisor_wait.json");outerref=rr("ram/route_"+str(index)+"/outer.json")
    ram[cp["path"]]=done
    launch=dict(pid=1001,input_ref=ir,ownership=dict(signals_use_pidfd_only=True,no_numeric_group_signals=True,owned_identities=[dict(pid=1001,start=1)]))
    wait=dict(schema="American_C3_full_original_CPU_actual_wait_v1",parent_id=parent,phase=q["phase"],input_ref=ir,completed_ref=cp,pid=1001,
        returncode=0,actual_OS_wait_status=0,actual_direct_worker_wait=True,actual_direct_launcher_wait=True,
        all_owned_processes_and_groups_drained=True,cleanup_errors=[],operational_failure=None,adopted_waits=[],
        group_observations=[dict(pgid=1001,absent=True)],known_owned_process_groups=[1001],known_owned_pids=[1001],
        supervisor_pid=1002,supervisor_process_group=1002)
    ram[lr["path"]]=launch;ram[wr["path"]]=wait
    ram[sw["path"]]=dict(actual_direct_wait=True,all_owned_processes_and_groups_drained=True,actual_OS_wait_status=0,actual_returncode=0,pid=1002)
    ram[outerref["path"]]=dict(schema="ROOT_batch_unit_actual_OS_terminal_v1",root_authorizer="root_agent",
        actual_supervisor_OS_terminal_consumed=True,actual_exit_code=0,UI_tool_terminal_claimed=False,batch_authority_ref=ar,
        actual_worker_wait_ref=wr,actual_supervisor_wait_ref=sw,completed_ref=cp,input_ref=ir,
        all_owned_processes_and_groups_drained=True,new_ancestry_score=False)
    k=load(REL+"/close_unit_ROOT_v1.py","_actual_close_metadata_"+str(index));real_kread=k.read;real_kraw=k.raw;real_kref=k.ref
    k.read=lambda r:copy.deepcopy(ram[r["path"]])if is_future(r["path"])and r["path"]in ram else real_kread(r)
    k.ref=lambda p:cref(p)
    k.raw=lambda r:json.dumps(ram[r["path"]],sort_keys=True).encode()if is_future(r["path"])and r["path"]in ram else real_kraw(r)
    k.support();support=k.support_module;support.read=read;support.metadata_ref=mref;support.guard=forbidden
    pins=k.pins()
    p=dict(pins,parent_id=parent,stage=stage,producer_source=actual_ref(REL+"/worker_entry_v1.py"),
        producer_manifest=actual_ref(REL+"/source_manifest.v1.json"),producer_review=reviewref,
        batch_authority_ref=ar,batch_control_input_ref=CONTROL,registry_row=w.registered_row(parent),
        reservation_key=own["key"],reservation_sha256="0"*64,input=ir,launch=lr,wait=wr,completed=cp,outer=outerref,
        supervisor_wait=sw,storage=storeref,original_request=qr,config=cfgref)
    joined=k.metadata_join(p)
    check(joined[-2:]==([1001,1002],[1001,1002]),"Actual ordinaryclose complete metadata/PID join "+parent+"/"+stage)
    identity=k.request_value(p,own,launch,rr("ram/targeted_observation"))["completed_identity"]
    check(("parent_id"in identity)==(stage=="native_paths"),"Actual infer/native completion identity distinction "+parent+"/"+stage)
    check(w._guard is None and support._guard is None,"Actual whole metadata route leaves guards absent "+parent+"/"+stage)
# Actual imported modules include guard definitions nowhere. Reading pinned guard source for its hash is allowed.
check(not any(str(getattr(m,"__file__","")).endswith("census_observation_retention_v1.py")for m in list(sys.modules.values())),
    "No current guard module imported anywhere")
result=dict(schema="TreeLoom_actual_source_metadata_routes_v1",passed=True,check_count=len(checks),checks=checks,
    actual_modules_and_original_contract_entrypoints=True,real_sealed_source_selector_runtime_registry_live_batch_metadata=True,
    future_documents_and_recorded_PID_values_are_explicit_RAM_doubles=True,actual_guard_import_or_call=False,
    actual_resource_process_lock_GPU_probes=False,actual_accounting=False,scientific_execution=False,
    actual_authority_capacity_review_wait_or_admission_claimed=False)
print(json.dumps(result,sort_keys=True,indent=2))
