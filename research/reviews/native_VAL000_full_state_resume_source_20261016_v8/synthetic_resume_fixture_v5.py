"""Explicit fake-only source/API fixture. No project scientific imports or devices."""
from pathlib import Path
from datetime import timedelta
import ast,builtins,copy,hashlib,inspect,json,types,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/native_VAL000_full_state_resume_source_20261016_v8"

def run():
    def require(value,message):
        if not value:raise ValueError(message)
    events=[]
    direct=ast.parse((ROOT/HERE/"direct_gpu0_v1.py").read_text())
    defs=[n for n in direct.body if isinstance(n,ast.FunctionDef) and n.name in ("bounded_capacity_sample_v5","capacity_v1")]
    identity=lambda pid:dict(pid=pid,uid=9,start_ticks=100+pid,state="R")
    gpu=dict(index=0,uuid="GPU_FAKE_ONLY",total_mib=81920,used_mib=0,compute=[])
    registry=dict(process_identities=[identity(2)])
    ns=dict(require=require,time=types.SimpleNamespace(time=lambda:1),os=types.SimpleNamespace(getuid=lambda:9),json=json,
        GPU0_UUID="GPU_FAKE_ONLY",CURRENT_CAP=dict(path="SYNTHETIC_ONLY",sha256="0"*64),LAST_CONTROLLER_LIVENESS=None,
        CAPACITY_EVIDENCE_SINK=lambda value:events.append(("persist_before_exception",copy.deepcopy(value))),
        controller_v1=lambda q,base:(None,dict(updated_unix=1),registry,None))
    exec(compile(ast.Module(body=defs,type_ignores=[]),"<actual-capacity-functions-with-fake-inventory-only>","exec"),ns)
    base=types.SimpleNamespace(real_inventory=lambda:[copy.deepcopy(gpu)],proc_identity=identity)
    q=dict(caps=dict(max_cuda_reserved_bytes=32*2**30))
    def set_external(amount):
        gpu["compute"]=[dict(pid=2,name="AI Auto Research Pipeline Test",used_mib=5304),
            dict(pid=1,name="AI Auto Research Pipeline Test",used_mib=10000),
            dict(pid=3,name="FAKE_EXTERNAL",used_mib=amount)]
        gpu["used_mib"]=5304+10000+amount
    set_external(16384)
    accepted=ns["capacity_v1"](q,base,worker_pid=1)
    assert accepted["projected_mib"]==81920 and accepted["capacity_fit"] is True and not events
    set_external(16385)
    rejected=False
    try:ns["capacity_v1"](q,base,worker_pid=1)
    except ValueError:rejected=True
    assert rejected and len(events)==1 and events[-1][1]["projected_mib"]==81921
    assert events[-1][1]["before_any_supervisor_stop_signal"] is True
    assert events[-1][1]["exact_owned_process_roster_known"] is False
    gpu["uuid"]="WRONG_FAKE_GPU"
    try:ns["capacity_v1"](q,base,worker_pid=1)
    except ValueError:pass
    assert len(events)==2 and events[-1][1]["GPU0"] is None and events[-1][1]["projected_mib"] is None
    many=[dict(pid=i,used_mib=1,name="x"*1000,classification="FAKE") for i in range(500)]
    bounded=ns["bounded_capacity_sample_v5"](dict(GPU0=dict(compute=many),external_compute=many,classification=many))
    assert bounded["GPU0"]["actual_compute_row_count"]==500 and bounded["GPU0"]["compute_rows_truncated"] is True
    assert len((json.dumps(bounded,sort_keys=True,indent=2)+"\n").encode())<=65536
    # Exact original native RLIMIT API, entirely fake resource module.
    original=ast.parse((ROOT/"research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/native_child_v1.py").read_text())
    run_native=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=="run_v1")
    calls=[n for n in ast.walk(run_native) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="setrlimit"]
    assert len(calls)==1
    limits=[]
    r=types.SimpleNamespace(RLIMIT_FSIZE="FAKE_LIMIT",setrlimit=lambda key,pair:limits.append((key,pair)))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=calls[0])],type_ignores=[])),"<actual-native-hard-RLIMIT-call-with-fake-resource>","exec"),
        dict(resource=r,cfg=dict(max_checkpoint_bytes=2*2**30)))
    assert limits==[("FAKE_LIMIT",(2*2**30,2*2**30))]
    # Full-state Composer downloader callback and restoration checks use fake
    # modules/data only. The installed original loader code is never imported.
    bridge=ast.parse((ROOT/HERE/"resume_scientific_bridge_v5.py").read_text())
    session_ast=next(n for n in bridge.body if isinstance(n,ast.ClassDef) and n.name=="ResumeSession")
    saved=dict(epoch=0,batch=6500,batch_in_epoch=6500,optimizer_unique_steps=[6500])
    loader_events=[]
    def download0(*args,**kwargs):raise AssertionError("Original fake download must be privately substituted")
    loader_ns=dict(download_checkpoint=download0,loader_events=loader_events)
    loader_source="""
def load_checkpoint(path,state,logger,object_store=None,load_weights_only=False,strict_model_weights=True,progress_bar=True,ignore_keys=None,exclude_algorithms=None,algorithm_passes=None):
    loader_events.append("actual_fake_original_code_entered")
    result=download_checkpoint(path,"FAKE_TMP",object_store,progress_bar,False,False)
    assert result==(path,None,False)
    return [{"FAKE_rng":1}]
"""
    exec(loader_source,loader_ns)
    checkpoint=types.SimpleNamespace(load_checkpoint=loader_ns["load_checkpoint"])
    utils=types.SimpleNamespace(checkpoint=checkpoint)
    def fake_import(name,globals=None,locals=None,fromlist=(),level=0):
        if name=="composer.utils":return utils
        return builtins.__import__(name,globals,locals,fromlist,level)
    c=types.SimpleNamespace(require=require,RESUME_STATE=dict(actual_saved_progress=saved,checkpoint_ref=dict(path="FAKE",sha256="0"*64)),
        ref=lambda name:dict(path="FAKE_source",sha256="1"*64))
    fake_native=types.SimpleNamespace(optimizer_counter_evidence=lambda state:dict(unique_step_values=[next(iter(state.values()))["step"]]))
    bn=dict(c=c,Path=Path,types=types,copy=copy,ast=ast,os=None,stat=None,hashlib=hashlib,timedelta=timedelta,SOURCE="FAKE_source",
        __builtins__=dict(vars(builtins),__import__=fake_import))
    exec(compile(ast.Module(body=[session_ast],type_ignores=[]),"<actual-session-class-with-fake-Composer-module>","exec"),bn)
    session=bn["ResumeSession"](fake_native,{})
    session.path="FAKE_READONLY_FD";session.stable=lambda:loader_events.append("fake_stable_identity_check")
    session.install_original_Composer_FD_loader()
    class FAKE_OPT:
        def state_dict(self):return dict(step=6500)
    state=types.SimpleNamespace(timestamp=types.SimpleNamespace(**{k:types.SimpleNamespace(value=saved[k]) for k in ("epoch","batch","batch_in_epoch")}),
        optimizers=[FAKE_OPT()],schedulers=[dict(FAKE=1)])
    result=checkpoint.load_checkpoint(session.path,state,None,load_weights_only=False)
    assert result==[dict(FAKE_rng=1)] and session.restore_receipt["load_weights_only"] is False
    assert "actual_fake_original_code_entered" in loader_events
    for change in [dict(load_weights_only=True),dict(ignore_keys=["rng"]),dict(strict_model_weights=False)]:
        rejected=False
        try:checkpoint.load_checkpoint(session.path,state,None,**change)
        except ValueError:rejected=True
        assert rejected
    state.optimizers[0].state_dict=lambda:dict(step=2)
    rejected=False
    try:checkpoint.load_checkpoint(session.path,state,None)
    except ValueError:rejected=True
    assert rejected
    # Native original full-run code is cloned without a modified science body.
    text=(ROOT/HERE/"resume_scientific_bridge_v5.py").read_text()
    assert "types.FunctionType(native.run_v1.__code__" in text
    assert 'cfg.load_path=self.path;cfg.load_weights_only=False;cfg.autoresume=False' in text
    # Exact nested timestamp schema accepts genuine format, rejects flat/partial
    # formats. All numbers here are explicitly synthetic.
    defs=[n for n in bridge.body if isinstance(n,ast.FunctionDef) and n.name=="validated_composer_timestamp_v1"]
    tn=dict(c=c,timedelta=timedelta)
    exec(compile(ast.Module(body=defs,type_ignores=[]),"<actual-timestamp-validator-with-fakes>","exec"),tn)
    integers={k:0 for k in ("iteration","epoch","batch","sample","token","epoch_in_iteration","token_in_iteration","batch_in_epoch","sample_in_epoch","token_in_epoch")}
    timestamp=dict(integers,**{k:timedelta() for k in ("total_wct","iteration_wct","epoch_wct","batch_wct")})
    assert tn["validated_composer_timestamp_v1"](dict(timestamp=dict(Timestamp=timestamp)))["batch"]==0
    for wrong in [timestamp,dict(Timestamp=dict(timestamp,batch=True))]:
        rejected=False
        try:tn["validated_composer_timestamp_v1"](dict(timestamp=wrong))
        except ValueError:rejected=True
        assert rejected
    # Actual ROOT prerequisite API with fake sealed receipts and authority.
    contract=ast.parse((ROOT/HERE/"amendment_contract_v1.py").read_text())
    constants={}
    for n in contract.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            try:constants[n.targets[0].id]=ast.literal_eval(n.value)
            except (ValueError,TypeError):pass
    records={}
    def rr(name):return dict(path="SYNTHETIC_ONLY/"+name,sha256=hashlib.sha256(name.encode()).hexdigest())
    def put(name,value):
        record=rr(name);records[record["path"]]=copy.deepcopy(value);return record
    qroot=dict(resume_policy_ref=constants["RESUME_POLICY_REF"],
        failed_v4_and_all_prior_launch_capabilities_revoked=True,original_checkpoint_and_all_failed_artifacts_preserved=True)
    records[constants["RESUME_POLICY_REF"]["path"]]=dict(original_whole_scope_bytes=7*2**30)
    qroot["failed_v4_stdout_observation_ref"]=put("failed_v4",dict(adapter_manifest_ref=constants["V4_MANIFEST_REF"],
        returncode=-15,actual_OS_wait_status=15,actual_direct_worker_wait=True,all_owned_processes_and_groups_drained=True,
        complete_owned_roster_known=True,completed_ref=None,
        operational_failure="ValueError: Actual external+all unfilled C3/native32 envelopes+4GiB headroom fitGPU0",
        known_identity_count=64,adopted_wait_count=41,adopted_waits_truncated=False,
        full_original_stdout_or_identity_array_not_reconstructed=True))
    qroot["failed_v4_outer_terminal_ref"]=put("outer",dict(actual_tool_session_id=24814,actual_exit_code=0,
        actual_terminal_consumed=True,actual_direct_worker_returncode=-15,all_owned_processes_and_groups_drained=True,
        actual_stdout_observation_ref=qroot["failed_v4_stdout_observation_ref"]))
    projected=records.pop(qroot["failed_v4_stdout_observation_ref"]["path"])
    outer=records.pop(qroot["failed_v4_outer_terminal_ref"]["path"])
    qroot["failed_v4_stdout_observation_ref"]=constants["V4_STDOUT_PROJECTION_REF"]
    qroot["failed_v4_outer_terminal_ref"]=constants["V4_OUTER_PROJECTION_REF"]
    outer["actual_stdout_observation_ref"]=qroot["failed_v4_stdout_observation_ref"]
    records[qroot["failed_v4_stdout_observation_ref"]["path"]]=projected
    records[qroot["failed_v4_outer_terminal_ref"]["path"]]=outer
    records[constants["V4_LAUNCH_REF"]["path"]]=dict(pid=999,parent_pid=1000,ownership=dict(owned_identities=[dict(pid=999,start=123)]))
    records[constants["V4_CUDA_OBSERVATION_REF"]["path"]]=dict(worker_identity_before=dict(pid=999,start_ticks=123),worker_identity_after=dict(pid=999,start_ticks=123))
    records[constants["V4_PRIOR_TARGET_REGISTRY_REF"]["path"]]=dict(known_old_targeted_absence=[dict(pid=1001,absent=True),dict(pgid=999,absent=True)])
    for name in ("PRIOR_WAIT","LATEST_METADATA_WAIT","LATEST_ROSTER_WAIT"):
        records[constants[name]["path"]]=dict(ownership=dict(retained_owned_identities=[]),group_observations=[],adopted_waits=[])
    qroot["resume_checkpoint_ref"]=constants["RESUME_CHECKPOINT_REF"]
    records[constants["ROOT_PRESERVED_METADATA_REF"]["path"]]=dict(checkpoint_ref=constants["RESUME_CHECKPOINT_REF"],
        actual_verified_optimizer_updates=6500,full_byte_read_and_sha256=True,pre_post_file_and_FD_identity_equal=True,full_budget_complete=False)
    qualify=dict(schema="ROOT_native_VAL000_full_state_resume_qualification_v5",root_authorizer="root_agent",status="qualified",
        checkpoint_ref=qroot["resume_checkpoint_ref"],checkpoint_full_SHA_via_own_readonly_FD_verified=True,
        genuine_full_state_model_optimizer_scheduler_RNG_timestamp_verified=True,
        whole_original7GiB_and6GiB_checkpoint_capacity_qualified=True,current_v4_complete64_identity_and_group_drain_verified=True,
        fresh_targeted_known_old_root_PID_start_and_group_absence_verified=True,all_cooperative_old_capabilities_permanently_revoked=True,
        historical_v3_descendant_coverage="UNKNOWN_TRUNCATED64",historical_v3_unknown_descendants_absent=None,
        private_FD_no_Composer_TMP_symlink_or_copy_route_reviewed=True,actual_source_manifest_ref=rr("manifest"),
        actual_ROOT_verifier_receipt_ref=constants["ROOT_PRESERVED_METADATA_REF"],whole_parent_capacity_observation_ref=rr("census"),
        checkpoint_identity=dict(bytes=2017931910,link_count=1),checkpoint_charge_bytes=2017932288,
        actual_saved_progress=dict(epoch=0,batch=6500,batch_in_epoch=6500,optimizer_unique_steps=[6500]))
    known=put("known_registry",dict(root_authorizer="root_agent",
        known_old_root_source_refs=[constants[key] for key in ("PRIOR_WAIT","LATEST_METADATA_WAIT","LATEST_ROSTER_WAIT","V4_STDOUT_PROJECTION_REF","V4_LAUNCH_REF","V4_CUDA_OBSERVATION_REF","V4_PRIOR_TARGET_REGISTRY_REF")],
        all_available_known_old_root_identities_from_listed_sources_and_ROOT_launchers_included=True,
        unknown_historical_descendants_claimed_listed=False,identities=[dict(pid=999+i,start_ticks=123+i) for i in range(64)],process_groups=[999]))
    stamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    pidobs=put("pid_absence",dict(root_authorizer="root_agent",known_identity_registry_ref=known,unknown_history_absence_asserted=False,
        observations=[dict(pid=999+i,start_ticks=123+i,known_identity_absent=True,numeric_PID_absent=True,method="targeted PID/start or numeric ESRCH",checked_utc=stamp) for i in range(64)]))
    groupobs=put("group_absence",dict(root_authorizer="root_agent",known_identity_registry_ref=known,unknown_history_absence_asserted=False,
        observations=[dict(pgid=999,absent=True,method="targeted os.killpg(pgid,0) ESRCH",checked_utc=stamp)]))
    revoke=put("cooperative_revoke",dict(schema="ROOT_native_VAL000_permanent_cooperative_old_capability_revocation_v6",
        root_authorizer="root_agent",source_manifest_ref=rr("manifest"),all_source_referenced_old_launch_capabilities_permanently_revoked=True,
        old_original_resume_entrypoints_and_automatic_retries_revoked=True,new_exception_is_exact_reviewed_v8_ROOT_amendment_only=True,
        historical_files_or_failures_mutated=False,unknown_old_processes_absent_asserted=False))
    owner=dict(schema="ROOT_native_VAL000_known_owner_observation_bundle_v6",root_authorizer="root_agent",
        source_manifest_ref=rr("manifest"),failed_v4_stdout_observation_ref=qroot["failed_v4_stdout_observation_ref"],
        failed_v4_outer_terminal_ref=qroot["failed_v4_outer_terminal_ref"],current_v4_complete_roster_identity_count=64,
        current_v4_complete_identity_and_group_drain_verified=True,historical_v3_descendant_coverage="UNKNOWN_TRUNCATED64",
        historical_v3_unknown_descendants_absent=None,historical_v3_full_ownership_reconstructed=False,
        known_identity_registry_ref=known,fresh_known_PID_start_absence_ref=pidobs,fresh_known_process_group_absence_ref=groupobs,
        permanent_cooperative_revocation_ref=revoke)
    qroot["ROOT_known_owner_observation_bundle_ref"]=put("owner_scope",owner)
    qualify["ROOT_known_owner_observation_bundle_ref"]=qroot["ROOT_known_owner_observation_bundle_ref"]
    qroot["ROOT_resume_qualification_ref"]=put("qualified",qualify)
    cn=dict(constants,require=require,read=lambda record:copy.deepcopy(records[record["path"]]),
        raw=lambda record:None,local=lambda path:Path(path),ref=lambda path:rr("manifest"),MANIFEST="manifest",
        time=types.SimpleNamespace(time=lambda:0),datetime=__import__("datetime").datetime,timezone=__import__("datetime").timezone,copy=copy)
    functions=[n for n in contract.body if isinstance(n,ast.FunctionDef) and n.name in ("validate_resume_prerequisites_v5","validate_known_owner_scope_v6")]
    exec(compile(ast.Module(body=functions,type_ignores=[]),"<actual-prerequisite-API-with-entirely-fake-authority-receipts>","exec"),cn)
    assert cn["validate_resume_prerequisites_v5"](qroot)["actual_saved_progress"]["batch"]==6500
    key=qroot["ROOT_resume_qualification_ref"]["path"]
    for changes in [dict(genuine_full_state_model_optimizer_scheduler_RNG_timestamp_verified=False),
        dict(current_v4_complete64_identity_and_group_drain_verified=False),
        dict(actual_saved_progress=dict(epoch=0,batch=6501,batch_in_epoch=6501,optimizer_unique_steps=[6501])),
        dict(template_only=True)]:
        records[key]=dict(qualify,**changes);rejected=False
        try:cn["validate_resume_prerequisites_v5"](qroot)
        except ValueError:rejected=True
        assert rejected
    records[key]=qualify
    ownerkey=qroot["ROOT_known_owner_observation_bundle_ref"]["path"]
    for changes in [dict(historical_v3_full_ownership_reconstructed=True),dict(historical_v3_unknown_descendants_absent=True)]:
        records[ownerkey]=dict(owner,**changes);rejected=False
        try:cn["validate_resume_prerequisites_v5"](qroot)
        except ValueError:rejected=True
        assert rejected
    records[ownerkey]=owner
    original_pid=copy.deepcopy(records[pidobs["path"]]);records[pidobs["path"]]["observations"]=[]
    rejected=False
    try:cn["validate_resume_prerequisites_v5"](qroot)
    except ValueError:rejected=True
    assert rejected;records[pidobs["path"]]=original_pid
    # Entire finite cleanup implementation stays AST-identical to sealed v4.
    blocks=[]
    for path in ["research/reviews/native_VAL000_GPU0_owned_roster_source_20261016_v4/execution_only_v1.py",HERE+"/execution_only_v1.py"]:
        t=ast.parse((ROOT/path).read_text());f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=="execute_v1")
        blocks.append(ast.Module(body=next(n.finalbody for n in f.body if isinstance(n,ast.Try) and n.finalbody),type_ignores=[]))
    assert ast.dump(blocks[0],include_attributes=False)==ast.dump(blocks[1],include_attributes=False)
    return dict(schema="native_VAL000_full_state_resume_v5_synthetic_source_API_fixture",status="PASS_SOFTWARE_ONLY",
        capacity16384MiB_boundary_accepted=True,capacity16385MiB_actual_sample_before_exception=True,
        missing_GPU_measurement_remains_unknown=True,actual_original_hard_RLIMIT_soft_and_hard2GiB_API_checked=True,
        private_original_full_state_loader_code_and_FD_callback_checked=True,
        weights_only_ignored_state_nonstrict_and_wrong_optimizer_step_rejected=True,
        exact_nested_Timestamp_schema_fake_fixture=True,actual_ROOT_prerequisite_API_fake_positive_and_seven_negatives=True,unknown_or_reconstructed_history_and_missing_known_observation_rejected=True,entire_v4_cleanup_finally_AST_identical=True,
        native_original_scientific_run_code_object_preserved=True,
        project_scientific_imports_payload_or_device_process_access=False,actual_science_restore_fit_or_training_claim=False)

if __name__=="__main__":
    if sys.argv[1:]!=["--run"]:raise SystemExit("Explicit fake-only --run required")
    print(json.dumps(run(),sort_keys=True,indent=2))
