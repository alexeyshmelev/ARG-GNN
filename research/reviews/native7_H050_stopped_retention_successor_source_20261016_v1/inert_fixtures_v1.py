"""Synthetic authorities/state/modules and owned tiny filesystem only; no real guard configure."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import ast,copy,hashlib,importlib.util,json,os,socket,sys,tempfile,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1"
def require(ok,message):
    if not ok:raise ValueError(message)
def load(path,name):
    sp=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def ref(path):return dict(path=str(path),sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
def fx(name):return dict(path="SYNTHETIC_FUTURE_ONLY/"+name+".json",sha256=hashlib.sha256(name.encode()).hexdigest())
def run():
    sys.path.insert(0,str(ROOT));checks=[]
    def good(name,ok=True):require(ok,name);checks.append(dict(name=name,status="PASS"))
    def rejects(name,fn):
        try:fn()
        except (ValueError,KeyError,AssertionError,RuntimeError,FileNotFoundError):good(name);return
        raise AssertionError("Invalid fixture accepted: "+name)
    c=load(HERE+"/stopped_retention_v1.py","_two_key_inert_contract")
    # Actual static source/compact-metadata authentication only. No configure/state/mutex/probe.
    m=c._manifest();good("all309_historical_source_metadata_refs_and_original45_64_bindings_authenticated",
        len(m["runtime_authenticated_historical_source_and_metadata_refs"])==309 and [len(m["targets"][k]["all_original_binding_refs"]) for k in c.TARGET_KEYS]==[45,64])
    c._manifest=lambda:copy.deepcopy(m)
    for key in c.TARGET_KEYS:good(key+"_genuine_failed_direct_and_consumed_outer_metadata",c._historical_stop(m["targets"][key]))
    rejects("malformed_H039_never_credit",lambda:c._target(c.MALFORMED_KEY))
    rejects("old14_never_re_retired",lambda:c._target("H041/American-C3-native-MLM-full30mb-mlm-fit-v8"))
    rejects("fresh_native12_and_other_keys_excluded",lambda:c._target("H041/native-VAL000-verified22000-continuation-v12_root01_oct16"))
    good("inactive_policy_has_no_actual_parent_projection",m["actual_parent_pins_and_projection"] is None)
    good("980GB_and1TB_and_event_metadata_unchanged",c.CEILING_BYTES==980000000000 and c.HUMAN_LIMIT_BYTES==1000000000000 and c.META_CHARGE==524288 and c.BOOTSTRAP_CHARGE==2097152)
    rejects("980GB_strict_ceiling",lambda:c.charge(980000000000,0,0,0))
    rejects("1TB_cannot_override980GB",lambda:c.charge(999000000000,0,0,0))
    original_json=c._json;docs={}
    c._json=lambda r:copy.deepcopy(docs[r["path"]]) if r["path"] in docs else original_json(r)
    c._paths=lambda own:[Path(own["output"]),*[Path(x) for x in own["auxiliary_scopes"]]]
    now=datetime.now(timezone.utc);revtime=(now-timedelta(seconds=5)).isoformat();stamp=now.isoformat()
    def facts(t):
        return dict(PID_observations=[[r["pid"],r["start_ticks"],True,stamp] for r in t["known_PID_start_rows"]],
            group_observations=[[v,True,stamp] for v in t["known_process_groups"]])
    fixtures={}
    for key in c.TARGET_KEYS:
        t=m["targets"][key];rr=fx(key+"_revocation");obsref=fx(key+"_observation")
        rev=c.stopped_revocation_draft_v1(key);rev.update(active=True,revoked_at_utc=revtime)
        obs=c.stopped_observation_draft_v1(key,revocation_ref=rr);obs.update(active=True,observed_at_utc=stamp,**facts(t))
        docs[rr["path"]]=rev;docs[obsref["path"]]=obs
        request=dict(key=key,reservation_sha256=t["original_reservation_sha256"],revocation_ref=rr,observation_ref=obsref)
        proof=c._stopped_proof(request,t["original_reservation"])
        good(key+"_synthetic_proof_requires_exact_key_reservation_revocation_fresh_known_roster",proof["scientific_success_relabelled"] is False)
        c._observe_targets=lambda target:facts(target)
        good(key+"_live_apply_path_uses_injected_synthetic_probes_only",c._stopped_proof(request,t["original_reservation"],observe=True)["key"]==key)
        malformed=copy.deepcopy(rev);malformed["active"]=False;docs[rr["path"]]=malformed
        rejects(key+"_inactive_revocation_blocks",lambda:c._stopped_proof(request,t["original_reservation"]))
        docs[rr["path"]]=rev
        malformed=copy.deepcopy(obs);malformed["PID_observations"]=malformed["PID_observations"][:-1];docs[obsref["path"]]=malformed
        rejects(key+"_omitted_known_target_blocks",lambda:c._stopped_proof(request,t["original_reservation"]))
        malformed=copy.deepcopy(obs);malformed["PID_observations"][0][2]=False;docs[obsref["path"]]=malformed
        rejects(key+"_live_or_reused_numeric_PID_blocks",lambda:c._stopped_proof(request,t["original_reservation"]))
        stale=copy.deepcopy(obs);stale["PID_observations"][0][-1]=(now-timedelta(seconds=300)).isoformat();docs[obsref["path"]]=stale
        rejects(key+"_stale_pre_measurement_ROOT_observation_blocks",lambda:c._stopped_proof(request,t["original_reservation"],observe=True))
        docs[obsref["path"]]=obs
        own=t["original_reservation"]
        measurement=dict(schema="two_key_exact_stopped_retained_node_measurement_v1",conservative_scope_bytes=16384,entries=4,
            scope_paths=t["scope_paths"],stat_only=True,per_scope=[dict(path=v,entries=2,conservative_scope_bytes=8192,node_types=dict(directory=1,regular=1,symlink=0,socket=0)) for v in t["scope_paths"]],
            inventory_sha256="a"*64,two_complete_passes_and_final_no_follow_revalidation=True,per_node_additional_metadata_charge_bytes=4096,
            regular_payloads_opened=False,symlink_targets_followed=False,unsafe_nodes_rejected=True,node_metadata_preserved=True,
            namespace_normalization_performed=False,cooperative_snapshot_not_atomic_quota=True)
        good(key+"_both_entire_scopes_conservatively_charged",c._retained_measurement(measurement,own)==16384)
        for field,value in (("scope_paths",t["scope_paths"][:1]),("regular_payloads_opened",True),("symlink_targets_followed",True),
            ("two_complete_passes_and_final_no_follow_revalidation",False),("conservative_scope_bytes",own["reserved_bytes"]+1)):
            bad=copy.deepcopy(measurement);bad[field]=value
            rejects(key+"_measurement_"+field,lambda bad=bad:c._retained_measurement(bad,own))
        bad=copy.deepcopy(measurement);bad["per_scope"][0]["node_types"]["socket"]=1
        rejects(key+"_retained_socket_preserved_and_blocks",lambda:c._retained_measurement(bad,own))
        c._recorded_stopped_proof(proof,proof,stamp)
        fixtures[key]=(request,proof,measurement)
    good("all_original_unknown_history_and_H050_TRAIN1_preserved",
        fixtures[c.TARGET_KEYS[0]][1]["historical_facts_preserved"]["historical_v3_descendants"]=="UNKNOWN_TRUNCATED64"
        and fixtures[c.TARGET_KEYS[1]][1]["historical_facts_preserved"]["TRAIN_direct_wait"]==1)
    # Pure synthetic parent/child replay: original state function, no journal/mutex/guard call.
    current=json.loads((ROOT/"research/reviews/C3_saved600_artifact_acceptance_source_20261016_v1/source_manifest.v1.json").read_text())["current_guard_roles"]
    parentbindings={str(i):r for i,r in enumerate({r["path"]:r for r in current.values()}.values())}
    p=c.activation_policy_draft_v1(repair_authority_ref=fx("repair"),base_pins=dict(base_pins={"SYNTHETIC":True},completed_overlay_prefix={"events":0,"tail_sha256":None}),
        base_projected_bytes=970000000000,implementation_ref=ref("scripts/project_budget_v9.py"),launcher_ref=ref("scripts/launch_reserved_gpu_v6.py"),
        root_activation_ref=fx("activation"),permanent_revocation_ref=fx("parent_revocation"),source_review_ref=fx("review"))
    p["active"]=True;c._POLICY_REF=fx("policy");c._policy=lambda:copy.deepcopy(p)
    original_raw=c._raw;c._raw=lambda r:b"synthetic source/metadata binding" if r["path"].startswith("SYNTHETIC_FUTURE_ONLY/") else original_raw(r)
    bindings=c._bindings(parentbindings,p);good("all30_current_guard_roles_remain_direct_and_child38_under64",len(bindings)==38 and all(r in bindings.values() for r in parentbindings.values()))
    purebudget=types.SimpleNamespace(**{k:getattr(c.budget,k) for k in ("digest","SOURCE_REL","LAUNCHER_REL")})
    purebudget.v4=types.SimpleNamespace(LEGACY="SYNTHETIC",canonical=lambda root,path:Path(path),read_json=lambda *a:({"jobs":{}},None),
        paths_overlap=lambda a,b:a==b or a in b.parents or b in a.parents,integer=c.budget.v4.integer,encoded=c.budget.v4.encoded,reference=c.budget.v4.reference,bound=c.budget.v4.bound,parse=c.budget.v4.parse)
    c.budget=purebudget
    reservations={k:copy.deepcopy(m["targets"][k]["original_reservation"]) for k in c.TARGET_KEYS}
    baseview=dict(projected_bytes=p["base_projected_bytes"],accounting_mode=c.BASE_MODE,limit_bytes=c.CEILING_BYTES,hard_limit_bytes=c.HUMAN_LIMIT_BYTES,
        new_reserved_bytes=123,completed_credit_bytes=0)
    c._BASE=types.SimpleNamespace(_state_locked=lambda *a,**kw:(copy.deepcopy(reservations),set(),dict(baseview),copy.deepcopy(parentbindings),[],[],{}),
        _pins=lambda view:copy.deepcopy(p["base_pins"]))
    events=[];hashes=[];c._events=lambda bound:(copy.deepcopy(events),list(hashes))
    for i,key in enumerate(c.TARGET_KEYS):
        request,proof,measurement=fixtures[key];authorityref=fx("retirement_"+str(i))
        before=dict(base_pins=p["base_pins"],completed_overlay_prefix=dict(events=i,tail_sha256=hashes[-1] if hashes else None))
        authority=c.operation_draft_v1(c.RETIRE_OPERATION,request,expected_before=before,bindings=bindings);authority.update(active=True,expires_at_utc=(now+timedelta(hours=1)).isoformat())
        docs[authorityref["path"]]=authority
        value=dict(request=request,reservation_sha256=m["targets"][key]["original_reservation_sha256"],measurement=measurement,
            proof=proof,closed_utc=stamp,future_writes_and_relaunch_revoked=True)
        events.append(dict(operation=c.RETIRE_OPERATION,authority=authorityref,value=value));hashes.append(str(i)*64)
        state=c._state_locked(p["implementation_ref"]["sha256"],p["launcher_ref"]["sha256"])
        good(key+"_synthetic_append_only_replay_once",key in state[1])
        full_event=dict(schema="two_key_exact_stopped_retention_event_v1",sequence=i+1,previous_sha256=hashes[i-1] if i else None,
            bindings=bindings,operation=c.RETIRE_OPERATION,authority=authorityref,value=value)
        good(key+"_full_event_with_all_known_rows_and38_bindings_under64KiB",len(purebudget.v4.encoded(full_event))+1<=65536)
    good("two_synthetic_retirements_preserve_frozen_parent_state",len(state[1])==2 and baseview["projected_bytes"]==970000000000)
    oldprojection=baseview["projected_bytes"];baseview["projected_bytes"]+=1
    rejects("parent_projection_drift_blocks_no_predicted_parent_snapshot",lambda:c._state_locked(p["implementation_ref"]["sha256"],p["launcher_ref"]["sha256"]))
    baseview["projected_bytes"]=oldprojection
    # Full new-consumer role models, including future selector instance, not actual authorities.
    v12="research/reviews/native_VAL000_full_state_resume_source_20261016_v12"
    nc=load(v12+"/amendment_contract_v1.py","_two_key_inert_native12_roles")
    req=dict(ROOT_directive_ref=fx("native_directive"),independent_source_review_ref=fx("native_review"),
        direct_attachment_authority_ref=json.loads((ROOT/nc.ORIGINAL_INPUT["path"]).read_text())["root_stage_request_ref"])
    # Preserve the actual original attachment authority ref.
    oldinput=json.loads((ROOT/nc.ORIGINAL_INPUT["path"]).read_text());oldrequest=json.loads((ROOT/oldinput["root_stage_request_ref"]["path"]).read_text())
    req["direct_attachment_authority_ref"]=oldrequest["direct_attachment_authority_ref"]
    req["config_ref"]=oldinput["refs"]["config"];req["runtime_ref"]=oldinput["refs"]["runtime"]
    native_raw=nc.raw;nc.raw=lambda r:b"synthetic future role" if r["path"].startswith("SYNTHETIC_FUTURE_ONLY/") else native_raw(r)
    needed=nc.fresh_required_roles_v11(fx("native_request"),req,fx("native_closure"))
    selector_instance=fx("explicit_ROOT_selector")
    nativeunion=c.consumer_bindings_v1(list(bindings.values()),needed,[selector_instance])
    good("future_native_complete_returned_needed_selector_union_fits64",len(nativeunion)<=64)
    ac=load("research/reviews/C3_saved600_artifact_acceptance_source_20261016_v1/acceptance_contract_v1.py","_two_key_inert_C3_roles")
    main=json.loads((ROOT/"research/reviews/C3_saved600_artifact_acceptance_source_20261016_v1/source_manifest.v1.json").read_text());ac.manifest_v1=lambda:copy.deepcopy(main)
    spec=dict(refs={"config":fx("C3config")},root_stage_request_ref=fx("C3request"))
    req=dict(phase="predict_val_000",caps_ref=fx("C3caps"),training_evidence_ref=fx("C3evidence"))
    required=ac.required_direct_refs_v1(spec,req)
    extra=[fx("C3directive"),fx("C3stage_evidence"),
        ref("research/reviews/C3_full30Mb_eval_admission_readiness_source_20261016_v2/prepare_evaluation_root_v2.py"),
        ref("research/reviews/C3_full30Mb_eval_admission_readiness_source_20261016_v2/source_handoff.v1.json"),
        fx("C3builder_review"),fx("C3main_review"),
        ref("research/reviews/C3_saved600_validator_ROOT_preparation_source_20261016_v1/prepare_root_v1.py"),
        ref("research/reviews/C3_saved600_artifact_acceptance_source_20261016_v1/source_handoff.v1.json")]
    by={r["path"]:r for r in [*bindings.values(),*required,*extra,selector_instance]};C3count=len(by)
    if C3count>64:rejects("unchanged_future_C3_adapter_full_union_over64_blocks",lambda:c.consumer_bindings_v1(list(by.values())))
    else:good("unchanged_future_C3_adapter_full_union_fits64",True)
    rolemodel=dict(guard_state=list(bindings.values()),guard_state_count=len(bindings),retirement_invocation_upper_count=len(bindings)+3,
        future_native_full_union=nativeunion,future_native_full_union_count=len(nativeunion),
        future_C3_full_union=list(by.values()),future_C3_full_union_count=C3count,unchanged_future_C3_fits64=C3count<=64,
        existing_native12_and_live_predictors_never_rebound=True,no_actual_capacity_or_future_adapter_approval=True)
    # Real helper on tiny fixture roots only: full two-pass metadata, stored symlink, unsafe socket rejection.
    helper=load(HERE+"/retained_nodes_v1.py","_two_key_real_retained_helper")
    with tempfile.TemporaryDirectory(prefix="tiny_nodes_",dir=ROOT/HERE) as temp:
        root=Path(temp);roots=[root/"out",root/"aux"]
        for path in roots:path.mkdir()
        (roots[0]/"state.json").write_text("{}");(roots[0]/"alias").symlink_to("missing-preserved-target")
        first=helper.measure_retained_nodes_v1(root,roots,expected_paths=[str(p) for p in roots],socket_is_absent=lambda p:False)
        good("real_helper_tiny_two_pass_no_payload_no_follow_link_preserved",first["regular_payloads_opened"] is False and (roots[0]/"alias").is_symlink())
        sock=socket.socket(socket.AF_UNIX);dfd=os.open(roots[1],os.O_RDONLY|os.O_DIRECTORY)
        try:sock.bind("/proc/self/fd/%d/tiny.sock"%dfd)
        finally:os.close(dfd)
        try:rejects("real_helper_tiny_retained_socket_blocks_without_cleanup",lambda:helper.measure_retained_nodes_v1(root,roots,expected_paths=[str(p) for p in roots],socket_is_absent=lambda p:False))
        finally:sock.close()
    good("no_scientific_runtime_imported",not any(n in sys.modules for n in ("torch","numpy","composer","tskit")))
    return dict(schema="two_key_retention_source_only_inert_fixtures_v1",status="PASS_SOURCE_ONLY",checks=checks,check_count=len(checks),
        full_role_models=rolemodel,actual_project_scope_measurement_or_process_GPU_probe_or_guard_configure_or_accounting_or_revocation_activation_admission_job_Git=False)
if __name__=="__main__":print(json.dumps(run(),sort_keys=True,indent=2))
