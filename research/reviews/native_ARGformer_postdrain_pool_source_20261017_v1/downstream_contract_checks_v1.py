"""RAM-only full downstream contract roundtrips; no production imports."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone,timedelta
import ast,copy,json,hashlib,time
HERE=Path(__file__).parent;ROOT=HERE.parents[2]
def need(ok,message="RAM fixture assertion"):
    if not ok:raise ValueError(message)
def rejected(fn):
    try:fn()
    except (ValueError,KeyError,TypeError):return True
    return False
def run_v1():
    records={}
    def ref(path):return dict(path=path,sha256=hashlib.sha256(path.encode()).hexdigest())
    def read(r):
        if r["path"] in records:return copy.deepcopy(records[r["path"]])
        need(r["path"].endswith(".json"),"Only bounded source metadata fixture")
        return json.loads((ROOT/r["path"]).read_text())
    def put(name,value):records[name]=value;return ref(name)
    def local(path):return ROOT/path
    original=SimpleNamespace(require=need,local=local,read=read,raw=lambda r:json.dumps(read(r)).encode(),ref=ref,
        load=lambda *a:None,save=lambda *a:None,ROOT=ROOT,GUARD=ref("RAM_guard"),POLICY=ref("RAM_policy"),
        SELECTOR_REF=ref("RAM_selector"),GPU0_UUID="RAM_UUID",BASE_POOL="RAM_base",GPU0_GUARD="RAM_GPU_guard",
        PIDFD="RAM_pidfd",OWNED="RAM_owned",NATIVE_PYTHON_REF=ref("RAM_python"),
        AUTHOR_ROOT="RAM_author",AUTHOR_MANIFEST_REF=ref("RAM_author_manifest"),RUNTIME_REF=ref("RAM_runtime"),
        PARENT_REGISTRY=str(HERE.relative_to(ROOT))+"/parent_registry.v1.json",
        cardinality_v1=lambda n,drop:dict(expected_optimizer_updates=5*(n//512 if drop else (n+511)//512)))
    # Execute only actual stdlib contract definitions; replace its single relative
    # production import with an explicitly bounded RAM metadata stand-in.
    tree=ast.parse((HERE/"downstream_contract_v1.py").read_text())
    tree.body=[n for n in tree.body if not isinstance(n,ast.ImportFrom)or n.level==0]
    ns=dict(original=original);exec(compile(tree,"<actual downstream contract; RAM original>","exec"),ns)
    registry=read(ref(original.PARENT_REGISTRY));parent="american_c3_val_000";row=registry["parents"][parent]
    counts_ref=dict(path="experiments/results/American_C3_native_val000_full30mb_counts980_namespace_v1/completed.json",sha256="502c111b443c46eea789dc03ee3ff8b6454546ce6e86e3f9fcddba25e82489a7")
    counts=read(counts_ref);prep=read(row["preparation_ref"]);pcfg_ref=counts["preparation_config_ref"]
    config=read(ref(str(HERE.relative_to(ROOT))+"/downstream_scientific_base.v1.json"));cfg=put("RAM_config",config)
    now=datetime.now(timezone.utc);start=(now-timedelta(minutes=1)).isoformat();end=(now+timedelta(hours=2)).isoformat()
    manifest=ref(ns["MANIFEST"]);records[manifest["path"]]=dict(refs=[])
    review=put("RAM_review",dict(independent=True,verdict="SOURCE_PASS",source_manifest_ref=manifest))
    records[ns["IO_REVIEW_REF"]["path"]]=dict(verdict="PASS_SOURCE_OPTION_ONLY_EXECUTION_NOT_AUTHORIZED")
    capability=put("RAM_capability",dict(jobs={}))
    original.pool_c=SimpleNamespace(validate_cap_v1=lambda r:read(r))
    base=dict(root_authorizer="root_agent",status="active",template_only=False,execution_authorized=True,created_at_utc=start,expires_at_utc=end)
    master=dict(base,schema="native_VAL000_downstream_ROOT_master_input_v1",job_id="RAM-master",parent_id=parent,
        source_manifest_ref=manifest,source_review_ref=review,source_review_passed=True,capacity_guard_ref=original.GUARD,
        capacity_policy_ref=original.POLICY,storage_policy_selector_ref=original.SELECTOR_REF,
        original_MLM7GiB_scope_extended=False,future_storage_reserved_bytes=65*2**30,
        operational_IO_source_ref=ns["IO_SOURCE_REF"],operational_IO_policy_ref=ns["IO_POLICY_REF"],
        owned_roster_policy_ref=ns["OWNED_ROSTER_POLICY_REF"],config_ref=cfg,runtime_ref=original.RUNTIME_REF,
        output_directory="experiments/results/native_postdrain_downstream_RAM",worker_auxiliary_scope="research/reviews/native_postdrain_downstream_AUX_RAM",
        preparation_ref=row["preparation_ref"],preparation_config_ref=pcfg_ref,counts_ref=counts_ref,
        postdrain_pool_capability_ref=capability,MLM_prerequisites=dict(RAM_only=True))
    storage=put("RAM_storage",dict(job_id=master["job_id"],output_directory=master["output_directory"],
        reserved_new_bytes=65*2**30,completed_overlay_policy_ref=original.POLICY,budget_components=dict(consumer_adapter=original.GUARD),
        reservation=dict(auxiliary_scopes=[str(local(master["worker_auxiliary_scope"]))],
            consumer_bindings=[manifest,ref(ns["DRIVER"]),cfg,original.RUNTIME_REF])))
    master["storage_ref"]=storage;master_ref=put("RAM_master",master)
    inputs=[]
    for phase in ("contrastive","finalize"):
        job=master["job_id"]+"--"+phase
        records[capability["path"]]["jobs"][job]=dict(worker_kind="native",parent_id=parent,native_phase=phase,cuda_cap_bytes=0 if phase=="finalize" else 32*2**30)
        q=dict(base,schema="native_VAL000_downstream_ROOT_phase_authority_v1",job_id=job,parent_id=parent,phase=phase,
            master_input_ref=master_ref,source_manifest_ref=manifest,source_review_ref=review,source_review_passed=True,
            config_ref=cfg,runtime_ref=original.RUNTIME_REF,capacity_guard_ref=original.GUARD,capacity_policy_ref=original.POLICY,
            storage_policy_selector_ref=original.SELECTOR_REF,postdrain_pool_capability_ref=capability,
            caps=dict(ns["BASE_CAPS"],max_cuda_reserved_bytes=0 if phase=="finalize" else 32*2**30),
            automatic_restart_authorized=False,partial_scientific_success_authorized=False,
            operational_IO_source_ref=ns["IO_SOURCE_REF"],operational_IO_policy_ref=ns["IO_POLICY_REF"],
            ROOT_context_source_ref=ref(ns["CONTEXT"]),keep1_rotation_of_new_scope_authorized=True,
            original_author_recipe_claimed_exact=False,prerequisites=dict(MLM=master["MLM_prerequisites"]),
            original74GiB_proposal_acknowledged_unmeasured=True,new32GiB_allocator_is_explicit_unqualified_resource_attempt=True,
            full_original_author_settings_and_workload_unchanged=True,no_batch_precision_workload_or_algorithm_reduction_or_fallback=True)
        qr=put("RAM_authority_"+phase,q)
        inputs.append(put("RAM_input_"+phase,dict(base,schema="native_VAL000_downstream_ROOT_phase_input_v1",
            job_id=job,parent_id=parent,phase=phase,root_stage_request_ref=qr,
            output_directory=master["output_directory"]+"/"+phase,worker_auxiliary_scope=master["worker_auxiliary_scope"]+"/"+phase,
            refs=dict(config=cfg,runtime=original.RUNTIME_REF,storage=storage))))
    for r in inputs:
        ns["inspect_input_v1"](r)
        need(ns["ROOT_INPUT_REF"] is None and ns["STATE"] is None and ns["PARENT"] is None,"Inspection restores all actual execution state")
    ns["set_amendment_v1"](inputs[0]);ns["validate_input_v1"](inputs[0])
    before=copy.deepcopy(ns["STATE"]);ns["inspect_input_v1"](inputs[1]);need(ns["STATE"]==before and ns["ROOT_INPUT_REF"]==inputs[0],"Inspection preserves already installed different actual phase")
    qr=records[inputs[0]["path"]]["root_stage_request_ref"];records[qr["path"]]["new32GiB_allocator_is_explicit_unqualified_resource_attempt"]=False
    need(rejected(lambda:ns["inspect_input_v1"](inputs[0])),"No implicit32GiB fit or fallback")
    records[qr["path"]]["new32GiB_allocator_is_explicit_unqualified_resource_attempt"]=True
    records["RAM_storage"]["reservation"]["consumer_bindings"]=[ref("RAM_role"+str(i))for i in range(65)]
    need(rejected(lambda:ns["inspect_input_v1"](inputs[0])),"Original64 direct ceiling")
    # Exercise the actual new pool prerequisite join with metadata-only records.
    cr=put("RAM_complete",dict(parent_id=parent,query_ancestry_truth_opened=False,child_input_ref=inputs[0]))
    wr=put("RAM_wait",dict(input_ref=inputs[0],returncode=0,actual_OS_wait_status=0,actual_direct_worker_wait=True,
        complete_owned_roster_known=True,all_owned_processes_and_groups_drained=True,operational_failure=None,
        completed_ref=cr,adapter_manifest_ref=manifest,group_observations=[dict(absent=True)]))
    job=master["job_id"]+"--contrastive"
    rr=put("RAM_registry",dict(capability_ref=capability,jobs=[dict(logical_job=job,state="succeeded",exit_code=0,leader_wait_status_known=True,owned_drain_observed=True)]))
    outer=put("RAM_outer",dict(root_authorizer="root_agent",job_id=job,actual_pool_logical_outer_OS_wait_consumed=True,
        actual_exit_code=0,actual_terminal_consumed=True,actual_wait_ref=wr,original_standalone_or_historical_wait_relabelled=False,
        pool_capability_ref=capability,immutable_pool_registry_snapshot_ref=rr))
    c=SimpleNamespace(read=read,require=need,PARENT=parent,ref=ref,MANIFEST=ns["MANIFEST"],STATE=before,COUNTS_REF=counts_ref,PREP_REF=row["preparation_ref"])
    fn=next(n for n in ast.parse((HERE/"downstream_context_v1.py").read_text()).body if isinstance(n,ast.FunctionDef)and n.name=="pool_prerequisite_bundle_v1")
    glue=dict(c=c);exec(compile(ast.Module(body=[fn],type_ignores=[]),"<actual pool prerequisite RAM>","exec"),glue)
    bundle=dict(completion_ref=cr,wait_ref=wr,outer_terminal_ref=outer)
    glue["pool_prerequisite_bundle_v1"](bundle,"contrastive")
    records[capability["path"]]["jobs"][job]["native_phase"]="mlm"
    need(rejected(lambda:glue["pool_prerequisite_bundle_v1"](bundle,"contrastive")),"Cross-phase declaration cannot be reused")
    records[capability["path"]]["jobs"][job]["native_phase"]="contrastive"
    records[cr["path"]]["child_input_ref"]=inputs[1]
    need(rejected(lambda:glue["pool_prerequisite_bundle_v1"](bundle,"contrastive")),"Different child input cannot be substituted")
    records[cr["path"]]["child_input_ref"]=inputs[0];records[wr["path"]]["group_observations"]=[dict(absent=False)]
    need(rejected(lambda:glue["pool_prerequisite_bundle_v1"](bundle,"contrastive")),"Undrained group cannot become genuine completion")
    return dict(schema="native_postdrain_full_downstream_contract_RAM_checks_v1",status="PASS_STATIC_RAM_ONLY",
        actual_contract_roundtrip_contrastive_and_finalizer=True,actual_pool_prerequisite_join_positive_and_cross_phase_input_drain_negatives=True,exact_original_VAL000_preparation_config_counts_metadata=True,
        inspect_restores_empty_and_installed_phase_state=True,implicit32GiB_fit_and_65_direct_refs_rejected=True,
        production_imports=0,scientific_calls_payloads_checkpoints=0,real_guard_process_device_journal_Git_operations=0)
if __name__=="__main__":
    result=run_v1();out=HERE/"downstream_contract_checks.final.v1.json"
    with out.open("x")as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
    print(json.dumps(dict(status=result["status"],result_path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest()),sort_keys=True))
