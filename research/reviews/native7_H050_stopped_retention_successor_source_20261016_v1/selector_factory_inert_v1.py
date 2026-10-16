"""Real exact configuration/selector/memo code on synthetic modules; no real guard configure."""
from pathlib import Path
import contextlib,copy,fcntl,hashlib,importlib.util,json,os,stat,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");HERE="research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1"
def load(path,name):
    sp=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def run():
    sys.path[:0]=[str(ROOT),str(ROOT/"src")];checks=[]
    def require(ok,msg):
        if not ok:raise ValueError(msg)
    def good(msg,ok=True):require(ok,msg);checks.append(dict(name=msg,status="PASS"))
    def rejects(msg,fn):
        try:fn()
        except (ValueError,RuntimeError,AssertionError):good(msg);return
        raise AssertionError("Missing failstop "+msg)
    def ref(path):return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
    def fx(name):return dict(path="SYNTHETIC_SELECTOR_ONLY/"+name+".json",sha256=hashlib.sha256(name.encode()).hexdigest())
    support=load("research/reviews/native_VAL000_full_state_resume_source_20261016_v12/dual_guard_memo_inert_v12.py","_selector_exact_code_support")
    v4=support.synthetic_module("src/arggnn/project_budget_v4.py","_selector_synthetic_v4",
        {"canonical","file_bytes","mutex","reference","scope_spec"},dict(Path=Path,contextlib=contextlib,stat=stat,os=os,hashlib=hashlib,fcntl=fcntl,MAX_JSON=16777216))
    budget=support.synthetic_module("scripts/project_budget_v9.py","_selector_synthetic_v9",{"successor_paths"},dict(v4=v4,Path=Path))
    v4.bound=lambda root,record:b"synthetic authenticated metadata only";v4.parse=json.loads
    events=[]
    parent=types.SimpleNamespace(configure_v2=lambda *a,**kw:events.append("parent_configured"))
    def import_namespace(module):
        return types.SimpleNamespace(spec_from_file_location=lambda *a:types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda m:None)),
            module_from_spec=lambda sp:module)
    guard=types.ModuleType("_selector_synthetic_child_guard")
    guard.__dict__.update(Path=Path,ROOT=ROOT,budget=budget,copy=copy,require=require,_POLICY_REF=None,_BASE=None,_REPAIR_AUTHORITY_REF=None,
        MODE="native7_H050_stopped_retention_v1",JOURNAL=".runtime/storage/native7_H050_stopped_retention_v1",
        BASE_REF=ref("research/reviews/native14_stopped_retention_successor_source_20261016_v3/native14_stopped_retention_v1.py"),
        BASE_POLICY_REF=ref("research/reviews/native14_stopped_retention_ROOT_activation_20261016_v3/activation_policy.root_active_v1.json"),
        RENEWAL_REF=ref("research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py"),RENEWAL_CERT_REF=fx("certificate"),_raw=lambda r:b"synthetic",
        _policy=lambda:events.append("child_policy_authenticated"),importlib=types.SimpleNamespace(util=types.SimpleNamespace(
            spec_from_file_location=lambda name,path:types.SimpleNamespace(path=Path(path),loader=types.SimpleNamespace(exec_module=lambda m:None)),
            module_from_spec=lambda sp:types.SimpleNamespace(install_renewal_v1=lambda r:events.append("renewal")) if sp.path.name=="marker_repair_renewal_v1.py" else parent)),
        load=lambda *a:types.SimpleNamespace(install_renewal_v1=lambda r:events.append("renewal")))
    support.definitions(HERE+"/stopped_retention_v1.py",{"configure_v2","configure_with_renewal_v1"},guard.__dict__)
    selector=load(HERE+"/explicit_policy_selector_v1.py","_synthetic_exact_selector")
    selector.budget=budget;selector.importlib=types.SimpleNamespace(util=import_namespace(guard))
    source=ref(HERE+"/stopped_retention_v1.py");manifest=ref(HERE+"/source_manifest.v1.json");measure=ref(HERE+"/retained_nodes_v1.py")
    policyref=fx("policy");reviewref=fx("review");repairref=fx("repair");sr=fx("selector")
    policy=dict(active=True,root_authorizer="root_agent",source_ref=source,source_manifest_ref=manifest,measurement_source_ref=measure,
        source_review_ref=reviewref,marker_repair_authority_ref=repairref,journal=guard.JOURNAL,scheduling_ceiling_bytes=980000000000,
        human_limit_bytes=1000000000000,warning_threshold_bytes=950000000000,base_ref=guard.BASE_REF,base_policy_ref=guard.BASE_POLICY_REF,
        base_pins={"SYNTHETIC_ACTUAL_ROOT_PIN":True},base_projected_bytes=970000000000,base_accounting_mode="native14_stopped_retention_v3")
    lineage={k:policy[k] for k in ("base_ref","base_policy_ref","base_pins","base_projected_bytes","base_accounting_mode")}
    value=selector.selector_draft_v1(source_ref=source,policy_ref=policyref,source_manifest_ref=manifest,measurement_source_ref=measure,
        source_review_ref=reviewref,repair_authority_ref=repairref,accounting_mode=guard.MODE,journal=guard.JOURNAL,frozen_parent_lineage=lineage)
    value["active"]=True
    review=dict(verdict="PASS_SOURCE_ONLY",source_ref=source,source_manifest_ref=manifest,measurement_source_ref=measure)
    docs={sr["path"]:value,policyref["path"]:policy,reviewref["path"]:review}
    selector._read=lambda r:copy.deepcopy(docs[r["path"]])
    first=selector.load_selected_guard_v1(sr);second=selector.load_selected_guard_v1(sr)
    good("real_selector_and_child_factory_initial_repeat_share_exact_module",first is guard and second is guard)
    good("one_parent_configure_no_memo_or_reinitialization",events.count("parent_configured")==1 and events.count("renewal")==1)
    good("exact_original_functions_still_unwrapped",all(getattr(v4,n).__module__==v4.__name__ for n in ("canonical","file_bytes","mutex")))
    for field,change in (("scheduling_ceiling_bytes",1000000000000),("journal","other"),("no_fallback_discovery_upgrade_or_in_run_swap",False)):
        old=docs[sr["path"]][field];docs[sr["path"]][field]=change
        rejects("selector_"+field+"_drift_blocks",lambda:selector.load_selected_guard_v1(sr));docs[sr["path"]][field]=old
    alternate=fx("selector_swap");docs[alternate["path"]]=copy.deepcopy(value)
    rejects("same_policy_new_selector_ref_swap_blocks",lambda:selector.load_selected_guard_v1(alternate))
    old=policy["base_pins"];policy["base_pins"]={"CHANGED":True};docs[policyref["path"]]=policy
    rejects("frozen_parent_lineage_drift_blocks",lambda:selector.load_selected_guard_v1(sr));policy["base_pins"]=old;docs[policyref["path"]]=policy
    namespace=load("research/reviews/current_mutex_contention_20261015_v1/consumer_namespace_memo_v1.py","_selector_real_pinned_namespace")
    handle=namespace.install_consumer_v1(v4,budget,consumer_role="read_only_consumer",expected_source_path=ROOT/"src/arggnn/project_budget_v4.py",
        expected_source_sha256="a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47")
    try:
        rejects("selector_reentry_after_real_singleton_memo_blocks_no_unwrap_or_reinstall",lambda:selector.load_selected_guard_v1(sr))
        rejects("real_child_factory_after_memo_blocks_strict_original_validation",lambda:guard.configure_v2(repairref,policy_ref=policyref))
        good("memo_lifecycle_stays_installed_and_no_second_configure",handle.receipt["installed"] is True and events.count("parent_configured")==1)
    finally:handle.uninstall()
    good("cached_selector_reuse_after_synthetic_explicit_lifecycle_no_reconfigure",selector.load_selected_guard_v1(sr) is guard and events.count("parent_configured")==1)
    return dict(schema="two_key_explicit_selector_real_factory_memo_synthetic_v1",status="PASS_SOURCE_ONLY",checks=checks,check_count=len(checks),
        real_source_code_objects_and_pinned_memo_installer=True,all_configured_wrapped_modules_synthetic=True,
        actual_guard_configure_policy_state_probe_accounting_revocation_activation_admission_job_or_Git=False)
if __name__=="__main__":print(json.dumps(run(),sort_keys=True,indent=2))
