"""Stdlib source/metadata/RAM-bound checks only. No project/runtime/guard imports."""
from pathlib import Path
import ast,copy,hashlib,json,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/H060_legacy_fast_full_training_source_20261017_v1"
BASE=ROOT/"research/reviews/H060_fast_observed_full_training_source_20261017_v1"
START=ROOT/"research/reviews/TreeLoom_observed_preparation_fast_source_20261017_v1/parent_startup_fast_v1.py"
COMP=ROOT/"research/reviews/H060_throughput_source_analysis_20261017_v1"


def need(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(path.read_text())


def ref(path):
    p=ROOT/path
    need(p.is_file() and p.resolve()==p and p.is_relative_to(ROOT),"Canonical contained file")
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def funcs(path):
    return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}


def same(a,b):return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)


def assignments(path):
    return {n.targets[0].id:n.value for n in ast.parse(path.read_text()).body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}


def clean_messages(node):
    node=copy.deepcopy(node)
    if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
        node.body[0].value.value="<doc>"
    for n in ast.walk(node):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="require" and len(n.args)>1:
            if isinstance(n.args[1],ast.Constant) and isinstance(n.args[1].value,str):n.args[1].value="<message>"
    return node


def first_party_import_closure(entry):
    def resolve(module):
        for folder in (ROOT/"src",ROOT):
            p=folder/(module.replace(".","/")+".py")
            if p.is_file():return p
            p=folder/module.replace(".","/")/"__init__.py"
            if p.is_file():return p
        return None
    pending=[entry];seen=set()
    while pending:
        path=pending.pop()
        key=str(path.relative_to(ROOT))
        if key in seen:continue
        seen.add(key)
        module=key.removesuffix(".py")
        if module.startswith("src/"):module=module[4:]
        parts=module.split("/")
        package=parts[:-1]
        for at in range(1,len(package)+1):
            p=resolve(".".join(package[:at]))
            if p is not None and p.name=="__init__.py":pending.append(p)
        for n in ast.walk(ast.parse(path.read_text())):
            names=[]
            if isinstance(n,ast.Import):names=[a.name for a in n.names]
            elif isinstance(n,ast.ImportFrom):
                prefix=(package[:len(package)-n.level+1] if n.level else [])+((n.module or "").split(".") if n.module else [])
                base=".".join(prefix);names=[base]+[base+"."+a.name for a in n.names]
            for name in names:
                p=resolve(name)
                if p is not None:pending.append(p)
    return seen


def run():
    checks=[]
    for p in HERE.glob("*.py"):ast.parse(p.read_text(),filename=str(p))
    b=read(HERE/"scientific_binding.v1.json");bindings=read(HERE/"source_bindings.v1.json")
    pins=read(HERE/"scientific_source_pins.v2.json")["refs"]
    pmap={r["path"]:r["sha256"]for r in pins}
    need(len(pmap)==len(pins),"Unique scientific source paths")
    for r in pins:need(ref(r["path"])==r,"Exact scientific source "+r["path"])
    original=read(BASE/"scientific_source_pins.v2.json")["refs"]
    need(len(original)==76 and all(pmap.get(r["path"])==r["sha256"]for r in original),"Retain every76 original scientific pins")
    need(read(HERE/"scientific_manifest.v1.json")["source_sha256"]==pmap,"Exact scientific manifest/pin map")
    for key,name in (("scientific_source_pins_ref","scientific_source_pins.v2.json"),("source_manifest_ref","scientific_manifest.v1.json"),("source_handoff_ref","scientific_handoff.v1.json"),("protocol_ref","protocol.inactive.v1.json"),("runner_source_ref","warmstart_train_v2.py")):
        need(b[key]==ref(str((HERE/name).relative_to(ROOT))),"Exact scientific join "+key)
    closure=first_party_import_closure(HERE/"warmstart_train_v2.py")
    need(all(p in pmap for p in closure),"Complete first-party runner closure; missing="+repr(sorted(closure-set(pmap))))
    checks.append("all76_original_scientific_pins_retained_complete_actual_runner_import_closure")

    old,new=funcs(BASE/"warmstart_train_v2.py"),funcs(HERE/"warmstart_train_v2.py")
    need(set(old)==set(new),"No added/removed training function")
    for name in old:
        need(same(clean_messages(old[name]),clean_messages(new[name])),"Whole original runner function AST retained: "+name)
    oa,na=assignments(BASE/"warmstart_train_v2.py"),assignments(HERE/"warmstart_train_v2.py")
    for key in ("DEFAULTS","CHECKPOINT_EPOCHS","SCHEMA","ACCEPTANCE","ENGINE","ENGINE_REVIEW","LEGACY_OBSERVATION_HERE"):
        need(same(oa[key],na[key]),"Original scientific constant "+key)
    module="research.reviews.H060_throughput_source_analysis_20261017_v1.observation_provider_candidate_v1"
    for path in (HERE/"warmstart_train_v2.py",HERE/"parent_startup_legacy_fast_v1.py"):
        imports=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ImportFrom)]
        need(any(n.module==module and any(a.name=="SealedObservationProviderV1" for a in n.names) for n in imports),"Same candidate provider in runner AND startup")
    need(any(isinstance(n,ast.ImportFrom) and n.level==1 and n.module=="parent_startup_legacy_fast_v1" for n in ast.parse((HERE/"warmstart_train_v2.py").read_text()).body),"Runner selects fresh startup")
    checks.append("entire_training_cache_targets_warmstart_600_each_1200_and60barrier_AST_unchanged")

    oldstart,newstart=funcs(START),funcs(HERE/"parent_startup_legacy_fast_v1.py")
    need(set(oldstart)==set(newstart),"Only original startup APIs")
    need(same(oldstart["fast_global_identity_v1"],newstart["fast_global_identity_v1"]),"Global ordered tensor digest byte-recipe unchanged")
    need(same(clean_messages(oldstart["_check_fast_sources_v1"]),clean_messages(newstart["_check_fast_sources_v1"])),"Original exact ref set/count/hash gate")
    normalized=copy.deepcopy(newstart["seal_then_prepare_one_parent_v1"])
    tr=next(n for n in normalized.body if isinstance(n,ast.Try))
    domain=[n for n in tr.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=="require" and any(isinstance(x,ast.Constant) and x.value=="Reviewed positive-stride NumPy int8 original genotype/allele axes only" for x in n.value.args)]
    need(len(domain)==1,"Explicit independently reviewed actual domain failstop")
    tr.body.remove(domain[0])
    added=[n for n in tr.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=="legacy_seal" for x in n.targets) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=="dict"]
    need(len(added)==1,"One genuine legacy record producer annotation after inherited real seal")
    sealmeta={k.arg:k.value for k in added[0].value.keywords}
    need(ast.literal_eval(sealmeta["legacy_block_seal_producer_unchanged"])is False,"Truthful legacy producer changed")
    tr.body.remove(added[0])
    globalnode=next(n for n in tr.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name)and x.id=="global_seal" for x in n.targets))
    kw={k.arg:k for k in globalnode.value.keywords}
    need(ast.literal_eval(kw["legacy_block_seal_producer_unchanged"].value)is False,"Truthful global record producer changed")
    need(ast.literal_eval(kw["global_FP32_preparation_implementation_unchanged"].value)is True,"Current FP32 global route retained")
    kw["legacy_block_seal_producer_unchanged"].value=ast.Constant(value=True)
    globalnode.value.keywords=[k for k in globalnode.value.keywords if k.arg not in ("legacy_block_seal_producer_source_refs","global_FP32_preparation_implementation_unchanged")]
    need(same(clean_messages(oldstart["seal_then_prepare_one_parent_v1"]),clean_messages(normalized)),"Original load/seal/release/global/hash/finally lifecycle except disclosed metadata/domain")
    a=assignments(HERE/"parent_startup_legacy_fast_v1.py")
    required=ast.literal_eval(a["REQUIRED_FAST_SOURCE_PATHS"]);producer=ast.literal_eval(a["LEGACY_PRODUCER_SOURCE_PATHS"])
    need(len(required)==len(set(required))==14 and len(producer)==len(set(producer))==8,"Complete nonambiguous preparation/producer sets")
    need({r["path"]for r in b["fast_preparation_source_refs"]}==set(required) and set(producer)<=set(required),"Every actual producer and complete preparation source is gated")
    need(all(ref(r["path"])==r and pmap[r["path"]]==r["sha256"] for r in b["fast_preparation_source_refs"]),"Actual complete refs match source/pins")
    checks.append("both_provider_bindings_actual_domain_new_producer_seals_complete14source_gates")

    legacy=read(ROOT/b["legacy_support_review_ref"]["path"])
    for key in ("legacy_support_review_ref","legacy_support_manifest_ref","predecessor_complete_review_ref","predecessor_complete_manifest_ref","predecessor_complete_runner_ref"):
        need(ref(b[key]["path"])==b[key],"Exact genuine component/predecessor evidence "+key)
    need(legacy["independent"]is True and legacy["verdict"]=="SOURCE_PASS" and legacy["blockers"]==[] and legacy["source_manifest_ref"]==b["legacy_support_manifest_ref"],"Genuine independent legacy component pass")
    need(legacy["source_pass_domain"]==b["legacy_support_domain"] and all(legacy[k]==b[k] for k in ("source_ref","encoder_ref","provider_ref")),"Exact reviewed component source/domain only")
    originalbinding=read(BASE/"scientific_binding.v1.json")
    for r in originalbinding["fast_preparation_source_refs"]:
        need(r in b["fast_preparation_source_refs"] and ref(r["path"])==r,"Retain every unchanged current global source")
    component=read(ROOT/b["fast_adapter_review_ref"]["path"]);fixture=read(ROOT/b["fast_adapter_ROOT_fixture_ref"]["path"])
    need(component["status"]=="SOURCE_PASS" and fixture["actual_returncode"]==0,"Retain genuine global adapter component evidence")
    checks.append("genuine_legacy_domain_review_and_current_FP32_global_component_evidence")

    current,base=funcs(HERE/"root_driver_v1.py"),funcs(BASE/"root_driver_v1.py")
    for name in ("driver","_make_root_mutation_memo","_memo_admission_node","_install_memo_admit","execute_or_admit","consumer_for_admitted","worker","append_selected_source"):
        need(same(current[name],base[name]),"Original transaction/owned lifecycle function "+name)
    need((HERE/"worker_entry_v1.py").read_bytes()==(BASE/"worker_entry_v1.py").read_bytes(),"Common worker byte-identical")
    need(set(bindings["consumers"])=={"H060"},"H060 consumer only")
    spec=bindings["consumers"]["H060"]
    for key in ("contract_ref","driver_ref","worker_ref","source_manifest_ref","source_handoff_ref"):
        need(ref(spec[key]["path"])==spec[key],"Original isolated CPU runtime component unchanged "+key)
    need(spec["scientific_binding_ref"]==ref(str((HERE/"scientific_binding.v1.json").relative_to(ROOT))),"Actual common-worker selected new binding")
    need(read(HERE/"storage_profiles.v1.json")==read(BASE/"storage_profiles.v1.json") and len(read(HERE/"storage_profiles.v1.json"))==1,"Only unchanged current selector profile")
    need(bindings["actual_selector_metadata_ref"]==read(BASE/"source_bindings.v1.json")["actual_selector_metadata_ref"] and ref(bindings["actual_selector_metadata_ref"]["path"])==bindings["actual_selector_metadata_ref"],"Exact unchanged current selector")
    need(all("legacy_fast_" in v for v in bindings["fresh_namespace_prefixes"].values()),"New disjoint namespace prefixes")
    need(all(e in bindings["excluded_scope_sets"]for e in read(BASE/"source_bindings.v1.json")["excluded_scope_sets"]),"Retain original excluded scopes")
    checks.append("unchanged_memo_three_admission_substitutions_worker_owned_gate_lifecycle_selector")

    caps=read(HERE/"caps.inactive.v1.json")
    need((HERE/"caps.inactive.v1.json").read_bytes()==(BASE/"caps.inactive.v1.json").read_bytes() and caps==spec["caps"],"Every cap byte-identical")
    fixture=read(HERE/"role_union_fixture.v1.json");guards=list(fixture["actual_guard_role_refs_by_name"].values())
    prior=read(ROOT/bindings["actual_admitted_predecessor_input_ref"]["path"])
    need(len(guards)==46 and len({json.dumps(r,sort_keys=True)for r in guards})==39 and all(r in prior["storage"]["reservation"]["consumer_bindings"]for r in guards),"Every46named/39distinct historical role retained")
    runtime=ROOT/spec["base_directory"];deps=read(runtime/"dependencies.v1.json")
    virtual=lambda label:dict(path="research/reviews/H060_isolated_CPU_ROOT_preparation_legacy_fast_FIXTURE/"+label+".json",sha256=hashlib.sha256(("symbolic-only:"+label).encode()).hexdigest())
    request=virtual("request");q={key:virtual(key)for key in ("source_review_ref","scientific_review_ref","runtime_ref","runtime_qualification_ref","root_directive_ref","registration_ref","registration_decisions_ref","training_context_ref")}
    q["storage_policy_selector_ref"]=bindings["actual_selector_metadata_ref"]
    ns=dict(binding=lambda:b,ref=virtual,MANIFEST="manifest",SOURCE="contract_driver",DRIVER="contract_driver",ENTRY="worker",HANDOFF="handoff",DEPS=deps,NAMESPACE=deps["NAMESPACE"],PIDFD=deps["PIDFD"],OWNED=deps["OWNED"],SUPERVISOR=deps["SUPERVISOR"])
    role=funcs(runtime/"root_contract_v1.py")["direct_roles"]
    exec(compile(ast.Module(body=[role],type_ignores=[]),"<pure-role-metadata>","exec"),ns)
    roles=ns["direct_roles"](request,q,virtual("config"),virtual("closure"))
    ns=dict(need=need,copy=copy);exec(compile(ast.Module(body=[current["append_selected_source"]],type_ignores=[]),"<pure-role-append>","exec"),ns)
    child=bindings["selected_child_source_ref"];roles=ns["append_selected_source"](roles,child,child)
    api=funcs(ROOT/child["path"])["consumer_bindings_v1"];ns=dict(require=need,copy=copy,_raw=lambda r:None)
    exec(compile(ast.Module(body=[api],type_ignores=[]),"<pure-bound-role-union>","exec"),ns)
    union=ns["consumer_bindings_v1"](guards,roles)
    need(len(union)==61 and all(r in union for r in guards),"Actual selected pure API full symbolic union61<=64")
    rejected=False
    try:ns["consumer_bindings_v1"](guards,roles+[virtual("overflow"+str(i))for i in range(4)])
    except ValueError:rejected=True
    need(rejected,"Actual bound pure API rejects65; no role dropping or cap change")
    admit=funcs(runtime/"root_driver_v1.py")["admit"]
    calls=[n for n in ast.walk(admit)if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    at=lambda name:min(n.lineno for n in calls if n.func.attr==name)
    need(at("consumer_bindings_v1")<at("scope_spec")<at("operation_draft_v1")<at("apply_reserve_v1"),"Real full-role union/scope preflight before authority/reserve mutations")
    checks.append("all46roles_preserved_actual_pure_union61_reject65_before_authority_reserve")

    helper=funcs(COMP/"legacy_support_exact_v1.py")["branch_concordance_support_exact_v1"]
    defaults={a.arg:ast.literal_eval(v)for a,v in zip(helper.args.kwonlyargs,helper.args.kw_defaults)}
    need(defaults==dict(marker_chunk_size=32,max_count_elements=8388608),"Exact reviewed chunk/count failstop defaults")
    need(8388608*8==67108864 and caps["rss_bytes"]==128*2**30 and caps["max_cached_parent_bytes"]==96*2**30,"64MiB count main payload; original128GiB RSS/96GiB cache caps retained")
    need(not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ("open","save","load","train","backward")for n in ast.walk(helper)),"No disk/truth/model operation in support helper")
    checks.append("bounded64MiB_main_count_payload_original_RAM_cache_failstops_no_fit_claim")

    for name in ("ROOT_request.INACTIVE.template.v1.json","ROOT_directive.INACTIVE.template.v1.json"):
        v=read(HERE/name);need(v["active"]is False and v["root_authorizer"]is None and v["created_at_utc"]is None and v["expires_at_utc"]is None,"No active authority/clock")
        need(v["scientific_config"]["execution_enabled"]is False,"Inactive science config")
    need(read(HERE/"runtime_qualification.INACTIVE.template.v1.json")["status"]=="inactive" and read(HERE/"registration.INACTIVE.template.v1.json")["status"]=="INACTIVE_NOT_REGISTERED","No fabricated runtime or registration")
    need(read(HERE/"ROOT_scientific_review.INACTIVE.template.v1.json")["status"]=="UNREVIEWED_INACTIVE","Complete independent review still required")
    reviewnode=current["fast_scientific_review"]
    text=ast.unparse(reviewnode)
    need("H060_legacy_fast_complete_scientific_review_v1" in text and "legacy_support_review_ref" in text and "legacy_support_domain" in text and "predecessor_complete_review_ref" in text,"Fresh complete scientific review gate binds new and predecessor/component sources")
    checks.append("fresh_complete_independent_review_registration_runtime_ROOT_authority_gates")
    # Execute only the actual isolated metadata review/ref gates. Symbolic
    # reviews exist solely in memory and are never published as real verdicts.
    template=read(HERE/"ROOT_scientific_review.INACTIVE.template.v1.json")
    probe=copy.deepcopy(template)
    probe.update(status="SOURCE_PASS",independent_review=True,blockers=[],
        candidate_outer_manifest_ref=ref(str((HERE/"source_manifest.v1.json").relative_to(ROOT))))
    marker=dict(path="symbolic-review-only.json",sha256="0"*64)
    payload=[probe]
    c=types.SimpleNamespace(binding=lambda:b,require=need,
        read=lambda r:payload[0] if r==marker else read(ROOT/r["path"]))
    ns=dict(metadata_ref=ref,MANIFEST=str((HERE/"source_manifest.v1.json").relative_to(ROOT)))
    exec(compile(ast.Module(body=[reviewnode],type_ignores=[]),"<pure-complete-review-gate>","exec"),ns)
    ns["fast_scientific_review"](c,marker)
    negatives=[template,dict(probe,schema="H060_fast_observed_complete_scientific_review_v1"),
        dict(probe,legacy_support_review_ref=dict(path="wrong.json",sha256="0"*64)),
        dict(probe,candidate_source_refs=[r for r in probe["candidate_source_refs"] if r!=b["provider_ref"]])]
    for value in negatives:
        payload[0]=value;rejected=False
        try:ns["fast_scientific_review"](c,marker)
        except (ValueError,KeyError,TypeError):rejected=True
        need(rejected,"Actual pure full-review gate rejects incomplete/inactive/stale source join")
    ns=dict(require=need,REQUIRED_FAST_SOURCE_PATHS=required,
        bound=lambda r:need(ref(r["path"])==r,"Exact ref gate"))
    exec(compile(ast.Module(body=[newstart["_check_fast_sources_v1"]],type_ignores=[]),"<pure-preparation-source-gate>","exec"),ns)
    refs=b["fast_preparation_source_refs"]
    need(len(ns["_check_fast_sources_v1"](refs))==14,"Actual exact preparation-ref gate accepts complete set")
    for rows in (refs[:-1],refs[:-1]+[refs[0]],[dict(r,sha256="0"*64) if at==0 else r for at,r in enumerate(refs)]):
        rejected=False
        try:ns["_check_fast_sources_v1"](rows)
        except ValueError:rejected=True
        need(rejected,"Actual preparation-ref gate rejects missing/duplicate/bad hash")
    checks.append("actual_pure_full_review_and_preparation_source_gates_positive_negative_checks")

    outer=read(HERE/"source_manifest.v1.json");baseouter=read(BASE/"source_manifest.v1.json")["source_sha256"]
    need(len(baseouter)==367 and all(outer["source_sha256"].get(p)==sha for p,sha in baseouter.items()),"Retain all367 current operational pins")
    need(all(outer["source_sha256"].get(p)==sha for p,sha in pmap.items()),"Every scientific pin in complete outer closure")
    for p,sha in outer["source_sha256"].items():need(ref(p)["sha256"]==sha,"Complete exact outer source closure "+p)
    need(len(outer["source_sha256"])==outer["source_sha256_count"],"Exact outer pin count")
    h=read(HERE/"source_handoff.v1.json");need(h["source_manifest_ref"]==ref(str((HERE/"source_manifest.v1.json").relative_to(ROOT))),"Acyclic handoff binds actual outer manifest")
    audit=read(HERE/"audit_manifest.v1.json")["source_sha256"]
    need(set(audit)=={str(p.relative_to(ROOT))for p in HERE.iterdir()if p.is_file()and p.name!="audit_manifest.v1.json"},"Audit covers every other package file")
    for path,sha in audit.items():need(ref(path)["sha256"]==sha,"Exact final package audit "+path)
    for name in ("torch","numpy","torch_geometric","tskit"):
        need(name not in sys.modules,"No project/scientific/runtime imports")
    checks.append("all367_operational_pins_plus_complete_new_immutable_outer_closure_no_runtime_operations")
    return dict(schema="H060_legacy_fast_source_only_checks_v1",status="STATIC_SOURCE_PASS_ONLY",checks=checks,checks_count=len(checks),scientific_pin_count=len(pmap),runner_first_party_modules=len(closure),outer_source_pin_count=len(outer["source_sha256"]),pure_metadata_role_union=61,role_ceiling=64,overflow65_rejected=True,main_count_payload_limit_bytes=67108864,production_fit_or_speed_measured=False,production_data_truth_model_steps_guard_mutex_process_admission_install_Git=False)


if __name__=="__main__":print(json.dumps(run(),indent=2,sort_keys=True))
