"""Real pinned factories and memo on synthetic modules/owned tiny filesystem only."""
from pathlib import Path
import ast,contextlib,copy,fcntl,hashlib,importlib.util,json,os,stat,sys,tempfile,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/native_VAL000_full_state_resume_source_20261016_v12"
OLD="research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/common_v1.py"
CAND=HERE+"/amendment_contract_v1.py"
V4="src/arggnn/project_budget_v4.py";V9="scripts/project_budget_v9.py"
NAMESPACE="research/reviews/current_mutex_contention_20261015_v1/consumer_namespace_memo_v1.py"
def require(ok,message):
    if not ok:raise ValueError(message)
def ref(path):return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
def definitions(path,names,namespace):
    tree=ast.parse((ROOT/path).read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.ImportFrom) and n.module=="__future__"]
    nodes += [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    require(len(nodes)==len(names)+sum(isinstance(n,ast.ImportFrom) for n in nodes),"Exact selected source definitions")
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROOT/path),"exec",dont_inherit=True,optimize=0),namespace)
    # Use exact full-file compile code objects, including inherited future flags
    # and nested-comprehension bytecode, while retaining definition defaults.
    compiled=compile((ROOT/path).read_bytes(),str(ROOT/path),"exec",dont_inherit=True,optimize=0)
    for node in nodes:
        if isinstance(node,ast.FunctionDef):
            matches=[c for c in compiled.co_consts if isinstance(c,types.CodeType) and c.co_name==node.name]
            require(len(matches)==1,"Exact source code object")
            fn=namespace[node.name]
            while hasattr(fn,"__wrapped__"):fn=fn.__wrapped__
            fn.__code__=matches[0]
def constants(path):
    result={}
    for n in ast.parse((ROOT/path).read_text()).body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            try:result[n.targets[0].id]=ast.literal_eval(n.value)
            except (ValueError,TypeError):pass
    return result
def pinned_module(path,name):
    sp=importlib.util.spec_from_file_location(name,ROOT/path)
    value=importlib.util.module_from_spec(sp);sp.loader.exec_module(value);return value
def synthetic_module(path,name,names,extra):
    module=types.ModuleType(name);module.__file__=str(ROOT/path);module.__dict__.update(extra)
    definitions(path,names,module.__dict__);return module
def rejects(check,fn,checks):
    try:fn()
    except (ValueError,RuntimeError,AssertionError) as error:checks.append(dict(check=check,passed=True,error=str(error)));return
    raise AssertionError("Required failstop missing: "+check)
def scenario(order,checks):
    cn=constants(CAND);on=constants(OLD);events=[];operations=[]
    memo=pinned_module(NAMESPACE,"_native12_fixture_real_namespace_"+order)
    with tempfile.TemporaryDirectory(prefix="synthetic_dual_guard_",dir=ROOT/HERE) as temp:
        scratch=Path(temp);(scratch/"synthetic.lock").write_text("fixture lock only")
        payload=scratch/"payload.json";payload.write_text('{"value":1}')
        v4=synthetic_module(V4,"_native12_fixture_original_v4_"+order,
            {"canonical","file_bytes","mutex","reference","scope_spec"},
            dict(Path=Path,contextlib=contextlib,contextmanager=contextlib.contextmanager,stat=stat,os=os,hashlib=hashlib,fcntl=fcntl,
                MAX_JSON=1048576,LOCK="synthetic.lock"))
        budget=synthetic_module(V9,"_native12_fixture_original_v9_"+order,
            {"successor_paths"},dict(v4=v4,Path=Path,JOURNAL_REL="synthetic_journal"))
        adapter=scratch/"src/arggnn/project_budget_v9_consumer.py"
        adapter.parent.mkdir(parents=True);adapter.write_text("# tiny synthetic binding only")
        v4.integer=lambda value,label,*a:require(type(value) is int and value>0,label)
        v4.paths_overlap=lambda a,b:a==b or a in b.parents or b in a.parents
        v4.accounting_paths=lambda root:[root/"synthetic.lock"]
        v4.bound=lambda root,record:v4.file_bytes(root,record["path"])
        reservation=dict(key="synthetic/key",output=str(scratch/"scope_out"),auxiliary_scopes=[str(scratch/"scope_aux")],
            reserved_bytes=1,consumer_bindings=[dict(path="src/arggnn/project_budget_v9_consumer.py",sha256="synthetic")])
        def make_guard(path,name):
            guard=synthetic_module(path,name,{"budget_check_v9","validate_storage_v9"},
                dict(budget=budget,ROOT=scratch,require=require,_raw=lambda r:r))
            def checked(storage,**kwargs):
                first=v4.file_bytes(scratch,"payload.json")[0]
                second=v4.file_bytes(scratch,"payload.json")[0]
                require(first==second,"Same-operation pinned source read agrees")
                require(budget.successor_paths(scratch,reservation)==budget.successor_paths(scratch,reservation),
                    "Exact pinned scope/successor namespace bodies execute twice within operation")
                operations.append(dict(guard=name,storage=storage["tag"],kwargs=dict(kwargs),payload=first.decode()))
                return dict(payload=first.decode(),operation=len(operations))
            guard.check_storage_locked_v1=checked
            def configure(*a,**kw):
                require(all(getattr(v4,n).__module__==v4.__name__ for n in ("canonical","file_bytes","mutex")),
                    "Synthetic configure sees original unwrapped identities")
                events.append(dict(event="configure",guard=name))
            guard.configure_v2=configure
            return guard
        oldguard=make_guard(on["GUARD"]["path"],"old")
        freshguard=make_guard(cn["FRESH_GUARD_REF"]["path"],"fresh")
        renewal=types.SimpleNamespace(install_renewal_v1=lambda record:events.append(dict(event="renewal")))
        policy=dict(source_ref=on["GUARD"],active=True,root_authorizer="root_agent",schema="prospective980_Census_fit2_retention_policy_v1")
        oldns=dict(on,ROOT=ROOT,require=require,sys=types.SimpleNamespace(path=[]),read=lambda r:copy.deepcopy(policy))
        def old_load(record,name):
            if record==on["RENEWAL"]:return renewal
            if record==on["GUARD"]:return oldguard
            if record==on["NAMESPACE_MEMO"]:return memo
            raise AssertionError("Unexpected original factory source")
        oldns["load"]=old_load
        definitions(OLD,{"guard_v1"},oldns)
        original=types.SimpleNamespace(guard_v1=oldns["guard_v1"])
        oldq=dict(capacity_guard_ref=on["GUARD"],capacity_policy_ref=on["POLICY"])
        old=dict(job_id="old",output_directory="old_output",worker_auxiliary_scope="old_aux",
            refs=dict(config={"config":"old"},runtime={"runtime":"old"}))
        spec=dict(output_directory="new_output",worker_auxiliary_scope="new_aux",job_id="fresh",
            actual_execution_amendment_ref={"amendment":"fixture"})
        def storage(tag,job,output):
            return dict(tag=tag,job_id=job,output_directory=output,reserved_new_bytes=(7 if tag=="old" else 5)*2**30,
                reservation=dict(consumer_bindings=[{"source":"old"},{"source":"new"},{"config":"old"},{"runtime":"old"}]))
        oldstorage=storage("old","old",old["output_directory"]);newstorage=storage("fresh","fresh",spec["output_directory"])
        original.ref=lambda path:{"source":"old"}
        original.DRIVER="original"
        original._sizes=lambda path,**kw:[(("metadata.json",),8192,"regular")]
        caps=dict(scientific_output_bytes=4*2**30+128*2**20+65536,auxiliary_bytes=768*2**20,
            reservation_bytes=5*2**30,directory_and_accounting_allowance_bytes=128*2**20-65536)
        g=dict(cn,__name__="_native12_fixture_contract_"+order,ROOT=ROOT,require=require,copy=copy,load=None,
            RENEWAL=on["RENEWAL"],CERT=on["CERT"],REPAIR=on["REPAIR"],NAMESPACE_MEMO=on["NAMESPACE_MEMO"],
            MEMO=on["MEMO"],V4=on["V4"],ORIGINAL_STATE=(old,oldq,oldstorage),FRESH_STATE=(spec,{},newstorage),
            RESUME_STATE={"fixture":True},CAPS=caps,ORIGINAL_CAPS=dict(caps,reservation_bytes=7*2**30,
                scientific_output_bytes=6*2**30+128*2**20+65536),original=original,
            AMENDMENT_REF={"amendment":"fixture"},active_amendment_v1=lambda:True,local=lambda path:path,
            ref=lambda path:{"source":"new"},ACTUAL_DRIVER="new",_FRESH_GUARD_V12=None,_CONSUMER_GUARDS_V12=None,
            ACCOUNTING_CHOICES_V11=(5*2**30,17*2**28))
        def load(record,name):
            if record==on["RENEWAL"]:return renewal
            if record==cn["FRESH_GUARD_REF"]:return freshguard
            if record==on["NAMESPACE_MEMO"]:return memo
            raise AssertionError("Unexpected candidate factory source "+str(record))
        g["load"]=load
        names={"_consumer_guard_integrity_v12","_consumer_guards_v12","fresh_guard_runtime_v11","guard_v1",
            "charged_census_v1","FreshContinuationGuardV11"}
        definitions(CAND,names,g)
        if order in ("prewrapped_initial","nonshared_initial","bad_policy_initial"):
            if order=="prewrapped_initial":
                saved=v4.canonical;v4.canonical=lambda *a:None
            elif order=="nonshared_initial":
                freshguard.budget=types.SimpleNamespace(v4=v4)
            else:oldq["capacity_policy_ref"]={"changed":True}
            rejects(order+"_strict_initial_original_runtime_validation",lambda:g["guard_v1"]({}),checks)
            require(g["_CONSUMER_GUARDS_V12"] is None,"No memo installed for invalid initial runtime")
            if order=="prewrapped_initial":v4.canonical=saved
            return
        if order=="ROOT_prepare":
            require(g["fresh_guard_runtime_v11"](consumer=False) is freshguard,"Initial ROOT factory")
            require(g["fresh_guard_runtime_v11"](consumer=False) is freshguard,"Repeated ROOT factory reuse")
            require(all(getattr(v4,n).__module__==v4.__name__ for n in ("canonical","file_bytes","mutex")),"ROOT remains unwrapped")
            checks.append(dict(check=order+"_initial_repeated_mutation_factory_unwrapped",passed=True))
        bridge=g["guard_v1"]({})
        require(bridge.retained is oldguard and bridge.fresh is freshguard,"Exact dual guard objects")
        handle=g["_CONSUMER_GUARDS_V12"][2]
        require(handle.receipt["installed"] is True and handle.active_count==0 and handle._state() is None,"One memo installed idle")
        bridge.validate_storage_v9(spec,dict(output_cap_bytes=5*2**30),newstorage,
            source_ref={"source":"new"},config_ref={"config":"old"},runtime_ref={"runtime":"old"})
        bridge.budget_check_v9(newstorage,boundary=True,required_write_bytes=123)
        before=len(operations);g["charged_census_v1"](spec,required_write_bytes=0)
        require(len(operations)==before+1 and operations[-1]["storage"]=="old"
            and operations[-1]["kwargs"]["required_write_bytes"]==0 and operations[-1]["kwargs"]["boundary"] is True,
            "Census retained guard still checks at every call without factory reconstruction")
        require(g["guard_v1"]({}).retained is oldguard and g["fresh_guard_runtime_v11"](consumer=True) is freshguard,
            "Repeated guard/fresh accessor reuses authenticated pair")
        bridge.budget_check_v9(newstorage,boundary=True,required_write_bytes=456)
        require(len([e for e in events if e.get("guard")=="old"])==1
            and len([e for e in events if e.get("guard")=="fresh"])==1,"Exactly one configuration per guard")
        require(all(x["kwargs"].get("required_write_bytes",0)==0 for x in operations if x["storage"]=="old"),
            "All retained scope operations have zero required write charge")
        require(any(x["kwargs"].get("required_write_bytes")==456 for x in operations if x["storage"]=="fresh"),
            "Fresh prospective write check still executes")
        require(handle._state() is None and handle.active_count==0
            and handle.receipt["cross_operation_cache"] is False and handle.last_operation_counts()["raw_hits"]>=1 and handle.last_operation_counts()["canonical_hits"]>=1,
            "Real memo state cleared after every actual synthetic operation")
        payload.write_text('{"value":2}')
        result=bridge.budget_check_v9(newstorage,boundary=True,required_write_bytes=0)
        require(result["payload"]=='{"value":2}',"Next operation rereads changed tiny metadata; no cross-operation cache")
        checks.append(dict(check=order+"_real_factory_and_memo_initial_repeated_validate_check_census_worker_order",passed=True,
            operations=len(operations),configurations=events,shared_memo_installations=1,namespace_and_result_cache_cross_operation=False))
        rejects(order+"_mutation_factory_after_consumer_install",lambda:g["fresh_guard_runtime_v11"](consumer=False),checks)
        saved=v4.canonical;v4.canonical=lambda *a:None
        rejects(order+"_cached_wrapper_identity_drift",g["_consumer_guards_v12"],checks);v4.canonical=saved
        saved_parent=v4.scope_spec;v4.scope_spec=lambda *a:None
        rejects(order+"_cached_namespace_parent_drift",g["_consumer_guards_v12"],checks);v4.scope_spec=saved_parent
        oldq["capacity_policy_ref"]={"changed":True}
        rejects(order+"_original_request_swap",g["_consumer_guards_v12"],checks);oldq["capacity_policy_ref"]=on["POLICY"]
        handle.uninstall()
        require(all(getattr(v4,n).__module__==v4.__name__ for n in ("canonical","file_bytes","mutex")),"Synthetic lifecycle restores originals")
        checks.append(dict(check=order+"_synthetic_memo_lifecycle_clean",passed=True))
def run():
    checks=[]
    for order in ("ROOT_prepare","execute_initial","worker_entry","prewrapped_initial","nonshared_initial","bad_policy_initial"):scenario(order,checks)
    require(not any(n in sys.modules for n in ("torch","numpy","composer","torch_geometric")),"No scientific payload/runtime imports")
    return dict(schema="native12_real_dual_guard_factory_memo_source_only_fixture_v1",status="PASS",checks=checks,
        check_count=len(checks),real_pinned_original_and_fresh_factory_code_used=True,real_pinned_namespace_and_transaction_memo_used=True,
        all_guard_modules_and_budget_modules_synthetic=True,all_operation_paths_within_owned_tiny_temporary_fixture=True,
        actual_guard_policy_read_scope_census_scheduler_lock_process_GPU_checkpoint_accounting_admission_job_or_Git=False,
        candidate_contract_ref=ref(CAND),source_refs=[ref(p) for p in (OLD,V4,V9,NAMESPACE,
            "research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py")])
if __name__=="__main__":print(json.dumps(run(),sort_keys=True,indent=2))
