"""Independent source-only fake checks; no project/scientific imports or runtime probes."""
from pathlib import Path
from datetime import timedelta
import ast,hashlib,json,re,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
CAND=ROOT/"research/reviews/native_VAL000_downstream_execution_source_20261016_v1"
def require(ok,message):
    if not ok:raise ValueError(message)
def load_ast(path):return ast.parse(path.read_text())
def top_function(tree,name):
    return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
def run():
    worker=load_ast(CAND/"worker_context_v1.py")
    context=next(n for n in worker.body if isinstance(n,ast.ClassDef) and n.name=="NativeContext")
    expression=next(n for n in ast.walk(worker) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="bootstrap_native_execution_v1").args[0]
    bootstrap=load_ast(ROOT/"research/reviews/American_C3_native_MLM_amendment_v12/abstract_unix_bootstrap_v1.py")
    def forbidden(*a,**k):raise AssertionError("Unexpected runtime use")
    bns=dict(_ORIGINAL=None,sys=types.SimpleNamespace(platform="linux"),re=re,mp=types.SimpleNamespace(get_start_method=forbidden))
    exec(compile(ast.Module(body=[top_function(bootstrap,"install_abstract_unix_v1")],type_ignores=[]),"<source-bootstrap-function-with-fake-runtime>","exec"),bns)
    phases=json.loads((CAND/"scientific_config.FROZEN.v1.json").read_text())["allowed_phases"]
    namespace=[]
    for phase in phases:
        if phase=="finalize":continue
        name=eval(compile(ast.Expression(expression),"<actual-namespace-argument>","eval"),{"self":types.SimpleNamespace(s={"phase":phase})})
        accepted=re.fullmatch(r"[A-Za-z0-9_-]{1,32}",name) is not None
        row=dict(phase=phase,namespace=name,length=len(name),regex_accepts=accepted)
        if not accepted:
            try:bns["install_abstract_unix_v1"](name)
            except ValueError as error:row["exact_function_error"]=str(error)
            else:raise AssertionError("Expected original bootstrap rejection")
        namespace.append(row)
    assert sum(not r["regex_accepts"] for r in namespace)==5
    # Execute the installed serialization methods with entirely fake state/time.
    croot=ROOT/".runtime/conda/envs/argformer_native_v1/lib/python3.11/site-packages/composer/core"
    stree=load_ast(croot/"state.py");ttree=load_ast(croot/"time.py")
    stateclass=next(n for n in stree.body if isinstance(n,ast.ClassDef) and n.name=="State")
    timeclass=next(n for n in ttree.body if isinstance(n,ast.ClassDef) and n.name=="Timestamp")
    smethod=next(n for n in stateclass.body if isinstance(n,ast.FunctionDef) and n.name=="state_dict")
    tmethod=next(n for n in timeclass.body if isinstance(n,ast.FunctionDef) and n.name=="state_dict")
    serial=ast.literal_eval(next(n.value for n in stree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="_STATE_DICT_SERIALIZED_ATTRIBUTES"))
    future=ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0)
    tns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[future,tmethod],type_ignores=[])),"<installed-Timestamp-state-dict-with-fake-state>","exec"),tns)
    keys=next(n for n in tmethod.body if isinstance(n,ast.Return)).value.keys
    actual_keys={ast.literal_eval(k) for k in keys}
    Timestamp=type("Timestamp",(),{"state_dict":tns["state_dict"]})
    timestamp=Timestamp()
    for k in actual_keys:setattr(timestamp,k,timedelta(0) if k.endswith("_wct") else types.SimpleNamespace(value=0))
    sns=dict(_STATE_DICT_SERIALIZED_ATTRIBUTES=serial,ensure_tuple=lambda x:(x,))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,smethod],type_ignores=[])),"<installed-State-state-dict-with-fake-state>","exec"),sns)
    fake=types.SimpleNamespace(serialized_attributes=["timestamp"],timestamp=timestamp,_get_integrations_state_dict=lambda:{},_get_state_metadata=lambda:{})
    serialized=sns["state_dict"](fake)
    assert set(serialized["timestamp"])=={"Timestamp"} and set(serialized["timestamp"]["Timestamp"])==actual_keys
    vns=dict(c=types.SimpleNamespace(require=require),timedelta=timedelta)
    exec(compile(ast.Module(body=[top_function(worker,"validated_composer_timestamp_v1")],type_ignores=[]),"<candidate-Timestamp-reader-only>","exec"),vns)
    assert vns["validated_composer_timestamp_v1"](serialized)==serialized["timestamp"]["Timestamp"]
    # Check the actual method's request/context key and signature compatibility.
    def record(name):return dict(path="SYNTHETIC_ONLY/"+name,sha256=hashlib.sha256(name.encode()).hexdigest())
    c=types.SimpleNamespace(require=require,active_amendment_v1=lambda:None,ref=record,CONTEXT="context",IO_SOURCE_REF=record("io_source"),IO_POLICY_REF=record("io_policy"),PARENT="american_c3_val_000")
    cns=dict(c=c,stable_reference_v1=record)
    method=next(n for n in context.body if isinstance(n,ast.FunctionDef) and n.name=="context_and_request")
    exec(compile(ast.Module(body=[method],type_ignores=[]),"<actual-context-request-with-fakes>","exec"),cns)
    context_cases=[]
    callback_names=["fast_guard","prepare_native","authenticate_inputs","preparation_callback","counts_callback","fit_callback","metadata_callback"]
    for phase in phases[:-1]:
        own=types.SimpleNamespace(s={"phase":phase},storage={"reservation":{"root_admission":record("admission")}},writer=types.SimpleNamespace(path=Path("/SYNTHETIC_ONLY/output"),save=lambda *a,**k:None,boundary=lambda *a,**k:None),input_ref=record("input"),MLM={"SYNTHETIC_ONLY":True},contrastive=None if phase=="contrastive" else {"SYNTHETIC_ONLY":True},prep={"SYNTHETIC_ONLY":True},verify_imports=lambda:None,q={})
        for name in callback_names:setattr(own,name,lambda *a,**k:None)
        if phase.startswith("embedding_"):
            for view in ("contrastive","query_paths"):
                for split in ("train","val"):
                    if phase=="embedding_"+view+"_"+split:own.q.update(view=view,split=split)
        ctx,req=cns["context_and_request"](own)
        expected="embedding" if phase.startswith("embedding_") else phase
        assert req["phase"]==expected and ctx["context_source_ref"]==req["ROOT_context_source_ref"]
        assert ctx["operational_IO_source_ref"]==req["operational_IO_source_ref"]
        assert ctx["operational_IO_policy_ref"]==req["operational_IO_policy_ref"]
        assert all(callable(ctx[k]) for k in ("validate_phase_authority_v1","fast_guard","reference","prepare_native_runtime_v1","verify_native_imports_v1","authenticate_native_inputs_once_v1","validated_VAL000_preparation_v1","validated_VAL000_counts_v1","validated_VAL000_fit_completion_v1","validated_VAL000_metadata"))
        ctx["validate_phase_authority_v1"](req)
        bad=dict(req,phase="wrong")
        try:ctx["validate_phase_authority_v1"](bad)
        except ValueError:pass
        else:raise AssertionError("Request mismatch accepted")
        if phase.startswith("embedding_"):assert (req["view"],req["split"])==(own.q["view"],own.q["split"])
        context_cases.append(dict(phase=phase,scientific_phase=expected,all_required_callbacks_callable=True,wrong_request_rejected=True))
    return dict(schema="independent_downstream_fake_source_checks_v1",namespace_cases=namespace,installed_State_Timestamp_nested_schema_accepted=True,installed_Timestamp_keys=sorted(actual_keys),actual_context_request_cases=context_cases,source_only=True,project_or_scientific_imports=False,real_payload_reads=False,device_process_or_accounting_probes=False)
if __name__=="__main__":print(json.dumps(run(),sort_keys=True,indent=2))
