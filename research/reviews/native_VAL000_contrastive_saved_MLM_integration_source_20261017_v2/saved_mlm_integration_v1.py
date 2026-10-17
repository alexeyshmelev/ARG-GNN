"""Inactive scientific integration; no runtime or payload access at import.
Only an independently reviewed/admitted existing ROOT context may call stages.
Original saved-MLM bridge and legacy full-fit qualifier stay immutable.
"""
from pathlib import Path
from datetime import timedelta
import ast,copy,hashlib,importlib.util,json,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/native_VAL000_contrastive_saved_MLM_integration_source_20261017_v2"
SOURCE=HERE+"/saved_mlm_integration_v1.py"
PINS={'bridge': {'path': 'research/reviews/native_VAL000_saved_MLM_acceptance_bridge_source_20261017_v1/saved_mlm_bridge_v1.py', 'sha256': '5810041797592fa414d57747607f646a3eee1050d0373df5412b7dd4b1b63d97'}, 'bridge_handoff': {'path': 'research/reviews/native_VAL000_saved_MLM_acceptance_bridge_source_20261017_v1/source_handoff.v1.json', 'sha256': 'd60c2f5b760f8d4ab7b735084229513fbdce2289654fb6d3c94c175562006e33'}, 'bridge_review': {'path': 'research/reviews/native_VAL000_saved_MLM_bridge_independent_20261017_v1/review.v1.json', 'sha256': '85e2d75ea0ccc83aea290700b2c445094900ab2063545dc119fa3bedc7114c60'}, 'stages': {'path': 'research/reviews/native_VAL000_downstream_scientific_stages_source_20261016_v1/scientific_stages_v1.py', 'sha256': '455704b88e9247021c09566609801838f88a118420cde856e487aaef4cee30da'}, 'io': {'path': 'research/reviews/native_VAL000_downstream_IO_efficiency_source_20261016_v1/efficient_downstream_v1.py', 'sha256': 'bab7057a958e9cfd51f503b5680034c0067226a661b820eb42bb16c7133a94b9'}, 'helper': {'path': 'research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/native_child_v1.py', 'sha256': 'f2770222f02979b6064a7f1ffea80953cd234114c96ddde0845372479292379e'}, 'legacy_contract': {'path': 'src/arggnn/american_c3_native_contract_v1.py', 'sha256': 'b14d7293b93904a9f23276b69ee66288e8b943285fe500eb22755aa288168212'}, 'legacy_run': {'path': 'src/arggnn/american_c3_native_run_v1.py', 'sha256': '65653b2522978d20341e35ab9e36610a75024a29931026961a5bc4b759fab2cc'}, 'nested_reference': {'path': 'research/reviews/native_VAL000_full_state_resume_source_20261016_v12/resume_scientific_bridge_v5.py', 'sha256': 'c52af58d30df80ca3888fac7f94c993847e35d03d0f5a6f7f0adc9467ca19ea4'}, 'original_downstream': {'path': 'research/reviews/native_VAL000_downstream_execution_source_20261016_v2/source_handoff.v2.json', 'sha256': '16931e9fe3663ac3460eb8cf9a33b35843fc65ddbc4dac9673539c3465db1553'}, 'current_selector': {'path': 'research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json', 'sha256': 'e27d47a22d0da1b2e8f0bdf0818cb2c6c473d8c9ddb5424618bf97e704928642'}}
_BRIDGE=None


def require(ok,message):
    if not ok:raise ValueError(message)

def _bytes(record,*,metadata=False):
    require(type(record) is dict and set(record)=={"path","sha256"},"Exact bounded source/metadata ref")
    name=record["path"];p=ROOT/name
    require(type(name) is str and not Path(name).is_absolute() and ".." not in Path(name).parts
        and p.resolve()==p and p.is_relative_to(ROOT) and p.is_file() and not p.is_symlink()
        and p.stat().st_size<=2**20 and (not metadata or p.suffix==".json"),"Only bounded canonical source or JSON metadata")
    raw=p.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==record["sha256"],"Exact source/metadata bytes")
    return raw

def own_ref_v1():
    p=ROOT/SOURCE
    return dict(path=SOURCE,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def load_source_v1(role):
    r=PINS[role];raw=_bytes(r)
    name="_saved_MLM_integration_v2_"+role
    spec=importlib.util.spec_from_file_location(name,ROOT/r["path"])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    exec(compile(raw,str(ROOT/r["path"]),"exec"),module.__dict__)
    return module

def bridge_v1():
    global _BRIDGE
    if _BRIDGE is None:
        review=json.loads(_bytes(PINS["bridge_review"],metadata=True))
        require(review["verdict"]=="SOURCE_PASS" and review["blocking_findings"]==[]
            and review["source_ref"]==PINS["bridge"],"Exact independently accepted original artifact bridge")
        _BRIDGE=load_source_v1("bridge")
    return _BRIDGE

def bind_stage_v1(ctx,request):
    require(ctx.get("storage_policy_selector_ref")==PINS["current_selector"]
        and request.get("storage_policy_selector_ref")==PINS["current_selector"]
        and ctx.get("saved_MLM_integration_source_ref")==own_ref_v1()
        and request.get("saved_MLM_integration_source_ref")==own_ref_v1(),
        "Current Relate64 selector and independently bound scientific integration source")

def saved_acceptance_v1(ctx,request):
    bridge=bridge_v1();eligible=bridge.read_eligibility_v1()
    record=request["ROOT_saved_MLM_artifact_acceptance_ref"]
    require(ctx.get("ROOT_saved_MLM_artifact_acceptance_ref")==record,"Same independently bound ROOT artifact acceptance")
    value=json.loads(_bytes(record,metadata=True))
    require(value.get("schema")=="ROOT_native_VAL000_saved_MLM_artifact_acceptance_v1"
        and value.get("root_authorizer")=="root_agent"
        and value.get("saved_MLM_checkpoint_ref")==bridge.CHECKPOINT_REF
        and value.get("saved_MLM_eligibility")==eligible.record()
        and value.get("historical_MLM_outer_exit_code")==1
        and value.get("contrastive_only_artifact_use_accepted") is True
        and value.get("MLM_replay_authorized") is False
        and value.get("launch_or_admission_authority") is False,
        "Separate ROOT saved-artifact acceptance, original outer1 and no launch/MLM replay authority")
    return bridge,eligible,bridge.CHECKPOINT_REF

def validated_composer_timestamp_v1(state):
    outer=state.get("timestamp")
    require(type(outer) is dict and set(outer)=={"Timestamp"},"Exact installed Composer nested Timestamp serialization")
    value=outer["Timestamp"]
    counters={"iteration","epoch","batch","sample","token","epoch_in_iteration","token_in_iteration","batch_in_epoch","sample_in_epoch","token_in_epoch"}
    walls={"total_wct","iteration_wct","epoch_wct","batch_wct"}
    require(type(value) is dict and set(value)==counters|walls
        and all(type(value[k]) is int and value[k]>=0 for k in counters)
        and all(isinstance(value[k],timedelta) and value[k].total_seconds()>=0 for k in walls),
        "Exact actual native timestamp keys, nonnegative integer counters and timedeltas")
    return value

def checkpoint_metadata_bridge_v1(helper):
    source=_bytes(PINS["helper"]).decode();tree=ast.parse(source)
    fn=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_final_checkpoint"))
    matches=[n for n in ast.walk(fn) if isinstance(n,ast.Assign) and len(n.targets)==1
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="timestamp"]
    expected=ast.parse('timestamp=state.get("timestamp",{})').body[0]
    require(len(matches)==1 and ast.dump(matches[0],include_attributes=False)==ast.dump(expected,include_attributes=False),
        "Only the existing final-reader timestamp lookup changes")
    matches[0].value=ast.Call(func=ast.Name(id="_actual_nested_timestamp_v1",ctx=ast.Load()),
        args=[ast.Name(id="state",ctx=ast.Load())],keywords=[])
    ns=dict(helper._final_checkpoint.__globals__,_actual_nested_timestamp_v1=validated_composer_timestamp_v1)
    ast.fix_missing_locations(fn)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(ROOT/SOURCE)+":nested_timestamp","exec"),ns)
    adapted=ns["_final_checkpoint"]
    def checked(*args,**kwargs):
        result=adapted(*args,**kwargs)
        return dict(result,actual_checkpoint_timestamp_schema="state.timestamp.Timestamp",
            checkpoint_metadata_parser_bridge_source_ref=own_ref_v1(),
            original_checkpoint_model_optimizer_and_selection_checks_preserved=True)
    return checked

def keep1_bridge_function_v1(bridge):
    """Private original function clone: sole expected-config difference is retention."""
    source=_bytes(PINS["bridge"]).decode()
    fn=copy.deepcopy(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)
        and n.name=="run_original_contrastive_train_v1"))
    anchor=ast.parse("expected.autoresume=False").body[0]
    indexes=[i for i,n in enumerate(fn.body) if ast.dump(n,include_attributes=False)==ast.dump(anchor,include_attributes=False)]
    require(len(indexes)==1,"Single authenticated original operational config anchor")
    fn.body.insert(indexes[0]+1,ast.parse("expected.save_num_checkpoints_to_keep=1").body[0])
    ns=dict(bridge.__dict__);ast.fix_missing_locations(fn)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(ROOT/SOURCE)+":explicit_keep1_only","exec"),ns)
    return ns["run_original_contrastive_train_v1"]

def run_original_keep1_contrastive_train_v1(cfg,eligibility,*,author_train):
    bridge=bridge_v1()
    result=keep1_bridge_function_v1(bridge)(cfg,eligibility,author_train=author_train)
    return dict(result,original_saved_MLM_bridge_source_ref=PINS["bridge"],
        keep1_compatibility_source_ref=own_ref_v1(),
        operational_checkpoint_retention=dict(original=-1,actual=1,save_interval="20ba"),
        all_scientific_configuration_fields_unchanged=True,
        no_extra_checkpoint_load_or_forward_or_optimizer_probe=True)

def _document_v1(bridge,record):
    return bridge.document_from_bytes_v1(record,_bytes(record,metadata=True))

def qualify_saved_MLM_for_embeddings_v1(*,initialization_ref,contrastive_completion_ref,contrastive_wait_ref,consumed_outer_ref):
    bridge=bridge_v1();eligible=bridge.read_eligibility_v1()
    init=_document_v1(bridge,initialization_ref)
    require(init.value.get("original_saved_MLM_bridge_source_ref")==PINS["bridge"]
        and init.value.get("keep1_compatibility_source_ref")==own_ref_v1()
        and init.value.get("operational_checkpoint_retention")==dict(original=-1,actual=1,save_interval="20ba")
        and init.value.get("all_scientific_configuration_fields_unchanged") is True
        and init.value.get("no_extra_checkpoint_load_or_forward_or_optimizer_probe") is True,
        "Explicit separately reviewed keep1 compatibility evidence")
    return bridge.qualify_for_native_embeddings_v1(eligible,initialization=init,
        contrastive_completion=_document_v1(bridge,contrastive_completion_ref),
        contrastive_wait=_document_v1(bridge,contrastive_wait_ref),
        consumed_outer=_document_v1(bridge,consumed_outer_ref))

def fit_completion_dispatch_v1(record,spec):
    selected=spec["refs"].get("saved_MLM_qualification")
    if selected is None:
        _bytes(PINS["legacy_contract"])
        from arggnn.american_c3_native_contract_v1 import fit_completion_v1
        return fit_completion_v1(record,spec)
    require(record==selected==spec["refs"]["fit_completion"],"Explicit separate qualification dispatch; no legacy wrapper claim")
    stored=json.loads(_bytes(record,metadata=True))
    require(stored.get("schema")=="native_VAL000_saved_MLM_then_original_contrastive_qualification_v1"
        and spec["parent_id"]==stored.get("parent_id")=="american_c3_val_000"
        and spec["refs"]["preparation"]==stored.get("preparation_ref")
        and spec["refs"]["counts"]==stored.get("counts_ref"),"Exact same-parent separate saved-MLM qualification")
    phase=stored["phases"]["contrastive"]
    actual=qualify_saved_MLM_for_embeddings_v1(initialization_ref=stored["initialization_ref"],
        contrastive_completion_ref=phase["child_completion_ref"],contrastive_wait_ref=phase["actual_wait_ref"],
        consumed_outer_ref=phase["consumed_outer_ref"])
    require(stored==actual,"Requalify actual new contrastive initialization/completion/wait/drain/consumedouter0")
    return actual

def saved_embedding_checkpoint_v1(ctx,request):
    selected=request["saved_MLM_qualification_ref"];spec=ctx["saved_MLM_qualification_spec"]
    require(spec["refs"].get("saved_MLM_qualification")==selected,"Same request/context separate fit qualification")
    return fit_completion_dispatch_v1(selected,spec)["phases"]["contrastive"]["checkpoint_ref"]

def embedding_v1(ctx,request):
    """Original complete single-split embedding body, separate qualifier dispatch only."""
    bind_stage_v1(ctx,request);base=load_source_v1("stages")
    if request.get("saved_MLM_qualification_ref") is None:
        return base.embedding_v1(ctx,request)
    source=_bytes(PINS["stages"]).decode()
    fn=copy.deepcopy(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="embedding_v1"))
    indexes=[i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id in ("mlm_checkpoint","checkpoint") for t in n.targets)]
    require(len(indexes)==2 and indexes[1]==indexes[0]+1,"Only adjacent original prerequisite dispatch statements change")
    expected=ast.parse('mlm_checkpoint=full_mlm_metadata_prerequisite_v1(**ctx["MLM_prerequisite_metadata"])\ncheckpoint=full_contrastive_metadata_prerequisite_v1(initializer_checkpoint_ref=mlm_checkpoint,**ctx["contrastive_prerequisite_metadata"])').body
    require(all(ast.dump(fn.body[i],include_attributes=False)==ast.dump(n,include_attributes=False)
        for i,n in zip(indexes,expected)),"Exact original scientific prerequisite boundary")
    fn.body[indexes[0]:indexes[1]+1]=ast.parse("checkpoint=_saved_checkpoint_v1(ctx,request)").body
    ns=dict(base.__dict__,_saved_checkpoint_v1=saved_embedding_checkpoint_v1);ast.fix_missing_locations(fn)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(ROOT/SOURCE)+":separate_embedding_qualifier","exec"),ns)
    return ns["embedding_v1"](ctx,request)

def saved_RAM_retrieval_function_v2(io,checkpoint):
    """Private original retrieval clone: only its two prerequisite assignments change."""
    source=_bytes(PINS["io"]).decode()
    fn=copy.deepcopy(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)
        and n.name=="RAM_retrieval_and_seal_v1"))
    indexes=[i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id in ("mlm","contrastive") for t in n.targets)]
    expected=ast.parse('mlm=base.full_mlm_metadata_prerequisite_v1(**ctx["MLM_prerequisite_metadata"])\ncontrastive=base.full_contrastive_metadata_prerequisite_v1(initializer_checkpoint_ref=mlm,**ctx["contrastive_prerequisite_metadata"])').body
    require(len(indexes)==2 and indexes[1]==indexes[0]+1
        and all(ast.dump(fn.body[i],include_attributes=False)==ast.dump(n,include_attributes=False)
            for i,n in zip(indexes,expected)),"Only the two exact adjacent original retrieval prerequisites change")
    fn.body[indexes[0]:indexes[1]+1]=ast.parse("contrastive=_qualified_contrastive_checkpoint_v2").body
    ns=dict(io.RAM_retrieval_and_seal_v1.__globals__,_qualified_contrastive_checkpoint_v2=checkpoint)
    ast.fix_missing_locations(fn)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(ROOT/SOURCE)+":separate_retrieval_qualifier","exec"),ns)
    return ns["RAM_retrieval_and_seal_v1"]

def RAM_retrieval_and_seal_v1(ctx,request,spec,cfg):
    """Original full RAM concat/retrieval/seal; explicit separately qualified checkpoint."""
    bind_stage_v1(ctx,request);io=load_source_v1("io")
    if spec["refs"].get("saved_MLM_qualification") is None:
        return io.RAM_retrieval_and_seal_v1(ctx,request,spec,cfg)
    qualified=fit_completion_dispatch_v1(spec["refs"]["fit_completion"],spec)
    checkpoint=qualified["phases"]["contrastive"]["checkpoint_ref"]
    ctx=dict(ctx,validated_VAL000_fit_completion_v1=fit_completion_dispatch_v1)
    return saved_RAM_retrieval_function_v2(io,checkpoint)(ctx,request,spec,cfg)
