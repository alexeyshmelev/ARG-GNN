"""Only source/metadata and RAM fakes. Never import Torch, guard or runtime."""
from pathlib import Path
from datetime import timedelta
from types import SimpleNamespace
import ast,copy,hashlib,importlib.util,json,stat,sys,types
D=Path("research/reviews")/Path(__file__).parent.name
checks=[]
def check(name,ok=True):
    assert ok,name
    checks.append(name)
def rejects(name,fn):
    try:fn()
    except (ValueError,KeyError,TypeError,AttributeError):check(name)
    else:raise AssertionError(name+" accepted")
def load(p,name):
    sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def fn_ast(path,name):
    return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)
def dump(node):return ast.dump(node,include_attributes=False)
def ref(p):
    p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
m=load(D/"saved_mlm_integration_v1.py","_saved_integration_RAM")
f=load(Path(m.PINS["bridge"]["path"]).parent/"ram_fixture_v1.py","_accepted_original_bridge_fixture_RAM")
check("accepted_bridge51_RAM_checks_reproduced",f.run()["count"]==51)
b=f.b;m._BRIDGE=b
# Strict exact nested timestamp view and unmodified final model/optimizer/selection logic.
counters={"iteration","epoch","batch","sample","token","epoch_in_iteration","token_in_iteration","batch_in_epoch","sample_in_epoch","token_in_epoch"}
walls={"total_wct","iteration_wct","epoch_wct","batch_wct"}
timestamp={k:0 for k in counters}|{k:timedelta(0)for k in walls}
timestamp.update(epoch=1,batch=2135)
state=dict(timestamp={"Timestamp":timestamp},model={"RAM_weight":"RAM_ONLY"},
    optimizers={"DecoupledAdamW":{"state":{0:dict(step=2135,exp_avg="RAM",exp_avg_sq="RAM")}}})
check("exact_nested_timestamp_accepted",m.validated_composer_timestamp_v1(state)==timestamp)
for key,value in [("epoch",True),("batch",-1),("total_wct",0),("epoch_wct",timedelta(seconds=-1))]:
    bad=copy.deepcopy(state);bad["timestamp"]["Timestamp"][key]=value
    rejects("bad_timestamp_"+key,lambda bad=bad:m.validated_composer_timestamp_v1(bad))
rejects("flat_timestamp_rejected",lambda:m.validated_composer_timestamp_v1(dict(timestamp=timestamp)))
bad=copy.deepcopy(state);bad["timestamp"]["Timestamp"]["extra"]=1
rejects("unknown_timestamp_key_rejected",lambda:m.validated_composer_timestamp_v1(bad))
ns=dict(Path=Path,os=SimpleNamespace(),stat=stat)
for name in ("_int_time","optimizer_counter_evidence","_final_checkpoint"):
    exec(compile(ast.Module(body=[copy.deepcopy(fn_ast(m.PINS["helper"]["path"],name))],type_ignores=[]),"<RAM original helper>","exec"),ns)
class P:
    def __init__(self,name="final.pt",size=123):self.name,self.size=name,size
    def lstat(self):return SimpleNamespace(st_size=self.size)
paths=[P()];alias=None;payloads={"final.pt":state}
ns["_checkpoint_entries"]=lambda directory:(paths,alias)
helper=SimpleNamespace(_final_checkpoint=ns["_final_checkpoint"])
adapted=m.checkpoint_metadata_bridge_v1(helper)
fake_torch=SimpleNamespace(load=lambda path,**kw:dict(state=copy.deepcopy(payloads[path.name])))
def reference(path):return dict(path="RAM/"+path.name,sha256=("1"if path.name=="final.pt"else"2")*64)
def final(**kw):
    args=dict(epochs=1,updates=2135,max_files=2,max_bytes=1000,torch=fake_torch,reference=reference);args.update(kw)
    return adapted("RAM_directory",**args)
check("original_final_reader_accepts_exact_nested_full2135",final()["actual_optimizer_updates"]==2135)
check("original_helper_code_unchanged",helper._final_checkpoint.__code__ is ns["_final_checkpoint"].__code__)
for key,value in [("model",{}),("optimizers",{}),("timestamp",{"Timestamp":dict(timestamp,batch=2134)})]:
    original=payloads["final.pt"];payloads["final.pt"]=dict(state,**{key:value})
    rejects("final_reader_rejects_"+key,final);payloads["final.pt"]=original
payloads["final.pt"]=copy.deepcopy(state);payloads["final.pt"]["optimizers"]["DecoupledAdamW"]["state"][0]["step"]=2134
rejects("final_reader_rejects_wrong_Adam_counter",final);payloads["final.pt"]=state
rejects("final_reader_file_cap_preserved",lambda:final(max_files=0))
rejects("final_reader_byte_cap_preserved",lambda:final(max_bytes=1))
paths=[P(),P("other.pt")];payloads["other.pt"]=state
rejects("final_reader_ambiguous_final_hash_rejected",final);paths=[P()]
alias=dict(target_text="other.pt")
rejects("final_reader_alias_must_name_verified_final",final);alias=dict(target_text="final.pt")
check("final_reader_matching_alias_accepted",final()["checkpoint_latest_alias"]==alias);alias=None
original_timestamp=fn_ast(m.PINS["nested_reference"]["path"],"validated_composer_timestamp_v1")
class CNormalize(ast.NodeTransformer):
    def visit_Attribute(self,node):
        if isinstance(node.value,ast.Name)and node.value.id=="c"and node.attr=="require":return ast.copy_location(ast.Name(id="require",ctx=ast.Load()),node)
        return self.generic_visit(node)
check("nested_timestamp_validator_exact_v12_AST",dump(CNormalize().visit(original_timestamp))==dump(fn_ast(D/"saved_mlm_integration_v1.py","validated_composer_timestamp_v1")))
# Exact clone of accepted wrapper: one insertion in expected operational retention.
clone=m.keep1_bridge_function_v1(b)
original=b.run_original_contrastive_train_v1
expected=copy.deepcopy(fn_ast(m.PINS["bridge"]["path"],"run_original_contrastive_train_v1"))
anchor=ast.parse("expected.autoresume=False").body[0]
index=next(i for i,n in enumerate(expected.body)if dump(n)==dump(anchor))
expected.body.insert(index+1,ast.parse("expected.save_num_checkpoints_to_keep=1").body[0]);ast.fix_missing_locations(expected)
en={};exec(compile(ast.Module(body=[expected],type_ignores=[]),"<expected sole retention clone>","exec"),en)
check("private_keep1_clone_is_exact_single_retention_insertion",clone.__code__.co_code==en[original.__name__].__code__.co_code)
class Dot(dict):
    def __getattr__(self,k):
        if k not in self:raise AttributeError(k)
        return self[k]
    def __setattr__(self,k,v):self[k]=v
def dot(v):return Dot({k:dot(x)for k,x in v.items()})if isinstance(v,dict)else v
base=dict(save_num_checkpoints_to_keep=-1,save_interval="20ba",data_local="ORIGINAL",
    run_name="ORIGINAL",save_folder="RAM_OUTPUT",load_path=None,autoresume=False,loggers={},
    device="gpu",model=dict(pretrained_model_name="ORIGINAL",pretrained_checkpoint="ORIGINAL",
    contrastive=dict(temperature=.30)),max_duration="1ep",global_train_batch_size=4096,
    device_train_microbatch_size=4096,precision="fp32",seed=36,learning_rate=.00001)
OM=SimpleNamespace(load=lambda p:dot(copy.deepcopy(base)),to_container=lambda cfg,resolve=True:copy.deepcopy(dict(cfg)))
class K:
    @classmethod
    def from_composer(cls):pass
flex=SimpleNamespace(create_flex_bert_classification=lambda:None,bert_layers_module=SimpleNamespace(FlexBertForSequenceClassification=K))
build=types.FunctionType((lambda:None).__code__,dict(flex_bert_module=flex))
events=[]
author=types.FunctionType((lambda cfg:None).__code__,dict(om=OM,build_contrastive_model=build))
backup={k:getattr(b,k)for k in ("_authenticate_function","_read_small","_instrument_training")}
def instrument(train,loader,vocabulary):
    check("keep1_preserves_exact_FD_loader_and_tokenizer_verifier",loader is b._load_once_from_exact_FD and vocabulary is b._runtime_reference_vocabulary)
    return lambda cfg:events.append("original_complete_train_RAM_stub"),dict(builder_calls=1,checkpoint_loads=1,verified={"RAM":"verified"})
b._authenticate_function=lambda *args:None;b._read_small=lambda *args:b"RAM_SOURCE"
b._instrument_training=instrument
cfg=copy.deepcopy(base)
cfg.update(save_num_checkpoints_to_keep=1,data_local=str(b.ROOT/Path(b.PREPARATION_REF["path"]).parent/"contrastive"),
    run_name="C3_native_contrastive",save_folder="RAM_OUTPUT",load_path=None,autoresume=False,loggers={},device="gpu")
cfg["model"].update(pretrained_model_name=str(b.ROOT/"research/reviews/C3_bert_base_configuration_root_v1"),pretrained_checkpoint=str(b.ROOT/b.CHECKPOINT_REF["path"]))
code_before=original.__code__
try:
    record=m.run_original_keep1_contrastive_train_v1(dot(copy.deepcopy(cfg)),f.eligibility,author_train=author)
    check("keep1_wrapper_accepts_only_disclosed_retention1",record["operational_checkpoint_retention"]==dict(original=-1,actual=1,save_interval="20ba"))
    check("keep1_wrapper_original_function_not_mutated",original.__code__ is code_before and len(events)==1)
    rejects("keep1_route_rejects_original_minus1_config",lambda:m.run_original_keep1_contrastive_train_v1(dot(dict(cfg,save_num_checkpoints_to_keep=-1)),f.eligibility,author_train=author))
    for key,value in [("seed",37),("global_train_batch_size",2048),("device_train_microbatch_size",2048),("max_duration","2ep"),("precision","amp_bf16"),("learning_rate",.02),("save_interval","100ba")]:
        rejects("keep1_rejects_changed_"+key,lambda key=key,value=value:m.run_original_keep1_contrastive_train_v1(dot(dict(cfg,**{key:value})),f.eligibility,author_train=author))
    original(dot(dict(cfg,save_num_checkpoints_to_keep=-1)),f.eligibility,author_train=author)
    check("original_all_retention_route_still_accepts_original_config")
finally:
    for key,value in backup.items():setattr(b,key,value)
# Separate real-document qualifier dispatch, with RAM-only document bytes.
b.read_eligibility_v1=lambda:f.eligibility
f.initialization_record.update(original_saved_MLM_bridge_source_ref=m.PINS["bridge"],keep1_compatibility_source_ref=m.own_ref_v1(),
    operational_checkpoint_retention=dict(original=-1,actual=1,save_interval="20ba"),
    all_scientific_configuration_fields_unchanged=True,no_extra_checkpoint_load_or_forward_or_optimizer_probe=True)
docs=f.bundle();byte_map={x.ref["path"]:x.json_bytes for x in docs.values()}
orig_bytes=m._bytes
m._bytes=lambda r,metadata=False:byte_map[r["path"]]if r["path"]in byte_map else orig_bytes(r,metadata=metadata)
refs=dict(initialization_ref=docs["initialization"].ref,contrastive_completion_ref=docs["contrastive_completion"].ref,
    contrastive_wait_ref=docs["contrastive_wait"].ref,consumed_outer_ref=docs["consumed_outer"].ref)
qualified=m.qualify_saved_MLM_for_embeddings_v1(**refs)
check("separate_qualification_keeps_MLM_outer1_and_new_contrastive_outer0",qualified["historical_MLM_outer_exit_code"]==1 and not qualified["completed_legacy_full_fit_wrapper_claimed"])
qr=dict(path=str(D/"RAM_ONLY_qualification.json"),sha256=hashlib.sha256(b._encoded(qualified)).hexdigest());byte_map[qr["path"]]=b._encoded(qualified)
spec=dict(parent_id=b.PARENT,refs=dict(saved_MLM_qualification=qr,fit_completion=qr,preparation=b.PREPARATION_REF,counts=b.COUNTS_REF))
check("explicit_dispatch_requalifies_separate_saved_pipeline",m.fit_completion_dispatch_v1(qr,spec)==qualified)
rejects("dispatch_rejects_conflicting_fit_slot",lambda:m.fit_completion_dispatch_v1(qr,dict(spec,refs=dict(spec["refs"],fit_completion=refs["contrastive_completion_ref"]))))
rejects("dispatch_rejects_cross_parent",lambda:m.fit_completion_dispatch_v1(qr,dict(spec,parent_id="other")))
bad=copy.deepcopy(qualified);bad["historical_MLM_outer_exit_code"]=0;byte_map[qr["path"]]=b._encoded(bad)
rejects("stored_qualification_outer0_relabel_rejected",lambda:m.fit_completion_dispatch_v1(qr,spec));byte_map[qr["path"]]=b._encoded(qualified)
for role,key,value in [("consumed_outer","actual_tool_terminal_exit_code",1),("consumed_outer","terminal_consumed",False),
    ("contrastive_wait","process_group_ESRCH",False),("contrastive_wait","operational_failure","RAM_FAIL"),
    ("contrastive_completion","actual_optimizer_updates",2134),("initialization","operational_checkpoint_retention",dict(original=-1,actual=-1,save_interval="20ba"))]:
    doc=docs[role];original_raw=byte_map[doc.ref["path"]];bad=dict(doc.value,**{key:value})
    # Authenticated RAM document with a new ref; preserve other joins to make rejection conservative.
    changed=f.document("RAM_mutation_"+role,bad);byte_map[changed.ref["path"]]=changed.json_bytes
    argkey={"consumed_outer":"consumed_outer_ref","contrastive_wait":"contrastive_wait_ref","contrastive_completion":"contrastive_completion_ref","initialization":"initialization_ref"}[role]
    rejects("separate_gate_rejects_"+role+"_"+key,lambda argkey=argkey,changed=changed:m.qualify_saved_MLM_for_embeddings_v1(**dict(refs,**{argkey:changed.ref})))
m._bytes=orig_bytes
# Source-level author call/guards are the accepted keep1 stage plus explicit joins only.
entry=ast.parse((D/"scientific_entry_v1.py").read_text())
stage=next(n for n in entry.body if isinstance(n,ast.FunctionDef))
check("contrastive_only_entry_has_no_MLM_author_call",not any(isinstance(n,ast.Attribute)and n.attr=="main"for n in ast.walk(stage)))
check("stage_uses_inline_keep1_bridge_and_nested_final_reader","run_original_keep1_contrastive_train_v1" in ast.unparse(stage) and "checkpoint_metadata_bridge_v1" in ast.unparse(stage))
for pin in m.PINS.values():check("exact_original_pin:"+pin["path"],ref(pin["path"])==pin)
check("no_Torch_runtime_guard_or_scientific_payload_import",not any(k=="torch"or k.startswith(("torch.","arggnn"))for k in sys.modules))
# Exact accepted keep1 stage AST after removing the four explicit integration joins.
old_stage=copy.deepcopy(fn_ast(m.PINS["io"]["path"],"contrastive_keep1_v1"))
new_stage=copy.deepcopy(stage);new_stage.name="contrastive_keep1_v1";new_stage.body=new_stage.body[2:]
class UndoStage(ast.NodeTransformer):
    def visit_Attribute(self,node):
        node=self.generic_visit(node)
        if isinstance(node.value,ast.Name)and node.value.id=="io"and node.attr in ("_policy_v1","_base_v1"):
            return ast.copy_location(ast.Name(id=node.attr,ctx=ast.Load()),node)
        return node
new_stage=UndoStage().visit(new_stage)
old_gate=next(n for n in old_stage.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="checkpoint"for t in n.targets))
old_train=next(n for n in old_stage.body if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Attribute)and n.value.func.attr=="train")
body=[]
for n in new_stage.body:
    if isinstance(n,ast.Assign)and any(isinstance(t,ast.Tuple)for t in n.targets)and "saved_acceptance_v1" in ast.unparse(n):
        body.append(copy.deepcopy(old_gate));continue
    if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="initialization"for t in n.targets):
        body.append(copy.deepcopy(old_train));continue
    if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="initialization_ref"for t in n.targets):continue
    if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="final"for t in n.targets):
        n.value.func=ast.parse("helper._final_checkpoint",mode="eval").body
    if isinstance(n,ast.Return):
        for call in ast.walk(n):
            if isinstance(call,ast.Call):call.keywords=[k for k in call.keywords if k.arg!="saved_MLM_initialization_ref"]
    body.append(n)
new_stage.body=body
check("keep1_stage_original_author_calls_guards_config_and_final_checks_unchanged",dump(old_stage)==dump(new_stage))
# All four original complete embedding calls, using only metadata-sized fake models/vectors.
base_module=m.load_source_v1("stages")
events=[]
class Writer:
    path="RAM"
    def save(self,name,value):
        events.append(("save",name,value))
        return dict(path="RAM/"+name,sha256="3"*64)
prepared_dir=Path("/RAM_ONLY")
base_module._context_v1=lambda ctx,request,phase,GPU:prepared_dir
base_module._native_v1=lambda ctx:(dict(OmegaConf=SimpleNamespace(save=lambda *a,**kw:None)),
    SimpleNamespace(_io_config=lambda *a,**kw:{},_verify_author_import_origins=lambda p:None))
embedding_module=SimpleNamespace(load_checkpoint_model=lambda *a:("RAM_MODEL",{}),
    embed_arg_dataset=lambda *a,**kw:(events.append(("embed",kw))or SimpleNamespace(shape=(base_module.ROW_COUNTS[kw["include_splits"]and current_key],768))))
base_module.importlib=SimpleNamespace(import_module=lambda name:embedding_module)
original_load=m.load_source_v1;original_bind=m.bind_stage_v1;original_checkpoint=m.saved_embedding_checkpoint_v1
m.load_source_v1=lambda role:base_module if role=="stages"else original_load(role)
m.bind_stage_v1=lambda *a:None
new_checkpoint=dict(path="RAM/new_contrastive.pt",sha256="4"*64)
m.saved_embedding_checkpoint_v1=lambda ctx,request:new_checkpoint
try:
    for view in ("query_paths","contrastive"):
        for split in ("train","val"):
            current_key=view+"/"+split
            ctx=dict(writer=Writer(),input_ref=dict(path="RAM/input.json",sha256="5"*64),
                prepared_metadata=dict(native_dataset_refs={current_key:dict(path="RAM/raw",sha256="6"*64)},
                    full_path_id_order_refs={current_key:dict(path="RAM/order",sha256="7"*64)}),
                verify_native_imports_v1=lambda:None,fast_guard=lambda event:None,
                reference=lambda p:dict(path=str(p),sha256="8"*64))
            request=dict(view=view,split=split,saved_MLM_qualification_ref=dict(path="RAM/qualification.json",sha256="9"*64))
            m.embedding_v1(ctx,request)
            call=[x for x in events if x[0]=="embed"][-1][1]
            done=[x for x in events if x[0]=="save"][-1][2]
            check("original_full_embedding_call_"+current_key,
                call["batch_size"]==256 and call["pooling_strategy"]=="head_norm"
                and call["layer_index"] is None and call["max_samples"]==0
                and call["include_splits"]==split and call["use_split_labels"]==(view=="contrastive")
                and done["checkpoint_ref"]==new_checkpoint and done["raw_dataset_ref"]==ctx["prepared_metadata"]["native_dataset_refs"][current_key])
    check("all_four_separate_embedding_products_preserved",len([x for x in events if x[0]=="embed"])==4)
finally:
    m.load_source_v1=original_load;m.bind_stage_v1=original_bind;m.saved_embedding_checkpoint_v1=original_checkpoint
check("original_native_run_four_embedding_loop_and_retrieval_source_unchanged",ref(m.PINS["legacy_run"]["path"])==m.PINS["legacy_run"])

result=dict(schema="saved_MLM_keep1_integration_AST_RAM_tests_v1",status="PASS_SOURCE_RAM_ONLY",
    tests=len(checks)+f.run()["count"],integration_checks=len(checks),accepted_bridge_checks=f.run()["count"],checks=checks,
    real_Torch_runtime_checkpoint_data_truth_GPU_guard_process_admission_calls=False)
(D/"ram_ast_result.v1.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items()if k!="checks"},sort_keys=True))
