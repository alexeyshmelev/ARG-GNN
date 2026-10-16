"""Independent resume checks using exact source functions and fake payload/runtime."""
from pathlib import Path
from datetime import timedelta
import ast,builtins,copy,hashlib,json,stat,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
B=ROOT/"research/reviews/native_VAL000_full_state_resume_source_20261016_v5"
def require(ok,message):
    if not ok:raise ValueError(message)
def function(t,name):return next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
def run():
    bridge=ast.parse((B/"resume_scientific_bridge_v5.py").read_text())
    native=ast.parse((ROOT/"research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/native_child_v1.py").read_text())
    original=ast.parse((ROOT/".runtime/conda/envs/argformer_native_v1/lib/python3.11/site-packages/composer/utils/checkpoint.py").read_text())
    nns={};exec(compile(ast.Module(body=[function(native,"optimizer_counter_evidence")],type_ignores=[]),"<original-optimizer-counter-source>","exec"),nns)
    ref=lambda n:dict(path="SYNTHETIC_ONLY/"+n,sha256=hashlib.sha256(n.encode()).hexdigest())
    payload_bytes=b"EXPLICITLY_FAKE_CHECKPOINT_BYTES"
    identity=types.SimpleNamespace(st_dev=1,st_ino=2,st_size=len(payload_bytes),st_mtime_ns=3,st_ctime_ns=4,st_nlink=1,st_mode=stat.S_IFREG|0o600)
    stamp=dict(device=1,inode=2,bytes=len(payload_bytes),mtime_ns=3,ctime_ns=4,link_count=1)
    saved=dict(epoch=0,batch=6500,batch_in_epoch=6500,optimizer_unique_steps=[6500])
    class FakePath:
        def lstat(self):return identity
    class FakeOS:
        O_RDONLY=0;O_CLOEXEC=1;O_NOFOLLOW=2;SEEK_SET=0
        def __init__(self):self.reads=0;self.opens=[]
        def open(self,p,flags):self.opens.append(flags);return 99
        def fstat(self,fd):assert fd==99;return identity
        def read(self,fd,n):
            assert fd==99;self.reads+=1
            return payload_bytes if self.reads==1 else b""
        def lseek(self,*a):return 0
        def close(self,fd):pass
    tkeys=("iteration","epoch","batch","sample","token","epoch_in_iteration","token_in_iteration","batch_in_epoch","sample_in_epoch","token_in_epoch")
    time0={k:0 for k in tkeys};time0.update(epoch=0,batch=6500,batch_in_epoch=6500)
    time0.update({k:timedelta(0) for k in ("total_wct","iteration_wct","epoch_wct","batch_wct")})
    checkpoint_value=dict(state=dict(timestamp=dict(Timestamp=time0),model={str(i):0 for i in range(78)},
        optimizers=dict(DecoupledAdamW=dict(state={i:dict(step=6500,exp_avg=0,exp_avg_sq=0) for i in range(77)})),
        schedulers=dict(LambdaLR=dict(last_epoch=6500)),callbacks=dict(SyntheticOnly={}),algorithms={},dataset_state={}),rng=[dict(SYNTHETIC_ONLY=True)])
    c=types.SimpleNamespace(require=require,active_amendment_v1=lambda:None,local=lambda p:FakePath(),ref=ref,
        ORIGINAL_INPUT=ref("input"),current_amendment_ref_v1=lambda:ref("amendment"),
        read=lambda r:dict(ROOT_resume_qualification_ref=ref("qualification")),
        RESUME_STATE=dict(checkpoint_identity=stamp,checkpoint_ref=dict(path="SYNTHETIC_ONLY/checkpoint",sha256=hashlib.sha256(payload_bytes).hexdigest()),actual_saved_progress=saved))
    fake_checkpoint_module=types.SimpleNamespace()
    fake_utils=types.SimpleNamespace(checkpoint=fake_checkpoint_module)
    def fake_import(name,globals=None,locals=None,fromlist=(),level=0):
        if name=="composer.utils":return fake_utils
        return builtins.__import__(name,globals,locals,fromlist,level)
    ns=dict(c=c,Path=Path,types=types,copy=copy,ast=ast,os=None,stat=stat,hashlib=hashlib,timedelta=timedelta,SOURCE="SYNTHETIC_ONLY/source",
        __builtins__=dict(vars(builtins),__import__=fake_import))
    defs=[n for n in bridge.body if isinstance(n,ast.FunctionDef) and n.name in ("validated_composer_timestamp_v1","_stamp")]
    defs.append(next(n for n in bridge.body if isinstance(n,ast.ClassDef) and n.name=="ResumeSession"))
    exec(compile(ast.Module(body=defs,type_ignores=[]),"<actual-session-source-with-fakes>","exec"),ns)
    native_helper=types.SimpleNamespace(optimizer_counter_evidence=nns["optimizer_counter_evidence"])
    def prepare(value):
        ns["os"]=FakeOS()
        session=ns["ResumeSession"](native_helper,{})
        torch=types.SimpleNamespace(load=lambda *a,**k:copy.deepcopy(value))
        result=session.prepare(torch)
        assert ns["os"].opens==[3]
        return session,result
    session,verified=prepare(checkpoint_value)
    assert verified["optimizer_counters"]["parameter_state_count"]==77 and verified["saved_progress"]["batch"]==6500
    negatives=[]
    for name,change in [
        ("missing_model",lambda v:v["state"].__setitem__("model",{})),
        ("missing_scheduler",lambda v:v["state"].__setitem__("schedulers",{})),
        ("missing_RNG",lambda v:v.__setitem__("rng",[])),
        ("wrong_optimizer_step",lambda v:v["state"]["optimizers"]["DecoupledAdamW"]["state"][0].__setitem__("step",2)),
        ("missing_Adam_moment",lambda v:v["state"]["optimizers"]["DecoupledAdamW"]["state"][0].pop("exp_avg")),
        ("missing_77th_parameter",lambda v:v["state"]["optimizers"]["DecoupledAdamW"]["state"].pop(76)),
        ("wrong_saved_timestamp",lambda v:v["state"]["timestamp"]["Timestamp"].__setitem__("batch",6501))]:
        value=copy.deepcopy(checkpoint_value);change(value)
        try:prepare(value)
        except ValueError:negatives.append(name)
        else:raise AssertionError(name+" was accepted")
    # Exercise actual installed Composer load/_restore source, no package import.
    class TemporaryContext:
        def __enter__(self):return "/SYNTHETIC_ONLY/temporary"
        def __exit__(self,*a):return False
    class FakeTensor:
        def __init__(self,x):self.data=x
    restoration=[]
    class DecoupledAdamW:
        def __init__(self,value):self.value=value
        def state_dict(self):return self.value
    DecoupledAdamW.__qualname__="DecoupledAdamW"
    class FakeState:
        fsdp_sharded_state_dict_enabled=False
        load_monolith_rank0_only=False
        run_name="SYNTHETIC_ONLY"
        model={}
        device=types.SimpleNamespace(tensor_to_device=lambda t:t)
        def load_state_dict(self,value,logger,**kwargs):
            restoration.append(dict(keys=sorted(value),kwargs=kwargs))
            stamp=value["timestamp"]["Timestamp"]
            self.timestamp=types.SimpleNamespace(**{k:types.SimpleNamespace(value=stamp[k]) for k in ("epoch","batch","batch_in_epoch")})
            self.optimizers=[DecoupledAdamW(value["optimizers"]["DecoupledAdamW"])]
            self.schedulers=list(value["schedulers"].values())
    dist=types.SimpleNamespace(get_local_rank=lambda:0,barrier=lambda:None,all_reduce=lambda *a,**k:None)
    def forbidden_download(*a,**k):raise AssertionError("Unadapted downloader reached")
    loadns=dict(partial_format=lambda path,**kwargs:path,dist=dist,
        tempfile=types.SimpleNamespace(TemporaryDirectory=TemporaryContext),
        log=types.SimpleNamespace(debug=lambda *a,**k:None,info=lambda *a,**k:None),
        _get_local_rank_zero_path=lambda p:p,is_model_deepspeed=lambda model:False,
        download_checkpoint=forbidden_download,safe_torch_load=lambda **kw:copy.deepcopy(checkpoint_value),
        torch=types.SimpleNamespace(tensor=lambda value,**kw:FakeTensor(value),int64="SYNTHETIC_INT64"))
    future=ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0)
    body=[future,function(original,"load_checkpoint"),function(original,"_restore_checkpoint")]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),"<exact-installed-Composer-load-restore-source-with-fake-runtime>","exec"),loadns)
    fake_checkpoint_module.load_checkpoint=loadns["load_checkpoint"]
    session,_=prepare(checkpoint_value)
    session.install_original_Composer_FD_loader()
    state=FakeState()
    rng=fake_checkpoint_module.load_checkpoint(session.path,state,None,load_weights_only=False)
    assert rng==checkpoint_value["rng"] and restoration[0]["keys"]==sorted(checkpoint_value["state"])
    assert session.restore_receipt["actual_restored_optimizer_counters"]["parameter_state_count"]==77
    assert session.restore_receipt["actual_restored_optimizer_counters"]["unique_step_values"]==[6500]
    assert session.restore_receipt["actual_scheduler_and_RNG_returned"] is True
    assert loadns["download_checkpoint"] is forbidden_download
    session.close()
    return dict(schema="independent_native_resume_v5_fullstate_source_fixture",status="PASS_SOFTWARE_ONLY",
        actual_prepare_with_77_fake_Adam_states_at6500=True,prepare_negative_cases_rejected=negatives,
        exact_installed_Composer_load_and_restore_AST_with_fake_runtime_exercised=True,
        all_serialized_state_keys_passed_to_original_restore_call=restoration[0]["keys"],
        full_optimizer77_step6500_scheduler_timestamp_and_RNG_before_training_checked=True,
        private_download_callback_only_original_global_untouched=True,
        project_scientific_imports_real_payload_process_device_or_accounting_access=False)
if __name__=="__main__":print(json.dumps(run(),sort_keys=True,indent=2))
