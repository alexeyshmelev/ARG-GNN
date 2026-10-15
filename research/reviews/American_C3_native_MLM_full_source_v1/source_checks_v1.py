
"""AST/source and pure synthetic checks only. Never imports project/model/guard modules."""
from pathlib import Path
import ast,builtins,copy,hashlib,json,symtable,argparse,contextlib,io,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/American_C3_native_MLM_full_source_v1"
BASE=ROOT/REL
OLD=ROOT/"research/reviews/American_C3_native_model_root_v1"
def text(path):return Path(path).read_text()
def node(path,name):
    source=text(path);return next(n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
def segment(path,name):return ast.get_source_segment(text(path),node(path,name))
def check(ok,label):
    if not ok:raise AssertionError(label)
    results.append(label)
def require(ok,message):
    if not ok:raise ValueError(message)
def pure(path,name,ns):
    exec(compile(ast.Module(body=[node(path,name)],type_ignores=[]),str(path)+":pure_synthetic_only","exec"),ns)
    return ns[name]
def rejects(fn,*args,**kwargs):
    try:fn(*args,**kwargs)
    except (ValueError,KeyError,TypeError):return True
    return False
results=[]
paths=sorted(BASE.glob("*.py"))+[ROOT/"src/arggnn/american_c3_native_mlm_full_v1.py"]
for p in paths:
    source=text(p);ast.parse(source);compile(source,str(p),"exec",dont_inherit=True)
    st=symtable.symtable(source,str(p),"exec");known=set(st.get_identifiers())|set(dir(builtins));unknown=[]
    def visit(q):
        for s in q.get_symbols():
            if s.is_referenced() and s.is_global() and s.get_name() not in known:unknown.append(s.get_name())
        for child in q.get_children():visit(child)
    visit(st);check(not unknown,"compile_and_static_globals:"+p.name)
for name in ("_early_read","_title_lease","_save_json","_int_time","_final_checkpoint","_verify_author_import_origins","_io_config"):
    check(segment(ROOT/"src/arggnn/american_c3_native_child_v1.py",name)==segment(BASE/"native_child_v1.py",name),"unchanged_child_helper:"+name)
for name in ("_RAMLog","_title_lease","_subreaper","_children","_group_exists","_signal_group","_reap_group"):
    check(segment(ROOT/"src/arggnn/american_c3_native_run_v1.py",name)==segment(BASE/"supervisor_v1.py",name),"unchanged_supervisor_helper:"+name)
original=segment(ROOT/"src/arggnn/american_c3_native_run_v1.py","_run_child")
successor=segment(BASE/"supervisor_v1.py","_run_child")
check(original.split('    ctx["boundary_guard"]("after_native_child_actual_wait_and_drain"')[0]==successor.split('    ctx["boundary_guard"]("after_native_child_actual_wait_and_drain"')[0],"unchanged_live_child_loop_and_directwait_reap_drain")
check(successor.index('wait_ref=writer.save')<successor.index('verify(directory,tmp,caps)'),"terminal_categories_after_actual_wait_receipt")
for name in ("_proc_reader","_fast_guard","_actual_context_uuid"):
    check(segment(OLD/"context_factory_v1.py",name)==segment(BASE/"context_factory_v1.py",name),"unchanged_context_guard:"+name)
for name in ("original_guard","consumer_guard","active"):
    check(segment(OLD/"root_native_contract_v1.py",name)==segment(BASE/"root_native_contract_v1.py",name),"unchanged_root_guard:"+name)
for name in ("_stat","_tree","_signal_owned","_absent_group","close"):
    check(segment(OLD/"root_native_driver_v1.py",name)==segment(BASE/"root_native_driver_v1.py",name),"unchanged_root_wait_close:"+name)
check('"contrastive"' not in segment(BASE/"supervisor_v1.py","run_context_v1"),"no_contrastive_stage_dispatch")
check('module.main(config,return_trainer=True)' in text(BASE/"native_child_v1.py") and '==224775' in text(BASE/"native_child_v1.py"),"full_author_main_and_exact_final_updates")
check('trainer.save_checkpoint' not in text(BASE/"native_child_v1.py"),"stock_Composer_final_checkpoint_lifecycle")
check('RLIMIT_FSIZE' in text(BASE/"native_child_v1.py"),"per_file_failstop_not_aggregate_quota")
consumer=BASE/"consumer_contract_v1.py";tree=ast.parse(text(consumer))
ns=dict(require=require)
default=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="DEFAULTS" for t in n.targets))
exec(compile(ast.Module(body=[default],type_ignores=[]),"synthetic_defaults","exec"),ns)
prep_ref=dict(path="v3",sha256="a"*64);ns.update(PREP_REF=prep_ref,GPU_PHASES=("qualify_mlm","mlm"),no_private_refs=lambda v:None)
counter=pure(consumer,"counts_v1",ns);prep=dict(unique_paths=25574587,vocabulary_nodes=106769,max_raw_path_nodes=29,native_train_fraction=.9)
check(counter(dict(refs=dict(preparation=prep_ref)),prep)["MLM_optimizer_updates"]==224775,"full_corpus_count_arithmetic")
bad=dict(prep,unique_paths=1000);check(rejects(counter,dict(refs=dict(preparation=prep_ref)),bad),"reject_subset_count")
validator=pure(consumer,"validate_config",ns);cfg=copy.deepcopy(ns["DEFAULTS"])
cfg.update(execution_enabled=True,phase="mlm",phase_write_budget_bytes=16*2**30,short_tmp_directory="synthetic",max_checkpoint_files=2,max_checkpoint_bytes=2950918528)
validator(cfg,"mlm");check(True,"accept_fixed_full_MLM_config")
bad=copy.deepcopy(cfg);bad["MLM"]["batch"]=256;check(rejects(validator,bad,"mlm"),"reject_smaller_batch")
bad=copy.deepcopy(cfg);bad["MLM"]["epochs"]=1;check(rejects(validator,bad,"mlm"),"reject_fewer_epochs")
caps=json.loads(text(BASE/"mlm_caps.inactive.json"));caps.update(active=True,runtime_ref=dict(path="runtime",sha256="b"*64))
root_ns=dict(require=require,DRIVER="driver",PARENT="american_c3_train_000",PREP_REF=caps["preparation_ref"],ref=lambda p:caps["driver_source_ref"])
cv=pure(BASE/"root_native_contract_v1.py","validate_caps",root_ns)
request=dict(phase="mlm",physical_gpu=4,refs=dict(runtime=caps["runtime_ref"]))
cv(caps,request);check(True,"accept_reviewed_18GiB_16GiB_caps")
bad=dict(caps,phase_write_budget_bytes=14*2**30);check(rejects(cv,bad,request),"reject_underallocated_14GiB_phase")
bad=dict(caps,active=False);check(rejects(cv,bad,request),"reject_inactive_caps")
bad=dict(caps,aggregate_cache_OS_quota_claimed=True);check(rejects(cv,bad,request),"reject_invented_aggregate_cache_quota")
objects={"complete":dict(status="completed_native_C3_path_preparation",parent_id="american_c3_train_000",observation_ref="obs"),
 "wait":dict(returncode=0,completed_ref=caps["preparation_ref"],actual_direct_worker_wait=True,owned_worker_group_ESRCH=True,cleanup_errors=[],operational_failure=None)}
root_ns.update(read=lambda r:objects["complete"] if r==caps["preparation_ref"] else objects["wait"],OBS_REF="obs")
uv=pure(BASE/"root_native_contract_v1.py","validate_upstream",root_ns)
uv(caps["preparation_ref"],"wait",preparation=True);check(True,"accept_genuine_prep_wait0_drain")
objects["wait"]["returncode"]=1;check(rejects(uv,caps["preparation_ref"],"wait",preparation=True),"reject_failed_prep_wait")
objects["wait"]["returncode"]=0;objects["wait"]["owned_worker_group_ESRCH"]=False
check(rejects(uv,caps["preparation_ref"],"wait",preparation=True),"reject_undrained_prep")
test_measure=lambda p:{"child/checkpoints":6*2**30,"child/write_scopes":8*2**30,"tmp":256*2**20,"child":14*2**30}.get(str(p),0)
tv=pure(BASE/"terminal_bounds_v1.py","verify",dict(measure=test_measure))
tv(Path("child"),Path("tmp"),caps);check(True,"terminal_allowance_categories_accept")
bad=dict(caps,cache_allowance_bytes=7*2**30);check(rejects(tv,Path("child"),Path("tmp"),bad),"terminal_cache_overage_reject")
for p in BASE.glob("*_request.inactive.json"):check(json.loads(text(p))["execution_authorized"] is False,"inactive_request:"+p.name)
check(6*2**30+8*2**30+256*2**20+256*2**20+64*2**20<16*2**30,"phase_allowance_sum_fits_16GiB")
check(16*2**30+128*2**20+2**20<18*2**30,"reservation_headroom_fits_18GiB")
check(12*(65536+8192)<=2**20 and 6*16*2**20+65536<=128*2**20,"root_packet_outer_RAM_log_allowances")
check(len(str(ROOT/".runtime/tmp/c3mlm/q/qualify_mlm/pymp-12345678/listener-12345678").encode())<=107,"short_native_IPC_path")
# High-impact regression: a successful producer may return a dict, but CLI exits0.
fake_spec=dict(phase="mlm")
cli_ns=dict(argparse=argparse,json=json,SOURCE="canonical.py",GPU_PHASES=("qualify_mlm","mlm"),
    require=require,metadata=lambda r:fake_spec,_title_lease=lambda:(None,"lease"),
    validate_active_input=lambda *a,**k:({},None),load_root_context=lambda *a:{},
    run_context_v1=lambda *a:dict(path="completed.json",sha256="f"*64),
    NativeChildFailure=type("FakeChildFailure",(Exception,),{}))
cli=pure(BASE/"supervisor_v1.py","main",cli_ns);captured=io.StringIO()
with contextlib.redirect_stdout(captured):
    status=cli(["--execute","--input","fixture.json","--expected-input-sha256","e"*64])
check(type(status) is int and status==0,"CLI_successful_dict_producer_returns_integer0")
check(json.loads(captured.getvalue())["completed_ref"]==dict(path="completed.json",sha256="f"*64),"CLI_emits_completed_ref_separately")
stub=types.SimpleNamespace(main=lambda argv:0)
safe_builtins=dict(vars(builtins))
def only_stub(name,*args,**kwargs):
    if name=="supervisor_v1":return stub
    raise AssertionError("Unexpected runtime import in canonical CLI stub test: "+name)
safe_builtins["__import__"]=only_stub
canonical_ns=dict(ROOT=ROOT,REL=REL,sys=types.SimpleNamespace(path=[]),__builtins__=safe_builtins)
canonical=pure(ROOT/"src/arggnn/american_c3_native_mlm_full_v1.py","main",canonical_ns)
check(type(canonical([])) is int and canonical([])==0,"canonical_CLI_forwards_integer0_without_runtime_imports")
result=dict(schema="american_c3_native_mlm_source_checks_v1",status="passed_source_and_synthetic_checks_only",
 checks=results,count=len(results),project_model_guard_modules_imported=False,
 payloads_opened=False,process_or_GPU_inspection=False,accounting_or_admission_or_qualifier_or_fit_executed=False,
 actual_guard_launcher_and_GPU_integration_qualified=False)
print(json.dumps(result,sort_keys=True,indent=2))
