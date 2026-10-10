from pathlib import Path
import ast,builtins,datetime,hashlib,importlib.util,json,os,sys,traceback
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=ROOT/'research/reviews/official_argformer_tsinfer_parent_v2_independent'
blocked=[]
def deny(reason):
    blocked.append(reason)
    raise RuntimeError('INDEPENDENT REVIEW GUARD: '+reason)
def canonical_path(value):
    if isinstance(value,int) or value is None:return None
    try:p=Path(os.fsdecode(value))
    except TypeError:return None
    return Path(os.path.abspath(p))
def owned(p):return p is not None and (p==OUT or OUT in p.parents)
def read_guard(p):
    if p is None or owned(p):return
    if p in {ROOT/'data/literature/argformer_btag438_source_check_v1/body.txt',ROOT/'data/literature/argformer_btag438_source_check_v1/source_receipt.json'}:return
    denied_roots=[ROOT/'data',ROOT/'.runtime/gpup',ROOT/'.runtime/locks',ROOT/'.runtime/conda/envs/argformer_native_v1']
    if any(p==x or x in p.parents for x in denied_roots):deny('actual data/runtime/live scheduler access '+str(p))
    if p in {ROOT/'research/experiment_ledger.md',ROOT/'research/experiment_decisions.json'}:deny('live research registry '+str(p))
    if p.suffix in {'.trees','.npz','.vcf','.pkl','.pt','.npy'}:deny('array read outside handmade review temp '+str(p))
    if any(x in p.parts for x in ['truth','oracle']):deny('truth path '+str(p))
def audit(event,args):
    if event=='open':
        p=canonical_path(args[0]);mode=args[1];flags=args[2]
        writes=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if writes and p is not None and not owned(p):deny('write outside independent review '+str(p))
        if not writes:read_guard(p)
    elif event in {'os.listdir','os.scandir'}:read_guard(canonical_path(args[0]))
    elif event in {'os.mkdir','os.remove','os.rmdir','os.unlink','os.chmod','os.chown','os.truncate','os.utime'}:
        p=canonical_path(args[0])
        if p is not None and not owned(p):deny('mutation outside independent review '+str(p))
    elif event in {'os.rename','os.link','os.symlink'}:
        for value in args[:2]:
            p=canonical_path(value)
            if p is not None and not owned(p):deny('mutation outside independent review '+str(p))
    elif event in {'subprocess.Popen','os.system','os.posix_spawn','socket.connect','socket.bind'}:deny('process/network event '+event)
    elif event=='import':
        name=args[0]
        if name.split('.')[0] in {'torch','tskit','tsinfer','tsdate','composer','transformers','omegaconf'} or name in {'arggnn.official_argformer_bridge','arggnn.train_official_argformer_biological'}:deny('native preparation/model import '+name)
original_import=builtins.__import__
def guarded_import(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','tskit','tsinfer','tsdate','composer','transformers','omegaconf'} or name in {'arggnn.official_argformer_bridge','arggnn.train_official_argformer_biological'}:deny('native preparation/model import '+name)
    return original_import(name,*args,**kwargs)
def profile(frame,event,arg):
    if event!='call':return
    c=frame.f_code
    if c.co_filename==str(ROOT/'src/arggnn/train_official_argformer_biological.py'):
        # The suite compiles only its explicitly patched handmade preparation stand-in here.
        if c.co_name=='prepare_native_paths' and 'execution_enabled' in c.co_consts and ('parent','output','inactive_bridge') in c.co_consts:return
        if c.co_name=='<module>':return
        deny('actual native preparation body execution '+c.co_name)
sys.dont_write_bytecode=True
sys.addaudithook(audit);builtins.__import__=guarded_import;sys.setprofile(profile)

import copy
try:
    spec=importlib.util.spec_from_file_location('independent_source_binding',ROOT/'scripts/prepare_official_argformer_tsinfer_parent_v2.py')
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    config=h.policy();h.verify_source(config)
    author=ROOT/'research/reviews/official_argformer_tsinfer_parent_v2_author'
    sealed=author/'review_receipt.json'
    assert h.sha(sealed)=='c5162c0f8d0e687dd6b26fd7b98008af804a1ebb08d41599edc5afe043b01a62'
    author_receipt=json.loads(sealed.read_text())
    templates={name:json.loads((author/name).read_text()) for name in ['root_admission_template.json','CPU_runtime_template.json','storage_template.json','experiment_registration_template.json','independent_review_schema_template.json']}
    expected=h.admission_bindings(config,config['default_parent'])
    admission=copy.deepcopy(templates['root_admission_template.json'])
    for k,v in expected.items():
        if k not in {'status','active'}:assert admission[k]==v,k
    assert admission['active'] is False and admission['actual_input_tree_roster_grid_validation_authorized'] is False
    assert templates['CPU_runtime_template.json']['python_executable_sha256'] is None
    assert templates['storage_template.json']['ledger_sha256_at_admission'] is None
    assert templates['experiment_registration_template.json']['hypothesis_id'] is None
    source=json.loads((OUT/'source_only_report.json').read_text())
    static=json.loads((OUT/'static_checks.json').read_text())
    seal=json.loads((OUT/'seal_checks.json').read_text())
    test=json.loads((OUT/'test_execution.json').read_text())
    assert test['returncode']==0 and test['blocked_events']==[] and seal['independent_tests']=={'tests':43,'failures':0,'errors':0,'skipped':0}
    binding={'kind':'official_argformer_tsinfer_parent_v1_independent_source_review','review_role':'independent_reviewer','verdict':'PASS','implementation':expected['implementation'],'config':expected['config'],'bridge':config['bridge']}
    assert {k:v for k,v in templates['independent_review_schema_template.json'].items() if k!='verdict'}=={k:v for k,v in binding.items() if k!='verdict'}
    # Authority activation exists only in memory to verify the source schema. No permit file or actual runtime proof is created.
    admission.update(expected);admission['actual_input_tree_roster_grid_validation_authorized']=True
    runtime=copy.deepcopy(templates['CPU_runtime_template.json']);runtime['python_executable_sha256']='0'*64
    assert h.validate_admission(config,config['default_parent'],admission,binding,runtime)==expected
    registration=copy.deepcopy(templates['experiment_registration_template.json']);registration['status']='registered_CPU_preparation_and_feasibility_only';registration['hypothesis_id']='IN_MEMORY_SCHEMA_CHECK_ONLY'
    h.validate_experiment_registration(config,config['default_parent'],registration)
    storage=templates['storage_template.json'];output,launch=h.selected_scopes(config,config['default_parent']);p=config['storage']
    required={'kind':'official_argformer_tsinfer_parent_v1_root_storage','authorizer':'root_agent','implementation':expected['implementation'],'config':expected['config'],'scope_parents':[config['default_parent']],'reservation_key':p['reservation_key_prefix']+'/'+config['default_parent'],'reserved_bytes':p['reserved_bytes'],'local_write_cap_bytes':p['local_write_cap_bytes'],'output':str(output),'write_scopes':[str(output),str(launch)],'snapshot':p['snapshot'],'external_reserve_bytes':p['external_reserve_bytes'],'warning_bytes':p['warning_bytes'],'hard_limit_bytes':p['hard_limit_bytes'],'stage_boundary_cap_not_OS_quota':True}
    assert all(storage.get(k)==v for k,v in required.items())
    checks={'review_binding_validate_admission_in_memory_passed':True,'root_admission_template_exact_fields_consistent':True,'root_registration_template_in_memory_schema_passed':True,'root_storage_template_static_scope_matches_Storage_expected':True,'templates_inactive_or_missing_executable_hash':True,'binding_guard_blocked_events':blocked,'actual_execution_admitted':False}
    assert blocked==[]
    (OUT/'binding_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    text='Independent verdict: PASS for the separate v2 source-only lock namespace repair.\n\nThe exact reviewed v2 implementation is scripts/prepare_official_argformer_tsinfer_parent_v2.py, SHA256 c75ef5231b30fd7faa34f5d9517f2047aeecd6f884aa1741f7e2840cf91be815. The exact reviewed config is configs/official_argformer_tsinfer_parent_v2.json, SHA256 7e0ef8bdab828201c28709afe7213b805355f3683e859a6f11cb47bed90aa76f.\n\nIndependent byte, JSON and AST checks found only CONFIG/SCRIPT namespace constants and the embedded/external storage.scheduler_lock repair. All 33 function/class ASTs and all bytes after the constants are unchanged. Current v1 source/config/docs/tests match the sealed initial snapshot. The original20 native-function ASTs, preparation tail and60-file author source tree match the v1 snapshot and sealed v2 source evidence. Scientific policy, 0.6.4 runtime, 1.0.3 writer canary, full-path/native-node identity, tokenizer, source4096 batch budget, retained support and40-billion pair guard are unchanged.\n\nThe pinned gpu_pool_v1.lock_file source puts scheduler.lock under .runtime/locks. The independent guarded handmade suite verifies that its returned descriptor and the configured lock path identify the same inode. The repaired dispatcher continues to open the existing mutex without creating/truncating/writing bytes. No real lock or ledger was inspected.\n\nAll43 handmade/source tests passed with zero failures/errors/skips in the independent project temporary directory. The successful run had zero guard violations. The guards deny real arrays/truth, native runtime/model/preparation imports, actual native preparation-body execution, live storage/registry access, external writes, network and process launches. Two exact literature text/receipt files were allowed for source hashes. Native scientific tokenizer/support definitions were exercised only on handmade data; the preparation reachability test used its explicit handmade stand-in.\n\nThe final review binding passes v2 validate_admission in memory using the retained v1 schema kind. Root runtime/storage/registration/admission templates agree with exact v2 source/config/bridge bindings, unchanged v1 output/reservation/launch-log scopes and CPU-only conditions. Templates remain inactive or omit executable/ledger hashes. In-memory shape checking creates no permit and supplies no real runtime or storage authority.\n\nThe sealed author artifact hashes and preserved initial failure snapshots were rechecked. Initial reviewer harness failures are retained: the blanket data guard blocked literature source text, pytest default /dev/null logging was rejected before collection, the seal-check harness used the wrong preserved-file key, and an initial static transport exited without a receipt. These were harness corrections, without dispatcher edits or actual data access.\n\nactual_execution_admitted=false. This verdict admits source review only. Root must issue fresh runtime/storage/registration/admission bindings, retain the initial v1 failure snapshot and append to the fixed launch log. No actual tree readability, native preparation or corpus feasibility, fitting, retrieval, score or paper reproduction success is established or promised.\n'
    (OUT/'review.txt').write_text(text)
    artifacts={str(p.relative_to(ROOT)):h.sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='review_receipt.json'}
    receipt={**binding,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reviewer_identity':'fresh_independent_source_only_reviewer','scope':'separate_v2_namespace_and_scheduler_mutex_repair_only','actual_execution_admitted':False,'active_project_permit_created':False,'author_seal':{'path':str(sealed.relative_to(ROOT)),'sha256':h.sha(sealed)},'tests':seal['independent_tests'],'narrow_delta_verified':True,'all33_dispatcher_function_class_ASTs_unchanged':True,'v1_source_config_docs_tests_bytes_preserved':True,'native20_preparation_tail_author60_unchanged':True,'canonical_library_scheduler_mutex':'.runtime/locks/scheduler.lock','canonical_mutex_handmade_descriptor_inode_test_passed':True,'fresh_review_binding_matches_v2_validate_admission':True,'root_admission_template_shape_consistent':True,'initial_actual_v1_failure_preserved':True,'successful_guarded_test_blocked_events':[],'actual_data_native_runtime_live_storage_registry_GPU_install_Git_access':False,'limitations':['No actual tree readability or prepared-corpus feasibility is established','Source PASS creates no active project permit and promises no execution success','Fresh root CPU runtime/storage/registration/admission and append-preserved fixed launch log remain required'],'artifact_sha256':artifacts}
    # Recheck the final extended receipt, whose evidence fields are tolerated by the exact binding validator.
    h.validate_admission(config,config['default_parent'],admission,receipt,runtime)
    (OUT/'review_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'verdict':receipt['verdict'],'review_receipt':str((OUT/'review_receipt.json').relative_to(ROOT)),'sha256':h.sha(OUT/'review_receipt.json'),'actual_execution_admitted':False,'root_admission_shape_consistent':True},indent=2))
except BaseException:
    error=traceback.format_exc();(OUT/'binding_failure.log').write_text(error);print(error)
    raise SystemExit(1)
finally:
    sys.setprofile(None)
