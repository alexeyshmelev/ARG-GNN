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
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
rc=1
try:
    spec=importlib.util.spec_from_file_location('independent_source_dispatcher',ROOT/'scripts/prepare_official_argformer_tsinfer_parent_v2.py')
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    config=h.policy();h.verify_source(config);bridge,bc=h.load_bridge(config)
    source=h.source_report(config,bridge,bc)
    (OUT/'source_only_report.json').write_text(json.dumps(source,indent=2)+'\n')
    import pytest
    rc=int(pytest.main(['-q','--log-file',str(OUT/'pytest_internal.log'),'-p','no:cacheprovider','--noconftest','-c','/dev/null','--rootdir',str(OUT),'--basetemp',str(OUT/'tmp/pytest'),'--junitxml',str(OUT/'pytest.xml'),str(ROOT/'tests/test_official_argformer_tsinfer_parent_v2.py')]))
except BaseException:
    (OUT/'guarded_execution_failure.log').write_text(traceback.format_exc())
    traceback.print_exc()
finally:
    sys.setprofile(None)
    receipt={'started_utc':started,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':rc,'blocked_events':blocked,'guarded_source_and_handmade_CPU_only':True,'real_native_runtime_or_live_storage_or_registry_inspected':False,'actual_arrays_or_truth_opened':False,'actual_native_preparation_fit_retrieval_executed':False,'GPU_used':False,'actual_execution_admitted':False,'python_executable':sys.executable,'pytest_options':'no third-party plugin autoload, no conftest, no cacheprovider, own basetemp and junitxml'}
    (OUT/'test_execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
raise SystemExit(rc)
