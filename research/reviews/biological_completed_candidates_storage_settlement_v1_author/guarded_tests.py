from pathlib import Path
import builtins,contextlib,datetime,hashlib,importlib.util,io,json,os,sys,traceback
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
AUTHOR=ROOT/'research/reviews/biological_completed_candidates_storage_settlement_v1_author'
blocked=[]
def denied(reason):
    blocked.append(reason);raise RuntimeError('AUTHOR GUARD: '+reason)
def pth(value):
    if value is None or isinstance(value,int):return None
    try:return Path(os.path.abspath(os.fsdecode(value)))
    except TypeError:return None
def own(p):return p is not None and (p==AUTHOR or AUTHOR in p.parents)
def reads(p):
    if p is None or own(p):return
    if p==Path('/proc') or Path('/proc') in p.parents:denied('live process inspection '+str(p))
    for directory in [ROOT/'data',ROOT/'results',ROOT/'experiments/results',ROOT/'.runtime/gpup',ROOT/'.runtime/locks',ROOT/'.runtime/conda/envs/argformer_native_v1']:
        if p==directory or directory in p.parents:denied('live state/product read '+str(p))
    if p in [ROOT/'research/experiment_ledger.md',ROOT/'research/experiment_decisions.json']:denied('live research registry '+str(p))
def audit(event,args):
    if event=='open':
        p=pth(args[0]);mode=args[1];flags=args[2]
        write=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if write and p is not None and not own(p):denied('write outside author fixture scope '+str(p))
        if not write:reads(p)
    elif event in {'os.listdir','os.scandir'}:reads(pth(args[0]))
    elif event in {'os.mkdir','os.remove','os.rmdir','os.unlink','os.chmod','os.chown','os.truncate','os.utime'}:
        p=pth(args[0])
        if p is not None and not own(p):denied('mutation outside author fixture scope '+str(p))
    elif event in {'os.rename','os.link','os.symlink'}:
        for value in args[:2]:
            p=pth(value)
            if p is not None and not own(p):denied('mutation outside author fixture scope '+str(p))
    elif event in {'subprocess.Popen','os.system','os.posix_spawn','socket.connect','socket.bind'}:denied('process/network action '+event)
original=builtins.__import__
def imports(name,*args,**kwargs):
    if name.split('.')[0] in {'numpy','torch','tskit','tsinfer','tsdate','composer','transformers','omegaconf'} or name in {'arggnn.biological_candidate','arggnn.official_argformer_bridge'}:denied('scientific array decoder/model import '+name)
    return original(name,*args,**kwargs)
sys.dont_write_bytecode=True;sys.addaudithook(audit);builtins.__import__=imports
rc=1;started=datetime.datetime.now(datetime.timezone.utc).isoformat()
try:
    spec=importlib.util.spec_from_file_location('author_source_only_check',AUTHOR/'audit_and_settle.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    buffer=io.StringIO()
    with contextlib.redirect_stdout(buffer):assert h.main([])==0
    (AUTHOR/'source_only_cli.json').write_text(buffer.getvalue())
    import pytest
    rc=int(pytest.main(['-q','--log-file',str(AUTHOR/'pytest_internal.log'),'-p','no:cacheprovider','--noconftest','-c','/dev/null','--rootdir',str(AUTHOR),'--basetemp',str(AUTHOR/'tmp/pytest'),'--junitxml',str(AUTHOR/'pytest.xml'),str(AUTHOR/'test_settlement.py')]))
except BaseException:
    error=traceback.format_exc();(AUTHOR/'test_guard_failure.log').write_text(error);print(error)
finally:
    result={'started_utc':started,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':rc,'blocked_events':blocked,'source_handmade_fixtures_only':True,'actual_live_inventory_inspect_apply':False,'actual_registry_PID_ledger_read_or_mutation':False,'scientific_array_decoders_GPU_install_Git':False,'only_own_project_fixture_writes':True}
    (AUTHOR/'test_execution.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
raise SystemExit(rc)
