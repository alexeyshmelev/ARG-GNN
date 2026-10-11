#!/usr/bin/env python3
from pathlib import Path
import ast, builtins, collections, copy, datetime, hashlib, importlib.util, json, os, sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OWN=ROOT/'research/reviews/official_argformer_tsinfer_parent_v3_independent'
CASE=OWN/'tmp/pytest_v6'
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
import tempfile
tempfile.tempdir=str(OWN/'tmp/tempfiles')
counts=collections.Counter(); denied=[]
ARRAYS={'.trees','.npz','.vcf','.pkl','.pt','.npy','.bcf'}
PUBLIC_SOURCE_READS={ROOT/'data/literature/argformer_btag438_source_check_v1/body.txt',ROOT/'data/literature/argformer_btag438_source_check_v1/source_receipt.json'}
BLOCKED_READS=[ROOT/'data',ROOT/'experiments',ROOT/'.runtime/gpup',ROOT/'.runtime/locks',ROOT/'.runtime/conda',ROOT/'research/experiment_ledger.md',ROOT/'research/experiment_decisions.json']
def path_value(value):
    if isinstance(value,int): return None
    if isinstance(value,bytes): value=os.fsdecode(value)
    if not isinstance(value,str): return None
    return Path(os.path.abspath(value))
def deny(why,p):
    denied.append({'reason':why,'path':str(p)})
    raise AssertionError(why+': '+str(p))
def check_path(value,write=False):
    p=path_value(value)
    if p is None:return
    owned=(p==OWN or OWN in p.parents)
    if write and not owned:deny('outside independent write',p)
    if not owned and p not in PUBLIC_SOURCE_READS and any(p==q or q in p.parents for q in BLOCKED_READS):deny('actual or live metadata access',p)
    if not owned and p.suffix in ARRAYS:deny('non-handmade array access',p)
    if not owned and any('truth' in x.lower() or 'oracle' in x.lower() for x in p.parts):deny('truth or oracle access',p)
def audit(event,args):
    if event=='open':
        mode=args[1]; flags=args[2]
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_APPEND|os.O_CREAT|os.O_TRUNC)))
        check_path(args[0],writing); counts['open_write' if writing else 'open_read']+=1
    elif event in {'os.listdir','os.scandir'}:check_path(args[0]);counts[event]+=1
    elif event in {'os.mkdir','os.remove','os.rmdir','os.chmod','os.utime','os.truncate'}:check_path(args[0],True);counts[event]+=1
    elif event in {'os.rename','os.link','os.symlink'}:
        check_path(args[0],True);check_path(args[1],True);counts[event]+=1
    elif event.startswith('subprocess.') or event in {'os.system','os.fork','os.posix_spawn','socket.connect','socket.bind'}:
        deny('process or network activity forbidden',event)
sys.addaudithook(audit)
real_import=builtins.__import__
def guarded_import(name,*args,**kwargs):
    if name.split('.')[0] in {'tsinfer','tsdate','composer','transformers','omegaconf','pynvml','nvml'}:
        deny('model GPU inference import forbidden',name)
    if name in {'arggnn.native_argformer_exec','arggnn.train_official_argformer_biological'}:
        deny('native model or training module forbidden',name)
    result=real_import(name,*args,**kwargs)
    torch=sys.modules.get('torch')
    if torch is not None and getattr(torch,'cuda',None) is not None and getattr(torch,'_C',None) is not None:
        def cuda_denied(*args,**kwargs):deny('CUDA query context or device operation forbidden','torch.cuda')
        for attr in ['init','_lazy_init','is_available','device_count','current_device','get_device_properties','get_device_name','set_device','synchronize','memory_allocated','memory_reserved','empty_cache','Stream','Event']:
            if hasattr(torch.cuda,attr):setattr(torch.cuda,attr,cuda_denied)
        for attr in ['_cuda_init','_cuda_getDeviceCount','_cuda_getDevice','_cuda_setDevice','_cuda_synchronize']:
            if hasattr(torch._C,attr):setattr(torch._C,attr,cuda_denied)
    return result
builtins.__import__=guarded_import
import pytest
class RedirectHandmadeHelper:
    def pytest_collection_modifyitems(self,items):
        modules={item.module for item in items}
        for module in modules:
            if hasattr(module,'smoke_helper'):
                original=module.smoke_helper
                def helper(original=original):
                    smoke=original()
                    smoke.ALLOWED=OWN/'tmp'
                    return smoke
                module.smoke_helper=helper
args=['-q','-p','no:cacheprovider','--log-file',str(OWN/'pytest_v6.log'),'--basetemp',str(CASE),'--junitxml',str(OWN/'pytest_v6.xml'),'tests/test_official_argformer_tsinfer_parent_v3.py']
returncode=pytest.main(args,plugins=[RedirectHandmadeHelper()])
record={'kind':'official_argformer_v3_independent_guarded_pytest','started_utc':started,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':returncode,'tests_requested':47,'argv':args,'source_changes':False,'public_source_read_exceptions':sorted(str(p) for p in PUBLIC_SOURCE_READS),'guarded_external_writes':True,'guarded_actual_arrays_metadata_ledger_registry':True,'guarded_models_GPU_inference_network_subprocess':True,'unchanged_CPU_bridge_torch_base_import_allowed':True,'CUDA_query_context_functions_guarded':True,'handmade_helper_scope_override':str(OWN/'tmp'),'denied_events':denied,'audit_event_counts':dict(counts),'actual_execution_admitted':False}
with (OWN/'pytest_v6_execution.json').open('x') as stream:stream.write(json.dumps(record,sort_keys=True,indent=2)+'\n')
raise SystemExit(returncode)
