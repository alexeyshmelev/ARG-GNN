"""Inactive, separately admitted native ARGformer tsinfer fit and retrieval.

No automatic authorisation, reservation, GPU selection, preparation, query truth,
scientific fallback, or existing receipt amendment exists in this executor.
--execute needs exact fresh independent review and root-issued stage authority.
"""
from __future__ import annotations
import argparse
import ast
from contextlib import contextmanager
import copy
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import re
import signal
import stat
import sys
import time
import traceback
import types

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
SCRIPT = ROOT / 'src/arggnn/native_argformer_tsinfer_fit_v2.py'
CONFIG = ROOT / 'configs/native_argformer_tsinfer_fit_v2.json'
CONFIG_SHA256 = '29c506b7dcb44a2544c5a0dcf8e2211c3ee9b242a09cc9e6984ffab5255b3129'
FIT = 'src/arggnn/train_official_argformer_biological.py'
RETRIEVAL = 'scripts/score_official_argformer_retrieval_biological_v1.py'
STAGES = ('probe-memory', 'fit', 'retrieve')
CACHE_VARS = ('XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME','HF_HOME',
    'TORCH_HOME','TRITON_CACHE_DIR','CUDA_CACHE_PATH','TORCHINDUCTOR_CACHE_DIR',
    'NUMBA_CACHE_DIR','MPLCONFIGDIR','WANDB_DIR','WANDB_CACHE_DIR','WANDB_CONFIG_DIR')
THREAD_VARS = ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
               'BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def exact(first,second):
    return canonical(first)==canonical(second)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def confined(value):
    p=Path(value)
    if '..' in p.parts or any('truth' in x.lower() or 'oracle' in x.lower() for x in p.parts):
        raise ValueError('canonical observed-only project path required')
    p=p if p.is_absolute() else ROOT/p
    if p==ROOT or ROOT not in p.parents or p.resolve()!=p:
        raise ValueError('non-symlink in-project path required')
    return p


def reference(path):
    p=confined(path)
    return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}


def bound(record,expected=None,json_only=False):
    if (not isinstance(record,dict) or set(record)!={'path','sha256'}
        or not isinstance(record['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',record['sha256'])):
        raise ValueError('exact immutable path/SHA256 reference required')
    p=confined(record['path'])
    if expected is not None and p!=confined(expected):raise ValueError('exact artifact path required')
    if not p.is_file() or p.is_symlink() or (json_only and p.suffix!='.json'):
        raise ValueError('regular immutable admitted artifact required')
    if sha(p)!=record['sha256']:raise ValueError('immutable artifact bytes changed: '+str(p))
    return p


def read_json(record,expected=None):
    p=bound(record,expected,json_only=True)
    if p.stat().st_size>32*1024*1024:raise ValueError('bounded JSON metadata required')
    return json.loads(p.read_text())


def policy(expected=None):
    if sha(CONFIG)!=CONFIG_SHA256 or (expected is not None and expected!=CONFIG_SHA256):
        raise ValueError('exact inactive reviewed fit configuration required')
    c=json.loads(CONFIG.read_text())
    if c['execution_enabled'] is not False:raise ValueError('config alone never activates execution')
    return c


def verify_source(c):
    for path,h in c['frozen_files_sha256'].items():
        if sha(confined(path))!=h:raise ValueError('frozen native authority changed: '+path)
    recipe=read_json(c['native_recipe'])
    # The bridge's source inventory includes the entire unchanged author release.
    bridge_path=ROOT/'src/arggnn/official_argformer_tsinfer_biological_v2.py'
    producer=read_json(c['prepared_producer']['config'])
    if sha(bridge_path)!=producer['bridge']['implementation']['sha256']:
        raise ValueError('unchanged observed bridge authority required')
    spec=importlib.util.spec_from_file_location('_native_fit_source_inventory',bridge_path)
    bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
    bridge.verify_author_source(recipe)
    return recipe


def named_nodes(c,path,names,classes=()):
    p=confined(path)
    if sha(p)!=c['frozen_files_sha256'][path]:raise ValueError('pinned AST source changed')
    tree=ast.parse(p.read_text())
    found={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    return [copy.deepcopy(found[n]) for n in (*classes,*names)]


def ast_proof(c):
    result={}
    for path,names,classes in ((FIT,c['source_AST']['fit_functions'],()),
        (RETRIEVAL,c['source_AST']['retrieval_functions'],c['source_AST']['retrieval_classes'])):
        result[path]={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
                      for n in named_nodes(c,path,names,classes)}
    return result


def compile_fit(c):
    """Private namespace, exact native scientific definitions and assignments."""
    tree=ast.parse(confined(FIT).read_text())
    body=[copy.deepcopy(n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign))]
    body+=named_nodes(c,FIT,c['source_AST']['fit_functions'])
    module=ast.Module(body=body,type_ignores=[]);ast.fix_missing_locations(module)
    ns={'__name__':'_admitted_native_tsinfer_fit_v2','__file__':str(SCRIPT)}
    exec(compile(module,str(confined(FIT)),'exec'),ns)
    ns.update(ROOT=ROOT,CONFIG=CONFIG,SCRIPT=SCRIPT)
    return ns


def compile_retrieval(c):
    """Exact prior reviewed normalisation, full-corpus, top-k and seal definitions."""
    body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
    body+=named_nodes(c,RETRIEVAL,c['source_AST']['retrieval_functions'],c['source_AST']['retrieval_classes'])
    module=ast.Module(body=body,type_ignores=[]);ast.fix_missing_locations(module)
    ns={'__name__':'_admitted_native_tsinfer_retrieval_v2','math':math,'time':time,
        'ROOT':ROOT,'Path':Path,'pickle':pickle,'os':os,'hashlib':hashlib,'json':json,'contextmanager':contextmanager,
        'RULE':copy.deepcopy(c['retrieval_rule']),'sha':sha,
        'local':lambda value,observed=False:confined(value),
        'bound':lambda record,observed=False:bound(record)}
    exec(compile(module,str(confined(RETRIEVAL)),'exec'),ns)
    return ns


def scopes(c,parent,stage,run_id):
    if parent not in c['parent_ids'] or stage not in STAGES or not re.fullmatch('[a-z0-9_]{1,48}',run_id):
        raise ValueError('one configured parent, stage and canonical fresh run ID required')
    parent_scope=confined(Path(c['result_root'])/parent)
    short_tmp=confined(Path(c['short_tmp_root'])/hashlib.sha256(str(parent_scope).encode()).hexdigest()[:12])
    output=parent_scope/stage/run_id
    log=parent_scope/'launch_logs'/(stage+'_'+run_id+'.log')
    return parent_scope,output,short_tmp,log


def require_fields(value,expected,label):
    if any(not exact(value.get(k),v) for k,v in expected.items()):raise ValueError(label)


def prepared_metadata(c,parent,dispatcher_ref):
    """Root-authorized scalar metadata only, before any scientific payload decode."""
    base=Path(c['prepared_root'])/parent
    if not exact(dispatcher_ref,c['prepared_dispatchers'][parent]):raise ValueError('exact completed immutable v3 parent dispatcher required')
    d=read_json(dispatcher_ref,base/'dispatcher_receipt.json')
    require_fields(d,{'kind':'official_argformer_tsinfer_parent_v1_receipt',
      'status':'qualified_full_native_preparation','parent_id':parent,'condition':c['condition'],
      'implementation':c['prepared_producer']['implementation'],'config':c['prepared_producer']['config'],
      'actual_native_preparation_completed':True,'model_fit_or_retrieval_executed':False,
      'query_truth_used':False,'scientific_fallback':False,'exit_code':0},'native v3 preparation-only completion required')
    if d['bridge']['config']!=c['native_recipe']:raise ValueError('original native recipe binding differs')
    if not exact(d['corpus_qualification'],c['qualified_corpus_metadata'][parent]):raise ValueError('exact completed full-corpus qualification required')
    q=read_json(d['corpus_qualification'],base/'corpus_qualification.json')
    require_fields(q,{'status':'qualified_full_native_preparation','native_original_node_IDs_used':True,
        'query_truth_used':False,'model_fit_or_retrieval_executed':False,
        'full_readout_occurrence_partitions_validated':True},'full native corpus qualification required')
    comp=read_json(d['input_tree_compatibility'],base/'input_tree_compatibility.json')
    require_fields(comp,{'status':'original_observed_tree_loaded_without_conversion',
        'native_IDs_preserved':True,'conversion_upgrade_fallback':False},'native original input IDs required')
    p=read_json(d['prepared_manifest'],base/'prepared/prepared_manifest.json')
    require_fields(p,{'schema':'official_argformer_prepared_observed_arg_v1','status':'prepared_observed_paths_only',
      'parent_id':parent,'policy_sha256':c['native_recipe']['sha256'],'namespace':d['namespace'],
      'true_query_ancestry_read':False,'within_ARG_transductive':True,'class_order':['ANA','CHG','WHG','EHG']},
      'same original recipe and per-ARG native namespace required')
    if p['namespace']!=parent+':'+comp['input_tree']['sha256'] or p['inferred_tree_sha256']!=comp['input_tree']['sha256']:
        raise ValueError('original per-ARG node vocabulary namespace differs')
    b=q['budget'];counts=b['actual_counts']
    needed={'unique_full_paths','MLM_train','MLM_val','pure_reference_train','pure_reference_val','query_full_paths_scored'}
    if set(counts)!=needed or any(type(v) is not int or v<1 for v in counts.values()):raise ValueError('typed exact full corpus counts required')
    if (counts['unique_full_paths']!=p['unique_full_paths'] or counts['MLM_train']+counts['MLM_val']!=p['unique_full_paths']
      or p['path_counts']!={'mlm':{'train':counts['MLM_train'],'val':counts['MLM_val']},
                         'contrastive':{'train':counts['pure_reference_train'],'val':counts['pure_reference_val']}}
      or counts['MLM_train']<512 or counts['pure_reference_train']<4096):raise ValueError('source native batches and full splits required')
    refs=counts['pure_reference_train']+counts['pure_reference_val']
    if (b['MLM_optimizer_updates']!=5*(counts['MLM_train']//512)
      or b['contrastive_optimizer_updates']!=counts['pure_reference_train']//4096
      or b['pure_reference_corpus_paths_all_train_plus_val']!=refs
      or b['exhaustive_query_corpus_pairs']!=refs*counts['query_full_paths_scored']
      or b['exhaustive_query_corpus_pairs']>c['retrieval_rule']['max_exhaustive_pairs_per_parent']
      or q['source_node_vocabulary_size']!=p['vocab_size']):raise ValueError('exact full-corpus fit/retrieval admission required')
    return {'dispatcher':d,'qualification':q,'compatibility':comp,'prepared':p,'budget':b}


def admission_bindings(c,parent,stage,run_id):
    parent_scope,output,tmp,log=scopes(c,parent,stage,run_id)
    return {'kind':'native_argformer_tsinfer_fit_v2_root_admission','authorizer':'root_agent',
      'active':True,'status':'root_authorized_exact_native_stage','condition':c['condition'],
      'scope_parents':[parent],'stage':stage,'run_id':run_id,'implementation':reference(SCRIPT),
      'config':reference(CONFIG),'native_recipe':c['native_recipe'],'prepared_producer':c['prepared_producer'],
      'launcher_contract':c['launcher_contract'] if stage!='retrieve' else None,
      'parent_storage_scope':str(parent_scope),'output':str(output),'short_tmp':str(tmp),'launch_log':str(log),
      'CPU_only':stage=='retrieve','query_truth_authorized':False,'scientific_fallback':False,
      'paper_reproduction_claimed':False,'contrastive_execution':'full_batch',
      'maximum_walltime_seconds':c['maximum_walltime_seconds'][stage]}


def validate_admission(c,parent,stage,run_id,a,review,runtime,registration):
    expected=admission_bindings(c,parent,stage,run_id)
    require_fields(a,expected,'fresh exact root stage admission required; preparation receipt grants no fitting authority')
    require_fields(review,{'kind':'native_argformer_tsinfer_fit_v2_independent_review',
      'review_role':'independent_reviewer','verdict':'PASS','implementation':expected['implementation'],
      'config':expected['config']},'fresh independent source/config PASS required')
    require_fields(runtime,{'kind':'native_argformer_tsinfer_fit_v2_root_runtime','authorizer':'root_agent',
      'active':True,'CPU_only':stage=='retrieve',**c['runtime']},'exact fresh isolated runtime authority required')
    if not re.fullmatch('[0-9a-f]{64}',str(runtime.get('python_executable_sha256'))):raise ValueError('exact interpreter SHA required')
    if not runtime.get('environment_inventory_receipt'):raise ValueError('root-bound unchanged native environment inventory receipt required')
    bound(runtime['environment_inventory_receipt'],json_only=True)
    additive_helper_paths(runtime)
    require_fields(registration,{'kind':'native_argformer_tsinfer_fit_v2_root_experiment_registration',
      'authorizer':'root_agent','active':True,'condition':c['condition'],'scope_parents':[parent],
      'stage':stage,'implementation':expected['implementation'],'config':expected['config'],
      'query_truth_authorized':False},'explicit H013 repeat condition registration required')
    if not registration.get('hypothesis_id') or not registration.get('repeat_reason'):raise ValueError('registered hypothesis and repeat reason required')
    return expected


def local_bytes(path):
    p=confined(path)
    if not p.exists():return 0
    if p.is_file():return p.stat().st_size
    total=0
    def onerror(error):raise error
    for parent,dirs,files in os.walk(p,followlinks=False,onerror=onerror):
        for name in dirs+files:
            q=Path(parent)/name
            if q.is_symlink():raise ValueError('owned scope symlinks forbidden')
        for name in files:
            q=Path(parent)/name
            if not q.is_file():raise ValueError('owned nonregular file forbidden')
            total+=q.stat().st_size
    return total


class Storage:
    """No reservation writes or opportunistic ledger rebase. First hash is exact.

    Later independent reservations may change ledger bytes. Schema, snapshot,
    baseline, external reserve, ceiling and this own reservation remain exact.
    Successor accounting is delegated only to separately bound reviewed code.
    """
    def __init__(self,c,parent,stage,run_id,receipt):
        self.c=c;self.receipt=receipt;self.first=True;self.high_water=0
        self.scope,self.output,self.tmp,self.log=scopes(c,parent,stage,run_id)
        self.key=c['storage']['reservation_key_prefix']+'/'+parent
        require_fields(receipt,{'kind':'native_argformer_tsinfer_fit_v2_root_storage','authorizer':'root_agent',
          'active':True,'implementation':reference(SCRIPT),'config':reference(CONFIG),'scope_parents':[parent],
          'reservation_key':self.key,'reserved_bytes':c['storage']['reserved_bytes_per_parent'],
          'output':str(self.scope),'write_scopes':[str(self.scope),str(self.tmp)],
          'stage_boundary_cap_not_OS_quota':True},'fresh own combined20GB storage authority required')
        if receipt.get('mode') not in ('legacy450','separately_audited_successor'):raise ValueError('explicit storage authority version required')
        self.companion=None
        if receipt['mode']=='separately_audited_successor':
            migration=read_json(receipt['migration_authority'])
            require_fields(migration,{'kind':'project_budget_v4_root_reserve_admission','root_authorizer':'root_agent',
              'active':True,'no_legacy_mutation_authorized':True,'scientific_execution_authorized':False,
              'global_rebase_authorized':False},'separate independently audited root supplemental budget authority required')
            companion=receipt['companion']
            if (not exact(companion,c['storage']['companion_contract'])
              or migration.get('implementation')!=companion['implementation'] or migration.get('config')!=companion['config']
              or receipt['reservation'].get('root_admission')!=receipt['migration_authority']):
                raise ValueError('exact audited companion and stored root reserve admission binding required')
            p=bound(companion['implementation'])
            bound(companion['config'],json_only=True)
            spec=importlib.util.spec_from_file_location('_exact_root_budget_successor',p)
            self.companion=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.companion)
        self.last=self.check()

    def check(self):
        r=self.receipt;c=self.c
        if self.companion is not None:
            reservation=r['reservation']
            require_fields(reservation,{'key':self.key,'output':str(self.scope),
              'auxiliary_scopes':[str(self.tmp)],'reserved_bytes':c['storage']['reserved_bytes_per_parent']},
              'fresh exact combined parent and short-tmp successor reservation required')
            pins=({'legacy_ledger_sha256':r['legacy_ledger_sha256_at_admission'],
                   'supplemental_ledger_sha256':r['supplemental_ledger_sha256_at_admission']} if self.first else None)
            account=self.companion.check_reservation(ROOT,reservation=reservation,
              expected_source_sha256=r['companion']['implementation']['sha256'],
              expected_config_sha256=r['companion']['config']['sha256'],initial_ledger_pins=pins,boundary=True)
            if account['projected_bytes']>=account['limit_bytes'] or account['limit_bytes']>c['storage']['maximum_project_budget_bytes']:
                raise ValueError('combined successor projection/authorized1TB ceiling exceeded')
        else:
            lock=confined(c['storage']['scheduler_lock']);ledger=confined(c['storage']['legacy_ledger_path'])
            fd=os.open(lock,os.O_RDWR|os.O_NOFOLLOW)
            with os.fdopen(fd,'r+') as stream:
                disk=lock.stat();held=os.fstat(stream.fileno())
                if not stat.S_ISREG(held.st_mode) or (disk.st_dev,disk.st_ino)!=(held.st_dev,held.st_ino):raise ValueError('exact existing scheduler lock required')
                fcntl.flock(stream,fcntl.LOCK_EX)
                h=sha(ledger)
                if self.first and h!=r['legacy_ledger_sha256_at_admission']:raise ValueError('exact live ledger required at first admission')
                state=json.loads(ledger.read_text())
                fixed={'schema','root','snapshot_sha256','baseline_bytes','external_reserve_bytes','limit_bytes'}
                if set(state)!=fixed|{'jobs','controllers'} or not isinstance(state['jobs'],dict) or not isinstance(state['controllers'],dict):
                    raise ValueError('exact legacy ledger schema and typed collections required')
                binding=r['legacy_binding']
                if set(binding)!=fixed:raise ValueError('complete exact legacy invariant binding required')
                require_fields(state,binding,'pinned schema/baseline/snapshot/external/ceiling requires separate audited migration')
                if binding.get('schema')!='gpu_pool_storage_ledger_v1' or binding.get('root')!=str(ROOT) or not 0<binding['limit_bytes']<=c['storage']['legacy_maximum_limit_bytes']:
                    raise ValueError('unchanged legacy450GB authority required')
                require_fields(state['jobs'].get(self.key,{}),{'reserved_bytes':c['storage']['reserved_bytes_per_parent'],'output':str(self.scope)},'new exact own reservation required; no borrowed slot')
                reserves=state['external_reserve_bytes']
                for collection in ('jobs','controllers'):
                    for value in state[collection].values():
                        fields={'reserved_bytes','output'} if collection=='jobs' else {'reserved_bytes','pid'}
                        if not isinstance(value,dict) or set(value)!=fields:raise ValueError('exact legacy reservation entry schema required')
                        n=value.get('reserved_bytes')
                        if type(n) is not int or n<0:raise ValueError('typed nonnegative reservations required')
                        reserves+=n
                projected=state['baseline_bytes']+reserves
                fs=os.statvfs(ROOT);free=fs.f_bavail*fs.f_frsize
                if projected>=state['limit_bytes'] or reserves>=free:raise ValueError('project or filesystem conservative headroom exceeded')
                if sha(ledger)!=h:raise ValueError('locked ledger changed during read')
                account={'legacy_ledger_sha256':h,'projected_bytes':projected,'limit_bytes':state['limit_bytes'],'reserve_bytes':reserves}
        used=local_bytes(self.scope)+local_bytes(self.tmp)
        self.high_water=max(self.high_water,used)
        if self.high_water>c['storage']['reserved_bytes_per_parent']:raise ValueError('all-stage output/tmp/cache/log20GB sampled cap exceeded')
        self.first=False
        self.last={**account,'local_bytes':used,'high_water_bytes':self.high_water,'readonly':True,'stage_boundary_cap_not_OS_quota':True}
        return self.last


def additive_helper_paths(runtime):
    """Explicit additive real library, never silent import from another environment."""
    helpers=runtime.get('additive_helpers',[])
    if not isinstance(helpers,list) or len(helpers)>1:raise ValueError('bounded explicit additive runtime helpers required')
    paths=[]
    for h in helpers:
        if set(h)!={'module','version','implementation','wheel','directory_inventory'} or h['module']!='threadpoolctl':
            raise ValueError('only separately root-bound threadpoolctl additive helper is supported')
        module=bound(h['implementation']);bound(h['wheel'])
        isolated=confined('.runtime/isolated_helpers')
        if isolated not in module.parents or module.name!='threadpoolctl.py':raise ValueError('isolated exact helper module required')
        inventory=read_json(h['directory_inventory'])
        if inventory.get('directory')!=str(module.parent) or not isinstance(inventory.get('files_sha256'),dict):
            raise ValueError('root-bound complete isolated helper directory inventory required')
        actual={}
        for p in module.parent.rglob('*'):
            if p.is_symlink():raise ValueError('helper directory symlinks forbidden')
            if p.is_file():actual[str(p.relative_to(module.parent))]=sha(p)
        if not exact(actual,inventory['files_sha256']):raise ValueError('isolated additive helper directory changed')
        if any(Path(p).suffix in ('.py','.so') and p!='threadpoolctl.py' for p in actual):raise ValueError('additive helper cannot shadow native scientific modules')
        paths.append(str(module.parent))
    return paths


def native_executable(c):
    # The pinned conda bin/python is a final-component symlink to python3.11.
    # Admit only that fixed alias and its confined same-bin target, never input
    # artifact symlink traversal or an interpreter from another environment.
    prefix=confined(c['runtime']['prefix']);p=Path(c['runtime']['interpreter'])
    p=p if p.is_absolute() else ROOT/p
    if p!=prefix/'bin/python' or p.resolve().parent!=prefix/'bin' or not p.is_file() or not os.access(p,os.X_OK):
        raise ValueError('fixed native Python alias with same-prefix canonical target required')
    return p


def runtime_preflight(c,runtime,CPU_only):
    executable=native_executable(c);prefix=confined(c['runtime']['prefix'])
    if (Path(sys.prefix).resolve()!=prefix or Path(sys.executable).resolve()!=executable.resolve()
      or list(sys.version_info[:3])!=c['runtime']['python_version'] or sha(executable)!=runtime['python_executable_sha256']):
        raise ValueError('exact isolated native interpreter required')
    if CPU_only and os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU retrieval must explicitly hide CUDA')
    helper_paths=additive_helper_paths(runtime)
    if os.environ.get('PYTHONPATH','')!=os.pathsep.join(helper_paths):raise ValueError('exact admitted additive helper PYTHONPATH required')
    for h in runtime.get('additive_helpers',[]):
        module=__import__(h['module'])
        if Path(module.__file__).resolve()!=bound(h['implementation']) or getattr(module,'__version__',None)!=h['version']:
            raise ValueError('actual imported additive helper origin/version differs')
    return {'interpreter':str(executable),'executable_sha256':sha(executable),'CPU_only':CPU_only}


def prepare_environment(c,output,tmp,CPU_only):
    # TMPDIR is deliberately short: source multiprocessing AF_UNIX suffix is32bytes.
    socket=tmp/'pymp-12345678/listener-12345678'
    if len(os.fsencode(socket))>107:raise ValueError('native multiprocessing socket path limit exceeded')
    tmp.mkdir(parents=True,exist_ok=True)
    os.environ.update(TMPDIR=str(tmp),TMP=str(tmp),TEMP=str(tmp),PYTHONNOUSERSITE='1',
      PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',
      HF_HUB_DISABLE_TELEMETRY='1',ARGGNN_BIOLOGICAL_NATIVE_CODE_SHA256=sha(SCRIPT))
    for key in CACHE_VARS:
        p=output/'write_scopes'/key.lower();p.mkdir(parents=True);os.environ[key]=str(p)
    for key in THREAD_VARS:os.environ[key]='4'
    os.environ.pop('PYTHONHOME',None)
    if CPU_only:os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['PYTHONPATH']=''
    sys.dont_write_bytecode=True


def bootstrap(c,args,a,runtime,storage,ownership):
    _,output,tmp,_=scopes(c,args.parent,args.stage,args.run_id)
    if output.exists():raise ValueError('fresh root-assigned stage output required')
    executable=native_executable(c)
    if (not executable.is_file() or sha(executable)!=runtime['python_executable_sha256']
      or executable.resolve().parent!=confined(c['runtime']['prefix'])/'bin'):
        raise ValueError('root bound isolated interpreter required before output ownership')
    guard=None;lease=None
    if args.stage!='retrieve':
        guard=load_frozen_module(c,'src/arggnn/native_argformer_exec.py','_native_root_gpu_guard')
        lease=guard.validate_lease();guard.apply_and_validate_title()
        if os.environ.get('ARGGNN_LOGICAL_JOB_NAME')!=a['logical_job_name']:
            raise ValueError('exact root logical shared-lease GPU job required')
    storage.check();output.parent.mkdir(parents=True,exist_ok=True);output.mkdir();ownership['owns_output']=True
    prepare_environment(c,output,tmp,args.stage=='retrieve')
    os.environ['PYTHONPATH']=os.pathsep.join(additive_helper_paths(runtime))
    admission={'implementation_sha256':sha(SCRIPT),'stage':args.stage,'lease':lease,
      'root_admission':reference(args.admission),'storage':storage.check(),
      'TMPDIR':str(tmp),'multiprocessing_socket_example_bytes':len(os.fsencode(tmp/'pymp-12345678/listener-12345678'))}
    with (output/'exec_admission.json').open('x') as stream:json.dump(admission,stream,indent=2)
    if lease is not None:
        os.set_inheritable(lease['lease_fd'],True)
        if guard.validate_lease()!=lease:raise ValueError('actual inherited lease changed before exec')
    argv=[str(executable),'-B',str(SCRIPT),'--native',*sys.argv[1:]]
    os.chdir(ROOT);os.execve(executable,argv,os.environ.copy())


def load_frozen_module(c,path,name):
    p=bound({'path':path,'sha256':c['frozen_files_sha256'][path]})
    spec=importlib.util.spec_from_file_location(name,p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def verify_completed_fit(c,prepared,receipt):
    require_fields(receipt,{'status':'passed','action':'fit','exit_code':0,'root_actual_execution':True,
      'true_query_ancestry_read':False,'implementation_sha256':sha(SCRIPT),'config_sha256':sha(CONFIG),
      'parent_id':prepared['dispatcher']['parent_id'],'namespace':prepared['prepared']['namespace'],
      'prepared_manifest_sha256':prepared['dispatcher']['prepared_manifest']['sha256'],
      'published_native_task_reproduction':False,'classifier_unchanged_unused':True},'same-ARG exact completed native fit required')
    for phase,mode,epochs,count in [('pretraining','mlm',5,prepared['budget']['MLM_optimizer_updates']),
        ('contrastive','contrastive',1,prepared['budget']['contrastive_optimizer_updates'])]:
        r=receipt[phase]
        require_fields(r,{'mode':mode,'execution':'full_batch','completed_epochs':epochs,'actual_optimizer_updates':count,
          'backbone_unchanged':True,'true_query_ancestry_read':False},'exact source epochs/batches/updates required')
        if r['initial_model_state_sha256']==r['final_model_state_sha256']:raise ValueError('actual optimizer changes required')
    if not receipt['encoder_initialization'].get('all_encoder_keys_loaded_equal') or receipt['encoder_initialization'].get('missing_encoder') or receipt['encoder_initialization'].get('unequal_encoder'):
        raise ValueError('complete exact trained MLM encoder initialization required')
    if receipt['contrastive']['execution']!='full_batch':raise ValueError('no contrastive fallback authorised')


def verify_memory(c,prepared,receipt,runtime):
    require_fields(receipt,{'status':'passed','action':'probe-memory','exit_code':0,
      'implementation_sha256':sha(SCRIPT),'config_sha256':sha(CONFIG),
      'prepared_manifest_sha256':prepared['dispatcher']['prepared_manifest']['sha256'],
      'parent_id':prepared['dispatcher']['parent_id'],'namespace':prepared['prepared']['namespace'],
      'full_batch_probe_attempted':True,'full_batch_probe_logical_batch':4096},'fresh actual same-ARG native4096 optimizer-memory PASS required')
    r=receipt['result']
    require_fields(r,{'logical_batch':4096,'execution':'actual_author_full_batch','optimizer_step':True,
      'probe_is_not_a_biological_fit':True,'temporary_probe_weights_restored':True},'genuine full4096 source objective/optimizer and restoration required')
    if not receipt.get('runtime',{}).get('actual_cuda') or type(r.get('peak_memory_bytes')) is not int:
        raise ValueError('actual CUDA memory measurement required')
    cap=runtime.get('maximum_GPU_memory_bytes')
    if type(cap) is not int or cap<=0 or r['peak_memory_bytes']>cap:raise ValueError('fresh exact reserved GPU memory cap required')
    if receipt['runtime']['device']['actual_uuid']!=runtime['qualified_GPU_UUID']:
        raise ValueError('source4096 probe and scheduled device qualification must match')


def execute_fit(c,args,a,prepared,storage,runtime,result):
    ns=compile_fit(c);recipe=read_json(c['native_recipe'])
    native_inputs=ns['prepared_inputs'];native_json=ns['json_new'];native_train=ns['train_phase']
    def prepared_inputs(directory,expected_sha,config_sha):
        if config_sha!=sha(CONFIG) or confined(directory)!=bound(prepared['dispatcher']['prepared_manifest']).parent:
            raise ValueError('new execution policy must bind original prepared recipe explicitly')
        value=native_inputs(directory,expected_sha,c['native_recipe']['sha256'])
        if not exact(value,prepared['prepared']):raise ValueError('immutable prepared metadata differs')
        return value
    def checked_json(path,value):
        storage.check();native_json(path,value)
    def checked_train(*arguments,**kwargs):
        storage.check();result=native_train(*arguments,**kwargs);storage.check();return result
    ns.update(prepared_inputs=prepared_inputs,json_new=checked_json,train_phase=checked_train,
      policy=lambda expected=None:recipe if expected==sha(CONFIG) else (_ for _ in ()).throw(ValueError('exact root execution policy SHA required')))
    guard=load_frozen_module(c,'src/arggnn/native_argformer_exec.py','_actual_native_gpu_guard')
    guard.apply_and_validate_title()
    if (guard.validate_lease()['canonical_uuid']!=runtime['qualified_GPU_UUID']
      or os.environ.get('ARGGNN_LOGICAL_JOB_NAME')!=a['logical_job_name']):
        raise ValueError('exact scheduled qualified GPU and logical job required before native model imports')
    helpers=load_frozen_module(c,'scripts/qualify_official_argformer_runtime.py','_actual_native_runtime_helpers')
    if args.stage=='fit':verify_memory(c,prepared,read_json(a['full_batch_memory_receipt']),runtime)
    native_args=types.SimpleNamespace(action=args.stage,expected_config_sha256=sha(CONFIG),
      prepared_dir=str(bound(prepared['dispatcher']['prepared_manifest']).parent),
      expected_prepared_sha256=prepared['dispatcher']['prepared_manifest']['sha256'],contrastive_execution='full_batch')
    result.update(status='failed',exit_code=1,true_query_ancestry_read=False,root_actual_execution=False)
    result=ns['execute_native'](native_args,scopes(c,args.parent,args.stage,args.run_id)[1],guard,helpers,result)
    if args.stage=='fit':verify_completed_fit(c,prepared,result)
    else:verify_memory(c,prepared,result,runtime)
    return result


def retrieval_parent(prepared,joins,roster,np):
    p=prepared['prepared'];coordinate=p['coordinate_contract']
    if len(roster)!=280 or [r['sample_index'] for r in roster]!=list(range(280)):raise ValueError('ordered complete280 observed roster required')
    query=[r for r in roster if r['role']=='query'];refs=[r for r in roster if r['role']=='pure_source_reference']
    if len(query)!=120 or len(refs)!=96 or sum(r['role']=='mixed_reference_only' for r in roster)!=64:
        raise ValueError('complete native observed roles required')
    if any(r['observed_reference_class'] is not None for r in roster if r['role']!='pure_source_reference'):
        raise ValueError('query and mixed reference ancestry labels forbidden')
    nodes=np.asarray([r['inferred_tree_node_id'] for r in roster],dtype=np.int64)
    if len(set(nodes.tolist()))!=280 or np.any(nodes<0):raise ValueError('unique actual internal sample IDs required')
    indexes=np.asarray(coordinate['retained_fixed100bp_indexes'],dtype=np.int64)
    coverage=np.asarray(coordinate['coverage_original_bp0'],dtype=np.float64)
    if coverage.shape!=(2,) or not np.isfinite(coverage).all() or not 0<=coverage[0]<coverage[1]<=2000000 or coordinate['local_to_original_offset_bp']!=coverage[0]:
        raise ValueError('original half-open observed support required')
    expected=np.flatnonzero((np.arange(50,2000000,100)>=coverage[0])&(np.arange(50,2000000,100)<coverage[1]))
    if not len(indexes) or not np.array_equal(indexes,expected) or np.any(np.diff(indexes)!=1):raise ValueError('all original retained centers required')
    positions=50+100*indexes
    checks={'grid_indexes':indexes,'original_positions_bp':positions,'local_positions_bp':positions-coverage[0],
      'query_nodes':np.asarray([r['inferred_tree_node_id'] for r in query]),
      'reference_nodes':np.asarray([r['inferred_tree_node_id'] for r in refs]),
      'reference_classes':np.asarray([r['observed_reference_class'] for r in refs])}
    if any(not np.array_equal(joins[k],v) for k,v in checks.items()):raise ValueError('exact physical/native ID/query/reference joins required')
    edges=100*np.r_[indexes,indexes[-1]+1].astype(np.float64)
    edges[0]=max(edges[0],coverage[0]);edges[-1]=min(edges[-1],coverage[1])
    if np.any(np.diff(edges)<=0):raise ValueError('positive clipped physical bins required')
    return {'parent_id':p['parent_id'],'positions':positions,'edges':edges,'retained_indexes':indexes,
      'coverage_original':coverage,'observation':{'nodes':nodes},
      'query_slots':np.asarray([r['sample_index'] for r in query],dtype=np.int64),
      'raw_query_nodes':np.asarray([r['source_observed_node_id'] for r in query],dtype=np.int64)}


def execute_retrieval(c,args,a,prepared,storage,runtime):
    fit_path=bound(a['completed_fit_receipt']);fit=read_json(a['completed_fit_receipt'])
    verify_completed_fit(c,prepared,fit)
    if fit_path.name!='receipt.json' or confined(Path(c['result_root'])/args.parent/'fit') not in fit_path.parents:
        raise ValueError('completed immutable fit in this new parent scope required')
    artifacts=fit['artifacts_sha256'];directory=bound(prepared['dispatcher']['prepared_manifest']).parent
    ns=compile_retrieval(c)
    with ns['cpu_libraries']({'retrieval_rule':c['retrieval_rule']}) as (np,torch,threads):
        join_path=bound({'path':str(directory/'coordinate_path_joins.npz'),'sha256':prepared['prepared']['files_sha256']['coordinate_path_joins.npz']})
        with np.load(join_path,allow_pickle=False) as z:
            expected={'grid_indexes','original_positions_bp','local_positions_bp','query_nodes','reference_nodes','reference_classes','query_path_ids','reference_path_ids'}
            if set(z.files)!=expected:raise ValueError('exact native coordinate join schema required')
            joins={k:z[k].copy() for k in z.files}
        roster=read_json({'path':str(directory/'observed_roster.json'),'sha256':prepared['prepared']['files_sha256']['observed_roster.json']})
        parent=retrieval_parent(prepared,joins,roster,np)
        metadata={'reference_full_path_count':prepared['budget']['pure_reference_corpus_paths_all_train_plus_val'],
          'embeddings':{'path':str(fit_path.parent/'native_path_embeddings.pt'),'sha256':artifacts['native_path_embeddings.pt']},
          'original_mean_readout':{'path':str(fit_path.parent/'downstream_C4_local_readout.npz'),'sha256':artifacts['downstream_C4_local_readout.npz']}}
        # Both native train and val path splits remain the entire pure reference corpus.
        for record in metadata.values():
            if isinstance(record,dict):bound(record)
        state={'prepared':prepared['prepared'],'records':{'prepared_manifest':prepared['dispatcher']['prepared_manifest'],'fit_receipt':a['completed_fit_receipt']},
               'joins':joins,'receipt':fit}
        storage.check()
        seal=ns['seal_parent'](parent,state,{'retrieval_rule':c['retrieval_rule']},metadata,scopes(c,args.parent,args.stage,args.run_id)[1],torch)
        storage.check()
        if seal['diagnostics']['unique_query_paths']!=prepared['budget']['actual_counts']['query_full_paths_scored'] or seal['diagnostics']['exhaustive_pairs']!=prepared['budget']['exhaustive_query_corpus_pairs']:
            raise ValueError('sealed retrieval must preserve every qualified query and reference pair')
        if torch.cuda.is_initialized():raise ValueError('CPU retrieval must never create CUDA context')
    return {'status':'passed','exit_code':0,'action':'retrieve','parent_id':args.parent,
      'namespace':prepared['prepared']['namespace'],'prepared_manifest_sha256':prepared['dispatcher']['prepared_manifest']['sha256'],
      'implementation_sha256':sha(SCRIPT),'config_sha256':sha(CONFIG),'prediction_seal':seal,
      'CPU_only':True,'BLAS_threads':threads,'primary_k':1,'sensitivity_k':5,
      'true_query_ancestry_read':False,'external_truth_access_authorized':False,
      'published_native_task_reproduction':False,'auxiliary_local_mean_readout_primary':False}


def source_report(c):
    verify_source(c)
    return {'kind':'native_argformer_tsinfer_fit_v2_source_report','status':'inactive_source_only',
      'implementation':reference(SCRIPT),'config':reference(CONFIG),'native_recipe':c['native_recipe'],
      'native_ASTs':ast_proof(c),'claims':c['claims'],'actual_arrays_opened':False,
      'native_model_executed':False,'GPU':False,'live_ledger_read':False}


def templates(c,parent='bio_paths_pilot_train_000',stage='probe-memory',run_id='root_assign_fresh_run'):
    expected=admission_bindings(c,parent,stage,run_id)
    inactive={**expected,'active':False,'status':'inactive_template_not_an_admission',
      'prepared_dispatcher':None,'independent_review':None,'runtime':None,'storage':None,
      'experiment_registration':None,'logical_job_name':None,'full_batch_memory_receipt':None,
      'completed_fit_receipt':None}
    parent_scope,output,tmp,log=scopes(c,parent,stage,run_id)
    return {'admission':inactive,
      'runtime':{'kind':'native_argformer_tsinfer_fit_v2_root_runtime','authorizer':'root_agent','active':False,
        'CPU_only':stage=='retrieve',**c['runtime'],'python_executable_sha256':None,
        'qualified_GPU_UUID':None,'maximum_GPU_memory_bytes':None,'environment_inventory_receipt':None,'additive_helpers':[]},
      'storage':{'kind':'native_argformer_tsinfer_fit_v2_root_storage','authorizer':'root_agent','active':False,
        'implementation':reference(SCRIPT),'config':reference(CONFIG),'scope_parents':[parent],
        'reservation_key':c['storage']['reservation_key_prefix']+'/'+parent,'reserved_bytes':c['storage']['reserved_bytes_per_parent'],
        'output':str(parent_scope),'write_scopes':[str(parent_scope),str(tmp)],'mode':None,
        'legacy_binding':None,'legacy_ledger_sha256_at_admission':None,'supplemental_ledger_sha256_at_admission':None,
        'companion':c['storage']['companion_contract'],'migration_authority':None,'reservation':None,'stage_boundary_cap_not_OS_quota':True},
      'experiment_registration':{'kind':'native_argformer_tsinfer_fit_v2_root_experiment_registration',
        'authorizer':'root_agent','active':False,'condition':c['condition'],'scope_parents':[parent],
        'stage':stage,'implementation':reference(SCRIPT),'config':reference(CONFIG),'hypothesis_id':'H013',
        'repeat_reason':'fresh native tsinfer/tsdate per-ARG representation, previous Relate inadequacy retained',
        'query_truth_authorized':False}}


def execute(c,args):
    if Path(__file__).resolve()!=SCRIPT or SCRIPT.is_symlink():raise ValueError('fixed new executor source required')
    if args.expected_source_sha256!=sha(SCRIPT) or args.expected_config_sha256!=sha(CONFIG):raise ValueError('exact reviewed source/config activation opt-in required')
    if not args.admission or not args.expected_admission_sha256:raise ValueError('fresh root stage admission required')
    a=read_json({'path':args.admission,'sha256':args.expected_admission_sha256})
    review=read_json(a['independent_review']);runtime=read_json(a['runtime']);registration=read_json(a['experiment_registration'])
    validate_admission(c,args.parent,args.stage,args.run_id,a,review,runtime,registration)
    # Old preparation admission is deliberately never accepted as new execution authority.
    prepared=prepared_metadata(c,args.parent,a['prepared_dispatcher'])
    storage=Storage(c,args.parent,args.stage,args.run_id,read_json(a['storage']))
    if not args.native:
        ownership={'owns_output':False}
        try:
            bootstrap(c,args,a,runtime,storage,ownership)
            raise RuntimeError('native exec unexpectedly returned')
        except BaseException as error:
            if not ownership['owns_output']:raise
            output=scopes(c,args.parent,args.stage,args.run_id)[1]
            failure={'status':'bootstrap_failed_partial_preserved','exit_code':1,'action':args.stage,
              'parent_id':args.parent,'implementation_sha256':sha(SCRIPT),'config_sha256':sha(CONFIG),
              'true_query_ancestry_read':False,'root_actual_execution':False,
              'failure':{'type':type(error).__name__,'message':str(error)[:4096]},
              'root_admission':reference(args.admission),'native_model_executed':False}
            with (output/'receipt.json').open('x') as stream:json.dump(failure,stream,sort_keys=True,indent=2)
            with (output/'exit_status').open('x') as stream:stream.write('1\n')
            return failure
    runtime_preflight(c,runtime,args.stage=='retrieve')
    _,output,tmp,_=scopes(c,args.parent,args.stage,args.run_id)
    admission=read_json(reference(output/'exec_admission.json'))
    require_fields(admission,{'implementation_sha256':sha(SCRIPT),'stage':args.stage,
      'root_admission':reference(args.admission),'TMPDIR':str(tmp)},'exact fresh owned child exec admission required')
    if (output/'receipt.json').exists() or os.environ.get('TMPDIR')!=str(tmp):raise ValueError('fresh native output and exact short temporary scope required')
    result={'status':'failed_partial_preserved','exit_code':1,'action':args.stage,'parent_id':args.parent,
      'implementation_sha256':sha(SCRIPT),'config_sha256':sha(CONFIG),'true_query_ancestry_read':False}
    started=time.monotonic()
    def timeout(signum,frame):raise TimeoutError('root-admitted stage walltime ceiling exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,c['maximum_walltime_seconds'][args.stage])
    try:
        result=(execute_retrieval(c,args,a,prepared,storage,runtime) if args.stage=='retrieve'
                else execute_fit(c,args,a,prepared,storage,runtime,result))
        verify_source(c);prepared_metadata(c,args.parent,a['prepared_dispatcher']);result['storage']=storage.check()
    except BaseException as error:
        result.update(status='failed_partial_preserved',exit_code=1,
          failure={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        result.update(elapsed_seconds=time.monotonic()-started,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          root_admission=reference(args.admission),prepared_dispatcher=a['prepared_dispatcher'],
          disclosed_additive_runtime_helpers=runtime.get('additive_helpers',[]),
          native_ASTs=ast_proof(c),paper_reproduction_claimed=False,scientific_fallback=False)
        result['artifacts_sha256']={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*'))
          if p.is_file() and 'write_scopes' not in p.relative_to(output).parts and p.name not in ('receipt.json','exit_status')}
        payload=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n'
        if len(payload.encode())>c['storage']['failure_receipt_max_bytes'] and result['exit_code']!=0:
            result['failure']['traceback']='omitted: bounded failure diagnostic allowance'
            payload=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n'
        with (output/'receipt.json').open('x') as stream:stream.write(payload)
        with (output/'exit_status').open('x') as stream:stream.write(str(result['exit_code'])+'\n')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--templates',action='store_true')
    parser.add_argument('--parent',default='bio_paths_pilot_train_000');parser.add_argument('--stage',choices=STAGES,default='probe-memory')
    parser.add_argument('--run-id',default='root_assign_fresh_run');parser.add_argument('--expected-source-sha256')
    parser.add_argument('--expected-config-sha256');parser.add_argument('--admission');parser.add_argument('--expected-admission-sha256')
    parser.add_argument('--native',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv);sys.dont_write_bytecode=True;c=policy();verify_source(c)
    if args.execute:
        result=execute(c,args);print(json.dumps(result,indent=2));return result['exit_code']
    print(json.dumps(templates(c,args.parent,args.stage,args.run_id) if args.templates else source_report(c),indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
