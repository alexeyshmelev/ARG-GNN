"""Inactive ROOT-only adapter for the exact reviewed single observed-hold qualifier.
No import-time project imports, probes, scientific loads, accounting or writes.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,collections,ctypes,hashlib,importlib.metadata,importlib.util,json,os,platform,resource,selectors,signal,stat,subprocess,sys,tempfile,time
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE='research/reviews/H058_one_hold_ROOT_execution_source_20261016_v4'
SOURCE=HERE+'/root_qualify_v4.py'
MANIFEST=HERE+'/source_manifest.v4.json'
PACKET=HERE+'/admitted_v4'
OUTPUT='experiments/results/H058_one_observed_hold_FP32_qualification_root_v4'
JOB='H058-one-observed-hold-FP32-qualification-v4'
RUNNER={'path':'research/reviews/same_marker_one_hold_FP32_qualification_interface_successor_source_20261016_v2/qualify_one_observed_hold_v2.py','sha256':'f9930bc1520b7e17cdb5d087d62081182c9775c1587bb86ecffe9553a4d3ec77'}
PIDFD={'path':'research/reviews/American_C3_native_MLM_full_source_v5/pidfd_backend_v1.py','sha256':'61be5d89357d1b95611e0ba26dd39d381ececf3ce6a36c347507cdc062373eca'}
OWNED={'path':'research/reviews/American_C3_native_MLM_amendment_v12/owned_lifecycle_v2.py','sha256':'93d71acc40f1037d0b37acb2ca0859300ea4bd9741409e9bd00ca21bf100d633'}
RENEWAL={'path':'research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py','sha256':'491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468'}
CERT={'path':'research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json','sha256':'66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7'}
GUARD={'path':'research/reviews/prospective980_H020_legacy_source_20261016_v3/prospective_980gb_H020_v3.py','sha256':'ec475ecf59f0335397ee8b310cd0cb406ce4e03161e7c791861c6583799e6635'}
POLICY={'path':'research/reviews/prospective980_H020_root_activation_20261016_v3/activation_policy.root_active_v1.json','sha256':'7360bd28f57c8eaef19336ccd77eba193afae554e98d2ef0726e8ec879d7e582'}
REPAIR={'path':'research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json','sha256':'f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db'}
RUNTIME={'path':'research/reviews/ARGformer_American_C3_implementation_20261015_v1/shared_cpu_runtime_metadata.json','sha256':'e0944a1c2c4b2a26d2edfe68fad11db691b4b6853ebeddd8e3b79692b43fa602'}
MEMO={'path': 'research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py', 'sha256': 'c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364'}
NAMESPACE_MEMO={'path': 'research/reviews/current_mutex_contention_20261015_v1/consumer_namespace_memo_v1.py', 'sha256': '17a1377320f61c3c0cc8d3816d0259f9dae3121ed91bf7206a7ee63fcab8eb7f'}
V4={'path': 'src/arggnn/project_budget_v4.py', 'sha256': 'a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47'}
CAPS=dict(max_seconds=600,max_owned_RSS_bytes=8*2**30,CPU_threads=8,reservation_cap_bytes=8*2**20,log_cap_bytes=256*2**10,receipt_cap_bytes=512*2**10,optimizer_updates=0,physical_GPUs=[],precision='FP32')
def require(ok,message):
    if not ok:raise ValueError(message)
def local(path):
    require(type(path) is str and not Path(path).is_absolute() and '..' not in Path(path).parts,'Contained project path')
    p=ROOT/path
    require(p.resolve()==p and p.is_relative_to(ROOT),'Canonical contained path')
    return p
def ref(path):
    p=local(path)
    require(p.is_file() and not p.is_symlink() and p.stat().st_size<=32*2**20,'Bounded source/metadata')
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(r):
    require(isinstance(r,dict) and set(r)=={'path','sha256'} and ref(r['path'])==r,'Exact actual source/JSON ref')
    return json.loads(local(r['path']).read_bytes())
def load(r,name):
    require(ref(r['path'])==r,'Exact source before private import')
    sp=importlib.util.spec_from_file_location(name,local(r['path']))
    m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def active(q):
    now=datetime.now(timezone.utc);a=datetime.fromisoformat(q['created_at_utc']);b=datetime.fromisoformat(q['expires_at_utc'])
    require(a.tzinfo and b.tzinfo and a<=now<b,'Current actual ROOT authority')
def unique(rows):
    d={}
    for r in rows:
        require(ref(r['path'])==r and (r['path'] not in d or d[r['path']]==r),'Complete exact unique bindings')
        d[r['path']]=r
    require(len(d)<=64,'Original64 direct-consumer ceiling')
    return list(d.values())
def source_refs_v1():
    manifest=read(ref(MANIFEST));rows=manifest['refs']
    require(all(ref(r['path'])==r for r in rows) and ref(SOURCE) in rows,'Exact full adapter/science/source closure')
    return rows
def guard_v1(*,consumer=True):
    require(type(consumer) is bool,'Explicit original-mutation or read-only consumer role')
    sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
    load(RENEWAL,'_H058_original_renewal').install_renewal_v1(CERT)
    g=load(GUARD,'_H058_actual_H020v3_guard');g.configure_v2(REPAIR,policy_ref=POLICY)
    require(all(getattr(g.budget.v4,n).__module__==g.budget.v4.__name__
        for n in ('canonical','file_bytes','mutex')),'Fresh original unwrapped guard before role selection')
    if not consumer:return g
    require(ref(MEMO['path'])==MEMO and ref(V4['path'])==V4,
        'Exact reviewed transaction memo2 and original v4 sources')
    namespace=load(NAMESPACE_MEMO,'_H058_exact_consumer_namespace_memo')
    # The namespace entrypoint installs memo2 itself exactly once. This process
    # performs read/check transactions only; root mutations use consumer=False.
    handle=namespace.install_consumer_v1(g.budget.v4,g.budget,
        consumer_role='read_only_consumer',expected_source_path=ROOT/V4['path'],
        expected_source_sha256=V4['sha256'])
    g._H058_readonly_memo_v1=handle
    return g
def validate_request_v1(request_ref):
    q=read(request_ref);active(q)
    require(set(q)=={'schema','status','root_authorizer','execution_authorized','GPU_authorized','job_id','output_directory','packet_directory','adapter_source_ref','runner_ref','source_manifest_ref','runtime_ref','caps_ref','created_at_utc','expires_at_utc'},'Exact narrow actual request')
    require(q['schema']=='H058_one_hold_ROOT_request_v1' and q['status']=='active' and q['root_authorizer']=='root_agent' and q['execution_authorized'] is True and q['GPU_authorized'] is False and q['job_id']==JOB and q['output_directory']==OUTPUT and q['packet_directory']==PACKET,'One fixed CPU-only qualification ROOT request')
    require(q['adapter_source_ref']==ref(SOURCE) and q['runner_ref']==RUNNER and q['source_manifest_ref']==ref(MANIFEST) and q['runtime_ref']==RUNTIME,'Exact reviewed source/runtime routes')
    caps=read(q['caps_ref'])
    require(caps==dict(schema='H058_one_hold_ROOT_caps_v1',root_authorizer='root_agent',active=True,**CAPS),'Exact600s/8GiB/8thread/8MiB caps')
    source_refs_v1();return q,caps

def worker_env_paths_v2():
    base=str(local(PACKET+'/worker_aux'))
    return dict(TMPDIR=base+'/tmp',TMP=base+'/tmp',TEMP=base+'/tmp',
        XDG_CACHE_HOME=base+'/cache',XDG_CONFIG_HOME=base+'/config',
        XDG_DATA_HOME=base+'/data',XDG_STATE_HOME=base+'/state',
        TORCH_HOME=base+'/cache/torch',TORCH_EXTENSIONS_DIR=base+'/cache/torch_extensions',
        PYTORCH_KERNEL_CACHE_PATH=base+'/cache/torch_kernels',
        TRITON_CACHE_DIR=base+'/cache/triton',NUMBA_CACHE_DIR=base+'/cache/numba',
        MPLCONFIGDIR=base+'/cache/matplotlib',PYTHONPYCACHEPREFIX=base+'/cache/pycache',
        CUDA_CACHE_PATH=base+'/cache/cuda',PIP_CACHE_DIR=base+'/cache/pip',
        HF_HOME=base+'/cache/huggingface',HF_HUB_CACHE=base+'/cache/huggingface/hub')
def validate_runtime_v2(runtime):
    require(runtime['schema']=='american_c3_cpu_runtime_metadata_v1'
        and runtime['metadata_only'] is True and runtime['CUDA_VISIBLE_DEVICES']=='',
        'Exact saved CPU runtime metadata contract')
    require(sys.prefix==runtime['prefix']==str(ROOT/'.runtime/env')
        and platform.python_version()==runtime['python']
        and Path(sys.executable).resolve()==Path(runtime['executable']).resolve(),
        'Exact saved runtime interpreter/prefix/Python')
    for module,distribution in (('numpy','numpy'),('torch','torch'),('torch_geometric','torch-geometric')):
        require(importlib.metadata.version(distribution)==runtime['packages'][module],
            'Installed distribution metadata matches saved runtime: '+module)
def charged_census_v2(*,required_write_bytes=0):
    """Bounded live metadata sample, never a stable-tree or payload snapshot.

    Both charged roots, directories, temporary files and caches count. Regular
    files use max(logical, allocated) bytes; links and special nodes fail closed.
    Vanished active temporary entries are allowed. Parent samples until drain
    and worker samples at the unchanged runner's resource callbacks.
    """
    require(type(required_write_bytes) is int and required_write_bytes>=0,'Nonnegative local planned growth')
    total=entries=vanished=0
    def add(info,regular):
        nonlocal total,entries
        entries+=1;require(entries<=4096,'Bounded charged-scope metadata entries')
        require(stat.S_ISREG(info.st_mode) if regular else stat.S_ISDIR(info.st_mode),
            'Only charged directories and regular files')
        if regular:require(info.st_nlink==1,'No charged file hardlinks')
        total+=max(info.st_size,info.st_blocks*512) if regular else info.st_blocks*512
        require(total+required_write_bytes<=CAPS['reservation_cap_bytes'],
            'Shared8MiB output/temp/cache allocation cap')
    def visit(fd):
        nonlocal vanished
        add(os.fstat(fd),False)
        with os.scandir(fd) as scan:
            for entry in scan:
                try:info=entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    vanished+=1;continue
                if stat.S_ISDIR(info.st_mode):
                    try:child=os.open(entry.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
                    except FileNotFoundError:
                        vanished+=1;continue
                    try:visit(child)
                    finally:os.close(child)
                else:add(info,True)
    for path in (OUTPUT,PACKET+'/worker_aux'):
        fd=os.open(local(path),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:visit(fd)
        finally:os.close(fd)
    return dict(charged_scope_bytes=total,entry_count=entries,vanished_active_entries=vanished,
        reservation_cap_bytes=CAPS['reservation_cap_bytes'],
        scopes=[OUTPUT,PACKET+'/worker_aux'],
        method='sampled lstat/fstat metadata; directories allocated; regular max(logical,allocated); no payload reads',
        simultaneous_stable_snapshot_claimed=False)

def exclusive_json_v1(directory,name,value,g,storage,*,cap=2**20):
    require(directory in (OUTPUT,PACKET+'/worker_aux'),'Exact charged compact-write scopes')
    require(Path(name).name==name and name not in ('','.','..'),'Flat exclusive output name')
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    require(len(raw)<=cap,'Compact fixed-cap JSON output')
    charged_census_v2(required_write_bytes=len(raw)+8192)
    # Every compact write is quiescent and invokes the unchanged strict guard.
    g.budget_check_v9(storage,boundary=True,required_write_bytes=len(raw)+8192)
    p=local(directory)/name
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno());opened=os.fstat(f.fileno())
    named=p.lstat()
    require(stat.S_ISREG(named.st_mode) and named.st_nlink==1 and (opened.st_dev,opened.st_ino)==(named.st_dev,named.st_ino) and named.st_size==len(raw),'Exact created exclusive output identity')
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest())
def admit_v1(request_ref):
    require(Path.cwd()==ROOT,'Exact ROOT cwd')
    q,caps=validate_request_v1(request_ref);packet=local(PACKET)
    require(not packet.exists() and not local(OUTPUT).exists(),'Fresh exclusive admission/output')
    g=guard_v1(consumer=False);packet.mkdir()
    def save(name,v):
        p=packet/name
        with p.open('x') as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
        return ref(str(p.relative_to(ROOT)))
    config_ref=save('config.json',dict(schema='H058_one_hold_ROOT_config_v1',output_cap_bytes=8*2**20,**CAPS))
    implementation=ref(g.budget.SOURCE_REL);launcher=ref(g.budget.LAUNCHER_REL)
    with g.budget.v4.mutex(ROOT):state=g._state_locked(implementation['sha256'],launcher['sha256'],fresh=True)
    bindings=unique([*state[3].values(),*source_refs_v1(),ref(MANIFEST),request_ref,q['caps_ref'],config_ref,RUNTIME])
    reserve=dict(key='H058/'+JOB,output=str(local(OUTPUT)),auxiliary_scopes=[str(packet/'worker_aux')],reserved_bytes=8*2**20,consumer_bindings=bindings)
    authority=g.operation_draft_v1('reserve',reserve,expected_before=g._pins(state[2]),bindings=state[3]);authority.update(active=True,expires_at_utc=q['expires_at_utc'])
    authority_ref=save('reserve_authority.json',authority)
    receipt=g.apply_reserve_v1(authority_ref,expected_source_sha256=implementation['sha256'],expected_launcher_sha256=launcher['sha256']);receipt_ref=save('reserve_receipt.json',receipt)
    storage=dict(kind='project_budget_v9_storage_record',active=True,consumer_uses_v9_guard=True,accounting_mode=g.MODE,job_id=JOB,output_directory=OUTPUT,reserved_new_bytes=8*2**20,reservation=receipt['value'],initial_ledger_pins=receipt['initial_ledger_pins'],expires_at_utc=q['expires_at_utc'],completed_overlay_policy_ref=POLICY,budget_components=dict(implementation=implementation,launcher=launcher,consumer_adapter=GUARD))
    storage_ref=save('storage.json',storage)
    spec=dict(schema='H058_one_hold_ROOT_input_v1',status='active',root_authorizer='root_agent',job_id=JOB,output_directory=OUTPUT,root_stage_request_ref=request_ref,caps_ref=q['caps_ref'],refs=dict(storage=storage_ref,config=config_ref,runtime=RUNTIME),adapter_source_ref=ref(SOURCE),runner_ref=RUNNER,created_at_utc=q['created_at_utc'],expires_at_utc=q['expires_at_utc'])
    input_ref=save('input.json',spec);save('admitted.json',dict(input_ref=input_ref,reserve_receipt_ref=receipt_ref,consumer_binding_count=len(bindings),execution_started=False))
    return input_ref

def validate_input_v1(input_ref):
    require(input_ref['path']==PACKET+'/input.json','Exact one admitted input path')
    s=read(input_ref);active(s);q,caps=validate_request_v1(s['root_stage_request_ref'])
    require(s['schema']=='H058_one_hold_ROOT_input_v1' and s['status']=='active' and s['root_authorizer']=='root_agent' and s['job_id']==JOB and s['output_directory']==OUTPUT and s['adapter_source_ref']==ref(SOURCE) and s['runner_ref']==RUNNER and s['caps_ref']==q['caps_ref'] and s['created_at_utc']==q['created_at_utc'] and s['expires_at_utc']==q['expires_at_utc'],'Exact actual admitted narrow input')
    storage=read(s['refs']['storage']);cfg=read(s['refs']['config'])
    require(storage['job_id']==JOB and storage['reserved_new_bytes']==8*2**20 and storage['output_directory']==OUTPUT and storage['completed_overlay_policy_ref']==POLICY and storage['budget_components']['consumer_adapter']==GUARD and s['refs']['runtime']==RUNTIME,'Actual original H020v3 separately charged8MiB scope')
    require(cfg==dict(schema='H058_one_hold_ROOT_config_v1',output_cap_bytes=8*2**20,**CAPS),'Exact actual config')
    needed=[*source_refs_v1(),ref(MANIFEST),s['root_stage_request_ref'],s['caps_ref'],s['refs']['config'],RUNTIME]
    require(len(storage['reservation']['consumer_bindings'])<=64 and all(r in storage['reservation']['consumer_bindings'] for r in needed),'All actual sources/request/caps/config/runtime direct')
    return s,q,caps,storage

# Exact original supervisor helper bodies are appended below by source-only AST extraction.

class _RAMLog:
    def __init__(self,cap):self.cap=cap;self.parts=collections.deque();self.kept=0;self.total=0
    def add(self,data):
        self.total+=len(data);self.parts.append(data);self.kept+=len(data)
        while self.parts and self.kept>self.cap:
            first=self.parts.popleft();over=self.kept-self.cap
            if len(first)>over:self.parts.appendleft(first[over:]);self.kept-=over
            else:self.kept-=len(first)
    def record(self):
        return dict(tail=b"".join(self.parts).decode("utf-8",errors="replace"),
            total_pipe_bytes=self.total,retained_raw_bytes=self.kept,cap_bytes=self.cap,
            discarded_earlier_bytes=self.total-self.kept,buffered_in_RAM_until_actual_drain=True)

def _subreaper():
    libc=ctypes.CDLL(None,use_errno=True)
    require(libc.prctl(36,1,0,0,0)==0,"Linux subreaper needed to adopt/drain native DataLoader descendants")

def _children():
    raw=Path("/proc/self/task/%d/children"%os.getpid()).read_text().strip()
    return [int(p) for p in raw.split()] if raw else []

def rss_identity_v1(owned,identity):
    if not owned._same(owned._stat(identity['pid']),identity):return 0
    try:pages=int(Path('/proc/%d/statm'%identity['pid']).read_text().split()[1])
    except (FileNotFoundError,ProcessLookupError):return 0
    require(owned._same(owned._stat(identity['pid']),identity),'RSS observation identity changed')
    return pages*os.sysconf('SC_PAGE_SIZE')
def worker_v1(input_ref,gate_fd):
    require(os.read(gate_fd,1)==b'G','No numerical work before parent records ownership');os.close(gate_fd)
    s,q,caps,storage=validate_input_v1(input_ref)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and Path.cwd()==ROOT and os.getpid()==os.getpgrp()==os.getsid(0),'Dedicated CPU-only owned worker')
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(allowed[:8]))
    paths=worker_env_paths_v2()
    require(all(os.environ.get(k)==v and Path(v).resolve()==Path(v)
        and Path(v).is_dir() for k,v in paths.items()),
        'Every worker temp/cache/config path is inside actual charged worker_aux')
    require(sys.dont_write_bytecode,'No worker bytecode writes')
    tempfile.tempdir=paths['TMPDIR']
    require(tempfile.gettempdir()==paths['TMPDIR'],'No host temporary fallback')
    runtime=read(RUNTIME);validate_runtime_v2(runtime)
    g=guard_v1();start=time.monotonic();written=False
    def check():
        active(s);active(q)
        require(time.monotonic()-start<=600 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=8*2**30,'Worker600s/8GiB self cap')
        charged_census_v2()
        g.budget_check_v9(storage,boundary=False)
    def save(value):
        nonlocal written
        require(not written,'Exactly one compact qualification receipt')
        result=exclusive_json_v1(OUTPUT,'qualification_receipt.json',value,g,storage,cap=512*2**10)
        written=True;return result
    runner=load(RUNNER,'_H058_exact_reviewed_qualifier')
    result=runner.run_qualification_v1(check_resources_v1=check,save_compact_receipt_v1=save)
    require(written and result['status']=='PASS','Only exact runner PASS is successful')
    for name in ('numpy','torch','torch_geometric'):
        require(name in sys.modules and sys.modules[name].__version__==runtime['packages'][name],'Exact current scientific runtime: '+name)
    return 0

def execute_v1(input_ref):
    require(Path.cwd()==ROOT,'Exact ROOT cwd');s,q,caps,storage=validate_input_v1(input_ref)
    runtime=read(RUNTIME);validate_runtime_v2(runtime)
    g=guard_v1();g.validate_storage_v9(s,dict(output_cap_bytes=8*2**20),storage,source_ref=ref(SOURCE),config_ref=s['refs']['config'],runtime_ref=RUNTIME)
    out,aux=local(OUTPUT),local(PACKET+'/worker_aux')
    require(not out.exists() and not aux.exists(),'Fresh exclusive output and supervisor auxiliary scope')
    paths=worker_env_paths_v2()
    directories=sorted({Path(p) for p in paths.values()},key=lambda p:(len(p.parts),str(p)))
    g.budget_check_v9(storage,boundary=True,required_write_bytes=(2+len(directories))*8192)
    out.mkdir();aux.mkdir()
    for directory in directories:directory.mkdir()
    last_census=charged_census_v2();peak_scope=last_census['charged_scope_bytes']
    load(PIDFD,'pidfd_backend_v1');owned=load(OWNED,'_H058_original_owned_lifecycle')
    sys.modules['pidfd_backend_v1'].self_probe_v1();require(not owned._children(os.getpid()),'Supervisor has no unrelated children');_subreaper()
    parent_identity=owned._stat(os.getpid());logs=_RAMLog(256*2**10);selector=selectors.DefaultSelector()
    proc=None;tree=None;failure=None;cancelled=None;peak=0;stop_at=None;adopted=[];original={};gate_read=gate_write=None;start=time.monotonic()
    def stop(sig,frame):
        nonlocal cancelled
        cancelled=sig
    try:
        for sig in (signal.SIGTERM,signal.SIGINT):original[sig]=signal.signal(sig,stop)
        gate_read,gate_write=os.pipe()
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',NUMEXPR_NUM_THREADS='8',VECLIB_MAXIMUM_THREADS='8',ARGGNN_LOGICAL_JOB_NAME=JOB)
        env.update(paths)
        command=[runtime['executable'],'-B',str(local(SOURCE)),'--worker-input-ref',json.dumps(input_ref,separators=(',',':')),'--gate-fd',str(gate_read)]
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=0,start_new_session=True,pass_fds=(gate_read,))
        os.close(gate_read);gate_read=None
        tree=owned.OwnedTree(proc.pid)
        exclusive_json_v1(PACKET+'/worker_aux','launch.json',dict(schema='H058_actual_child_launch_v1',pid=proc.pid,parent_pid=os.getpid(),input_ref=input_ref,command=command,ownership=tree.evidence(),gate_still_closed=True),g,storage)
        os.set_blocking(proc.stdout.fileno(),False);selector.register(proc.stdout,selectors.EVENT_READ)
        require(cancelled is None,'Cancelled before gate release');os.write(gate_write,b'G');os.close(gate_write);gate_write=None
        while proc.poll() is None or selector.get_map() or tree.live() or owned._children(os.getpid()):
            tree.refresh();peak=max(peak,rss_identity_v1(owned,parent_identity)+sum(rss_identity_v1(owned,r) for r in tree.known.values()))
            try:
                last_census=charged_census_v2();peak_scope=max(peak_scope,last_census['charged_scope_bytes'])
            except Exception as exc:failure=failure or 'Charged scope census: '+type(exc).__name__+': '+str(exc)[:1000]
            if peak>8*2**30:failure=failure or 'Combined owned RSS cap exceeded'
            if time.monotonic()-start>600:failure=failure or 'ROOT600s operational deadline exceeded'
            if datetime.now(timezone.utc)>=datetime.fromisoformat(s['expires_at_utc']):failure=failure or 'Actual ROOT authority expired'
            if cancelled is not None:failure=failure or 'ROOT cancellation signal '+str(cancelled)
            if proc.poll() is not None and tree.live():failure=failure or 'Direct worker exited with live owned descendants'
            if failure:
                if stop_at is None:stop_at=time.monotonic()
                tree.signal_all(signal.SIGKILL if time.monotonic()-stop_at>=20 else signal.SIGTERM)
            for key,_ in selector.select(timeout=.2):
                data=os.read(key.fileobj.fileno(),65536)
                if data:logs.add(data)
                else:selector.unregister(key.fileobj)
            if proc.poll() is not None:
                proc.wait()
                while True:
                    try:pid,status=os.waitpid(-1,os.WNOHANG)
                    except ChildProcessError:break
                    if pid==0:break
                    adopted.append(dict(pid=pid,status=status))
    except BaseException as exc:failure=failure or type(exc).__name__+': '+str(exc)[:2000]
    finally:
        for fd in (gate_read,gate_write):
            if fd is not None:os.close(fd)
        if proc is not None:
            if tree is None:
                # Gate is still closed and its writer just closed. The child can
                # only observe EOF and fail before any qualification or imports.
                rc=proc.wait()
            else:
                while proc.poll() is None or tree.live() or owned._children(os.getpid()):
                    tree.signal_all(signal.SIGKILL)
                    if proc.poll() is not None:
                        proc.wait()
                        while True:
                            try:pid,status=os.waitpid(-1,os.WNOHANG)
                            except ChildProcessError:break
                            if pid==0:break
                            adopted.append(dict(pid=pid,status=status))
                    for key,_ in selector.select(timeout=.1):
                        data=os.read(key.fileobj.fileno(),65536)
                        if data:logs.add(data)
                        else:selector.unregister(key.fileobj)
                rc=proc.wait()
            # Drain any final pipe bytes without an on-disk active log.
            if proc.stdout is not None:
                while True:
                    try:data=os.read(proc.stdout.fileno(),65536)
                    except BlockingIOError:break
                    if not data:break
                    logs.add(data)
                proc.stdout.close()
        else:rc=None
        selector.close()
        for sig,handler in original.items():signal.signal(sig,handler)
    require(proc is not None and tree is not None,'No fabricated wait or ownership evidence')
    absent=tree.identities_absent() and not owned._children(os.getpid())
    groups=sorted({r['group'] for r in tree.known.values()});group_observations=[]
    for group in groups:
        try:os.killpg(group,0)
        except ProcessLookupError:gone=True
        else:gone=False
        group_observations.append(dict(pgid=group,absent=gone,method='targeted os.killpg(pgid,0) ESRCH'))
    drained=absent and all(r['absent'] for r in group_observations)
    require(absent,'No output/receipt inspection before all owned identities are absent')
    if not drained:failure=failure or 'Owned group absence not established'
    if rc!=0:failure=failure or 'Actual worker exit '+str(rc)
    if any(not os.WIFEXITED(r['status']) or os.WEXITSTATUS(r['status'])!=0 for r in adopted):failure=failure or 'Nonzero adopted descendant wait'
    # Persist the genuine wait/drain first, without asserting success, so a
    # later resource-check failure cannot erase the observed child code when
    # this strict quiescent compact write itself is admitted by original guard.
    minimal_wait_ref=exclusive_json_v1(PACKET+'/worker_aux','actual_wait_minimal.json',dict(
        schema='H058_actual_direct_wait_minimal_v1',pid=proc.pid,returncode=rc,
        actual_direct_worker_wait=True,input_ref=input_ref,ownership=tree.evidence(),
        all_owned_processes_and_groups_drained=drained,group_observations=group_observations,
        operational_failure=failure,success_not_yet_asserted=True),g,storage)
    # Additional strict original guard after actual wait/drain; the worker
    # receipt also uses its own strict quiescent boundary after numerical work.
    g.budget_check_v9(storage,boundary=True,required_write_bytes=2*2**20)
    receipt_path=out/'qualification_receipt.json';qualification_ref=ref(str(receipt_path.relative_to(ROOT))) if receipt_path.is_file() else None
    qualification=read(qualification_ref) if qualification_ref else None
    if not qualification or qualification.get('status')!='PASS':failure=failure or 'No exact original qualification PASS receipt'
    log_ref=exclusive_json_v1(PACKET+'/worker_aux','worker_log.json',logs.record(),g,storage,cap=2*2**20)
    final_census=charged_census_v2();peak_scope=max(peak_scope,final_census['charged_scope_bytes'])
    wait=dict(sampled_peak_charged_scope_bytes=peak_scope,last_active_scope_census=last_census,final_pre_wait_scope_census=final_census,worker_env_paths=paths,schema='H058_actual_child_wait_v1',pid=proc.pid,returncode=rc,actual_direct_worker_wait=True,input_ref=input_ref,completed_ref=qualification_ref,all_owned_processes_and_groups_drained=drained,ownership=tree.evidence(),group_observations=group_observations,adopted_waits=adopted,peak_combined_owned_RSS_bytes=peak,elapsed_seconds=time.monotonic()-start,operational_failure=failure,cancellation_signal=cancelled,log_ref=log_ref,actual_wait_minimal_ref=minimal_wait_ref,finished_at_utc=datetime.now(timezone.utc).isoformat())
    wait_ref=exclusive_json_v1(PACKET+'/worker_aux','actual_wait.json',wait,g,storage)
    g.budget_check_v9(storage,boundary=True)
    if failure:return dict(status='FAILED',actual_wait_ref=wait_ref,qualification_ref=qualification_ref,returncode=rc if rc else 1)
    completed_ref=exclusive_json_v1(OUTPUT,'completed.json',dict(status='completed_H058_one_observed_hold_FP32_qualification',phase='qualify',qualification_ref=qualification_ref,actual_wait_ref=wait_ref,input_ref=input_ref,ancestry_improvement_established=False,full_parent_feasibility_established=False,training_authorized=False),g,storage)
    g.budget_check_v9(storage,boundary=True)
    return dict(status='PASS',completed_ref=completed_ref,actual_wait_ref=wait_ref,returncode=0)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--admit-request-ref');p.add_argument('--execute-input-ref');p.add_argument('--worker-input-ref');p.add_argument('--gate-fd',type=int);a=p.parse_args(argv)
    require(sum(x is not None for x in (a.admit_request_ref,a.execute_input_ref,a.worker_input_ref))<=1,'One explicit ROOT action')
    if a.worker_input_ref:
        require(a.gate_fd is not None,'Owned parent gate FD required');return worker_v1(json.loads(a.worker_input_ref),a.gate_fd)
    if a.admit_request_ref:print(json.dumps(admit_v1(json.loads(a.admit_request_ref))));return 0
    if a.execute_input_ref:
        result=execute_v1(json.loads(a.execute_input_ref));print(json.dumps(result));return result['returncode']
    print(json.dumps(dict(status='SOURCE_ONLY_INACTIVE',admission=False,execution=False)));return 0
if __name__=='__main__':raise SystemExit(main())
