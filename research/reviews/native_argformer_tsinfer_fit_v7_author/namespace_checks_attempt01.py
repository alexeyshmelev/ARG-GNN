"""Private genuine AF_UNIX/stat metadata and unchanged-v4 accounting fixtures."""
from pathlib import Path
import builtins,contextlib,copy,datetime,hashlib,io,json,os,shutil,socket,stat,sys,types,uuid
from unittest.mock import patch
sys.dont_write_bytecode=True
R=Path('/workspace-SR003.nfs2/ARG-GNN');O=R/'research/reviews/native_argformer_tsinfer_fit_v7_author';FROOT=R/'.runtime/tmp/n7f'
sys.path.insert(0,str(R/'src'))
from arggnn import native_argformer_tsinfer_fit_v7 as m,project_budget_v4 as b
SOURCE=m.SCRIPT.read_bytes();BUDGET=(R/'src/arggnn/project_budget_v4.py').read_bytes();CFG=m.policy();PINS=json.loads((O/'baseline_pins.json').read_text())
checks=[];details={};guard=[];counterexamples=[];socket_manifests=[]
FROOT.mkdir(exist_ok=True);FIXTURES=[]
def sha(raw):return hashlib.sha256(raw).hexdigest()
def check(name,value=True):
 if not bool(value):raise AssertionError(name)
 checks.append(name)
def path(value):
 if isinstance(value,int):return None
 try:return Path(os.path.abspath(os.fsdecode(value)))
 except (TypeError,ValueError):return None
def own(p):return p is not None and (p==O or O in p.parents or p==FROOT or FROOT in p.parents)
def deny(why):guard.append(why);raise RuntimeError('HANDMADE ONLY: '+why)
def audit(event,args):
 if event=='open':
  p=path(args[0]);mode,flags=args[1:3];write=isinstance(mode,str) and any(c in mode for c in 'wax+') or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)
  if p is not None and write and not own(p):deny('outside ownership write '+str(p))
  if p is not None and not write and not own(p):
   for base in (R/'data',R/'experiments/results',R/'.runtime/conda/envs',R/'.runtime/gpup',R/'.runtime/locks',R/'.runtime/storage',Path('/proc')):
    if p==base or base in p.parents:deny('actual native/data/resource read '+str(p))
 elif event in ('os.mkdir','os.remove','os.rmdir','os.unlink','os.chmod','os.truncate','os.utime','os.chown'):
  p=path(args[0])
  if p is not None and not own(p):deny('outside ownership mutation '+str(p))
 elif event in ('os.rename','os.link'):
  for arg in args[:2]:
   p=path(arg)
   if p is not None and not own(p):deny('outside ownership mutation '+str(p))
 elif event=='os.symlink':
  p=path(args[1])
  if p is not None and not own(p):deny('outside ownership symlink '+str(p))
 elif event=='socket.bind':
  sock,address=args[:2]
  if sock.family!=socket.AF_UNIX or not own(path(address)):deny('outside private AF_UNIX bind')
 elif event in ('socket.connect','subprocess.Popen','os.system','os.posix_spawn'):deny(event)
sys.addaudithook(audit)
old_import=builtins.__import__
def imported(name,*args,**kwargs):
 if name.split('.')[0] in ('numpy','torch','tskit','tsinfer','transformers','composer','streaming'):deny('scientific import '+name)
 return old_import(name,*args,**kwargs)
builtins.__import__=imported
class Fixture:
 def __init__(self):
  self.F=FROOT/uuid.uuid4().hex[:8];self.F.mkdir();FIXTURES.append(self.F)
  self.scope=self.F/'r';self.tmp=self.F/'t';self.scope.mkdir();self.tmp.mkdir();self.handles=[]
 def sock(self,relative='pymp-12345678/listener-abcdefgh',root=None,mode=0o700,relative_bind=False):
  p=(self.tmp if root is None else root)/relative;p.parent.mkdir(parents=True,exist_ok=True);p.parent.chmod(mode)
  sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);self.handles.append(sock)
  if relative_bind:
   cwd=Path.cwd()
   try:os.chdir(self.F);sock.bind(str(p.relative_to(self.F)))
   finally:os.chdir(cwd)
  else:sock.bind(str(p))
  st=p.lstat();socket_manifests.append({'fixture':str(self.F.relative_to(R)),'path':str(p),'absolute_path_bytes':len(os.fsencode(p)),'mode':st.st_mode,'uid':st.st_uid,'nlink':st.st_nlink,'inode':st.st_ino,'family':'AF_UNIX','handmade':True})
  return p
 def measure(self):
  with patch.object(m,'ROOT',self.F):return m.native_owned_scope_measurement(self.scope,self.tmp)
 def close(self):
  for handle in self.handles:handle.close()
def rejects(name,call,reason=None):
 try:call()
 except m.NativeOwnedScopeError as error:
  data=error.namespace_metadata
  check(name,reason is None or reason in data['reason'])
  check(name+'_offender_diagnostic',set(data)=={'kind','reason','path','object'} and 'present' in data['object'])
  if data['object']['present']:check(name+'_inode_type_UID_nlink_metadata',all(k in data['object'] for k in ('type','uid','nlink','inode','mode','dev')))
  counterexamples.append({'case':name,'error':data});return data
 raise AssertionError('unsafe namespace accepted: '+name)
for p,h in PINS.items():check('pin_before_'+p,sha((R/p).read_bytes())==h)
# A truly canonical short absolute AF_UNIX listener, with no path-length seam.
f=Fixture();p=f.sock();r=f.measure()
check('real_canonical_short_AF_UNIX_accepted',r['accepted_AF_UNIX_socket_inodes']==1 and r['entries']==4 and len(os.fsencode(p))<=107)
check('socket_conservative4096_floor',r['conservative_scope_bytes']>=4096+sum(max(x.lstat().st_size,x.lstat().st_blocks*512) for x in (f.scope,f.tmp,p.parent)))
with patch.object(b,'canonical',side_effect=lambda root,value:Path(value) if Path(value).is_absolute() else root/str(value)):
 try:b.measure_scopes(f.F,[f.scope,f.tmp])
 except ValueError as error:check('unchanged_v4_regular_only_reproduces_guard','unsafe scope namespace/hardlink' in str(error))
 else:raise AssertionError('unchanged v4 accepted socket')
details['allowed_socket_measurement']=r
# No payload file is opened by the private scanner.
with patch.object(builtins,'open',side_effect=AssertionError('payload open')),patch.object(os,'open',side_effect=AssertionError('payload FD')):
 check('stat_only_scanner_uses_no_file_content_open',f.measure()['accepted_AF_UNIX_socket_inodes']==1)
for label,relative,where,mode in [
 ('result_socket','pymp-12345678/listener-abcdefgh','result',0o700),
 ('wrong_parent','pymp-1234567/listener-abcdefgh','tmp',0o700),
 ('wrong_listener','pymp-12345678/listener-abcdefg','tmp',0o700),
 ('extra_depth','extra/pymp-12345678/listener-abcdefgh','tmp',0o700),
 ('uppercase','pymp-1234567A/listener-abcdefgh','tmp',0o700),
 ('dash_alphabet','pymp-1234567-/listener-abcdefgh','tmp',0o700),
 ('wrong_parent_mode','pymp-12345678/listener-abcdefgh','tmp',0o755),
 ('parent_sticky','pymp-12345678/listener-abcdefgh','tmp',0o1700)]:
 f=Fixture();f.sock(relative,root=f.scope if where=='result' else f.tmp,mode=mode)
 rejects(label,f.measure)
f=Fixture();p=f.sock()
with patch.object(m.os,'getuid',return_value=os.getuid()+1):rejects('wrong_socket_UID',f.measure,'UID')
f=Fixture();p=f.sock();os.link(p,f.tmp/'second-link');rejects('socket_hardlink',f.measure)
for kind in ('FIFO','regular_hardlink','symlink','directory_alias','root_regular','root_symlink','root_FIFO','dangling_symlink'):
 f=Fixture()
 if kind=='FIFO':os.mkfifo(f.scope/'fifo')
 elif kind=='regular_hardlink':
  p=f.scope/'file';p.write_bytes(b'handmade');os.link(p,f.scope/'link')
 elif kind=='symlink':
  p=f.scope/'file';p.write_bytes(b'handmade');(f.scope/'alias').symlink_to(p)
 elif kind=='directory_alias':(f.scope/'alias').symlink_to(f.tmp,target_is_directory=True)
 elif kind=='root_regular':f.scope.rmdir();f.scope.write_bytes(b'file')
 elif kind=='root_symlink':f.scope.rmdir();f.scope.symlink_to(f.tmp,target_is_directory=True)
 elif kind=='root_FIFO':f.scope.rmdir();os.mkfifo(f.scope)
 else:(f.scope/'dangling').symlink_to(f.F/'missing')
 rejects(kind,f.measure)
# Real path byte boundaries, including an overlong absolute socket created via a relative bind.
for size,want in [(107,True),(108,False)]:
 f=Fixture();base_length=len(os.fsencode(f.F/'x'/'pymp-12345678/listener-abcdefgh'));name='x'*(1+size-base_length);f.tmp=f.F/name;f.tmp.mkdir()
 p=f.sock(relative_bind=not want)
 check('absolute_socket_length_exact_'+str(size),len(os.fsencode(p))==size)
 if want:check('socket_length107_accepted',f.measure()['accepted_AF_UNIX_socket_inodes']==1)
 else:rejects('socket_length108_rejected',f.measure,'107bytes')
# Directory membership, file/socket identity, and absent-root appearance races.
f=Fixture();(f.scope/'file').write_bytes(b'one');original_scandir=os.scandir;done=[]
class ChangedListing:
 def __init__(self,manager,path):self.manager=manager;self.path=path
 def __enter__(self):return self.manager.__enter__()
 def __exit__(self,*args):
  result=self.manager.__exit__(*args)
  if self.path==f.scope and not done:(f.scope/'new-entry').write_bytes(b'new');done.append(True)
  return result
def listing(path):
 return ChangedListing(original_scandir(path),Path(path))
with patch.object(m.os,'scandir',side_effect=listing):rejects('directory_membership_race',f.measure,'membership')
f=Fixture();p=f.sock();original_lstat=Path.lstat;seen=[]
def socket_identity(path,*args,**kwargs):
 value=original_lstat(path,*args,**kwargs)
 if path==p:
  seen.append(True)
  if len(seen)>1:
   changed={k:getattr(value,k) for k in ('st_dev','st_ino','st_mode','st_uid','st_gid','st_nlink','st_size','st_blocks','st_mtime_ns','st_ctime_ns')};changed['st_ino']+=1;return types.SimpleNamespace(**changed)
 return value
with patch.object(Path,'lstat',new=socket_identity):rejects('socket_inode_race',f.measure,'identity')
f=Fixture();file=f.scope/'file';file.write_bytes(b'one');seen=[]
def file_identity(path,*args,**kwargs):
 value=original_lstat(path,*args,**kwargs)
 if path==file:
  seen.append(True)
  if len(seen)>1:
   changed={k:getattr(value,k) for k in ('st_dev','st_ino','st_mode','st_uid','st_gid','st_nlink','st_size','st_blocks','st_mtime_ns','st_ctime_ns')};changed['st_nlink']=2;return types.SimpleNamespace(**changed)
 return value
with patch.object(Path,'lstat',new=file_identity):rejects('regular_file_hardlink_race',f.measure,'identity')
f=Fixture();f.scope.rmdir();done=[]
def absent_root_listing(path):
 manager=original_scandir(path)
 if Path(path)==f.tmp and not done:f.scope.mkdir();done.append(True)
 return manager
with patch.object(m.os,'scandir',side_effect=absent_root_listing):rejects('absent_root_creation_race',f.measure,'appeared')
f=Fixture();f.scope.rmdir();f.tmp.rmdir()
check('both_fresh_absent_scopes_measure_zero',f.measure()['entries']==0 and f.measure()['conservative_scope_bytes']==0)
# Genuine unchanged v4 code/policy with only fake-root identities and authorities.
class Accounting(Fixture):
 def __init__(self,cap=20000000000):
  super().__init__();F=self.F
  def put(name,value):
   p=F/name;p.parent.mkdir(parents=True,exist_ok=True);raw=value if isinstance(value,bytes) else (json.dumps(value,indent=2,sort_keys=True)+'\n').encode();p.write_bytes(raw)
   return {'path':name,'sha256':sha(raw)}
  self.put=put;source=put('s.py',SOURCE);helper=put('b.py',BUDGET);legacy_src=put('l.py',(R/'src/arggnn/gpu_pool_v1.py').read_bytes())
  api=types.ModuleType('unchanged_v4_'+F.name);exec(compile(BUDGET,str(F/'b.py'),'exec'),api.__dict__)
  api.ROOT=F;api.SOURCE=F/'b.py';api.CONFIG=F/'bc.json';api.FIXED_POLICY=copy.deepcopy(b.FIXED_POLICY)
  human=put('h.json',{'kind':'human_project_storage_authorization','project_root':str(F),'authorized_hard_limit_bytes':1000000000000,'future_scheduling_ceiling_bytes':950000000000,'handmade_only':True})
  api.FIXED_POLICY.update(root=str(F),human_authorization=human,legacy_pool_source=legacy_src);bc=put('bc.json',api.FIXED_POLICY)
  self.cfg=copy.deepcopy(CFG);self.cfg.update(result_root='r',short_tmp_root='t');self.cfg['storage']['reserved_bytes_per_parent']=cap;config=put('c.json',self.cfg)
  bindings={'implementation':helper,'config':bc,'human_authorization':human}
  review=put('rv.json',{'kind':'project_budget_v4_independent_review','review_role':'independent_reviewer','verdict':'PASS',**bindings,'actual_apply_performed':False,'handmade_only':True})
  parent=self.cfg['parent_ids'][0]
  with patch.object(m,'ROOT',F):self.scope,self.output,self.tmp,self.log=m.scopes(self.cfg,parent,'probe-memory','handmade')
  self.scope.mkdir(parents=True,exist_ok=True);self.tmp.mkdir(parents=True,exist_ok=True)
  key=self.cfg['storage']['reservation_key_prefix']+'/'+parent
  request={'key':key,'output':str(self.scope),'auxiliary_scopes':[str(self.tmp)],'reserved_bytes':cap,'consumer_bindings':[source,config]}
  root=put('reserve.json',{'kind':'project_budget_v4_root_reserve_admission','root_authorizer':'root_agent','active':True,**bindings,'all_budget_writers_use_scheduler_mutex':True,'no_legacy_mutation_authorized':True,'GPU_authorized':False,'scientific_execution_authorized':False,'global_rebase_authorized':False,'reservations':[request],'independent_review':review,'handmade_only':True})
  reservation={**request,'root_admission':root}
  self.legacy={**api.FIXED_POLICY['legacy_invariants'],'root':str(F),'jobs':{},'controllers':{'handmade':{'reserved_bytes':1,'pid':1}}}
  supplemental=api.empty_ledger(F,api.FIXED_POLICY,bindings);supplemental['reservations']={key:reservation}
  lr=put(api.LEGACY,self.legacy);sr=put(api.SUPPLEMENTAL,supplemental);lock=F/api.LOCK;lock.parent.mkdir(parents=True,exist_ok=True);lock.write_bytes(b'private handmade lock\n')
  self.store=object.__new__(m.Storage);self.store.c=self.cfg;self.store.receipt={'reservation':reservation,'companion':{'implementation':helper,'config':bc},'legacy_ledger_sha256_at_admission':lr['sha256'],'supplemental_ledger_sha256_at_admission':sr['sha256']}
  self.store.first=True;self.store.high_water=0;self.store.scope=self.scope;self.store.output=self.output;self.store.tmp=self.tmp;self.store.log=self.log;self.store.key=key;self.store.companion=api;self.api=api
  self.calls=[];actual=api.check_reservation
  def record(*args,**kwargs):
   self.calls.append(kwargs);return actual(*args,**kwargs)
  api.check_reservation=record
 def check(self):
  with patch.object(m,'ROOT',self.F):return self.store.check()
f=Accounting();p=f.sock();before={name:(f.F/name).read_bytes() for name in (f.api.LEGACY,f.api.SUPPLEMENTAL,f.api.LOCK)}
with patch.object(f.api,'measure_scopes',side_effect=AssertionError('global v4 boundary should not run')):r=f.check()
check('actual_v4_accounting_boundary_false_before_private_scan',f.calls[0]['boundary'] is False and f.calls[0]['initial_ledger_pins'] is not None and r['scope_measurement'] is None and r['native_owned_scope_measurement']['accepted_AF_UNIX_socket_inodes']==1)
check('real_v4_and_own_stat_count_charges_socket',r['local_bytes']==r['native_owned_scope_measurement']['conservative_scope_bytes'] and r['high_water_bytes']>=4096)
f.check();check('later_real_v4_own_reservation_rechecked_without_initial_pins',f.calls[-1]['initial_ledger_pins'] is None and f.calls[-1]['boundary'] is False)
check('private_accounting_and_lock_bytes_unchanged',all((f.F/name).read_bytes()==raw for name,raw in before.items()))
details['actual_v4_boundary_false_socket_result']=r
f=Accounting();f.store.receipt['legacy_ledger_sha256_at_admission']='0'*64
with patch.object(m,'native_owned_scope_measurement',side_effect=AssertionError('private stat before accounting')):
 try:f.check()
 except ValueError as error:check('v4_stale_pins_reject_before_private_stat','first-read' in str(error))
 else:raise AssertionError('stale ledger pin accepted')
f=Accounting();f.legacy['controllers']['handmade']['reserved_bytes']=True;lr=f.put(f.api.LEGACY,f.legacy);f.store.receipt['legacy_ledger_sha256_at_admission']=lr['sha256']
with patch.object(m,'native_owned_scope_measurement',side_effect=AssertionError('private stat before accounting')):
 try:f.check()
 except ValueError as error:check('v4_typed_bad_charge_reject_before_private_stat','integer' in str(error))
 else:raise AssertionError('boolean fake charge accepted')
f=Accounting(cap=20000);(f.scope/'large').write_bytes(b'0'*16000)
try:f.check()
except ValueError as error:check('private_conservative_scope_cap_overflow','20GB sampled cap' in str(error))
else:raise AssertionError('fixture-scaled owned cap accepted')
f=Accounting();os.mkfifo(f.scope/'blocked_fifo')
bad=rejects('real_v4_pass_then_private_FIFO_rejection',f.check,'unsafe')
check('FIFO_only_private_boundary_reject_after_accounting',f.calls[-1]['boundary'] is False)
# Existing terminal error path stores the complete new diagnostic without opening the offender.
terminal=f.F/'terminal';terminal.mkdir()
result={'status':'failed_partial_preserved','exit_code':1,'failure':{'type':'NativeOwnedScopeError','message':json.dumps(bad)}}
m.terminal_receipt(CFG,terminal,result)
saved=json.loads((terminal/'receipt.json').read_text())
check('offending_metadata_survives_existing_terminal_receipt',json.loads(saved['failure']['message'])==bad)
for p,h in PINS.items():check('pin_after_'+p,sha((R/p).read_bytes())==h)
check('production20GB_four_parent80GB_and_measured500GB_preserved',CFG['storage']['reserved_bytes_per_parent']==20000000000 and len(CFG['parent_ids'])==4 and CFG['launcher_contract']['measured_project_hard_limit_bytes']==500000000000)
check('no_actual_offender_socket_claim', 'not captured or identified' in CFG['namespace_correction_scope'])
check('zero_forbidden_access_events',not guard)
for fixture in FIXTURES:
 # Handles are process-local handmade AF_UNIX endpoints; final close/cleanup below.
 pass
report={'kind':'native_argformer_tsinfer_fit_v7_author_namespace_handmade_checks','status':'PASS','count':len(checks),'checks':checks,'details':details,'counterexamples':counterexamples,'real_AF_UNIX_fixture_stat_manifests':socket_manifests,'guard_events':guard,'source_sha256':sha(SOURCE),'config_sha256':m.sha(m.CONFIG),'test_source_sha256':m.sha(__file__),'unchanged_v4_code_sha256':sha(BUDGET),'actual_archive_data_native_runtime_capture_GPU_resource_or_Git':False,'handmade_fixture_root':str(FROOT),'path_length_monkeypatch':False}
(O/'namespace_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'PASS','checks':len(checks),'guard_events':guard},indent=2),flush=True)
# Keep compact metadata and reproducible code, no expanded trees in author seal.
for fixture in FIXTURES:shutil.rmtree(fixture)
