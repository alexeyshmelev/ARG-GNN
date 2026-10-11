"""Independent handmade metadata and admission review. No actual archive/live state."""
from pathlib import Path
import ast,builtins,contextlib,copy,datetime,hashlib,io,json,os,stat,struct,sys,traceback,types,uuid,warnings,zipfile,zlib
from unittest.mock import patch
sys.dont_write_bytecode=True
R=Path('/workspace-SR003.nfs2/ARG-GNN')
O=R/'research/reviews/gst_uspto_archive_inventory_v2_independent'
sys.path.insert(0,str(R/'src'))
from arggnn import gst_uspto_archive_inventory_v2 as m,project_budget_v4 as b
PIN=json.loads((O/'baseline_pins.json').read_text())
ORIG={k:getattr(m,k) for k in ('ROOT','SOURCE','CONFIG','CONFIG_SHA256','RAW','RAW_SHA','RAW_BYTES')}
SOURCE=ORIG['SOURCE'].read_bytes();BUDGET=(R/'src/arggnn/project_budget_v4.py').read_bytes()
RESULTS=[];GUARD=[];PAYLOAD=[];START=datetime.datetime.now(datetime.timezone.utc).isoformat()
def sh(raw):return hashlib.sha256(raw).hexdigest()
def pth(value):
 if isinstance(value,int):return None
 try:return Path(os.path.abspath(os.fsdecode(value)))
 except (TypeError,ValueError):return None
def owned(p):return p is not None and (p==O or O in p.parents)
def deny(message):GUARD.append(message);raise RuntimeError('INDEPENDENT HANDMADE GUARD: '+message)
def audit(event,args):
 if event=='open':
  p=pth(args[0]);mode=args[1];flags=args[2]
  write=isinstance(mode,str) and any(c in mode for c in 'wax+') or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)
  if p is not None and write and not owned(p):deny('outside-ownership write '+str(p))
  if p is not None and not write and not owned(p):
   for base in (R/'data',R/'experiments/results',R/'.runtime/gpup',R/'.runtime/storage',R/'.runtime/locks',Path('/proc')):
    if p==base or base in p.parents:deny('actual dataset/live-state read '+str(p))
 elif event in ('os.mkdir','os.remove','os.rmdir','os.unlink','os.chmod','os.truncate','os.utime'):
  p=pth(args[0])
  if p is not None and not owned(p):deny('outside-ownership mutation '+str(p))
 elif event in ('os.rename','os.link'):
  for value in args[:2]:
   p=pth(value)
   if p is not None and not owned(p):deny('outside-ownership mutation '+str(p))
 elif event=='os.symlink':
  p=pth(args[1])
  if p is not None and not owned(p):deny('outside-ownership symlink '+str(p))
 elif event in ('subprocess.Popen','os.system','os.posix_spawn','socket.bind','socket.connect'):deny(event)
sys.addaudithook(audit)
old_import=builtins.__import__
def imported(name,*args,**kwargs):
 if name.split('.')[0] in ('numpy','torch','rdkit','torch_geometric','tskit','tsinfer'):deny('scientific import '+name)
 return old_import(name,*args,**kwargs)
builtins.__import__=imported

def require(ok,message):
 if not ok:raise AssertionError(message)
def case(name,fn):
 try:
  value=fn();RESULTS.append({'case':name,'status':'PASS','evidence':value})
 except Exception as error:
  RESULTS.append({'case':name,'status':'FAIL','error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()})
 print(name,RESULTS[-1]['status'],flush=True)

def payload_tripwire(label):
 def fail(*args,**kwargs):
  PAYLOAD.append(label);raise AssertionError('Member payload/decoder API invoked: '+label)
 return fail
@contextlib.contextmanager
def no_payload():
 with contextlib.ExitStack() as stack:
  for name in ('open','read','extract','extractall'):stack.enter_context(patch.object(zipfile.ZipFile,name,side_effect=payload_tripwire('ZipFile.'+name)))
  for name in ('read','read1','readline','readlines','_read1','_read2'):stack.enter_context(patch.object(zipfile.ZipExtFile,name,side_effect=payload_tripwire('ZipExtFile.'+name)))
  for name in ('decompress','decompressobj'):stack.enter_context(patch.object(zlib,name,side_effect=payload_tripwire('zlib.'+name)))
  yield
def archive(extras=(),compression=zipfile.ZIP_STORED,names=None):
 out=io.BytesIO()
 if names is None:names=[('native/',b''),('native/train.txt',b'train handmade\n'),('native/valid.txt',b'valid handmade\n'),('native/test.txt',b'test handmade\n')]
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',UserWarning)
  with zipfile.ZipFile(out,'w',compression=compression) as z:
   for name,data in [*names,*extras]:z.writestr(name,data)
 return out.getvalue()
def central(raw,index=0):
 positions=[];start=0
 while True:
  off=raw.find(b'PK\x01\x02',start)
  if off<0:break
  positions.append(off);start=off+4
 return positions[index]
def edit(raw,index,offset,fmt,value):
 out=bytearray(raw);struct.pack_into(fmt,out,central(raw,index)+offset,value);return bytes(out)
def flags(raw,value,index=1):return edit(raw,index,8,'<H',value)
def extra(identity,body,name='safe-metadata'):
 info=zipfile.ZipInfo(name);info.extra=struct.pack('<HH',identity,len(body))+body;return info
def asi(mode,name='safe-metadata',sizdev=0,crc_ok=True):
 tail=struct.pack('<HIHH',mode,sizdev,7,9);crc=zlib.crc32(tail)&0xffffffff
 return extra(0x756e,struct.pack('<I',crc if crc_ok else crc^1)+tail,name)
def unicode_extra(body,name='safe-metadata'):return extra(0x7075,body,name)
def run_inventory(raw):
 before=len(PAYLOAD)
 with no_payload():value=m.inventory(raw)
 require(len(PAYLOAD)==before,'payload tripwire fired')
 return value
def accepts(raw):
 r=run_inventory(raw)
 require(r['status']=='metadata_only_complete','unexpected inventory status')
 for k in ('member_payload_opened','data_extracted','dataset_rows_parsed','member_content_SHA256_known','CRC_payload_verified','native_model_run','original_author_used_snapshot_proven'):require(r[k] is False,k)
 return {'members':r['member_count'],'archive_sha256':r['archive_sha256'],'payload_apis':0}
def rejects(raw,label=None):
 before=len(PAYLOAD)
 try:run_inventory(raw)
 except (ValueError,TypeError,zipfile.BadZipFile,NotImplementedError,UnicodeError) as error:
  require(len(PAYLOAD)==before,'rejection touched member payload')
  if label:require(label in str(error),'wrong rejection: '+str(error))
  return {'rejected':type(error).__name__,'message':str(error),'payload_apis':0}
 raise AssertionError('Unsafe metadata accepted')
def with_limits(raw,values,want):
 with patch.dict(m.LIMITS,values):return accepts(raw) if want else rejects(raw)
def changed_name(raw,index,replacement):
 off=central(raw,index);n=struct.unpack_from('<H',raw,off+28)[0]
 require(len(replacement)==n,'name patch must preserve size')
 out=bytearray(raw);out[off+46:off+46+n]=replacement;return bytes(out)

class Private:
 def __init__(self):
  self.F=O/'fixtures'/uuid.uuid4().hex;self.F.mkdir(parents=True);F=self.F
  self.raw=archive();self.events=[];self.checks=[];self.reads=[];self.after_opaque=None;self.after_check=None
  self.rawref=self.put('handmade/tiny.zip',self.raw)
  self.source=self.put('src/arggnn/gst_uspto_archive_inventory_v2.py',SOURCE)
  helper=self.put('src/arggnn/project_budget_v4.py',BUDGET)
  require((F/helper['path']).read_bytes()==BUDGET,'budget code changed')
  proto=self.put('configs/gst_uspto_native_data_protocol_v1.json',(R/'configs/gst_uspto_native_data_protocol_v1.json').read_bytes())
  legacy_source=self.put('src/arggnn/gpu_pool_v1.py',(R/'src/arggnn/gpu_pool_v1.py').read_bytes())
  api=types.ModuleType('actual_v4_private_'+F.name)
  exec(compile(BUDGET,str(F/helper['path']),'exec'),api.__dict__)
  api.ROOT=F;api.SOURCE=F/helper['path'];api.CONFIG=F/'configs/project_budget_v4.json'
  api.FIXED_POLICY=copy.deepcopy(b.FIXED_POLICY)
  human=self.put('meta/human.json',{'kind':'human_project_storage_authorization','project_root':str(F),'authorized_hard_limit_bytes':1000000000000,'future_scheduling_ceiling_bytes':950000000000,'fixture_only':True})
  api.FIXED_POLICY.update(root=str(F),human_authorization=human,legacy_pool_source=legacy_source)
  bc=self.put('configs/project_budget_v4.json',api.FIXED_POLICY);self.api=api
  self.cfg=json.loads(ORIG['CONFIG'].read_text());self.cfg.update(raw={**self.rawref,'bytes':len(self.raw)},budget_source=helper,budget_config=bc,native_protocol=proto)
  self.config=self.put('configs/gst_uspto_archive_inventory_v2.json',self.cfg)
  self.review={'kind':'gst_uspto_archive_inventory_v2_independent_review','review_role':'independent_reviewer','verdict':'PASS','implementation':self.source,'config':self.config,'source_handmade_metadata_verified':True,'actual_archive_opened':False,'actual_member_payload_opened':False,'fixture_mode':False,'actual_admission':False,'tested_scope':['explicit independent handmade fixture capability only'],'fixture_only_never_use_for_actual_admission':True}
  rr=self.put('meta/review.json',self.review)
  self.reg={'kind':'gst_uspto_archive_inventory_v2_root_registration','authorizer':'root_agent','hypothesis_id':'H022','archive_metadata_run_registered':True,'implementation':self.source,'config':self.config,'raw':self.cfg['raw'],'payload_read_authorized':False,'native_model_authorized':False,'experiment_id':'private_handmade_metadata_only','fixture_only':True}
  reg=self.put('meta/registration.json',self.reg)
  bindings={'implementation':helper,'config':bc,'human_authorization':human}
  br=self.put('meta/budget_review.json',{'kind':'project_budget_v4_independent_review','review_role':'independent_reviewer','verdict':'PASS',**bindings,'actual_apply_performed':False,'fixture_only':True})
  self.output=F/'fresh_output'
  self.request={'key':'gst-v2/independent-handmade','output':str(self.output),'auxiliary_scopes':[],'reserved_bytes':20000000,'consumer_bindings':[self.source,self.config,rr]}
  self.rootreserve={'kind':'project_budget_v4_root_reserve_admission','root_authorizer':'root_agent','active':True,**bindings,'all_budget_writers_use_scheduler_mutex':True,'no_legacy_mutation_authorized':True,'GPU_authorized':False,'scientific_execution_authorized':False,'global_rebase_authorized':False,'reservations':[self.request],'independent_review':br,'fixture_only':True}
  rootref=self.put('meta/root_reserve.json',self.rootreserve)
  self.reservation={**self.request,'root_admission':rootref};resref=self.put('meta/reservation.json',self.reservation)
  self.legacy={**api.FIXED_POLICY['legacy_invariants'],'root':str(F),'controllers':{'fixture_pool':{'pid':777,'reserved_bytes':67108864}},'jobs':{}}
  self.supp=api.empty_ledger(F,api.FIXED_POLICY,bindings);self.supp['reservations']={self.request['key']:self.reservation}
  lr=self.put(api.LEGACY,self.legacy);sr=self.put(api.SUPPLEMENTAL,self.supp)
  now=datetime.datetime.now(datetime.timezone.utc)
  self.auth={'kind':'gst_uspto_archive_inventory_v2_root_admission','authorizer':'root_agent','active':True,'metadata_only':True,'payload_read_authorized':False,'native_model_authorized':False,'implementation':self.source,'config':self.config,'raw':self.cfg['raw'],'limits':copy.deepcopy(m.LIMITS),'created_utc':(now-datetime.timedelta(minutes=1)).isoformat(),'expires_utc':(now+datetime.timedelta(hours=1)).isoformat(),'registration':reg,'independent_review':rr,'output_directory':'fresh_output','reservation':resref,'initial_ledger_pins':{'legacy_ledger_sha256':lr['sha256'],'supplemental_ledger_sha256':sr['sha256']}}
  self.ar=self.put('meta/admission.json',self.auth)
  (F/api.LOCK).parent.mkdir(parents=True,exist_ok=True);(F/api.LOCK).write_bytes(b'independent handmade mutex\n')
 def put(self,name,value):
  q=self.F/name;q.parent.mkdir(parents=True,exist_ok=True)
  raw=value if isinstance(value,bytes) else (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
  q.write_bytes(raw);return {'path':name,'sha256':sh(raw)}
 def refresh(self):self.ar=self.put('meta/admission.json',self.auth)
 def review_refresh(self):self.auth['independent_review']=self.put('meta/review.json',self.review);self.refresh()
 def reg_refresh(self):self.auth['registration']=self.put('meta/registration.json',self.reg);self.refresh()
 def ledger_refresh(self):
  self.auth['initial_ledger_pins']['legacy_ledger_sha256']=self.put(self.api.LEGACY,self.legacy)['sha256']
  self.auth['initial_ledger_pins']['supplemental_ledger_sha256']=self.put(self.api.SUPPLEMENTAL,self.supp)['sha256']
  self.refresh()
 def reservation_refresh(self):
  self.request={k:v for k,v in self.reservation.items() if k!='root_admission'}
  self.rootreserve['reservations']=[self.request]
  self.reservation['root_admission']=self.put('meta/root_reserve.json',self.rootreserve)
  self.auth['reservation']=self.put('meta/reservation.json',self.reservation)
  self.supp['reservations']={self.request['key']:self.reservation}
  self.ledger_refresh()
 @contextlib.contextmanager
 def context(self):
  with contextlib.ExitStack() as stack:
   for k,v in {'ROOT':self.F,'SOURCE':self.F/self.source['path'],'CONFIG':self.F/self.config['path'],'CONFIG_SHA256':self.config['sha256'],'RAW':Path(self.rawref['path']),'RAW_SHA':self.rawref['sha256'],'RAW_BYTES':len(self.raw)}.items():stack.enter_context(patch.object(m,k,v))
   stack.enter_context(patch.object(Path,'cwd',return_value=self.F))
   opaque=m._opaque_bytes
   def checked(*args,**kwargs):
    self.events.append('storage_enter');value=self.api.check_reservation(*args,**kwargs)
    self.events.append('storage_pass');self.checks.append({'kwargs':kwargs,'view':value})
    if self.after_check:self.after_check(len(self.checks))
    return value
   def read():
    self.events.append('opaque_enter');self.reads.append(True);value=opaque();self.events.append('opaque_pass')
    if self.after_opaque:self.after_opaque()
    return value
   stack.enter_context(patch.object(b,'check_reservation',side_effect=checked))
   stack.enter_context(patch.object(m,'_opaque_bytes',side_effect=read))
   stack.enter_context(no_payload())
   yield
 def run(self):
  with self.context(),contextlib.redirect_stdout(io.StringIO()) as out:code=m.main(['--execute','--admission',self.ar['path'],'--expected-admission-sha256',self.ar['sha256']])
  self.stdout=out.getvalue();return code
 def denied(self,want=None):
  try:self.run()
  except (ValueError,TypeError,FileNotFoundError) as error:
   require(self.reads==[],'opaque access preceded admission failure')
   require(not self.output.exists(),'output created before admission failure')
   if want:require(want in str(error),'wrong gate rejection: '+str(error))
   return {'rejected':type(error).__name__,'message':str(error),'opaque_reads':0,'events':self.events,'fixture_root':str(self.F.relative_to(R))}
  raise AssertionError('Malformed authority/storage admitted')
 def success(self):
  before={k:(self.F/k).read_bytes() for k in (self.api.LEGACY,self.api.SUPPLEMENTAL,self.api.LOCK)}
  require(self.run()==0,'valid private run failed')
  require(self.events==['storage_enter','storage_pass','opaque_enter','opaque_pass','storage_enter','storage_pass'],'wrong storage/opaque order')
  require([p.name for p in self.output.iterdir()]==['inventory.json'],'non-JSON products')
  r=json.loads((self.output/'inventory.json').read_text())
  require(r['status']=='metadata_only_complete','wrong successful status')
  require(r['logical_split_member_paths']=={n:'native/'+n for n in m.LOGICAL},'wrong native paths')
  for k in ('member_payload_opened','data_extracted','dataset_rows_parsed','member_content_SHA256_known','CRC_payload_verified','native_model_run'):require(r[k] is False,k)
  require(self.checks[0]['kwargs']['boundary'] is True and self.checks[0]['kwargs']['required_write_bytes']==1008192,'initial storage boundary not exact')
  require(self.checks[1]['kwargs']['boundary'] is True,'final boundary omitted')
  for k,raw in before.items():require((self.F/k).read_bytes()==raw,'accounting or mutex mutated')
  return {'events':self.events,'budget_code_sha256':sh(BUDGET),'fixture_root':str(self.F.relative_to(R)),'JSON_only':True,'ledger_and_mutex_bytes_unchanged':True}
def auth_change(key,value):
 f=Private();f.auth[key]=value;f.refresh();return f.denied()
def review_change(key,value):
 f=Private();f.review[key]=value;f.review_refresh();return f.denied()
def reg_change(key,value):
 f=Private();f.reg[key]=value;f.reg_refresh();return f.denied()
def ledger_change(fn,want=None):
 f=Private();fn(f);f.ledger_refresh();return f.denied(want)
def reservation_change(key,value):
 f=Private();f.reservation[key]=value;f.reservation_refresh();return f.denied()
def opaque_fail(fn):
 f=Private();fn(f)
 require(f.run()==1,'bad handmade opaque input did not return preserved failure')
 r=json.loads((f.output/'inventory.json').read_text())
 require(r['status']=='failed_preserved','failure receipt missing')
 require(r['member_payload_opened'] is False and len(f.reads)==1,'invalid failure receipt/read count')
 return {'status':r['status'],'error':r['error'],'events':f.events,'fixture_root':str(f.F.relative_to(R))}
def duplicate_record(kind):
 f=Private();name={'admission':'meta/admission.json','review':'meta/review.json','registration':'meta/registration.json','legacy':f.api.LEGACY,'supplemental':f.api.SUPPLEMENTAL}[kind]
 value={'admission':f.auth,'review':f.review,'registration':f.reg,'legacy':f.legacy,'supplemental':f.supp}[kind]
 key={'admission':'active','review':'verdict','registration':'archive_metadata_run_registered','legacy':'baseline_bytes','supplemental':'root'}[kind]
 raw=json.dumps(value).encode();raw=b'{'+json.dumps(key).encode()+b':null,'+raw[1:]
 ref=f.put(name,raw)
 if kind=='admission':f.ar=ref
 elif kind=='review':f.auth['independent_review']=ref;f.refresh()
 elif kind=='registration':f.auth['registration']=ref;f.refresh()
 else:f.auth['initial_ledger_pins'][kind+'_ledger_sha256']=ref['sha256'];f.refresh()
 return f.denied('duplicate JSON key')

def source_only():
 with patch.object(m,'_opaque_bytes',side_effect=AssertionError('default read archive')),patch.object(b,'check_reservation',side_effect=AssertionError('default budget access')),contextlib.redirect_stdout(io.StringIO()) as out:require(m.main([])==0,'source-only error')
 r=json.loads(out.getvalue());require(r['archive_opened'] is False and r['live_budget_or_lock_access'] is False,'default accesses')
 require(r['raw']==json.loads(ORIG['CONFIG'].read_text())['raw'],'default raw identity drift')
 (O/'source_only_cli.json').write_text(out.getvalue());return r
def pins():
 for name,h in PIN.items():require(sh((R/name).read_bytes())==h,'pin changed: '+name)
 require(m.LIMITS=={'members':512,'member_bytes':256000000,'total_bytes':1000000000,'compression_ratio':1000},'caps drift')
 require(m.RAW_BYTES==3446572 and m.RAW_SHA=='66b4f5cb6b5de4bd7c3d72bdb7fa0c5073c4b97984109f0dae0b10997a9aa723','raw drift')
 source=ast.parse(SOURCE);inventory=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='inventory')
 require(not any(isinstance(n,ast.Attribute) and n.attr in ('open','read','extract','extractall') for n in ast.walk(inventory)),'inventory payload call')
 return {'pins':len(PIN),'original_caps':m.LIMITS,'actual_raw_opened':False}

case('source_config_predecessor_author_budget_pins',pins)
case('default_actual_constants_source_only',source_only)
for compression in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED):case('valid_compression_'+str(compression),lambda c=compression:accepts(archive([('ancillary',b'')],c)))
for name in ('../x','/x','C:x','x\\y','x//y','x/./y','x/../y','x\x01y','x\x7fy','native','Native/train.txt','other/train.txt','other/TRAIN.TXT','other/Valid.Txt','other/TEST.txt'):
 case('reject_name_'+ascii(name),lambda n=name:rejects(archive([(n,b'')])))
case('central_NUL_untruncated_rejected',lambda:rejects(changed_name(archive([('nullXname',b'')]),4,b'null\x00name'),'control'))
case('central_empty_name_rejected',lambda:rejects(archive(names=[('',b'')])))
case('NFC_collision',lambda:rejects(archive([('x/caf\u00e9',b''),('x/cafe\u0301',b'')]),'collision'))
case('Unicode_casefold_collision',lambda:rejects(archive([('x/Stra\u00dfe',b''),('x/STRASSE',b'')]),'collision'))
case('prefix_collision_reverse_order',lambda:rejects(archive([('branch/leaf',b''),('branch',b'')]),'prefix'))
case('missing_logical_split',lambda:rejects(archive(names=[('train.txt',b''),('valid.txt',b'')]),'three'))
case('logical_replacement_wrong_case',lambda:rejects(archive(names=[('TRAIN.TXT',b''),('valid.txt',b''),('test.txt',b'')]),'spelling'))
case('directory_payload',lambda:rejects(archive([('extra/',b'data')]),'Directory payload'))
for mode in (stat.S_IFLNK|0o777,stat.S_IFIFO|0o600,stat.S_IFSOCK|0o600,stat.S_IFBLK|0o600,stat.S_IFCHR|0o600):
 def mode_case(v=mode):
  x=zipfile.ZipInfo('special');x.create_system=3;x.external_attr=v<<16;return rejects(archive([(x,b'')]),'Special/link')
 case('Unix_external_special_'+str(mode),mode_case)
for name,system,attrs in [('file',0,0x10),('folder/',0,0),('file',3,(stat.S_IFDIR|0o700)<<16),('folder/',3,(stat.S_IFREG|0o600)<<16)]:
 def attr_case(n=name,s=system,a=attrs):
  x=zipfile.ZipInfo(n);x.create_system=s;x.external_attr=a;return rejects(archive([(x,b'')]),'disagreement')
 case('type_name_disagreement_'+name+'_'+str(system),attr_case)
for flag in (1,0x40,0x2000,0x2041):case('encryption_flag_'+str(flag),lambda v=flag:rejects(flags(archive(),v),'Encrypted'))
for compression in (zipfile.ZIP_BZIP2,zipfile.ZIP_LZMA):case('unsupported_compression_'+str(compression),lambda v=compression:rejects(edit(archive(),1,10,'<H',v),'compression'))
for body in (b'',b'x'*11,b'x'*13,struct.pack('<IIHH',0,0,2,3)+b'target'):
 case('PKWARE_reject_length_'+str(len(body)),lambda v=body:rejects(archive([(extra(0x000d,v),b'')]),'PKWARE'))
case('PKWARE_fixed_metadata_retained',lambda:accepts(archive([(extra(0x000d,struct.pack('<IIHH',0,0,2,3)),b'')])))
for mode in (stat.S_IFLNK|0o777,stat.S_IFIFO|0o600,stat.S_IFCHR|0o600):
 case('ASi_link_special_'+str(mode),lambda v=mode:rejects(archive([(asi(v),b'')]),'ASi Unix special'))
case('ASi_valid_regular',lambda:accepts(archive([(asi(stat.S_IFREG|0o644),b'')])))
case('ASi_valid_directory',lambda:accepts(archive([(asi(stat.S_IFDIR|0o755,'extra/'),b'')])))
case('ASi_bad_CRC',lambda:rejects(archive([(asi(stat.S_IFREG|0o644,crc_ok=False),b'')]),'CRC'))
case('ASi_directory_file_disagreement',lambda:rejects(archive([(asi(stat.S_IFDIR|0o755),b'')]),'disagreement'))
case('ASi_regular_directory_disagreement',lambda:rejects(archive([(asi(stat.S_IFREG|0o644,'extra/'),b'')]),'disagreement'))
case('ASi_link_tail',lambda:rejects(archive([(extra(0x756e,b'x'*15),b'')]),'ASi'))
base=b'\x01'+struct.pack('<I',zlib.crc32(b'safe-metadata')&0xffffffff)
for label,body in [('identity',base+b'safe-metadata'),('wrong-path',base+b'other/train.txt'),('invalid-UTF8',base+b'\xff'),('bad-version',b'\x02'+base[1:]+b'safe-metadata'),('bad-crc',b'\x01'+b'\x00'*4+b'safe-metadata'),('truncated',b'\x01')]:
 case('Unicode_path_'+label,lambda n=label,v=body:accepts(archive([(unicode_extra(v),b'')])) if n=='identity' else rejects(archive([(unicode_extra(v),b'')])))
for label,data in [('header',b'x'),('body',struct.pack('<HH',0x5455,4)+b'x'),('duplicate',struct.pack('<HH',0xffff,0)*2)]:
 def malformed(v=data):
  x=zipfile.ZipInfo('malformed-extra');x.extra=v;return rejects(archive([(x,b'')]))
 case('extra_malformed_'+label,malformed)
case('unknown_bounded_extra_retained',lambda:accepts(archive([(extra(0xffff,b'opaque-metadata'),b'')])))
raw=archive();total=sum(len(data) for _,data in [('native/',b''),('native/train.txt',b'train handmade\n'),('native/valid.txt',b'valid handmade\n'),('native/test.txt',b'test handmade\n')])
case('member_count_cap_inclusive',lambda:with_limits(raw,{'members':4},True))
case('member_count_cap_over',lambda:with_limits(raw,{'members':3},False))
case('member_bytes_cap_inclusive',lambda:with_limits(raw,{'member_bytes':15},True))
case('member_bytes_cap_over',lambda:with_limits(raw,{'member_bytes':14},False))
case('total_cap_inclusive',lambda:with_limits(raw,{'total_bytes':total},True))
case('total_cap_over',lambda:with_limits(raw,{'total_bytes':total-1},False))
case('ratio_cap_inclusive',lambda:with_limits(raw,{'compression_ratio':1},True))
case('ratio_cap_over',lambda:with_limits(archive([('compressible',b'a'*4096)],zipfile.ZIP_DEFLATED),{'compression_ratio':2},False))
case('declared_ratio_zero_compressed',lambda:rejects(edit(edit(raw,1,20,'<I',0),1,24,'<I',1001),'ratio'))
case('independent_valid_main_actual_unchanged_v4',lambda:Private().success())

for key,value in [('active',1),('metadata_only',1),('payload_read_authorized',0),('native_model_authorized',0),('kind','author_receipt'),('authorizer','other_agent'),('implementation',{'path':'src/arggnn/gst_uspto_archive_inventory_v2.py','sha256':'0'*64}),('raw',{'path':'wrong.zip','sha256':'0'*64,'bytes':1}),('limits',{'members':512,'member_bytes':256000000,'total_bytes':1000000000,'compression_ratio':True}),('unknown_execute_scope',True)]:
 case('admission_exact_'+key,lambda k=key,v=value:auth_change(k,v))
for key,value in [('kind','gst_uspto_archive_inventory_v2_author_source_receipt'),('review_role','author_not_independent_reviewer'),('verdict','FAIL'),('source_handmade_metadata_verified',1),('actual_archive_opened',0),('actual_member_payload_opened',0),('fixture_mode',True),('actual_admission',True),('tested_scope',[]),('tested_scope',['']),('tested_scope',[1]),('implementation',{'path':'src/arggnn/gst_uspto_archive_inventory_v2.py','sha256':'0'*64}),('config',{'path':'configs/gst_uspto_archive_inventory_v2.json','sha256':'0'*64})]:
 case('independent_review_exact_'+key+'_'+repr(value),lambda k=key,v=value:review_change(k,v))
for key,value in [('kind','wrong'),('authorizer','source_author'),('hypothesis_id','H000'),('archive_metadata_run_registered',1),('archive_metadata_run_registered',False),('payload_read_authorized',0),('native_model_authorized',True),('experiment_id',''),('experiment_id',3),('implementation',{'path':'src/arggnn/gst_uspto_archive_inventory_v2.py','sha256':'0'*64}),('raw',{'path':'handmade/tiny.zip','sha256':'0'*64,'bytes':1})]:
 case('registration_exact_'+key+'_'+repr(value),lambda k=key,v=value:reg_change(k,v))
for kind in ('admission','review','registration','legacy','supplemental'):case('duplicate_JSON_'+kind,lambda k=kind:duplicate_record(k))
for value in (float('nan'),float('inf'),float('-inf')):case('nonfinite_JSON_'+str(value),lambda v=value:auth_change('active',v))
now=datetime.datetime.now(datetime.timezone.utc)
for label,key,value in [('expired','expires_utc',(now-datetime.timedelta(seconds=1)).isoformat()),('future','created_utc',(now+datetime.timedelta(hours=1)).isoformat()),('too-wide','expires_utc',(now+datetime.timedelta(days=2)).isoformat()),('naive','created_utc',now.replace(tzinfo=None).isoformat()),('typed','created_utc',1)]:
 case('authority_dates_'+label,lambda k=key,v=value:auth_change(k,v))
for key,value in [('schema','unknown'),('root','wrong'),('snapshot_sha256','0'*64),('baseline_bytes',0),('external_reserve_bytes',0),('limit_bytes',450000000001),('baseline_bytes',True),('jobs',[]),('controllers',[]),('unknown',0)]:
 case('legacy_invariant_'+key+'_'+repr(value),lambda k=key,v=value:ledger_change(lambda f:f.legacy.update({k:v})))
for key,value in [('reserved_bytes',True),('reserved_bytes',-1),('reserved_bytes',1.0),('pid',True),('pid',0),('extra',0)]:
 case('legacy_controller_type_'+key+'_'+repr(value),lambda k=key,v=value:ledger_change(lambda f:f.legacy['controllers']['fixture_pool'].update({k:v})))
case('legacy_exact_450GB_ceiling',lambda:ledger_change(lambda f:f.legacy['controllers']['fixture_pool'].update(reserved_bytes=450000000000-f.legacy['baseline_bytes']-f.legacy['external_reserve_bytes']),'450GB'))
case('legacy_job_relative_output',lambda:ledger_change(lambda f:f.legacy['jobs'].update({'job':{'reserved_bytes':1,'output':'relative'}}),'absolute'))
case('legacy_job_accounting_overlap',lambda:ledger_change(lambda f:f.legacy['jobs'].update({'job':{'reserved_bytes':1,'output':str(f.F/'.runtime/storage')}}),'accounting'))
case('legacy_job_ancestor_overlap',lambda:ledger_change(lambda f:f.legacy['jobs'].update({'j1':{'reserved_bytes':1,'output':str(f.F/'occupied')},'j2':{'reserved_bytes':1,'output':str(f.F/'occupied/child')}}),'overlap'))
case('supplemental_header_drift',lambda:ledger_change(lambda f:f.supp.update(hard_limit_bytes=1),'header'))
case('supplemental_reservation_missing',lambda:ledger_change(lambda f:f.supp['reservations'].clear(),'absent'))
case('supplemental_reserved_bool',lambda:ledger_change(lambda f:f.supp['reservations'][f.request['key']].update(reserved_bytes=True)))
for key,value in [('reserved_bytes',True),('reserved_bytes',19999999),('auxiliary_scopes',['/different']),('consumer_bindings',[])]:
 case('reservation_exact_'+key+'_'+repr(value),lambda k=key,v=value:reservation_change(k,v))
def stale(key):
 f=Private();f.auth['initial_ledger_pins'][key]='0'*64;f.refresh();return f.denied('first-read')
for key in ('legacy_ledger_sha256','supplemental_ledger_sha256'):case('stale_initial_'+key,lambda k=key:stale(k))
def combined_ceiling():
 f=Private();other={**copy.deepcopy(f.request),'key':'other-charge','output':str(f.F/'other_fresh'),'reserved_bytes':950000000000-(f.legacy['baseline_bytes']+f.legacy['external_reserve_bytes']+67108864+20000000)}
 f.rootreserve['reservations']=[f.request,other];ref=f.put('meta/root_reserve.json',f.rootreserve)
 f.reservation['root_admission']=ref;f.auth['reservation']=f.put('meta/reservation.json',f.reservation)
 f.supp['reservations']={f.request['key']:f.reservation,other['key']:{**other,'root_admission':ref}};f.ledger_refresh()
 return f.denied('ceiling')
case('combined_supplemental_exact_950GB_ceiling_before_raw',combined_ceiling)
def lock_bad(kind):
 f=Private();p=f.F/f.api.LOCK;p.unlink();other=f.F/'handmade/other-lock';other.write_bytes(b'fixture')
 if kind=='symlink':p.symlink_to(other)
 else:os.link(other,p)
 return f.denied()
for kind in ('symlink','hardlink'):case('canonical_mutex_'+kind+'_before_raw',lambda k=kind:lock_bad(k))
def existing_output():
 f=Private();f.output.mkdir();f.output.rmdir();target=f.F/'handmade/occupied';target.mkdir();f.output.symlink_to(target)
 try:f.run()
 except ValueError as error:
  require(not f.reads,'output symlink reached raw');return {'error':str(error),'opaque_reads':0}
 raise AssertionError('existing output symlink admitted')
case('exclusive_output_symlink_before_raw',existing_output)
def tampered_source():
 f=Private();(f.F/f.source['path']).write_bytes(SOURCE+b'\n# fixture tamper\n');return f.denied()
case('changed_source_bytes_before_raw',tampered_source)
def tampered_bound(kind):
 f=Private();ref=f.cfg[kind];(f.F/ref['path']).write_bytes((f.F/ref['path']).read_bytes()+b' ');return f.denied()
for kind in ('budget_source','budget_config','native_protocol'):case('changed_bound_'+kind+'_before_raw',lambda k=kind:tampered_bound(k))
case('opaque_wrong_hash_failure_preserved',lambda:opaque_fail(lambda f:(f.F/f.rawref['path']).write_bytes(f.raw[:-1]+b'X')))
case('opaque_wrong_size_failure_preserved',lambda:opaque_fail(lambda f:(f.F/f.rawref['path']).write_bytes(f.raw+b'X')))
def raw_link(f,kind):
 p=f.F/f.rawref['path'];other=f.F/'handmade/other-raw'
 if kind=='hardlink':os.link(p,other)
 else:p.rename(other);p.symlink_to(other)
for kind in ('symlink','hardlink'):case('opaque_'+kind+'_failure_preserved',lambda k=kind:opaque_fail(lambda f:raw_link(f,k)))
def final_ledger_change():
 f=Private()
 def mutate():f.legacy['controllers']['fixture_pool']['reserved_bytes']=True;f.put(f.api.LEGACY,f.legacy)
 f.after_opaque=mutate
 try:f.run()
 except ValueError as error:
  require(len(f.reads)==1 and f.output.exists() and not (f.output/'inventory.json').exists(),'final boundary wrote invalid receipt')
  return {'error':str(error),'events':f.events,'failure_directory_preserved':True}
 raise AssertionError('final storage invalidity accepted')
case('final_real_v4_boundary_blocks_write_after_ledger_drift',final_ledger_change)
def final_authority_change():
 f=Private()
 def mutate():f.put('meta/admission.json',{**f.auth,'active':False})
 f.after_opaque=mutate
 try:f.run()
 except ValueError as error:
  require(len(f.reads)==1 and f.output.exists() and not (f.output/'inventory.json').exists(),'changed final admission wrote receipt')
  return {'error':str(error),'events':f.events,'failure_directory_preserved':True}
 raise AssertionError('changed final admission accepted')
case('final_admission_byte_rebind_blocks_write',final_authority_change)
case('final_implementation_and_predecessor_pins',pins)
END=datetime.datetime.now(datetime.timezone.utc).isoformat()
report={'kind':'gst_uspto_archive_inventory_v2_independent_handmade_test_report','status':'PASS' if all(r['status']=='PASS' for r in RESULTS) and not GUARD and not PAYLOAD else 'FAIL','started_utc':START,'completed_utc':END,'checks':len(RESULTS),'passing':sum(r['status']=='PASS' for r in RESULTS),'failed':sum(r['status']=='FAIL' for r in RESULTS),'guard_events':GUARD,'payload_api_events':PAYLOAD,'actual_archive_opened':False,'actual_member_payload_opened':False,'actual_dataset_TEST_scientific_model_access':False,'actual_live_ledger_mutex_access':False,'budget_code_sha256':sh(BUDGET),'budget_functions_unchanged_private_policy_constants_only':True,'results':RESULTS}
(O/'independent_attempt01.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
raise SystemExit(0 if report['status']=='PASS' else 1)
