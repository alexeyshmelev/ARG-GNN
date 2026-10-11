"""Guarded handmade metadata and real-v4-on-private-root admission checks."""
from pathlib import Path
import ast,builtins,contextlib,copy,datetime,hashlib,io,json,os,stat,struct,sys,types,unittest,uuid,zipfile,zlib
from unittest.mock import patch
R=Path('/workspace-SR003.nfs2/ARG-GNN');O=R/'research/reviews/gst_uspto_archive_inventory_v2_author';EVENTS=[]
sys.dont_write_bytecode=True;sys.path.insert(0,str(R/'src'))
from arggnn import gst_uspto_archive_inventory_v2 as m,project_budget_v4 as b
ORIGINAL={k:getattr(m,k) for k in ('ROOT','SOURCE','CONFIG','CONFIG_SHA256','RAW','RAW_SHA','RAW_BYTES')}
def path(v):
 if isinstance(v,int):return None
 try:return Path(os.path.abspath(os.fsdecode(v)))
 except (TypeError,ValueError):return None
def own(p):return p is not None and (p==O or O in p.parents)
def deny(x):EVENTS.append(x);raise RuntimeError('HANDMADE GUARD: '+x)
def audit(event,args):
 if event=='open':
  p=path(args[0]);mode=args[1];flags=args[2];writing=isinstance(mode,str) and any(x in mode for x in 'wax+') or isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  if writing and p is not None and not own(p):deny('write outside author ownership '+str(p))
  if not writing and p is not None and not own(p):
   for base in (R/'data',R/'experiments/results',R/'.runtime/gpup',R/'.runtime/locks',R/'.runtime/storage',Path('/proc')):
    if p==base or base in p.parents:deny('actual dataset/live state read '+str(p))
 elif event in ('os.mkdir','os.remove','os.rmdir','os.unlink','os.chmod','os.truncate','os.utime'):
  p=path(args[0])
  if p is not None and not own(p):deny('mutation outside ownership '+str(p))
 elif event in ('subprocess.Popen','os.system','os.posix_spawn','socket.bind','socket.connect'):deny(event)
sys.addaudithook(audit);old_import=builtins.__import__
def imported(name,*args,**kwargs):
 if name.split('.')[0] in ('numpy','torch','rdkit','torch_geometric','tskit','tsinfer'):deny('scientific import '+name)
 return old_import(name,*args,**kwargs)
builtins.__import__=imported

def sh(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fixture(extra=()):
 out=io.BytesIO()
 with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for name in ('root/','root/train.txt','root/valid.txt','root/test.txt'):z.writestr(name,b'' if name.endswith('/') else b'explicit handmade row\n')
  for item,data in extra:z.writestr(item,data)
 return out.getvalue()
def inv(raw):
 with contextlib.ExitStack() as stack:
  for method in ('open','read','extract','extractall'):stack.enter_context(patch.object(zipfile.ZipFile,method,side_effect=AssertionError('Payload API '+method)))
  return m.inventory(raw)
def extra_info(identity,body,name='metadata'):
 item=zipfile.ZipInfo(name);item.extra=struct.pack('<HH',identity,len(body))+body;return item
class Metadata(unittest.TestCase):
 def test_exact_original_caps_raw_identity_and_inverse_inventory_delta(self):
  pins=json.loads((O/'baseline_pins.json').read_text())
  for p,h in pins.items():self.assertEqual(sh(R/p),h)
  proof=json.loads((O/'inventory_exact_delta.json').read_text());old=(R/'src/arggnn/gst_uspto_archive_inventory_v1.py').read_text();tree=ast.parse(old);original=ast.get_source_segment(old,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='inventory'))
  for item in proof['changes']:original=original.replace(item['old'],item['new'])
  current=ast.get_source_segment(m.SOURCE.read_text(),next(n for n in ast.parse(m.SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='inventory'))
  self.assertEqual(current,original);self.assertEqual(m.LIMITS,{'members':512,'member_bytes':256000000,'total_bytes':1000000000,'compression_ratio':1000})
  self.assertEqual(m.RAW_SHA,'66b4f5cb6b5de4bd7c3d72bdb7fa0c5073c4b97984109f0dae0b10997a9aa723');self.assertEqual(m.RAW_BYTES,3446572)
 def test_valid_all_members_order_no_payload_APIs(self):
  r=inv(fixture([('ancillary',b'')]))
  self.assertEqual([x['archive_order'] for x in r['members']],list(range(5)));self.assertEqual(r['logical_split_member_paths'],{x:'root/'+x for x in m.LOGICAL});self.assertFalse(r['member_payload_opened']);self.assertFalse(r['CRC_payload_verified']);self.assertFalse(r['dataset_rows_parsed'])
 def test_DOS_directory_disagreement_repaired(self):
  x=zipfile.ZipInfo('file');x.create_system=0;x.external_attr=0x10
  with self.assertRaises(ValueError):inv(fixture([(x,b'')]))
 def test_PKWARE_link_extra_repaired(self):
  x=extra_info(0x000d,struct.pack('<IIHH',0,0,1000,1000)+b'root/train.txt')
  with self.assertRaises(ValueError):inv(fixture([(x,b'')]))
 def test_PKWARE_benign_fixed_metadata_retained(self):
  r=inv(fixture([(extra_info(0x000d,struct.pack('<IIHH',0,0,1000,1000)),b'')]))
  self.assertEqual(r['members'][-1]['extra_field_ids'],[0x000d])
 def test_ASI_special_link_and_valid_regular_metadata(self):
  for mode in (stat.S_IFLNK|0o777,stat.S_IFIFO|0o600):
   tail=struct.pack('<HIHH',mode,0,1000,1000);body=struct.pack('<I',zlib.crc32(tail))+tail
   with self.assertRaises(ValueError):inv(fixture([(extra_info(0x756e,body),b'')]))
  tail=struct.pack('<HIHH',stat.S_IFREG|0o644,0,1000,1000);body=struct.pack('<I',zlib.crc32(tail))+tail;inv(fixture([(extra_info(0x756e,body),b'')]))
 def test_masked_header_and_other_encryption_repaired(self):
  for flag in (1,0x40,0x2000):
   raw=bytearray(fixture());offset=raw.index(b'PK\x01\x02');old=struct.unpack_from('<H',raw,offset+8)[0];struct.pack_into('<H',raw,offset+8,old|flag)
   with self.subTest(flag=flag),self.assertRaises(ValueError):inv(bytes(raw))
 def test_logical_casefold_ambiguity_repaired(self):
  with self.assertRaises(ValueError):inv(fixture([('other/TRAIN.TXT',b'')]))
 def test_Unicode_path_alias_conflict_and_good_identity(self):
  name='metadata';base=b'\x01'+struct.pack('<I',zlib.crc32(name.encode()))
  with self.assertRaises(ValueError):inv(fixture([(extra_info(0x7075,base+b'other/train.txt',name),b'')]))
  inv(fixture([(extra_info(0x7075,base+name.encode(),name),b'')]))
 def test_malformed_duplicate_extra_records_rejected(self):
  for extra in (b'\x01',struct.pack('<HH',0x5455,10)+b'x',struct.pack('<HH',0xffff,0)*2):
   x=zipfile.ZipInfo('metadata');x.extra=extra
   with self.subTest(extra=extra),self.assertRaises((ValueError,zipfile.BadZipFile)):inv(fixture([(x,b'')]))
 def test_unsafe_paths_prefix_and_normalization_retained(self):
  for name in ('../x','/x','C:x','a\\x','a//x','a/./x','a/../x','a\x7fx','ROOT/TRAIN.TXT','other/train.txt','root'):
   with self.subTest(name=name),self.assertRaises(ValueError):inv(fixture([(name,b'')]))
  with self.assertRaises(ValueError):inv(fixture([('x/caf\u00e9',b''),('x/cafe\u0301',b'')]))
 def test_directory_payload_member_total_ratio_caps_retained(self):
  with self.assertRaises(ValueError):inv(fixture([('x/',b'payload')]))
  for field,value in (('members',3),('member_bytes',1),('total_bytes',1)):
   with self.subTest(field=field),patch.dict(m.LIMITS,**{field:value}),self.assertRaises(ValueError):inv(fixture())
  with patch.dict(m.LIMITS,compression_ratio=2),self.assertRaises(ValueError):inv(fixture([('compressible',b'x'*1000)]))
 def test_default_actual_constants_source_only(self):
  with patch.object(m,'_opaque_bytes',side_effect=AssertionError('No ZIP')),patch.object(b,'check_reservation',side_effect=AssertionError('No live accounting')),contextlib.redirect_stdout(io.StringIO()) as out:self.assertEqual(m.main([]),0)
  r=json.loads(out.getvalue());self.assertFalse(r['archive_opened']);self.assertFalse(r['live_budget_or_lock_access'])
class Gate(unittest.TestCase):
 def setUp(self):
  self.f=O/'f'/uuid.uuid4().hex;self.f.mkdir(parents=True);F=self.f;self.raw=fixture();self.reads=[]
  def put(p,data):
   q=F/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data if isinstance(data,bytes) else (json.dumps(data,sort_keys=True,indent=2)+'\n').encode());return {'path':p,'sha256':sh(q)}
  self.put=put;self.rawref=put('handmade/archive.zip',self.raw);source=put('src/arggnn/gst_uspto_archive_inventory_v2.py',ORIGINAL['SOURCE'].read_bytes());helper=put('src/arggnn/project_budget_v4.py',(R/'src/arggnn/project_budget_v4.py').read_bytes())
  proto=put('configs/gst_uspto_native_data_protocol_v1.json',(R/'configs/gst_uspto_native_data_protocol_v1.json').read_bytes());legacy_source=put('src/arggnn/gpu_pool_v1.py',(R/'src/arggnn/gpu_pool_v1.py').read_bytes())
  api=types.ModuleType('_private_handmade_v4');exec(compile((R/'src/arggnn/project_budget_v4.py').read_text(),'unchanged_v4_private_constants','exec'),api.__dict__);api.ROOT=F;api.SOURCE=F/helper['path'];api.CONFIG=F/'configs/project_budget_v4.json';api.FIXED_POLICY=copy.deepcopy(b.FIXED_POLICY)
  human=put('meta/human.json',{'kind':'human_project_storage_authorization','project_root':str(F),'authorized_hard_limit_bytes':1000000000000,'future_scheduling_ceiling_bytes':950000000000});api.FIXED_POLICY.update(root=str(F),human_authorization=human,legacy_pool_source=legacy_source);budget_config=put('configs/project_budget_v4.json',api.FIXED_POLICY);self.api=api
  cfg=json.loads(ORIGINAL['CONFIG'].read_text());cfg.update(raw={'path':self.rawref['path'],'sha256':self.rawref['sha256'],'bytes':len(self.raw)},budget_source=helper,budget_config=budget_config,native_protocol=proto);config=put('configs/gst_uspto_archive_inventory_v2.json',cfg)
  self.review={'kind':'gst_uspto_archive_inventory_v2_independent_review','review_role':'independent_reviewer','verdict':'PASS','implementation':source,'config':config,'source_handmade_metadata_verified':True,'actual_archive_opened':False,'actual_member_payload_opened':False,'fixture_mode':False,'actual_admission':False,'tested_scope':['explicit handmade capability only'],'fixture_only_not_actual_review':True};reviewref=put('meta/review.json',self.review)
  reg=put('meta/registration.json',{'kind':'gst_uspto_archive_inventory_v2_root_registration','authorizer':'root_agent','hypothesis_id':'H022','archive_metadata_run_registered':True,'implementation':source,'config':config,'raw':cfg['raw'],'payload_read_authorized':False,'native_model_authorized':False,'experiment_id':'handmade_metadata_only'})
  bindings={'implementation':helper,'config':budget_config,'human_authorization':human}
  br=put('meta/budget_review.json',{'kind':'project_budget_v4_independent_review','review_role':'independent_reviewer','verdict':'PASS',**bindings,'actual_apply_performed':False,'fixture_only':True})
  self.output=F/'new_output';request={'key':'gst_v2/handmade','output':str(self.output),'auxiliary_scopes':[],'reserved_bytes':20000000,'consumer_bindings':[source,config,reviewref]}
  reserve=put('meta/root_reserve.json',{'kind':'project_budget_v4_root_reserve_admission','root_authorizer':'root_agent','active':True,**bindings,'all_budget_writers_use_scheduler_mutex':True,'no_legacy_mutation_authorized':True,'GPU_authorized':False,'scientific_execution_authorized':False,'global_rebase_authorized':False,'reservations':[request],'independent_review':br,'fixture_only':True});self.reservation={**request,'root_admission':reserve};resref=put('meta/reservation.json',self.reservation)
  self.legacy={**api.FIXED_POLICY['legacy_invariants'],'root':str(F),'controllers':{'pool':{'pid':123,'reserved_bytes':67108864}},'jobs':{}}
  self.supp=api.empty_ledger(F,api.FIXED_POLICY,bindings);self.supp['reservations']={request['key']:self.reservation};lr=put(api.LEGACY,self.legacy);sr=put(api.SUPPLEMENTAL,self.supp)
  now=datetime.datetime.now(datetime.timezone.utc);self.auth={'kind':'gst_uspto_archive_inventory_v2_root_admission','authorizer':'root_agent','active':True,'metadata_only':True,'payload_read_authorized':False,'native_model_authorized':False,'implementation':source,'config':config,'raw':cfg['raw'],'limits':m.LIMITS.copy(),'created_utc':(now-datetime.timedelta(minutes=1)).isoformat(),'expires_utc':(now+datetime.timedelta(hours=1)).isoformat(),'registration':reg,'independent_review':reviewref,'output_directory':str(self.output.relative_to(F)),'reservation':resref,'initial_ledger_pins':{'legacy_ledger_sha256':lr['sha256'],'supplemental_ledger_sha256':sr['sha256']}}
  self.ar=put('meta/admission.json',self.auth);(F/api.LOCK).parent.mkdir(parents=True,exist_ok=True);(F/api.LOCK).write_bytes(b'handmade mutex')
  self.stack=contextlib.ExitStack();self.addCleanup(self.stack.close)
  values={'ROOT':F,'SOURCE':F/source['path'],'CONFIG':F/config['path'],'CONFIG_SHA256':config['sha256'],'RAW':Path(self.rawref['path']),'RAW_SHA':self.rawref['sha256'],'RAW_BYTES':len(self.raw)}
  for k,v in values.items():self.stack.enter_context(patch.object(m,k,v))
  self.stack.enter_context(patch.object(Path,'cwd',return_value=F));self.stack.enter_context(patch.object(b,'check_reservation',side_effect=api.check_reservation))
  opaque=m._opaque_bytes
  def tracked():self.reads.append('handmade_opaque_read');return opaque()
  self.stack.enter_context(patch.object(m,'_opaque_bytes',side_effect=tracked))
 def refresh(self):self.ar=self.put('meta/admission.json',self.auth)
 def run_cli(self):
  with contextlib.redirect_stdout(io.StringIO()),contextlib.ExitStack() as stack:
   for method in ('open','read','extract','extractall'):stack.enter_context(patch.object(zipfile.ZipFile,method,side_effect=AssertionError('Payload forbidden')))
   return m.main(['--execute','--admission',self.ar['path'],'--expected-admission-sha256',self.ar['sha256']])
 def denied_before_raw(self):
  with self.assertRaises((ValueError,TypeError)):self.run_cli()
  self.assertEqual(self.reads,[]);self.assertFalse(self.output.exists())
 def test_valid_real_v4_private_root_and_JSON_only(self):
  before={p:(self.f/p).read_bytes() for p in (self.api.LEGACY,self.api.SUPPLEMENTAL)};self.assertEqual(self.run_cli(),0);self.assertEqual(self.reads,['handmade_opaque_read']);self.assertEqual([p.name for p in self.output.iterdir()],['inventory.json']);r=json.loads((self.output/'inventory.json').read_text());self.assertEqual(r['status'],'metadata_only_complete');self.assertFalse(r['member_payload_opened'])
  for p,raw in before.items():self.assertEqual((self.f/p).read_bytes(),raw)
 def test_typed_active_1_repaired(self):self.auth['active']=1;self.refresh();self.denied_before_raw()
 def test_duplicate_authority_keys_repaired(self):
  p=self.f/self.ar['path'];p.write_text(json.dumps(self.auth).replace('{','{"active": false,',1));self.ar['sha256']=sh(p);self.denied_before_raw()
 def test_nonfinite_authority_rejected(self):
  self.auth['active']=float('nan');self.refresh();self.denied_before_raw()
 def test_author_role_and_fixture_review_repaired(self):
  self.review['review_role']='source_author';self.auth['independent_review']=self.put('meta/review.json',self.review);self.refresh();self.denied_before_raw()
 def test_fixture_review_rejected(self):
  self.review['fixture_mode']=True;self.auth['independent_review']=self.put('meta/review.json',self.review);self.refresh();self.denied_before_raw()
 def test_legacy_rebase_repaired_and_before_raw(self):
  self.legacy.update(schema='unknown',root='wrong-root',snapshot_sha256='0'*64,baseline_bytes=0,external_reserve_bytes=0);r=self.put(self.api.LEGACY,self.legacy);self.auth['initial_ledger_pins']['legacy_ledger_sha256']=r['sha256'];self.refresh();self.denied_before_raw()
 def test_bool_legacy_charge_repaired_and_before_raw(self):
  self.legacy['controllers']['pool']['reserved_bytes']=True;r=self.put(self.api.LEGACY,self.legacy);self.auth['initial_ledger_pins']['legacy_ledger_sha256']=r['sha256'];self.refresh();self.denied_before_raw()
 def test_stale_ledger_pin_denies_before_raw_repaired(self):self.auth['initial_ledger_pins']['legacy_ledger_sha256']='0'*64;self.refresh();self.denied_before_raw()
 def test_missing_own_reservation_denies_before_raw(self):
  self.supp['reservations'].clear();r=self.put(self.api.SUPPLEMENTAL,self.supp);self.auth['initial_ledger_pins']['supplemental_ledger_sha256']=r['sha256'];self.refresh();self.denied_before_raw()
 def test_expired_authority_denies_before_raw(self):self.auth['expires_utc']=(datetime.datetime.now(datetime.timezone.utc)-datetime.timedelta(seconds=1)).isoformat();self.refresh();self.denied_before_raw()
 def test_missing_run_registration_denies_before_raw(self):
  r=json.loads((self.f/self.auth['registration']['path']).read_text());r['archive_metadata_run_registered']=False;self.auth['registration']=self.put('meta/registration.json',r);self.refresh();self.denied_before_raw()
 def test_unknown_authority_field_denies_before_raw(self):self.auth['allow_any_archive']=True;self.refresh();self.denied_before_raw()
 def test_actual_opaque_mismatch_preserves_failure_receipt(self):
  (self.f/self.rawref['path']).write_bytes(self.raw[:-1]+b'X');self.assertEqual(self.run_cli(),1);r=json.loads((self.output/'inventory.json').read_text());self.assertEqual(r['status'],'failed_preserved');self.assertFalse(r['member_payload_opened'])
if __name__=='__main__':
 buff=io.StringIO();suite=unittest.TestSuite([Metadata('test_directory_payload_member_total_ratio_caps_retained')]);result=unittest.TextTestRunner(stream=buff,verbosity=2).run(suite)
 (O/'tests_attempt02.log').write_text(buff.getvalue());report={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'guard_events':EVENTS,'failure_cases':[x.id() for x,_ in result.failures],'error_cases':[x.id() for x,_ in result.errors],'actual_archive_opened':False,'actual_member_payload_opened':False,'actual_dataset_or_TEST_or_scientific_models':False,'actual_live_ledger_or_lock_access':False,'real_v4_code_used_only_with_private_handmade_constants':True}
 (O/'tests_attempt02.json').write_text(json.dumps(report,indent=2)+'\n');print(buff.getvalue());print(json.dumps(report,indent=2));raise SystemExit(0 if result.wasSuccessful() and not EVENTS else 1)
