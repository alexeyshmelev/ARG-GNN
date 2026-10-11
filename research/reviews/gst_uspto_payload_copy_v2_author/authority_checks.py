"""Handmade V1 counterexample and V2 sticky-denial/failure diagnostics checks."""
import contextlib
import datetime
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import stat
import sys
import uuid
import zipfile
from arggnn import gst_uspto_payload_copy_v1 as old
from arggnn import gst_uspto_payload_copy_v2 as new
from arggnn import gst_uspto_archive_inventory_v2 as inv

OUT=Path(__file__).resolve().parent;cases=OUT/('authority-'+uuid.uuid4().hex);cases.mkdir()
checks=[];forbidden=[];actual=new.ROOT/inv.RAW

def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  name=os.fsdecode(args[0]);p=Path(name)
  if name==str(inv.RAW) or (p.is_absolute() and (p==actual or str(p).startswith(str(new.ROOT/'data')))):
   forbidden.append(name);raise AssertionError('No actual archive/data')
sys.addaudithook(audit)
def ck(name,value):
 checks.append({'check':name,'passed':bool(value)})
 if not value:raise AssertionError(name)
def reject(name,fn):
 try:fn()
 except ValueError:ck(name,True)
 else:ck(name,False)
def bytes_sha(raw):return hashlib.sha256(raw).hexdigest()
data=[('data/',b''),('data/connor_original.tsv',bytes(range(256))*130),('data/test.txt',b'opaque_test_bytes'),('data/train.txt',b'opaque_train_bytes'),('data/valid.txt',b'opaque_valid_bytes')]
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w') as archive:
 for name,payload in data:
  info=zipfile.ZipInfo(name);info.create_system=3;info.external_attr=((stat.S_IFDIR|0o755)<<16)|0x10 if name.endswith('/') else (stat.S_IFREG|0o644)<<16;info.compress_type=zipfile.ZIP_STORED;archive.writestr(info,payload)
raw=buf.getvalue();frozen=inv.inventory(raw)
reproduction=[]
for label,api,fn in [('sealed_V1',old,old.copy_payload_v1),('corrected_V2',new,new.copy_payload_v2)]:
 destination=cases/label;state={'denial':None,'guard_calls':0}
 def guard(planned):
  state['guard_calls']+=1
  copied=destination/data[1][0]
  if copied.exists() and copied.stat().st_size==len(data[1][1]):
   if state['denial'] is None:
    progress=(destination/'progress.jsonl').read_bytes();state['denial']={'progress_bytes':len(progress),'progress_sha256':bytes_sha(progress),'guard_call':state['guard_calls']}
   raise ValueError('handmade admission/reservation guard explicitly denied')
 result,receipt=fn(raw,frozen,destination,guard=guard)
 after=(destination/'progress.jsonl').read_bytes();delta=len(after)-state['denial']['progress_bytes']
 reproduction.append({'case':label,'before':state['denial'],'after_progress_bytes':len(after),'after_progress_sha256':bytes_sha(after),'bytes_appended_after_denial':delta,'result_status':result['status'],'final_receipt_sha256':receipt,'guard_calls':state['guard_calls']})
 ck(label+' exact existing payload preserved',(destination/data[1][0]).read_bytes()==data[1][1])
 ck(label+' later payload absent and final receipt refused',not (destination/data[2][0]).exists() and receipt is None and result['final_receipt_written'] is False)
 if label=='sealed_V1':ck('sealed V1 independently reproduced151-byte failure append',delta==151)
 else:
  ck('V2 zero progress bytes after explicit denial',delta==0 and bytes_sha(after)==state['denial']['progress_sha256'])
  ck('V2 original denial preserved in result',result['error']=='handmade admission/reservation guard explicitly denied' and result['authority_rejected'])
  ck('V2 sticky latch prevents later guard reattempt',state['guard_calls']==state['denial']['guard_call'])
# Deny specifically the failure event after an independent CRC rejection.
with zipfile.ZipFile(io.BytesIO(raw)) as archive:item=archive.infolist()[1]
import struct
header=struct.unpack_from('<IHHHHHIIIHH',raw,item.header_offset);payload_start=item.header_offset+30+header[-2]+header[-1]
corrupt=bytearray(raw);corrupt[payload_start]^=1;corrupt=bytes(corrupt)
destination=cases/'deny_failure_event';state={'before':None,'denied_calls':0}
def fail_event_guard(planned):
 check_frame=inspect.currentframe().f_back;caller=check_frame.f_back
 if caller.f_code.co_name=='event' and caller.f_locals.get('kind')=='copy_failed':
  progress=(destination/'progress.jsonl').read_bytes();state['before']=progress;state['denied_calls']+=1
  raise ValueError('handmade authority revoked before failure event')
result,receipt=new.copy_payload_v2(corrupt,inv.inventory(corrupt),destination,guard=fail_event_guard)
ck('failure event itself is guarded before write',state['denied_calls']==1 and state['before']==(destination/'progress.jsonl').read_bytes())
ck('CRC original cause survives failure logging denial','Actual member CRC32 differs' in result['error'] and 'authority revoked before failure event' in result['failure_event_error'])
ck('failure-event denial forbids final receipt',receipt is None and result['authority_rejected'] and result['final_receipt_written'] is False)
# Guard recovery cannot reopen destination effects after any explicit rejection.
destination=cases/'direct_latch';calls=[]
def once_denied(planned):
 calls.append(planned)
 if len(calls)==1:raise ValueError('handmade one-time denial')
scope=new.CopyScopeV2(destination,once_denied,new.RESERVATION_BYTES)
reject('initial direct event explicitly denied',lambda:scope.event('original_failure'))
before=(destination/'progress.jsonl').read_bytes();entries=sorted(p.name for p in destination.iterdir())
for label,fn in [
 ('second event',lambda:scope.event('retry_failure')),
 ('direct write',lambda:scope.write(scope.progress,b'forbidden')),
 ('normal file',lambda:scope.file('forbidden.txt')),
 ('unguarded internal file',lambda:scope.file('forbidden-internal.txt',guarded=False)),
 ('directory',lambda:scope.directories(['forbidden-dir'])),
 ('final receipt',lambda:scope.finish({'forbidden':True}))]:
 reject('sticky denial rejects '+label,fn)
 ck('sticky denial '+label+' has no owned effect',before==(destination/'progress.jsonl').read_bytes() and entries==sorted(p.name for p in destination.iterdir()))
ck('recovered callback never consulted after latch',len(calls)==1)
scope.close()
# Main's preflight fallback also latches denial. All admission/opaque/budget
# hooks here are explicit handmade stubs, not real authorities or archive IO.
real_admit=new.admit_v2;real_opaque=inv._opaque_bytes;real_bound=new.budget.bound;real_check=new.budget.check_reservation
now=datetime.datetime.now(datetime.timezone.utc)
for label,opaque_error in [('initial_copy_denial',False),('opaque_failure_then_denial',True)]:
 destination=cases/label;guard_calls=[]
 admission={'created_utc':now.isoformat(),'expires_utc':(now+datetime.timedelta(minutes=20)).isoformat(),'implementation':{'handmade':True},'config':{'handmade':True},'registration':{'handmade':True},'independent_review':{'handmade':True}}
 cfg={'budget_source':{'sha256':'b'*64},'budget_config':{'sha256':'c'*64},'inventory':{'handmade':True}}
 new.admit_v2=lambda _: (admission,cfg,frozen,{'handmade':True},destination,{'handmade':True})
 def opaque():
  if opaque_error:raise ValueError('handmade original opaque preflight failure')
  return raw
 inv._opaque_bytes=opaque;new.budget.bound=lambda *a,**k:None
 def denied_check(*args,**kwargs):
  guard_calls.append(kwargs);raise ValueError('handmade original exact authority rejection')
 new.budget.check_reservation=denied_check
 buffer=io.StringIO()
 try:
  with contextlib.redirect_stdout(buffer):code=new.main(['--execute','--admission','handmade','--expected-admission-sha256','a'*64])
 finally:
  new.admit_v2=real_admit;inv._opaque_bytes=real_opaque;new.budget.bound=real_bound;new.budget.check_reservation=real_check
 report=json.loads(buffer.getvalue())
 ck(label+' no output directory or guard retry',not destination.exists() and len(guard_calls)==1 and code==1)
 ck(label+' safe caller stdout marks denial',report['receipt_sha256'] is None and report['failure_diagnostics']['authority_rejected'])
 expected='handmade original opaque preflight failure' if opaque_error else 'handmade original exact authority rejection'
 ck(label+' original cause retained on caller stdout',report['failure_diagnostics']['error']==expected)
ck('zero forbidden actual archive/data attempts',forbidden==[])
result={'kind':'gst_uspto_payload_copy_v2_authority_checks','status':'PASS','checks':checks,'checks_passed':len(checks),'decisive_cases':reproduction,'actual_archive_opened':False,'actual_member_payload_opened':False,'actual_budget_or_authority_used':False,'handmade_main_hook_stubs_only':True,'forbidden_access_events':forbidden}
(OUT/'authority_checks.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':'PASS','checks_passed':len(checks),'V1_bytes_after_denial':reproduction[0]['bytes_appended_after_denial'],'V2_bytes_after_denial':reproduction[1]['bytes_appended_after_denial']},sort_keys=True))
