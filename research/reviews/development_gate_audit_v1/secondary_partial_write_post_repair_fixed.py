"""Independent publication faults. All positive admission is a non-executable stub.
This isolates public API emission only; no cohort, producer selection, or fitting is claimed.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib,json,os
from arggnn import secondary_control_bindings as b
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');OUT=ROOT/'research/reviews/development_gate_audit_v1'
TEMP=ROOT/'.runtime/tmp/development_gate_audit_v1/secondary_partial_post_fixed';TEMP.mkdir(parents=True,exist_ok=False)
payload=b'{"audit_fixture_non_executable": true, "kind": "synthetic_fault_injection"}\n'
existing=b'{"audit_fixture_non_executable": true, "owned_by_racer": true}\n'
checks={}
for api in ('binder','synthesizer'):
 for fault in ('write','fsync','check','race'):
  directory=TEMP/(api+'_'+fault);directory.mkdir();target=directory/'interrupted.json'
  calls=[]
  def check():
   calls.append(None)
   # The selector calls check once before serializing and once before publication.
   if api=='synthesizer' and len(calls)==1:return
   if fault=='check':raise RuntimeError('Independent synthetic admission interruption')
   if fault=='race':target.write_bytes(existing)
  if api=='binder':
   admission=SimpleNamespace(content=payload,check=check)
   mocked=patch.object(b,'_bind',return_value=admission)
   invoke=lambda:b.bind_secondary_development({}, {}, {}, output_path=target)
  else:
   config=directory/'non_executable_input.json';config.write_bytes(payload)
   settings={key:'a'*64 for key in b.main.BINDINGS}
   settings.update(models=['synthetic_lr0','synthetic_lr1'],controls=['synthetic_control'],
    model_specs={name:{'control':'synthetic_control'} for name in ('synthetic_lr0','synthetic_lr1')},
    bound_implementation_sha256={},validation_parents=[],main_selection_plan={'path':'synthetic','sha256':'b'*64},
    main_selection_receipt={'path':'synthetic','sha256':'c'*64},selected_candidate={'synthetic':True},
    primary_strongest_baseline={'synthetic':True},optimizer_starts=6)
   admission=SimpleNamespace(settings=settings,capture=b.main._Capture(ROOT),
     dataset=SimpleNamespace(cases=lambda _:[]),check=check)
   mocked=patch.object(b,'admit_secondary',return_value=admission)
   invoke=lambda:b.synthesize_secondary_selection(config,hashlib.sha256(payload).hexdigest(),output_path=target)
  original=b.tempfile.NamedTemporaryFile
  class Broken:
   def __init__(self,handle):self.handle=handle;self.name=handle.name
   def __enter__(self):return self
   def fileno(self):return self.handle.fileno()
   def __exit__(self,*args):return self.handle.__exit__(*args)
   def write(self,value):
    self.handle.write(value[:17]);raise OSError('Independent synthetic staged partial-write interruption')
  emitted_error=None
  with mocked:
   if fault=='write':
    with patch.object(b.tempfile,'NamedTemporaryFile',lambda **kwargs:Broken(original(**kwargs))):
     try:invoke()
     except Exception as error:emitted_error={'type':type(error).__name__,'reason':str(error)}
   elif fault=='fsync':
    with patch.object(b.os,'fsync',side_effect=OSError('Independent synthetic fsync interruption')):
     try:invoke()
     except Exception as error:emitted_error={'type':type(error).__name__,'reason':str(error)}
   else:
    try:invoke()
    except Exception as error:emitted_error={'type':type(error).__name__,'reason':str(error)}
  pending=list(directory.glob('.secondary_pending_*'))
  assert emitted_error is not None and not pending,(api,fault,emitted_error)
  assert target.exists()==(fault=='race'),(api,fault,emitted_error)
  if fault=='race':assert target.read_bytes()==existing
  checks[api+'_'+fault]={'error':emitted_error,'output_exists':target.exists(),
   'pending_files':len(pending),'raced_bytes_preserved':target.read_bytes()==existing if target.exists() else None,
   'check_calls':len(calls)}
report={'positive_admission':'non-executable synthetic stubs for public emission only',
 'no_actual_cohort_access':True,'checks':checks,
 'source_sha256':hashlib.sha256((ROOT/'src/arggnn/secondary_control_bindings.py').read_bytes()).hexdigest()}
(OUT/'secondary_partial_write_post_repair_fixed.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,sort_keys=True))

