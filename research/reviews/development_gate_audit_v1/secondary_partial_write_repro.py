"""Controlled write failure; admission stub is explicitly non-executable bytes."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json,hashlib
from arggnn import secondary_control_bindings as b
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');OUT=ROOT/'research/reviews/development_gate_audit_v1'
TEMP=ROOT/'.runtime/tmp/development_gate_audit_v1/secondary_partial';TEMP.mkdir(parents=True,exist_ok=False)
target=TEMP/'synthetic_interrupted_config.json'
payload=b'{"audit_fixture_non_executable": true, "kind": "synthetic_fault_injection"}\n'
admission=SimpleNamespace(content=payload,check=lambda:None)
original=Path.open
class Broken:
    def __init__(self,handle):self.handle=handle
    def __enter__(self):return self
    def __exit__(self,*args):return self.handle.__exit__(*args)
    def write(self,value):
        self.handle.write(value[:17]);raise OSError('Independent synthetic config write interruption')
def broken(path,*args,**kwargs):
    handle=original(path,*args,**kwargs)
    return Broken(handle) if path==target and args and args[0]=='xb' else handle
error=None
with patch.object(b,'_bind',return_value=admission),patch.object(Path,'open',broken):
    try:b.bind_secondary_development({}, {}, {}, output_path=target)
    except Exception as caught:error={'type':type(caught).__name__,'reason':str(caught)}
exists=target.exists();content=target.read_bytes() if exists else b''
report={'admission_stub':'non-executable synthetic bytes, no actual cohort or config','error':error,
    'partial_file_survived':exists,'surviving_bytes':len(content),'surviving_sha256':hashlib.sha256(content).hexdigest(),
    'output_file':str(target),'source_sha256':hashlib.sha256((ROOT/'src/arggnn/secondary_control_bindings.py').read_bytes()).hexdigest()}
if exists:(OUT/'secondary_partial_surviving_bytes.txt').write_bytes(content)
(OUT/'secondary_partial_write_repro.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,sort_keys=True))
