"""Exact candidate focal seal on synthetic RAM NPZ; no scientific payloads."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,io,json,sys
import numpy as np
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL=sys.argv[1]
SOURCE=REL+"/worker_entry_v1.py";MANIFEST=REL+"/source_manifest.v1.json"
IDENTITY_SOURCE="src/arggnn/american_c3_gnn_targets_v1.py";PARENT="american_c3_train_002"
checks=[]
def need(x,msg):
 if not x:raise ValueError(msg)
def check(x,msg):need(x,msg);checks.append(msg)
def ref(path):return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
m=json.loads((ROOT/MANIFEST).read_text())
def extract(path,names,env):
 nodes=[n for n in ast.parse((ROOT/path).read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names]
 need(len(nodes)==len(names),"All exact candidate functions")
 exec(compile(ast.Module(body=nodes,type_ignores=[]),path+"#independent-RAM","exec"),env)
 return env
class RAMNPZ(io.BytesIO):
 def read_bytes(self):return self.getvalue()
 def stat(self):return SimpleNamespace(st_size=len(self.getvalue()))
payload=None;done=None;writes=[]
def local(path):
 if path=="RAM_SYNTHETIC/focals.npz":return payload
 need(path==IDENTITY_SOURCE,"No access to existing science files")
 return ROOT/path
def metadata_ref(path):
 need(path in (IDENTITY_SOURCE,MANIFEST),"Only allowed immutable source/manifest")
 return ref(path)
def read(record):need(record==ref(MANIFEST),"Only immutable source manifest read");return copy.deepcopy(m)
def write(name,value):writes.append((name,copy.deepcopy(value)));return value
env=dict(need=need,local=local,metadata_ref=metadata_ref,read=read,hashlib=hashlib,json=json,IDENTITY_SOURCE=IDENTITY_SOURCE,MANIFEST=MANIFEST,PARENT=PARENT)
extract(SOURCE,{"canonical_ref_shape","publish_focal_seal"},env)
nodes=ast.parse((ROOT/IDENTITY_SOURCE).read_text()).body
keep=[n for n in nodes if isinstance(n,ast.Assign)and any(isinstance(v,ast.Name)and v.id=="IDENTITY"for v in n.targets)or isinstance(n,ast.FunctionDef)and n.name=="identity_sha"]
original=dict(np=np,hashlib=hashlib,json=json)
exec(compile(ast.Module(body=keep,type_ignores=[]),IDENTITY_SOURCE+"#independent-synthetic-reference","exec"),original)
base=dict(record_ids=np.arange(10000,dtype=np.int64)*31+3,sample_nodes=np.arange(10000,dtype=np.int64)%1000+600,
 sample_indices=np.arange(10000,dtype=np.int64)%1000+600,positions_bp0=np.arange(10000,dtype=np.float64)*3000+.5,
 parent_id=np.asarray(PARENT),class_order=np.asarray(["EUR","EAS","AFR"]))
def prepare(records,hash_wrong=False):
 global payload,done
 with io.BytesIO()as stream:
  np.savez(stream,**records);payload=RAMNPZ(stream.getvalue())
 done=dict(focals_ref=dict(path="RAM_SYNTHETIC/focals.npz",sha256="0"*64 if hash_wrong else hashlib.sha256(payload.getvalue()).hexdigest()),
  observation_ref=dict(path="RAM_SYNTHETIC/inferred_manifest.json",sha256="1"*64),status="completed_native_C3_path_preparation")
prepare(base);before=copy.deepcopy(done);seal=env["publish_focal_seal"](done,SimpleNamespace(json=write))
check(seal["focal_identity_sha256"]==original["identity_sha"](base),"Exact original identity hash on all10000 RAM records")
check(done==before,"Original completed metadata unchanged")
check(len(writes)==1 and writes[0][0]=="focal_seal.json"and seal["query_truth_opened"]is False,"One separate metadata seal, no ancestry truth")
for name,change in [
 ("duplicate_record_id",lambda a:a["record_ids"].__setitem__(99,a["record_ids"][0])),
 ("wrong_parent",lambda a:a.update(parent_id=np.asarray("american_c3_train_003"))),
 ("wrong_class_order",lambda a:a.update(class_order=np.asarray(["AFR","EAS","EUR"]))),
 ("short_position_array",lambda a:a.update(positions_bp0=a["positions_bp0"][:-1])),
 ("missing_identity_field",lambda a:a.pop("sample_nodes")),
 ("wrong_NPZ_hash",lambda a:None)]:
 records={k:v.copy()for k,v in base.items()};change(records);prepare(records,hash_wrong=name=="wrong_NPZ_hash");n=len(writes)
 try:env["publish_focal_seal"](done,SimpleNamespace(json=write))
 except (ValueError,KeyError):check(len(writes)==n,"No seal after invalid synthetic focal metadata: "+name)
 else:raise AssertionError("Invalid synthetic focal metadata accepted: "+name)
print(json.dumps(dict(schema="independent_original_focal_RAM_checks_v1",source_ref=ref(SOURCE),source_manifest_ref=ref(MANIFEST),checks_passed=len(checks),checks=checks,
 synthetic_NPZ_only=True,existing_scientific_payload_reads_or_hashes=0,guard_imports_probes_operations=0),sort_keys=True,indent=2))
