"""Pure bound checks; no scope scan or write."""
from pathlib import Path
from types import SimpleNamespace
import ast,json
R=Path("/workspace-SR003.nfs2/ARG-GNN")
C="research/reviews/American_C3_VAL004_fresh7500MB_authority_source_20261016_v1"
def require(ok,msg):
 if not ok:raise ValueError(msg)
def reject(fn):
 try:fn()
 except ValueError:return
 raise AssertionError("Expected prospective bound rejection")
text=(R/C/"root_driver_v1.py").read_text()
node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=="boundary")
s=dict(output_directory="SYNTHETIC_OUTPUT",worker_auxiliary_scope="SYNTHETIC_AUX",storage={"SYNTHETIC":True})
calls=[]
g=SimpleNamespace(budget_check_v9=lambda *a,**kw:calls.append(kw))
cap=7500000000;aux=167772160;used=123456
ns=dict(need=require,current=lambda *a:None,CAPS=dict(reservation_bytes=cap,aux_bytes=aux),
 sample_scope=lambda spec:{"SYNTHETIC_OUTPUT":used,"SYNTHETIC_AUX":0})
exec(compile(ast.Module(body=[node],type_ignores=[]),"<exact_bound_no_scope_scan>","exec"),ns)
exact=cap-used-8192
ns["boundary"](g,s,exact);assert calls[-1]["required_write_bytes"]==exact+8192
before=len(calls);reject(lambda:ns["boundary"](g,s,exact+1));assert len(calls)==before
ns["boundary"](g,s,aux-8192,target="aux")
before=len(calls);reject(lambda:ns["boundary"](g,s,aux-8192+1,target="aux"));assert len(calls)==before
# The original scientific Writer forwards the full serializer plan; it does not clip to lower accounting.
text=(R/"src/arggnn/american_c3_common_v1.py").read_text()
node=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef)and n.name=="Writer")
prepared=[]
class VirtualPath:
 def __init__(self,name):self.name=name
 def __truediv__(self,name):return VirtualPath(self.name+"/"+name)
 @property
 def parent(self):return VirtualPath(self.name.rsplit("/",1)[0])
 def exists(self):return True if self.name=="SYNTHETIC_OUTPUT"else False
def storage_check(storage,**kw):
 prepared.append(kw)
 require(used+kw.get("required_write_bytes",0)<=cap,"Exact selected7.5B failstop")
gns=dict(local=lambda p:p,require=require)
exec(compile(ast.Module(body=[node],type_ignores=[]),"<exact_original_Writer_no_IO>","exec"),gns)
w=object.__new__(gns["Writer"]);w.storage={};w.adapter=SimpleNamespace(budget_check_v9=storage_check);w.path=VirtualPath("SYNTHETIC_OUTPUT")
w.prepare("original_full_product.pkl",cap-used-8192)
assert prepared[-1]["required_write_bytes"]==cap-used
reject(lambda:w.prepare("original_full_product.pkl",16000000000))
assert prepared[-1]["required_write_bytes"]==16000000000+8192
result=dict(schema="VAL004_fresh7500_independent_prospective_bound_checks_v1",status="PASS_SOURCE_ONLY",
 whole_output_plus_AUX_exact7_5B_boundary_accepts_next_byte_rejects=True,
 exact_AUX_boundary_accepts_next_byte_rejects=True,
 original_Writer_forwards_unmodified_full_serializer_plan_and_accounting_rejects_without_clipping=True,
 no_actual_guard_scope_scan_product_write_or_payload=True)
