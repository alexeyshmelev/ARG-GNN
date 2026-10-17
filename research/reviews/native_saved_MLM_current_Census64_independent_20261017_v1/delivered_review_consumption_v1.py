"""Exact delivered SOURCE_BLOCKED review must be rejected by actual candidate consumers, in RAM."""
from pathlib import Path
import copy,hashlib,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");OUT=ROOT/"research/reviews/native_saved_MLM_current_Census64_independent_20261017_v1"
rpath=OUT/"review.v1.json";data=rpath.read_bytes()
assert hashlib.sha256(data).hexdigest()=="b9389dfcc068d944712a72d35bfb70656deaf44f7ef231ac771380cc1e0c3ed0"
prefix=(OUT/"independent_checks_v1.py").read_text().split("# Full worker entry",1)[0];env={"__name__":"RAM_delivered_review_consumption"};exec(compile(prefix,"<audited_RAM_support>","exec"),env)
c=env["c"];builder=env["builder"];put=env["put"];read=env["read"]
delivered=put(str(rpath.relative_to(ROOT)),json.loads(data))
assert delivered["sha256"]==hashlib.sha256(data).hexdigest()
rejections=[]
def rejected(fn,label):
 try:fn()
 except ValueError as ex:rejections.append(dict(consumer=label,reason=str(ex)));return
 raise AssertionError("Blocked review accepted by "+label)
q=dict(env["q"],source_review_ref=delivered,action="prepare_master")
rejected(lambda:builder._request_v1(put("RAM/delivered_review_master_request.json",q),"prepare_master"),"actual ROOT builder request")
s=read(env["phase_prepared"]["contrastive"]["input_ref"]);a=read(s["root_stage_request_ref"]);m=read(a["master_input_ref"])
m["source_review_ref"]=delivered;a["source_review_ref"]=delivered
a["master_input_ref"]=put("RAM/delivered_review_master.json",m)
s["root_stage_request_ref"]=put("RAM/delivered_review_authority.json",a)
sr=put("RAM/delivered_review_phase_input.json",s);c.ROOT_INPUT_REF=None;c.set_amendment_v1(sr)
rejected(c.active_amendment_v1,"actual active phase contract")
result=dict(schema="native_Census64_delivered_blocked_review_consumption_v1",review_ref=delivered,rejected_count=len(rejections),rejections=rejections,actual_guard_calls=False,all_RAM=True)
(OUT/"delivered_review_consumption.result.v1.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps(result,sort_keys=True))
