"""Delivered exact independent v2 SOURCE_PASS is consumed by full candidate ROOT/contract paths in RAM."""
from pathlib import Path
import hashlib,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");OUT=ROOT/"research/reviews/native_saved_MLM_current_Census64_independent_20261017_v2"
rpath=OUT/"review.v2.json";data=rpath.read_bytes();expected="740ff5c7a65363ac4f2b01dba39ba2d891739c2bc5de3ab0dc6bbd923af23417"
assert hashlib.sha256(data).hexdigest()==expected
prefix=(OUT/"independent_checks_v1.py").read_text().split("# Full worker entry",1)[0];env={"__name__":"RAM_delivered_v2_pass"};exec(compile(prefix,"<audited_v2_RAM_support>","exec"),env)
c=env["c"];builder=env["builder"];put=env["put"];read=env["read"]
delivered=put(str(rpath.relative_to(ROOT)),json.loads(data));assert delivered["sha256"]==expected
q=dict(env["q"],instance="delivered_independent_pass",source_review_ref=delivered,action="prepare_master")
rr=put("RAM/delivered_v2_master_request.json",q)
assert builder._request_v1(rr,"prepare_master")["source_review_ref"]==delivered
admitted=builder.prepare_master_v1(rr);master=read(admitted["master_input_ref"])
assert master["source_review_ref"]==delivered
accepted=[]
for phase in c.PHASES:
 p=env["phase_request"](q,admitted,phase);c.ROOT_INPUT_REF=None
 prepared=builder.prepare_phase_v1(put("RAM/delivered_v2_"+phase+"_request.json",p))
 s,a,cfg,storage,_=c.validate_input_v1(prepared["input_ref"])
 assert a["source_review_ref"]==delivered and s["phase"]==phase
 accepted.append(phase)
# A copied prior blocked verdict cannot stand in for the delivered v2 pass.
bad=dict(json.loads(data),verdict="SOURCE_BLOCKED",blockers=[dict(id="predecessor")])
qbad=dict(q,source_review_ref=put("RAM/delivered_v2_blocked_review.json",bad))
try:builder._request_v1(put("RAM/delivered_v2_blocked_request.json",qbad),"prepare_master")
except ValueError:blocked_rejected=True
else:raise AssertionError("Blocked source review accepted")
result=dict(schema="native_Census64_delivered_SOURCE_PASS_consumption_v2",review_ref=delivered,actual_candidate_request_master_and_seven_phase_contracts_accepted=accepted,blocked_review_rejected=blocked_rejected,actual_guard_state_resource_calls=False,all_RAM=True)
(OUT/"delivered_review_consumption.result.v2.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps(result,sort_keys=True))
