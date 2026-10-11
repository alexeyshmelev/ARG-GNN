"""Reserve-only helper mocks. No actual journal, scope, process or GPU access."""
import ast,copy,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
root=Path.cwd();out=root/"research/reviews/full_reservation_admission_v1_independent"
path="research/reviews/full_reservation_admission_v1_root/admission.py"
source={"path":path,"sha256":hashlib.sha256((root/path).read_bytes()).hexdigest()}
assert source["sha256"]=="ec247bcd834f8d31bffd7f3e2c2f76a4dde60981158e5a041338ba8138ea82e2"
tree=ast.parse((root/path).read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="apply")
adapter_ref={"path":"src/arggnn/project_storage_full_reservation_v1.py","sha256":"adapter"}
authority_ref={"path":"mock-authority","sha256":"authority"}
review_ref={"path":"mock-review","sha256":"review"}
class Scope:
    def __init__(self,p,exists=False):self.p=p;self.existing=exists
    def exists(self):return self.existing
def run(change=None):
    lock={"held":False};calls=[];appended=[]
    before={"v4_projected_bytes":940000000000,"v9_reserved_bytes":10000000000,"projected_bytes":900000000000,"hard_limit_bytes":1000000000000,"journal_events":2,"journal_tail_sha256":"tail","legacy_ledger_sha256":"legacy","supplemental_ledger_sha256":"supplemental"}
    expected_pins={k:before[k] for k in ("legacy_ledger_sha256","supplemental_ledger_sha256","journal_tail_sha256")}
    req={"key":"new","output":"mock-new","auxiliary_scopes":[],"reserved_bytes":2000000000,"consumer_bindings":[adapter_ref]}
    authority={"operation":"reserve","accounting_mode":"full_reserved_no_closed_scope_credit","actual_reserve_helper":source,"full_reservation_helper_review":review_ref,"expected_before":expected_pins,"reservation":req}
    review={"verdict":"PASS","source":source}
    legacy={"jobs":{"legacy":{"output":"mock-legacy"}}};old={"reservations":{"old":{"key":"old","output":"mock-old","reserved_bytes":1}}};new={"prior":{"key":"prior","output":"mock-prior","reserved_bytes":10000000000}}
    paths=[Scope("mock-new")];closed={}
    if change=="close":authority["operation"]="close"
    if change=="rebase":authority["operation"]="global_rebase"
    if change=="wrong_mode":authority["accounting_mode"]="credited"
    if change=="wrong_helper":authority["actual_reserve_helper"]={"path":path,"sha256":"wrong"}
    if change=="wrong_review":review["verdict"]="FAIL"
    if change=="wrong_review_source":review["source"]={"path":path,"sha256":"wrong"}
    if change=="stale_pins":authority["expected_before"]={}
    if change=="missing_full_adapter":req["consumer_bindings"]=[]
    if change=="duplicate_key":req["key"]="prior"
    if change=="existing_scope":paths[0].existing=True
    if change=="overlap":paths[0].p="mock-old/child"
    if change=="max_records":old["reservations"]={str(i):{"key":str(i),"output":f"mock-old-{i}","reserved_bytes":1} for i in range(4096)}
    if change=="at_cap":before["v4_projected_bytes"]=988000000000
    if change=="over_cap":before["v4_projected_bytes"]=989000000000
    if change=="below_cap":before["v4_projected_bytes"]=987000000000
    if change=="wrong_hard_limit":before["hard_limit_bytes"]=999999999999
    original_maps=copy.deepcopy((legacy,old,new))
    class Mutex:
        def __enter__(self):assert not lock["held"];lock["held"]=True;calls.append("lock")
        def __exit__(self,*a):assert lock["held"];lock["held"]=False;calls.append("unlock")
    def held(label):assert lock["held"];calls.append(label)
    bindings={"source":"core","launcher":"launcher"};config={"mock":True}
    def reviewed(r,s,l):held("reviewed");assert r==root and s=="core" and l=="launcher";return config,bindings
    def check_authority(r,ref,operation,b):
        held("authority");assert ref==authority_ref and operation=="reserve" and b==bindings
        if authority["operation"]!=operation:raise ValueError("wrong operation")
        return authority
    def state(r,c,b,*,verify_closed):
        held("state");assert verify_closed is False
        return legacy,old,new,closed,before,[]
    def successor(r,res,stored=False):
        held("scope")
        return paths if res is req else [Scope(res["output"])]
    def canonical(r,p):return Scope(p)
    def overlap(a,b):return a.p==b.p or a.p.startswith(b.p+"/") or b.p.startswith(a.p+"/")
    def reference(r,p):return source if p==path else adapter_ref
    def bound(r,ref,required):held("review");assert ref==review_ref and required is True;return review
    def reconcile(base,o,n,cl):
        held("reconcile");assert base["projected_bytes"]==before["v4_projected_bytes"] and o is old["reservations"] and cl is closed
        reserved=sum(v["reserved_bytes"] for v in n.values())
        return {**base,"v9_reserved_bytes":reserved,"projected_bytes":1}
    def append(r,event,b,c,bind):
        held("append");assert event["operation"]=="reserve" and event["schema"]=="project_budget_v9_event"
        assert event["sequence"]==3 and event["previous_sha256"]=="tail"
        appended.append(event);return "new-tail"
    fake=SimpleNamespace(v4=SimpleNamespace(mutex=lambda r:Mutex(),reference=reference,bound=bound,canonical=canonical,scope_spec=lambda r,res,stored:[Scope(res["output"])],paths_overlap=overlap),reviewed=reviewed,check_authority=check_authority,state=state,pins=lambda v:{k:v[k] for k in expected_pins},successor_paths=successor,reconcile=reconcile,append_event=append)
    env={"budget":fake,"Path":Path,"copy":copy,"SOURCE":path}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),path,"exec"),env)
    result=error=None
    try:result=env["apply"](root,authority_ref,expected_source_sha256="core",expected_launcher_sha256="launcher")
    except ValueError as caught:error=caught
    assert not lock["held"] and (legacy,old,new)==original_maps
    if error is not None:assert not appended
    return result,error,calls,appended
r,e,c,a=run()
assert e is None and r["full_reserved_bytes"]==952000000000 and r["closed_credit_applied_to_full_charge"]==0 and len(a)==1
assert c.count("reviewed")==2 and c.count("authority")==2 and c[-1]=="unlock"
r,e,c,a=run("below_cap");assert e is None and r["full_reserved_bytes"]==999000000000
failures=["close","rebase","wrong_mode","wrong_helper","wrong_review","wrong_review_source","stale_pins","missing_full_adapter","duplicate_key","existing_scope","overlap","max_records","at_cap","over_cap","wrong_hard_limit"]
for change in failures:
    r,e,c,a=run(change);assert r is None and e is not None and not a,change
report={"verdict":"PASS","source":source,"tested_conditions":["reserve only;close/rebase rejected","same mutex and verify_closed=False original state","exact helper/accounting/review/source and fresh pins required","actual full-charge adapter binding required","fresh exclusive key/nonoverlap/nonexistent scopes/max4096","full original after-charge ignores credits;952GB/999GB pass;>=1TB and changed hard limit fail","original review/authority rechecked before unchanged reserve event append","failure appends nothing and historical maps remain unchanged"],"scope":"synthetic mocks only;no actual journal/proc/scope/data/GPU/admission operations"}
(out/"checks.json").write_text(json.dumps(report,indent=2)+"\n")
receipt={**report,"schema":"full_reservation_admission_v1_independent_review","created_at_utc":datetime.now(timezone.utc).isoformat(),"review_role":"independent_reviewer","actual_apply_performed":False,"checks_ref":{"path":"research/reviews/full_reservation_admission_v1_independent/checks.json","sha256":hashlib.sha256((out/"checks.json").read_bytes()).hexdigest()},"checks_script_ref":{"path":"research/reviews/full_reservation_admission_v1_independent/checks.py","sha256":hashlib.sha256((out/"checks.py").read_bytes()).hexdigest()},"blockers":[]}
(out/"receipt.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
print(json.dumps({"verdict":"PASS","receipt_path":"research/reviews/full_reservation_admission_v1_independent/receipt.json","receipt_sha256":hashlib.sha256((out/"receipt.json").read_bytes()).hexdigest(),"source":source},indent=2))
