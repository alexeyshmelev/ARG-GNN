"""Independent targeted FLARE conversion entry/source/memo2 bridge checks. No real resource or scientific calls."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import ast,copy,contextlib,hashlib,io,json,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");D=ROOT/"research/reviews/FLARE_original_VAL002004_conversion_execution_source_20261017_v1"
OUT=ROOT/"research/reviews/FLARE_original_VAL002004_conversion_execution_independent_20261017_v1"
checks=[];negative=[]
def check(ok,label):
 if not ok:raise AssertionError(label)
 checks.append(label)
def rejects(fn,label):
 try:fn()
 except(ValueError,KeyError,TypeError,AssertionError):negative.append(label);return
 raise AssertionError("Invalid accepted: "+label)
def reference(p):
 p=Path(p);p=p if p.is_absolute()else ROOT/p
 check(p.suffix in(".py",".json",".md",".sh",".txt",".patch",".diff")and p.stat().st_size<=32*2**20,"Bounded explicit source/metadata only")
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
check(reference(D/"source_handoff.v1.json")["sha256"]=="98079974083fbf6e9dfaf731dd217099e75d8062f8319cb6506198ef097e4d7f","Exact sealed handoff")
check(reference(D/"source_manifest.v1.json")["sha256"]=="ac1d30529084e84a3712b87e8b4cac2b55529feb0a65a980e998fa5f6ff4c23e","Exact sealed manifest")
hand=json.loads((D/"source_handoff.v1.json").read_text());manifest=json.loads((D/"source_manifest.v1.json").read_text());b=json.loads((D/"source_bindings.v1.json").read_text())
for path,sha in manifest["source_sha256"].items():check(reference(path)["sha256"]==sha,"Exact source pin")
for r in manifest["metadata_refs"]:check(reference(r["path"])==r,"Exact operational metadata pin")
prior=json.loads((ROOT/b["predecessor_manifest_ref"]["path"]).read_text())
check(all(manifest["source_sha256"].get(k)==v for k,v in prior["source_sha256"].items()),"All326 predecessor source pins preserved")
check(all(r in manifest["metadata_refs"]for r in prior["metadata_refs"]),"All154 predecessor metadata refs preserved")
# Inspected author harness executes only inert stdlib entry modules and RAM operations.
# Remove only its final exclusive RESULT publication so the sealed candidate is untouched.
author=(D/"entry_helper_checks_v1.py").read_text();tree=ast.parse(author)
removed=[n for n in tree.body if isinstance(n,ast.With)and "source_checks.result.v2.json"in ast.get_source_segment(author,n)]
check(len(removed)==1,"Only author harness result publication redirected to RAM")
tree.body=[n for n in tree.body if n not in removed];ast.fix_missing_locations(tree);env={"__name__":"independent_FLARE_entry_RAM_rerun"}
with contextlib.redirect_stdout(io.StringIO())as output:exec(compile(tree,"<inspected_actual_entry_harness_RAM_result_only>","exec"),env)
rerun=json.loads(output.getvalue());check(rerun["checks"]==47,"Inspected47 targeted actual entry checks rerun")
check(rerun["full_role_unions"]==dict.fromkeys(("american_c3_val_002","american_c3_val_003","american_c3_val_004"),62),"Three full62/64 actual RAM unions")
check(rerun["guard_resource_process_payload_truth_numerical_admission_launch_or_Git_operations"]is False,"No actual resource or payload operation")
c=env["c"];d=env["d"];h=env["h"];read=env["read"];documents=env["documents"];key=env["key"];mref=env["mref"];calls=env["calls"]
def changed(r,field,value,fn,label):
 old=copy.deepcopy(documents[key(r)]);documents[key(r)][field]=value;before=len(calls)
 try:rejects(fn,label);check(len(calls)==before,"Rejection precedes any RAM operation")
 finally:documents[key(r)]=old
for parent,adm in env["admitted"].items():
 inp=read(adm["input_ref"]);qref=inp["root_stage_request_ref"];q=read(qref)
 check(inp["parent_id"]==parent and q["parent_id"]==parent and inp["storage"]["reserved_new_bytes"]==b["original_targets"][parent]["whole_scope_reservation_bytes"],"Complete original count-derived parent admission")
 check(adm["execution_argv"][5]==d.HERE.replace("/",".")+".root_driver_v1","Exact emitted driver module")
 for stage in("native","seal"):
  changed(qref,"stage",stage,lambda:c.validate_request(qref),"Conversion-only request rejects "+stage)
 changed(qref,"parent_id","american_c3_val_000",lambda:c.validate_request(qref),"VAL000 rerun rejected")
 changed(qref,"source_review_ref",env["parent_selector"],lambda:c.validate_request(qref),"Incomplete or unrelated source review rejected")
 changed(qref,"created_at_utc",(datetime.now(timezone.utc)-timedelta(days=2)).isoformat(),lambda:c.validate_request(qref),"Original source clocks cannot reset or disagree with ROOT instruction")
 changed(qref,"proposed_reservation_key","H041/reused",lambda:c.validate_request(qref),"Another reservation key cannot substitute")
changed(env["policy"],"active",False,lambda:c.selected(env["selector_ref"]),"Inactive actual policy rejected")
changed(env["activation"],"ROOT_explicitly_releases_current_selector_for_this_child",False,lambda:c.selected(env["selector_ref"]),"Missing explicit ROOT Census release rejected")
changed(env["selector_ref"],"measurement_source_ref",b["predecessor_worker_ref"],lambda:c.selected(env["selector_ref"]),"Mixed measurement source rejected")
changed(env["authority"],"launch_authorized",True,lambda:h.construct_metadata_v1(env["authority"],c=c,b=b,put_metadata=env["put"],own_ref=mref(h.SOURCE)),"Metadata instruction cannot acquire launch authority")
changed(env["authority"],"expires_at_utc",(datetime.now(timezone.utc)+timedelta(hours=48)).isoformat(),lambda:c.validate_batch_instruction_v1(env["authority"]),"ROOT metadata clock must retain exact finite8h interval")
# Exact lifecycle and worker bodies, original scientific converter hash and no new stage fallback.
old=ast.parse((ROOT/b["predecessor_driver_ref"]["path"]).read_text());new=ast.parse((D/"root_driver_v1.py").read_text())
nodes=lambda t:{n.name:n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for name,node in nodes(old).items():
 if name not in("contract","admit"):check(ast.dump(node,include_attributes=False)==ast.dump(nodes(new)[name],include_attributes=False),"Exact retained lifecycle "+name)
check((D/"worker_entry_v1.py").read_bytes()==(ROOT/b["predecessor_worker_ref"]["path"]).read_bytes(),"Entire proven original worker byte-identical")
check(reference("src/arggnn/american_c3_flare_convert_v1.py")["sha256"]=="6e55e295226a183548a23258e7230855842bb876b4d2a90875d8f2ad53181ff5","Exact original scientific converter")
# New memo _check body itself is executed against RAM module/manifest readers.
bridge=env["actual_load"](b["ROOT_memo_bridge_ref"],"_independent_Census_memo_bridge")
base=env["actual_load"](b["ROOT_memo_source_ref"],"_independent_accepted_memo2")
qref=read(env["admitted"]["american_c3_val_002"]["input_ref"])["root_stage_request_ref"];q=read(qref)
uncached=bridge.current_uncached_paths_v1(c,q,qref,b)
current=[qref,*[q[k]for k in("ROOT_metadata_preparation_instruction_ref","source_manifest_ref","source_review_ref","root_directive_ref","registration_ref","context_ref","runtime_ref","storage_policy_selector_ref")]]
check(all(r["path"]in uncached for r in current),"Every current direct ROOT/control JSON explicitly excluded")
check(all(r["path"]in uncached for r in manifest["metadata_refs"]if Path(r["path"]).suffix==".json"),"All declared JSON metadata excluded from dynamic historical cache")
auth=[];source_auth=[]
def authenticated_functions(module,record,names):
 expected=b["selected_child_source_ref"]if module is env["g"]else base.PINS["v4"]if module is env["v4"]else base.PINS["budget"]
 if record!=expected:raise ValueError("Wrong source operand")
 auth.append((record,tuple(names)));return {n:(None,None)for n in names}
# Only dependencies of the derived body are RAM doubles; source body and AST transformation are actual.
real_source_bytes=base.source_bytes;base.functions_from_source=authenticated_functions
def source_bytes(r):
 source_auth.append(r)
 return real_source_bytes(r)
base.source_bytes=source_bytes
methods,count=bridge.derive_memo_methods_v1(base,env["raw"](b["ROOT_memo_source_ref"]),b["selected_child_source_ref"],uncached,b["ROOT_memo_bridge_ref"])
check(count==[1,1],"Exactly two memo source operands change")
fake=types.SimpleNamespace(guard=env["g"],v4=env["v4"],budget=env["budget"],manifest_ref=mref(d.MANIFEST),adapter_ref=b["ROOT_memo_source_ref"])
old_bound=getattr(env["v4"],"bound",None);env["v4"].bound=lambda root,r,as_json:read(r)
methods["_check"](fake)
check(auth[0][0]==b["selected_child_source_ref"]and len(auth)==3,"Actual derived check authenticates exact Census guard and unchanged v4/budget sources")
# Actual stable_source shim is used by derived state and reserve before returning/appending.
shim=methods["_run"].__globals__["source_bytes"];shim(b["ROOT_memo_source_ref"])
check(source_auth[-2:]==[b["ROOT_memo_source_ref"],b["ROOT_memo_bridge_ref"]],"Memo and bridge sources both freshly authenticated before return/append")
modified=dict(b["selected_child_source_ref"],sha256="0"*64)
wrong,_=bridge.derive_memo_methods_v1(base,env["raw"](b["ROOT_memo_source_ref"]),modified,uncached,b["ROOT_memo_bridge_ref"])
rejects(lambda:wrong["_check"](fake),"Altered guard authentication operand fails")
if old_bound is None:delattr(env["v4"],"bound")
else:env["v4"].bound=old_bound
# Original admission delta is exact-shape bound and every old reserve-body statement is retained.
original_driver=(ROOT/b["original_driver_ref"]["path"]).read_text();node=next(n for n in ast.parse(original_driver).body if isinstance(n,ast.FunctionDef)and n.name=="admit")
derived=bridge.derive_admission_v1(node)
check(not any(isinstance(n,ast.With)for n in derived.body),"No caller-held outer mutex in derived admission")
check(sum(isinstance(n,ast.Attribute)and n.attr=="read_state_v1"for n in ast.walk(derived))==1 and sum(isinstance(n,ast.Attribute)and n.attr=="reserve_v1"for n in ast.walk(derived))==1,"Exactly separate state and reserve transaction call sites")
# Existing inspected harness executed both real derived _run branches with cold transaction/final-append RAM sentinels.
check(env["trace"].count("lock")==env["trace"].count("unlock")==2 and env["trace"].count("restore_before_unlock")==2,"Two cold original mutex transactions restore before unlock")
check(env["trace"].index("original_append_fsync_postread")>max(i for i,x in enumerate(env["trace"])if x=="uncached_final"),"Original append/fsync/postread only after uncached final validation")
for p in D.glob("*.py"):compile(p.read_bytes(),str(p),"exec");check(True,"Compiles "+p.name)
check(reference(D/"source_handoff.v1.json")["sha256"]=="98079974083fbf6e9dfaf731dd217099e75d8062f8319cb6506198ef097e4d7f","Sealed candidate unchanged")
result=dict(schema="independent_FLARE_VAL002004_conversion_source_entry_checks_v1",independent=True,source_only=True,positive_count=len(checks),negative_count=len(negative),checks=checks,negative_cases=negative,inspected_author_entry_checks=47,complete_parent_routes=3,full_direct_role_unions=rerun["full_role_unions"],actual_resource_state_process_payload_model_or_numerical_calls=0,production_edits=False)
OUT.mkdir(exist_ok=True);(OUT/"independent_checks.v1.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items()if k not in("checks","negative_cases")},sort_keys=True))
