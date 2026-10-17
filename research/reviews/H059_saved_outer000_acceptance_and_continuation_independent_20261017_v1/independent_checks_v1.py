"""Narrow independent H059 source and actual entry joins; all operational dependencies remain RAM doubles."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import ast,copy,contextlib,hashlib,io,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
D=ROOT/"research/reviews/H059_saved_outer000_acceptance_and_continuation_source_20261017_v1"
OUT=ROOT/"research/reviews/H059_saved_outer000_acceptance_and_continuation_independent_20261017_v1"
checks=[];negatives=[]
def check(ok,label):
 if not ok:raise AssertionError(label)
 checks.append(label)
def rejects(fn,label):
 try:fn()
 except (ValueError,KeyError,AssertionError,TypeError):negatives.append(label);return
 raise AssertionError("Invalid accepted:"+label)
def reference(p):return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
pins={"source_handoff.v1.json":"11b554b30d66a8f5f1d0a0e49c43370df3e94f9d95b1387d99efa9ce70a02eaa","source_manifest.v1.json":"7794bfcf29cbacacc24f1388a10f7b48642c1e5f822094c877b246d285d0e88e","audit_manifest.v1.json":"db32a87a9a79cd0bee20d08dc56d6fcd6cc2def8df750ec827a20af122621ec2"}
for name,h in pins.items():check(reference(D/name)["sha256"]==h,"Exact candidate "+name)
handoff=json.loads((D/"source_handoff.v1.json").read_text());manifest=json.loads((D/"source_manifest.v1.json").read_text())
audit=json.loads((D/"audit_manifest.v1.json").read_text())
allowed={".py",".json",".md",".sh",".patch",".diff"}
for r in manifest["refs"]:
 p=ROOT/r["path"];check(p.suffix in allowed,"Only explicit bounded source/metadata closure")
 check(p.stat().st_size<=32*2**20 and reference(p)==r,"Candidate immutable closure pin")
for r in audit["files"]:check(reference(ROOT/r["path"])==dict(path=r["path"],sha256=r["sha256"]),"Exact artifact audit ref")
prior=json.loads((ROOT/manifest["predecessor_manifest_ref"]["path"]).read_text())
check(all(r in manifest["refs"]for r in prior["refs"]),"All330 inherited refs preserved")
check(manifest["expected_original_metadata_and_assets"]==prior["expected_original_metadata_and_assets"],"Six original expected assets carried without reads")
# Author's complete eight-entry and64-dispatch harness is inspected before this rerun.
# It imports only inert stdlib candidate modules; real guard/process/science/payload calls are explicit RAM replacements.
author=(D/"source_check_v1.py").read_text()
author=author.replace('check(len(prediction["all_data_ready_v1"]','prediction["importlib"]=Imports()\ncheck(len(prediction["all_data_ready_v1"]')
env={"__name__":"H059_author_harness_independent_rerun"}
with contextlib.redirect_stdout(io.StringIO())as output:exec(compile(author,"<inspected_source_RAM_entry_harness>","exec"),env)
rerun=json.loads(output.getvalue())
check(rerun["actual_guard_state_process_resource_GPU_admission_journal_payload_operations"]==0,"No actual resource or payload operation in entry harness")
check(rerun["full_RAM_request_admit_input_count"]==8,"Complete actual eight builder/admit/input joins")
check(rerun["full_source_derived_role_union_counts"]==[62]*7+[58],"Full source-derived actual RAM role unions<=64")
# Independent negative joins using the complete already-connected RAM fixtures.
saved=env["saved"];pred=env["prediction"];read=env["read"];ref=env["ref"];docs=env["docs"];put=env["putram"]
def changed(r,field,value,call,label):
 old=copy.deepcopy(docs[r["path"]]);docs[r["path"]][field]=value
 try:rejects(call,label)
 finally:docs[r["path"]]=old
for r,field,value in[(env["wait"],"complete_owned_roster_known",False),(env["wait"],"actual_OS_wait_status",1),(env["terminal"],"actual_tool_terminal_consumed",False),(env["result"],"original_guard_monitoring_interval_recovered",True),(env["rev"],"read_only_artifact_reuse_preserved",False)]:
 changed(r,field,value,lambda:saved["acceptance_values_v1"](env["accept"],read,ref),"Saved acceptance rejects "+field)
check(saved["HISTORY"]["original_outer_exit_code"]==1 and saved["HISTORY"]["original_worker_OS_wait"]==saved["HISTORY"]["original_owned_descendant_drain"]=="UNKNOWN","Original34504outer1 and unknown wait/drain immutable")
def exact_original_metadata(r):
 path=ROOT/r["path"]
 check(path.suffix==".json"and not any(x.lower()in("truth","oracle")for x in path.parts),"Only bounded original operational JSON metadata, no scientific payload")
 raw=path.read_bytes();check(len(raw)<=32*2**20 and hashlib.sha256(raw).hexdigest()==r["sha256"],"Original exact metadata ref")
 return json.loads(raw)
complete,inp,cfg,row,generation=saved["metadata_v1"](exact_original_metadata)
inference,entry,commands=saved["native_metadata_v1"](complete,exact_original_metadata)
check(inference["outer_command_returncodes"]==[0]*5 and entry["samples"]==952,"Five genuine helper waits0/full952 metadata joins, original outer history still unknown")
for index,(ns,qref,answer,union)in enumerate(env["answers"][:7],1):
 q=read(qref);b=read(ref(ns["HERE"]+"/source_bindings.outer%03d.v1.json"%index))
 check(ns["PARENT"]=="ap_expanded_outer_%03d"%index and ns["OUTPUT"]=="data/expanded_full_panel_v1/"+ns["PARENT"],"Each exact original fixed row and canonical output")
 check(b["proposed_reservation_key"]==q["proposed_reservation_key"],"Each exact unused-key request binding")
 changed(qref,"parent_id","ap_expanded_outer_000",lambda ns=ns,qref=qref:ns["request"](qref),"Old000 cannot substitute remaining row")
 changed(qref,"automatic_retry_or_reseed",True,lambda ns=ns,qref=qref:ns["request"](qref),"No reseed or automatic retry")
 old=read(qref)["saved_outer000_acceptance_ref"]
 changed(qref,"saved_outer000_acceptance_ref",env["wait"],lambda ns=ns,qref=qref:ns["request"](qref),"Unaccepted old saved artifact cannot start remaining generation")
 # Original scientific worker remains exact after only parent-number normalization.
 oldsource=(ROOT/"research/reviews/H059_outer000_Relate64_execution_source_20261017_v1/prepare_outer000_v1.py").read_text()
 newsource=(D/("prepare_outer%03d_v1.py"%index)).read_text().replace("outer%03d"%index,"outer000").replace("outer_%03d"%index,"outer_000")
 fn=lambda s,name:next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and n.name==name)
 check(ast.dump(fn(oldsource,"worker"),include_attributes=False)==ast.dump(fn(newsource,"worker"),include_attributes=False),"Full original scientific worker AST preserved")
vns,vqref,_,_=env["answers"][-1]
for key,value in[("original_whole_reservation_bytes",1),("original_charge_retained",False),("complete_original_output_and_packet_measured",False),("historical_monitoring_gap_recovered",True)]:
 changed(env["readonly_ref"],key,value,lambda:vns["request"](vqref),"Original readonly qualification rejects "+key)
for parent in pred["PARENTS"]:
 method=pred["METHODS"][-1];fitref=env["previous"][parent][method]["fit"]["completion_ref"]
 changed(fitref,"query_truth_used",True,lambda parent=parent:pred["build_stage_v1"](env["intentref"],parent,method,"predict",read,ref),"Same-parent native fit cannot use truth")
 fit=docs[fitref["path"]];backup=copy.deepcopy(fit);fit["MLM"]["optimizer_updates"]-=1
 try:rejects(lambda parent=parent:pred["build_stage_v1"](env["intentref"],parent,method,"predict",read,ref),"Native MLM full update budget cannot shrink")
 finally:docs[fitref["path"]]=backup
 prepref=env["previous"][parent][method]["prepare"]["completion_ref"]
 changed(prepref,"reference_haplotypes",265,lambda parent=parent:pred["build_stage_v1"](env["intentref"],parent,method,"fit",read,ref),"Full952 source roster cannot shrink")
for bad in[env["cells"][:-1],env["cells"]+[env["cells"][0]]]:
 rejects(lambda bad=bad:pred["build_matrix_bundle_v1"](env["intentref"],bad,read,ref),"No missing or duplicate matrix cells")
needfn=fn((D/"prediction_stage_v1.py").read_text(),"full_prediction_barrier_v1")
check("validator.validate_matrix" in ast.unparse(needfn),"Full matrix delegates exact original observed-only validator")
check("score_outer"not in ast.unparse(needfn)and "resolve_target"not in ast.unparse(needfn),"No scoring or target resolver in barrier")
for p in D.glob("*.py"):compile(p.read_bytes(),str(p),"exec");check(True,"Compiles "+p.name)
for name,h in pins.items():check(reference(D/name)["sha256"]==h,"Candidate seal unchanged after review "+name)
result=dict(schema="independent_H059_saved000_continuation_source_checks_v1",source_only=True,independent=True,checks=checks,negative_cases=negatives,
 positive_count=len(checks),negative_count=len(negatives),inspected_author_rerun=dict(positive_count=rerun["positive_checks"],negative_count=rerun["negative_fixtures"],actual_entry_joins=8,later_dispatch_joins=64),
 actual_operations=0,actual_guard_imports=False,process_GPU_resource_or_payload_inspection=False,numerical_execution=False)
OUT.mkdir(exist_ok=True)
(OUT/"independent_checks.v1.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items()if k not in("checks","negative_cases")},sort_keys=True))
