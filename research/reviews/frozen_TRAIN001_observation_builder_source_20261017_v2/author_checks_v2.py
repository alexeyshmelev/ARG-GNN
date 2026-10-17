"""Author source/AST/RAM checks; no actual predecessor, capacity, guard/runtime or new request."""
from pathlib import Path
from types import FunctionType,CodeType
from datetime import datetime,timezone,timedelta
from contextlib import redirect_stdout
import ast,argparse,copy,hashlib,io,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");HERE="research/reviews/frozen_TRAIN001_observation_builder_source_20261017_v2";OLD=HERE[:-1]+"1"
OBSERVE="research/reviews/frozen_TRAIN001_observe_ready_20261017_v2";PREVOBS=OBSERVE[:-1]+"1"
p=HERE+"/build_ROOT_request_v1.py";s=(ROOT/p).read_text();old=(ROOT/OLD/"build_ROOT_request_v1.py").read_text()
checks=[]
def need(ok,msg="RAM"):
 if not ok:raise ValueError(msg)
def yes(name,ok=True):need(ok,name);checks.append(name)
def ref(p):return dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
norm=s.replace(HERE,OLD).replace(OBSERVE,PREVOBS).replace("08fe20361a2294ecc00afa8c36d780cd620faf75562685b664102ae33d6c09c2","d672b38e5d317ad9844104e1e0f741c6e338ac0f33b39e7d0f897f124a01a912").replace("4ff3ac5b8e18faca841fd235e4203ca7f7569096169906e7eafa5c435ebb9530","ff208ebdf593811aacd876257aaa5c329d904b18522fec97bd306b5f58bea1a5").replace("research/reviews/frozen_TRAIN001_observation_independent_20261017_v2/review.observation.v2.json","research/reviews/frozen_TRAIN001_observation_independent_20261017_v1/review.observation.v1.json").replace("a15d5fe5d3aee474fbb11a7183525a6ef1b23bc8b36421d6ae31c0daa0f2f577","a39a3cfe7403d0c6e6809490ee9355d3f1fb57efdf8b8ed7e02542db88f31b5d").replace("outer_terminal.root_actual_v2.json","outer_terminal.root_actual_v1.json")
yes("entire_builder_AST_only_namespace_pins_and_outer_filename",ast.dump(ast.parse(norm),include_attributes=False)==ast.dump(ast.parse(old),include_attributes=False))
code=compile(s,p,"exec")
def fn(name,env):
 n=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and n.name==name)
 return FunctionType(next(c for c in code.co_consts if isinstance(c,CodeType)and c.co_name==name),env,name,tuple(ast.literal_eval(x)for x in n.args.defaults))
actual_input=dict(path="research/reviews/frozen_benchmark_training_ready_20261017_v1/train001_simulate_v5/ROOT_actual_TRAIN001_root02_oct17_v5/input.json",sha256="91743348259ade65ccf83017ece27a01ba61df0147c8f5ee192e9f05b289c491")
env=dict(ROOT=ROOT,HERE=HERE,OBSERVE=OBSERVE,SOURCE=ref(OBSERVE+"/worker_entry_v1.py"),MANIFEST=ref(OBSERVE+"/source_manifest.v1.json"),
 REVIEW=ref("research/reviews/frozen_TRAIN001_observation_independent_20261017_v2/review.observation.v2.json"),ACTUAL_SIM_INPUT=actual_input,
 Path=Path,need=need,datetime=datetime,timedelta=timedelta)
rows={};lookups=[];reads=[]
def rr(p):lookups.append(str(p));return dict(path=str(p),sha256="0"*64)
inp=dict(parent_id="american_c3_train_001",phase="simulate_train_001",stage="simulate",output_directory="experiments/results/RAM_ONLY_SIM",
 worker_auxiliary_scope="research/reviews/RAM_ONLY/worker_aux",refs=dict(config=rr("research/reviews/RAM_ONLY/config.json")))
obs=rr(inp["output_directory"]+"/observed/manifest.json");roster=rr(inp["output_directory"]+"/observed/roster.json")
rows[actual_input["path"]]=inp;rows[inp["output_directory"]+"/completed.json"]=dict(observation_ref=obs);rows[obs["path"]]=dict(roster_ref=roster)
def read(r):reads.append(r["path"]);return copy.deepcopy(rows[r["path"]])
env.update(read=read,ref=rr)
prior=fn("actual_prior",env)()
yes("actual_prior_selects_only_additive_outer2",prior["outer_ref"]["path"]==str(Path(actual_input["path"]).parent/"outer_terminal.root_actual_v2.json")and not any(x.endswith("outer_terminal.root_actual_v1.json")for x in lookups))
yes("actual_prior_only_compact_RAMmetadata_reads",reads==[actual_input["path"],inp["output_directory"]+"/completed.json",obs["path"]])
yes("actual_prior_all_seven_roles",set(prior)=={"input_ref","config_ref","completion_ref","wait_ref","outer_ref","observation_ref","roster_ref"})
def missing_outer(p):
 if str(p).endswith("outer_terminal.root_actual_v2.json"):raise FileNotFoundError("RAM missing additive outer2")
 return rr(p)
env["ref"]=missing_outer
try:fn("actual_prior",env)()
except FileNotFoundError:yes("no_fallback_to_immutable_outer1")
else:raise AssertionError("Unexpected outer fallback")
env["ref"]=rr
m=json.loads((ROOT/OBSERVE/"source_manifest.v1.json").read_text());now=datetime.now(timezone.utc);end=now+timedelta(hours=4)
capacity=dict(root_authorizer="root_agent",actual_current_observation=True,source_only_estimate=False,parent_id="american_c3_train_001",
 created_at_utc=(now-timedelta(seconds=1)).isoformat(),expires_at_utc=end.isoformat(),cpu_ids=list(range(8)),admissible_new_bytes=13000000000,available_owned_RSS_bytes=68719476736)
compose=fn("compose_request",env)
q=compose(capacity=capacity,scope_id="RAM_ONLY",prior=prior,manifest=m,expires_at=end,now=now)
yes("new_request_exact_v2_source_manifest_review_namespace",q["source_manifest_ref"]==env["MANIFEST"]and q["source_review_ref"]==env["REVIEW"]and q["packet_directory"].startswith(OBSERVE+"/ROOT_actual_TRAIN001_"))
yes("new_request_original_science_13GBwhole_12point5GBprimary",q["caps"]["reservation_bytes"]==13000000000 and q["scientific_config"]==dict(m["condition"]["inference_template_inactive"],execution_enabled=True,observation_ref=prior["observation_ref"])and q["scientific_config"]["output_cap_bytes"]==12500000000)
yes("new_request_all7prior_actual_inline_closure_and_no_truth_training",q["prior"]==prior and q["bindings"]["simulation_prerequisites"]==prior and q["source_sha256"]=={r["path"]:r["sha256"]for r in m["refs"]}and q["query_truth_access_authorized"]is False and q["GNN_optimization_authorized"]is False and q["TEST_access_authorized"]is False)
for name,mut in[
 ("capacity",lambda c:c.update(admissible_new_bytes=1)),("RSS",lambda c:c.update(available_owned_RSS_bytes=1)),("duplicateCPU",lambda c:c.update(cpu_ids=[0]*8)),
 ("estimated",lambda c:c.update(source_only_estimate=True)),("expired",lambda c:c.update(expires_at_utc=(now-timedelta(seconds=1)).isoformat())),
 ("wrongparent",lambda c:c.update(parent_id="american_c3_train_000"))]:
 cap=copy.deepcopy(capacity);mut(cap)
 try:compose(capacity=cap,scope_id="RAM_ONLY",prior=prior,manifest=m,expires_at=end,now=now)
 except ValueError:yes("reject_"+name)
 else:raise AssertionError(name)
env.update(argparse=argparse,json=json,__doc__="RAM",build_only=lambda **kw:need(False,"Unexpected active metadata generation"))
cap=io.StringIO()
with redirect_stdout(cap):yes("inactive_default_returns0",fn("main",env)([])==0)
yes("inactive_no_admit_execute",json.loads(cap.getvalue())==dict(status="SOURCE_ONLY_INACTIVE",admission=False,execution=False))
print(json.dumps(dict(schema="TRAIN001_observation_outer2_builder_author_RAM_checks_v2",status="AUTHOR_PASS",independent=False,count=len(checks),checks=checks,
 builder_ref=ref(p),source_ref=env["SOURCE"],source_manifest_ref=env["MANIFEST"],review_ref=env["REVIEW"],actual_predecessor_or_capacity_or_outer_reads=0,
 actual_guard_imports_runtime_configurations_probes_admissions_launches_closes_request_writes=0),sort_keys=True,indent=2))
