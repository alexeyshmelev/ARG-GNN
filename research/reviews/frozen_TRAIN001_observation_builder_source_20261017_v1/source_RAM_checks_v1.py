"""Builder AST/RAM checks. No active request, actual source import or operations."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
from contextlib import redirect_stdout
import ast,copy,hashlib,io,json,argparse
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
BASE="research/reviews/frozen_TRAIN001_observation_builder_source_20261017_v1"
OBS="research/reviews/frozen_TRAIN001_observe_ready_20261017_v1"
SOURCE=BASE+"/build_ROOT_request_v1.py"
CHECKS=[]
def need(ok,msg="RAM assertion"):
 if not ok:raise ValueError(msg)
def ref(p):
 p=Path(p);p=p if p.is_absolute()else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def extract(path,name,env):
 s=(ROOT/path).read_text();n=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and n.name==name)
 exec(compile(ast.Module(body=[n],type_ignores=[]),path+"#RAM","exec"),env);return env[name]
def reject(name,call):
 try:call()
 except(ValueError,KeyError,TypeError):CHECKS.append(name);return
 raise AssertionError(name)
s=(ROOT/SOURCE).read_text();t=ast.parse(s)
constants={"Path":Path}
for n in t.body:
 if isinstance(n,ast.Assign):
  exec(compile(ast.Module(body=[n],type_ignores=[]),SOURCE+"#static_constants","exec"),constants)
need(ref(constants["ACTUAL_SIM_INPUT"]["path"])==constants["ACTUAL_SIM_INPUT"])
for key in("SOURCE","MANIFEST","REVIEW"):need(ref(constants[key]["path"])==constants[key])
CHECKS.append("all_actual_source_review_and_predecessor_input_pins_authenticated")
# Reuse the prior observed-stage author's RAM metadata fixture, not its tests.
a=OBS+"/author_RAM_checks_v1.py";at=ast.parse((ROOT/a).read_text())
allowed={"require","ref","doc","v5_scope","function","original_refs","fixture"}
nodes=[n for n in at.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign))
 or isinstance(n,ast.FunctionDef)and n.name in allowed]
ae={"__name__":"builder_fixture_data_only"};exec(compile(ast.Module(body=nodes,type_ignores=[]),a+"#RAM_data_only","exec"),ae)
f=ae["fixture"]()
env=dict(constants,datetime=datetime,timezone=timezone,timedelta=timedelta,need=need)
compose=extract(SOURCE,"compose_request",env)
now=datetime.now(timezone.utc);capacity=copy.deepcopy(f.q["actual_capacity"]);expiry=datetime.fromisoformat(capacity["expires_at_utc"])
q=compose(capacity=capacity,scope_id="builder_RAM",prior=f.prior,manifest=f.read(f.rref(OBS+"/source_manifest.v1.json")),expires_at=expiry,now=now)
rr=dict(path="RAM_builder_request.json",sha256="0"*64);f.rows[rr["path"]]=q
need(f.scope["validate_request"](rr)[0]==q)
CHECKS.append("composed_complete_request_accepted_by_actual_reviewed_observation_validator_RAM")
need(set(q["prior"])=={"input_ref","config_ref","completion_ref","wait_ref","outer_ref","observation_ref","roster_ref"})
need(q["scientific_config"]["output_cap_bytes"]==12500000000 and q["caps"]["reservation_bytes"]==13000000000)
need(q["runtime_ref"]==f.condition["inference_runtime_ref"]and q["GPU_authorized"]is False)
CHECKS.append("all7_predecessor_refs_original_science_runtime_and13GBwhole_cap_preserved")
for name,mut in[
 ("reject_authored_capacity_estimate",lambda c:c.update(source_only_estimate=True)),
 ("reject_capacity_too_small",lambda c:c.update(admissible_new_bytes=12999999999)),
 ("reject_RSS_too_small",lambda c:c.update(available_owned_RSS_bytes=1)),
 ("reject_duplicate_CPU",lambda c:c.update(cpu_ids=[0]*8))]:
 c=copy.deepcopy(capacity);mut(c)
 reject(name,lambda:compose(capacity=c,scope_id="RAM",prior=f.prior,manifest=f.read(f.rref(OBS+"/source_manifest.v1.json")),expires_at=expiry,now=now))
reject("reject_expiring_clock",lambda:compose(capacity=capacity,scope_id="RAM",prior=f.prior,manifest=f.read(f.rref(OBS+"/source_manifest.v1.json")),expires_at=now+timedelta(seconds=600),now=now))
reject("reject_scope_escape",lambda:compose(capacity=capacity,scope_id="../else",prior=f.prior,manifest=f.read(f.rref(OBS+"/source_manifest.v1.json")),expires_at=expiry,now=now))
calls=[n for n in ast.walk(t)if isinstance(n,ast.Call)]
need(not any(isinstance(n.func,ast.Attribute)and n.func.attr in("admit","guard","execute","apply_reserve_v1","worker")for n in calls))
need(not any(isinstance(n,ast.Import)and any(x.name in("torch","subprocess")for x in n.names)for n in ast.walk(t)))
CHECKS.append("builder_has_no_guard_admit_process_launch_or_scientific_call")
ce=dict(argparse=argparse,json=json,__doc__="RAM")
stream=io.StringIO()
with redirect_stdout(stream):need(extract(SOURCE,"main",ce)([])==0)
need(json.loads(stream.getvalue())["status"]=="SOURCE_ONLY_INACTIVE")
CHECKS.append("default_CLI_inert")
print(json.dumps(dict(status="AUTHOR_RAM_PASS",checks=CHECKS,count=len(CHECKS),source_ref=ref(SOURCE),
 actual_guard_imports_or_probes_or_active_request_writes_or_admission_or_science=0,
 support_reuse="Prior author RAM metadata fixture constructor only, no author tests or conclusions reused."),sort_keys=True,indent=2))
