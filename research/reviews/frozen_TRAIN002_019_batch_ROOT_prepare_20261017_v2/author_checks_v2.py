"""Author source/AST/RAM checks only. No actual capacity, guard imports or generated ROOT authority."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import FunctionType,CodeType
from contextlib import redirect_stdout
import ast,argparse,copy,hashlib,io,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/frozen_TRAIN002_019_batch_ROOT_prepare_20261017_v2"
OLD="research/reviews/frozen_TRAIN002_019_batch_ROOT_prepare_20261017_v1"
BATCH="research/reviews/frozen_TRAIN002_019_batch_continuation_source_20261017_v3"
source=(ROOT/HERE/"ROOT_batch_admin_builder_v1.py").read_text();tree=ast.parse(source);code=compile(source,HERE+"/ROOT_batch_admin_builder_v1.py","exec")
def need(ok,message):
 if not ok:raise ValueError(message)
def ref(p):return dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
env=dict(ROOT=ROOT,HERE=HERE,BATCH=BATCH,datetime=datetime,need=need,
 MANIFEST=ref(BATCH+"/source_manifest.v1.json"),
 REVIEW=ref("research/reviews/frozen_TRAIN002_019_batch_independent_20261017_v3/review.v3.json"),
 RECIPE=ref(BATCH+"/ROOT_RECIPE.v1.md"),argparse=argparse,json=json,__doc__="RAM")
env["MODULE"]=BATCH.replace("/",".")+".batch_ROOT_v1"
def fn(name):
 n=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
 f=FunctionType(next(c for c in code.co_consts if isinstance(c,CodeType)and c.co_name==name),env,name,tuple(ast.literal_eval(x)for x in n.args.defaults))
 return f
checks=[]
def yes(n,ok=True):need(ok,n);checks.append(n)
oldtree=ast.parse((ROOT/OLD/"ROOT_batch_admin_builder_v1.py").read_text())
for name in("authority_value","build_once","argv","main","local","ref","read","save"):
 a=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef)and n.name==name);b=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
 yes("unchanged_builder_function_"+name,ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False))
now=datetime.now(timezone.utc);template=json.loads((ROOT/BATCH/"ROOT_batch_authority.template.inactive.v1.json").read_text())
p=dict(schema="ROOT_TRAIN002_019_batch_actual_parameters_v1",root_authorizer="root_agent",once_authorized=True,scope_id="RAM_NOT_ACTUAL",
 CPU_lanes=[list(range(8)),list(range(8,16))],expires_at_utc=(now+timedelta(hours=44)).isoformat(),actual_capacity_observed_at_utc=(now-timedelta(seconds=1)).isoformat(),
 admissible_new_bytes=10**11,actual_current_capacity_observed=True,source_only_estimate=False,
 ROOT_exclusive_initial_admin_admission_until_coordinator_registered=True,ROOT_every_manual_mutation_uses_same_coordinator=True,use_corrected_v2_commands=True)
a=fn("authority_value")(p,template,now)
yes("v3_value_exact_manifest_review_all18",a["source_manifest_ref"]==env["MANIFEST"]and a["source_review_ref"]==env["REVIEW"]and a["allowed_parents"]==["american_c3_train_%03d"%i for i in range(2,20)])
yes("v3_run_admission_namespaces",a["run_directory"].startswith(BATCH+"/ROOT_actual_batch_")and a["admission_directory"].startswith(BATCH+"/ROOT_batch_admission_"))
for name,mut in[
 ("CPU",lambda p:p["CPU_lanes"][1].__setitem__(0,0)),("capacity",lambda p:p.update(admissible_new_bytes=1)),
 ("stale",lambda p:p.update(actual_capacity_observed_at_utc=(now-timedelta(minutes=6)).isoformat())),("future",lambda p:p.update(actual_capacity_observed_at_utc=(now+timedelta(seconds=1)).isoformat())),
 ("deadline",lambda p:p.update(expires_at_utc=(now+timedelta(days=3)).isoformat())),
 ("exclusivity",lambda p:p.update(ROOT_exclusive_initial_admin_admission_until_coordinator_registered=False)),
 ("coordinator",lambda p:p.update(ROOT_every_manual_mutation_uses_same_coordinator=False)),("estimate",lambda p:p.update(source_only_estimate=True)),("once",lambda p:p.update(once_authorized=False))]:
 q=copy.deepcopy(p);mut(q)
 try:fn("authority_value")(q,template,now)
 except ValueError:yes("reject_"+name)
 else:raise AssertionError(name)
expected=env["MODULE"];rr=dict(path="RAM",sha256="0"*64)
yes("actual_argv_module_v3",fn("argv")("--admit-batch-ref",rr)[4]==expected and "_v2.batch_ROOT"not in expected)
# Pure source/compact metadata validation: all file reads are source/manifest/review/template/recipe only.
env.update(read=lambda r:json.loads((ROOT/r["path"]).read_text()),ref=ref,local=lambda p:ROOT/p,TEMPLATE=BATCH+"/ROOT_batch_authority.template.inactive.v1.json")
yes("v3_static_packet_exact_source_review_recipe",fn("static_packet")()==template)
env.update(build_once=lambda p:need(False,"Unexpected metadata write"),run_argv=lambda r:need(False,"Unexpected active argv"))
cap=io.StringIO()
with redirect_stdout(cap):yes("inactive_default_return0",fn("main")([])==0)
yes("inactive_no_writes_probes_guard",json.loads(cap.getvalue())==dict(status="SOURCE_ONLY_INACTIVE",resource_probes=False,guard_imported=False,metadata_written=False))
inactive=json.loads((ROOT/HERE/"ROOT_actual_parameters.template.INACTIVE.v1.json").read_text())
yes("actual_template_inactive_null_capacity_CPU_time_scope",inactive["once_authorized"]is False and inactive["actual_current_capacity_observed"]is False and all(inactive[k]is None for k in("scope_id","CPU_lanes","expires_at_utc","actual_capacity_observed_at_utc","admissible_new_bytes")))
print(json.dumps(dict(schema="ROOT_TRAIN002_019_batch_v3_metadata_builder_author_checks_v1",status="AUTHOR_PASS",independent=False,count=len(checks),checks=checks,
 builder_ref=ref(HERE+"/ROOT_batch_admin_builder_v1.py"),review_ref=env["REVIEW"],actual_capacity_or_authority_used=False,
 actual_guard_imports_configurations_resource_probes_admissions_launches_closes=0),sort_keys=True,indent=2))
