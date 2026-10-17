"""Source AST namespace drift check. Never import or execute project sources."""
import ast,hashlib,json,sys
from pathlib import Path
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");OWN=ROOT/"research/reviews/admission_repairs_independent_20261017_v1";base=ROOT/sys.argv[1];tag=sys.argv[2]
def ref(p):return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
relative=str(base.relative_to(ROOT));driver=base/"root_driver_v1.py";worker=base/"worker_entry_v1.py"
d=ast.parse(driver.read_text());w=ast.parse(worker.read_text())
here=next(ast.literal_eval(n.value)for n in d.body if isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id=="HERE"for x in n.targets))
imports=[n.module for n in w.body if isinstance(n,ast.ImportFrom)and any(a.name=="root_driver_v1"and a.asname=="d"for a in n.names)]
assert here==relative
expected=relative.replace("/",".");matches=imports==[expected]
blockers=[]if matches else[dict(priority=1,consumers=["H060","H061"],title="Successor worker imports predecessor driver",file=str(worker.relative_to(ROOT)),line=3,actual_imports=imports,expected_import=expected,impact="New admitted driver identity is rejected by predecessor consumer_for_admitted before original worker. No scientific run is permitted by SOURCE review.")]
result=dict(schema="independent_child_role_worker_namespace_checks_v1",candidate_source_ref=ref(driver),candidate_worker_ref=ref(worker),driver_HERE=here,worker_driver_imports=imports,expected_worker_driver_import=expected,same_successor_namespace=matches,blockers=blockers,source_only=True,project_imports_or_source_functions_executed=False)
p=OWN/("child_role_namespace."+tag+".json");assert not p.exists();p.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n");print(json.dumps(dict(result_ref=ref(p),blockers=blockers)))
