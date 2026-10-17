"""Authorized actual inactive entries only; deny active project/scientific imports."""
import sys,runpy,importlib,json,io,contextlib
from pathlib import Path
sys.path.insert(0, '/workspace-SR003.nfs2/ARG-GNN') # Same ROOT search entry as the genuine -m invocation.
BASE='research.reviews.H058_transaction_admission_source_20261017_v1'
allowed={'research','research.reviews',BASE,BASE+'.root_driver_v1',BASE+'.worker_entry_v1'}
forbidden_roots={'arggnn','torch','torch_geometric','numpy','scipy','pandas','numba','tskit'}
assert not any(n.split('.')[0]in forbidden_roots for n in sys.modules)
seen=[]
def audit(event,args):
 if event=='import':
  name=args[0];seen.append(name)
  if name.split('.')[0]in forbidden_roots or(name.startswith('research') and name not in allowed):raise RuntimeError('Forbidden inactive import '+name)
sys.addaudithook(audit)
before=set(sys.modules);records=[]
for entry in ('root_driver_v1','worker_entry_v1'):
 module=BASE+'.'+entry;sys.argv=[module];sink=io.StringIO();code=None
 with contextlib.redirect_stdout(sink):
  try:runpy.run_module(module,run_name='__main__')
  except SystemExit as e:code=e.code
 assert code==0
 rows=[json.loads(x)for x in sink.getvalue().splitlines()];assert rows[0]['status']=='SOURCE_ONLY_INACTIVE' and rows[0]['guard_imported']is False
 records.append(dict(module=module,actual_main_returncode=code,outputs=rows))
d=importlib.import_module(BASE+'.root_driver_v1');w=importlib.import_module(BASE+'.worker_entry_v1')
assert w.d is d
project=sorted(n for n in set(sys.modules)-before if n.startswith(('research','arggnn','_H058','_fresh_ROOT')))
assert set(project)<=allowed
receipt=dict(schema='H058_transaction_integration_independent_inactive_entry_check_v1',interpreter=sys.executable,dont_write_bytecode=sys.dont_write_bytecode,no_user_site=sys.flags.no_user_site,optimization=sys.flags.optimize,actual_noargs_entries=records,worker_relative_driver_identity=True,new_project_modules=project,forbidden_imports=[],no_original_contract_memo_guard_or_numerical_import=True)
assert receipt['dont_write_bytecode'] and receipt['no_user_site'] and receipt['optimization']==0
out=Path('/workspace-SR003.nfs2/ARG-GNN/research/reviews/H058_fresh_CPU_lifetime_independent_review_20261017_v1/transaction_integration_v1/inactive_entry_checks.v1.json')
out.write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
print(json.dumps(receipt,sort_keys=True))
