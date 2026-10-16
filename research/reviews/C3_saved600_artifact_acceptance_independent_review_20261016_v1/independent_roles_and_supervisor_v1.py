"""Reconstruct exact current guard roles from pure source binding ASTs only."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');SRC=ROOT/'research/reviews/C3_saved600_artifact_acceptance_source_20261016_v1'
OUT=ROOT/'research/reviews/C3_saved600_artifact_acceptance_independent_review_20261016_v1'
sys.path.insert(0,str(SRC))
import acceptance_contract_v1 as c
def ref(path):
 p=ROOT/path
 assert p.suffix in ('.py','.json','.sh') and p.stat().st_size<=32*2**20
 return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def authenticate(r):assert ref(r['path'])==r
def constants(path):
 result={}
 for n in ast.parse((ROOT/path).read_text()).body:
  if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
   try:result[n.targets[0].id]=ast.literal_eval(n.value)
   except (ValueError,TypeError):pass
 return result
roles={name:ref(path) for name,path in dict(
 implementation='scripts/project_budget_v9.py',launcher='scripts/launch_reserved_gpu_v6.py',
 cpu_overlay='src/arggnn/project_storage_cpu_closed_scopes_v1.py',
 terminal_overlay='src/arggnn/project_storage_terminal_scopes_v1.py',
 terminal_success_overlay='src/arggnn/project_storage_terminal_success_scopes_v1.py',
 terminal_retired_overlay='src/arggnn/project_storage_terminal_retired_scopes_v1.py',
 H039_failed_scope_overlay='research/reviews/H039_failed_scope_capacity_v2/failed_scope_capacity_v2.py').items()}
stages=[
 ('research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py','research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json'),
 ('research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py','research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json'),
 ('research/reviews/prospective980_H020_legacy_source_20261016_v3/prospective_980gb_H020_v3.py','research/reviews/prospective980_H020_root_activation_20261016_v3/activation_policy.root_active_v1.json'),
 ('research/reviews/prospective980_Census_fit2_retention_source_20261016_v1/prospective_980gb_Census_fit2_v1.py','research/reviews/prospective980_Census_fit2_root_activation_20261016_v1/activation_policy.root_active_v1.json'),
 (c.GUARD_REF['path'],c.POLICY_REF['path'])]
counts=[];historical_authenticated=[]
for source,policy in stages:
 s=(ROOT/source).read_text();tree=ast.parse(s);p=json.loads((ROOT/policy).read_text())
 ns=dict(constants(source),copy=copy,_ref=ref,_POLICY_REF=ref(policy),_REPAIR_AUTHORITY_REF=c.REPAIR_REF,
  _raw=lambda r:authenticate(r),require=c.require)
 for var,key in dict(BASE_REF='base_ref',BASE_POLICY_REF='base_policy_ref',REPAIR_REF='repair_ref',CAPACITY_V1_REF='capacity_v1_ref',MARKER_REF='marker_ref',AUDIT_REF='evidence_audit_ref',RENEWAL_REF='renewal_ref',RENEWAL_CERT_REF='renewal_certificate_ref').items():
  if key in p:ns[var]=p[key]
 ns['SOURCE']=source
 if 'source_manifest_ref' in p:ns['MANIFEST']=p['source_manifest_ref']['path']
 if 'measurement_source_ref' in p:ns['MEASURE_SOURCE']=p['measurement_source_ref']['path']
 if source==c.GUARD_REF['path']:
  ns['_native_manifest']=lambda:json.loads((ROOT/ns['MANIFEST']).read_text())
  def raw(r):authenticate(r);historical_authenticated.append(r)
  ns['_raw']=raw
  nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_bindings','consumer_bindings_v1')]
 else:nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_bindings']
 exec(compile(ast.Module(body=nodes,type_ignores=[]),source+':pure-source-only-bindings','exec'),ns)
 roles=ns['_bindings'](roles,p);counts.append(dict(source=source,roles=len(roles),unique_refs=len({r['path'] for r in roles.values()})))
manifest=c.manifest_v1();assert roles==manifest['current_guard_roles']
assert len(roles)==37 and len({r['path'] for r in roles.values()})==30
checks=[dict(name='actual_ancestor_binding_function_AST_chain_equals_all37_declared_roles30refs',status='PASS_SOURCE_CHECK')]
for phase in c.PHASES:
 req=dict(phase=phase,source_review_ref=dict(path='FUTURE_actual_review.json',sha256='0'*64),
  training_evidence_ref=dict(path='FUTURE_actual_acceptance.json',sha256='1'*64),
  caps_ref=c.ref(SRC/('caps_'+phase+'.INACTIVE.v1.json')))
 spec=dict(refs=dict(config=dict(path='FUTURE_actual_config.json',sha256='2'*64)),root_stage_request_ref=dict(path='FUTURE_actual_request.json',sha256='3'*64))
 direct=c.required_direct_refs_v1(spec,req)
 assert len(direct)==49 and all(r in direct for r in roles.values())
 checks.append(dict(name='complete_union_'+phase,status='PASS_SOURCE_CHECK',unique_refs=len(direct)))
# No hypothetical future role may be dropped to make the ceiling fit.
original=c.manifest_v1
try:
 expanded=copy.deepcopy(manifest)
 for i in range(20):expanded['current_guard_roles']['SYNTHETIC_EXTRA_'+str(i)]=dict(path='SYNTHETIC_EXTRA_'+str(i)+'.py',sha256='0'*64)
 c.manifest_v1=lambda:expanded
 try:c.required_direct_refs_v1(spec,req)
 except ValueError:pass
 else:raise AssertionError('Original64 direct ceiling raised')
finally:c.manifest_v1=original
checks.append(dict(name='complete_union_over64_blocks_without_dropping_roles',status='PASS_REJECTED'))
# Real supervisor binding is source compilation only; launch is never invoked.
sp=importlib.util.spec_from_file_location('_independent_saved600_ROOT_driver',SRC/'root_driver_v1.py')
driver=importlib.util.module_from_spec(sp);sp.loader.exec_module(driver)
launch=driver.bind_reviewed_supervisor_v1()
assert launch.__globals__['c'] is c and c.HERE==str(SRC.relative_to(ROOT))
assert launch.__code__.co_filename==str(ROOT/c.BASE_DRIVER_REF['path'])
assert launch.__globals__['STATUSES'][c.ACCEPT_PHASE]=='completed_C3_saved600_artifact_validation'
checks.append(dict(name='actual_exact_supervisor_binds_new_contract_runtime_and_phase',status='PASS_SOURCE_ONLY'))
assert not any(n in sys.modules for n in ('torch','numpy','torch_geometric','composer'))
result=dict(schema='C3_saved600_independent_current_guard_roles_and_supervisor_v1',status='PASS_SOURCE_ONLY',checks=checks,
 ancestor_chain_counts=counts,exact_current_roles=roles,historical_roles_metadata_authenticated=len(historical_authenticated),
 current_guard_operation_or_resource_probe_or_launch_performed=False)
(OUT/'independent_roles_and_supervisor.v1.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status='PASS',checks=len(checks),roles=len(roles),unique=30)))
