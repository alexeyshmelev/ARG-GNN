import datetime,hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');D=ROOT/'research/reviews/native_argformer_tsinfer_fit_v5_independent'
sys.dont_write_bytecode=True
def ref(p):return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
r=json.loads((D/'review_receipt.json').read_text());s=json.loads((D/'independent_review_schema_template.json').read_text());checks=[]
def check(n,v):
 if not v:raise AssertionError(n)
 checks.append(n)
for k in ('kind','review_role','implementation','config'):check('schema_'+k,r[k]==s[k])
check('final_source_PASS',r['verdict']=='PASS' and r['template_not_review'] is False)
for k in ('actual_model_fit_executed','actual_scientific_arrays_opened','actual_GPU_inventory_context_or_live_ledger_read','actual_native_runtime_inventory_captured','authority_activation_performed'):check('source_only_'+k,r[k] is False)
for record in r['files']:check('evidence_'+record['path'],ref(ROOT/record['path'])==record)
for k in ('implementation','config','author_source_receipt','preserved_v4_REJECT'):check('pin_'+k,ref(ROOT/r[k]['path'])==r[k])
h=json.loads((D/'handmade_checks.json').read_text());check('fresh146_delta_checks',h['check_count']==146 and not h['issues'])
spec=importlib.util.spec_from_file_location('_v5_verdict_validator',ROOT/r['implementation']['path']);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);c=m.policy()
check('all30_ASTs_bound',m.ast_proof(c)==h['details']['scientific_ASTs'])
for p in c['parent_ids']:
 for stage in m.STAGES:
  a=m.admission_bindings(c,p,stage,'review_validator')
  try:m.validate_admission(c,p,stage,'review_validator',a,r,{}, {})
  except ValueError as e:check('PASS_does_not_supply_runtime_'+p+'_'+stage,str(e)=='exact fresh isolated runtime authority required')
  else:raise AssertionError('source PASS must not supply runtime')
  pending=dict(r,verdict='PENDING')
  try:m.validate_admission(c,p,stage,'review_validator',a,pending,{}, {})
  except ValueError as e:check('PENDING_denies_'+p+'_'+stage,str(e)=='fresh independent source/config PASS required')
  else:raise AssertionError('pending must deny')
result={'kind':'native_argformer_tsinfer_fit_v5_independent_review_validator_receipt','status':'passed','check_count':len(checks),'checks':checks,'review_receipt':ref(D/'review_receipt.json'),'validator':ref(Path(__file__)),'executor_receipt_contract_validated':True,'source_PASS_does_not_supply_runtime_or_execution_authority':True,'actual_execution_or_authority_activation':False,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (D/'validator_receipt.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2)
print(json.dumps({'status':'passed','check_count':len(checks),'review_receipt':result['review_receipt']},indent=2))
