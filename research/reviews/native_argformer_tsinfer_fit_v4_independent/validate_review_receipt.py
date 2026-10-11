"""Validate the exact v4 independent REJECT evidence and all12 stage denials."""
import datetime,hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');REVIEW=ROOT/'research/reviews/native_argformer_tsinfer_fit_v4_independent'
sys.dont_write_bytecode=True
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
r=json.loads((REVIEW/'review_receipt.json').read_text());s=json.loads((REVIEW/'independent_review_schema_template.json').read_text());checks=[]
def check(name,value):
 if not value:raise AssertionError(name)
 checks.append(name)
for k in ('kind','review_role','implementation','config'):check('schema_'+k,r[k]==s[k])
check('final_REJECT',r['verdict']=='REJECT' and r['template_not_review'] is False)
for k in ('actual_model_fit_executed','actual_scientific_arrays_opened','actual_GPU_inventory_context_or_live_ledger_read','actual_native_runtime_inventory_captured','authority_activation_performed'):check('source_only_'+k,r[k] is False)
for record in r['files']:check('evidence_'+record['path'],sha(ROOT/record['path'])==record['sha256'])
for k in ('implementation','config','author_source_receipt','preserved_v3_REJECT','preserved_v3_handmade_evidence'):
 record=r[k];check('pin_'+k,sha(ROOT/record['path'])==record['sha256'])
h=json.loads((REVIEW/'handmade_checks.json').read_text())
check('fresh179_delta_checks',h['check_count']==179 and len(h['issues'])==1)
check('exact_terminal_blocker',h['issues'][0]['id']=='DANGLING_EXIT_STATUS_NOT_TREATED_AS_EXISTING_TERMINAL')
check('entry_policy_fix_verified',all(h['details']['entry_cases'][case]['ownership_proven'] for case in ('byte_drift','missing_config','injected_policy')))
check('bounded_fresh_native_kernel',h['details']['fresh_handmade_kernel']['exhaustive_pairs']==24 and h['details']['fresh_handmade_kernel']['native_fit_evidence'] is False)
spec=importlib.util.spec_from_file_location('_v4_exact_verdict_validator',ROOT/r['implementation']['path']);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);c=m.policy()
check('all30_scientific_ASTs_bound',m.ast_proof(c)==h['details']['native_scientific_ASTs'])
for p in c['parent_ids']:
 for stage in m.STAGES:
  a=m.admission_bindings(c,p,stage,'review_validator')
  try:m.validate_admission(c,p,stage,'review_validator',a,r,{}, {})
  except ValueError as error:check('REJECT_blocks_'+p+'_'+stage,str(error)=='fresh independent source/config PASS required')
  else:raise AssertionError('REJECT must deny')
result={'kind':'native_argformer_tsinfer_fit_v4_independent_review_validator_receipt','status':'passed','checks':checks,'check_count':len(checks),'review_receipt':ref(REVIEW/'review_receipt.json'),'schema_template':ref(REVIEW/'independent_review_schema_template.json'),'validator':ref(Path(__file__)),'executor_receipt_contract_validated':True,'REJECT_blocks_all12_parent_stages':True,'actual_execution_or_authority_activation':False,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (REVIEW/'validator_receipt.json').open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2)
print(json.dumps({'status':'passed','check_count':len(checks),'review_receipt':result['review_receipt']},indent=2))
