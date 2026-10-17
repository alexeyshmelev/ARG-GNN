"""Concrete join review: bounded source/immutable metadata and pure AST only.
No guard/source module imports, clocks filled, current state, process or resource
probes, journal operations, numerical payloads, or operation callbacks execute.
"""
from pathlib import Path
import ast
import copy
import hashlib
import json
import types

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/reviews/Census_numerical_concrete_whole_join_independent_20261017_v1'
V33='research/reviews/research_execution_ROOT_20261017_v33/'
PINNED={
 'proposal':{'path':V33+'Census_numerical_concrete_whole_join_PROPOSED_INACTIVE.root_v1.json','sha256':'7cc6e1a1fcbea2fdc153ff16524a27353572bddf90cb3586f7732be687bf420b'},
 'staging':{'path':V33+'Census_numerical_actual_fixed_source_staging_ready.root_v1.json','sha256':'4bef8f62aebc92926480307c8c3df5e79ab1a9aa783c2bf09b4d8dbf10fe775e'},
 'source_review':{'path':'research/reviews/Census_numerical_inline_independent_20261017_v1/review.v1.json','sha256':'5be56849b4005c581f7f47d20b5e0d6b4381df9b9bf9da64b63a692cef121d43'},
}
checks=[]
def check(ok,message):
 if not ok:raise AssertionError(message)
 checks.append(message)
def raw(record):
 check(type(record)is dict and set(record)=={'path','sha256'},'Exact ref shape: '+str(record.get('path')))
 p=ROOT/record['path']
 check(not Path(record['path']).is_absolute() and '..'not in Path(record['path']).parts,'Contained supplied immutable source/metadata ref')
 data=p.read_bytes()
 check(len(data)<=32*2**20 and hashlib.sha256(data).hexdigest()==record['sha256'],'Authentic bounded source/metadata SHA: '+record['path'])
 return data
def document(record):return json.loads(raw(record))
def ref(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
proposal=document(PINNED['proposal']);staging=document(PINNED['staging']);source_review=document(PINNED['source_review'])
check(source_review['independent']is True and source_review['verdict']=='SOURCE_PASS' and source_review['blockers']==[], 'Authentic independent complete numerical source review passes')
for flag in ('actual_callback_complete_source_review','orchestration_complete_source_review','complete_transitive_runtime_and_control_path_source_review'):
 check(source_review[flag]is True,'Passed closure coverage: '+flag)
for value in source_review['evidence_refs'].values():raw(value)
for field in ('generic_helper_independent_review_ref','inspect_predecessor_independent_review_ref'):
 prior=document(source_review[field]);check(prior['independent']is True and prior['verdict']=='SOURCE_PASS' and prior['blockers']==[], 'Authenticated inherited pass: '+field)
check(staging['proposed_whole_join_ref']==PINNED['proposal'] and staging['actual_source_review_ref']==PINNED['source_review'],'ROOT staging binds exact proposal and genuine source review')
check(staging['root_authorizer']=='root_agent' and staging['all_ROOT_accounting_mutations_serialized']is True and staging['no_guard_import_or_retirement_or_credit']is True,'ROOT staging serialization assertion authenticated, not independently probed')
for pair in staging['byte_identical_fixed_copies']:
 check(raw(pair['from'])==raw(pair['to']),'Fixed stage is byte-identical to passed source: '+pair['to']['path'])
check(len(staging['byte_identical_fixed_copies'])==3,'Exactly the three ROOT-reported fixed copies reviewed; no directory census performed')
manifest=document(proposal['source_manifest_ref'])
check(proposal['source_manifest_ref']['sha256']==source_review['integration_manifest_template_ref']['sha256'],'Concrete staged integration manifest equals the passed authored manifest bytes')
check(len(manifest['source_sha256'])==218,'Complete passed 218-entry integration map retained')
source_count=0
for path,sha in manifest['source_sha256'].items():
 check(Path(path).suffix in ('.py','.sh','.json','.md','.txt','.patch') and not path.startswith('.runtime/'),'Map entry is source or bounded immutable metadata, never runtime journal or scientific array')
 data=raw({'path':path,'sha256':sha})
 if Path(path).suffix=='.py':
  compile(ast.parse(data,filename=path),path,'exec',dont_inherit=True,optimize=0)
  source_count+=1
check(source_count==68,'All68 mapped Python sources parse/compile without module import')
for field,manifest_field in (('callback_source_ref','callback_source_ref'),('orchestration_source_ref','orchestration_source_ref'),('guard_ref','guard_ref'),('source_ref','helper_ref'),('selector_ref','selector_ref'),('policy_ref','policy_ref')):
 check(proposal[field]==manifest[manifest_field],'Concrete proposal agrees with manifest: '+field)
 check(manifest['source_sha256'][proposal[field]['path']]==proposal[field]['sha256'],'Concrete source/control pin retained in map: '+field)
for field in ('callback_source_ref','orchestration_source_ref'):
 check(proposal[field]['sha256']==source_review[field]['sha256'],'Concrete staged source byte-equivalent to independently passed numerical source: '+field)
for field in ('memo_ref','bridge_ref','permanent_revocation_ref'):raw(proposal[field])
certificate=document(staging['latest_prefix_and_roles_ref'])
check(certificate['root_authorizer']=='root_agent' and certificate['exact17original_chain_links_authenticated']is True and certificate['no_pending_ROOT_accounting_mutations']is True and certificate['ROOT_holds_further_accounting_until_numerical_concrete_review_and_operation']is True,'Genuine ROOT immutable prefix/hold certificate authenticated; no new current state claim')
check(proposal['expected_before']==certificate['expected_before'],'Concrete expected_before exactly equals ROOT captured prefix')
check(proposal['expected_guard_roles']==certificate['expected_guard_roles'] and len(proposal['expected_guard_roles'])==47,'Concrete all47 roles exactly equal ROOT captured map')
check(proposal['expected_before']['completed_overlay_prefix']==dict(events=17,tail_sha256='d5398d93ac6eaa9e61bc5fa047ea2b19f0a2b0b86b18faa1adcb6a5027dc1c63'),'Exact original17-event child prefix fixed, no selection or fallback')
receipt=document(certificate['prefix_source_receipt_ref'])
check(receipt['initial_ledger_pins']==proposal['expected_before'],'Genuine current reserve receipt independently agrees with the captured complete prefix')
check(receipt['projected_bytes']==certificate['projected_commitment_bytes']==933188579176,'Captured receipt projection agrees; no physical-usage or new projection inference')
for record in proposal['expected_guard_roles'].values():
 raw(record)
 check(record in receipt['value']['consumer_bindings'],'Captured guard role also retained in genuine reserve receipt bindings')
revocation=document(proposal['permanent_revocation_ref'])
check(revocation['key']==proposal['key']==manifest['key']=='census_tsinfer_observation_v1/census-tsinfer-numerical-v1','Only exact numerical key throughout')
check(revocation['reservation_sha256']==source_review['original_reservation_sha256']=='e79a7c7b3806576efda20f606eb16fd0dbfd8f5fd4e7d14511cfe2a5785c16e5','Exact genuine original numerical reservation remains bound')
check(revocation['original_expiry_unchanged']==source_review['original_expiry_unchanged']=='2026-10-11T11:40:36.315986+00:00','Historical original expiry is preserved, never reset')
check(revocation['unknown_historical_descendants_absent_claimed']is False and revocation['ROOT_accepts_cooperative_revocation_and_residual_unknown_historical_external_NFS_writers']is True,'Revocation preserves residual historical uncertainty and claims no absence')
check(revocation['all_original_launch_resume_retry_write_aliases_for_exact_key_permanently_cooperatively_revoked']is True,'All exact old numerical writer routes permanently cooperatively revoked')
callback_tree=ast.parse(raw(proposal['callback_source_ref']))
orchestration_tree=ast.parse(raw(proposal['orchestration_source_ref']))
helper_tree=ast.parse(raw(proposal['source_ref']))
def constants(tree):
 out={}
 def literal(node):
  if isinstance(node,ast.Constant):return node.value
  if isinstance(node,ast.Name)and node.id in out:return copy.deepcopy(out[node.id])
  if isinstance(node,ast.BinOp)and isinstance(node.op,ast.Add):return literal(node.left)+literal(node.right)
  if isinstance(node,ast.Dict):return {literal(k):literal(v)for k,v in zip(node.keys,node.values)}
  if isinstance(node,ast.Tuple):return tuple(literal(v)for v in node.elts)
  if isinstance(node,ast.List):return [literal(v)for v in node.elts]
  raise ValueError('Not a literal source constant')
 for node in tree.body:
  if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
   try:out[node.targets[0].id]=literal(node.value)
   except (ValueError,TypeError):pass
 return out
oc=constants(orchestration_tree)
check(proposal['fresh_metadata_paths']==oc['FRESH'] and proposal['outcome_metadata_paths']==oc['NAMES'] and proposal['actual_intent_path']==oc['INTENT'],'All three future helper names, four outcomes and intent path equal exact staged source')
check(oc['SOURCE']==proposal['orchestration_source_ref']['path'] and oc['CALLBACK']==proposal['callback_source_ref']['path'] and oc['MANIFEST']==proposal['source_manifest_ref']['path'],'Exact staged source/callback/manifest aliases, no alternate namespace')
check(oc['KEY']==proposal['key'] and oc['GUARD']==proposal['guard_ref'] and oc['HELPER']==proposal['source_ref'] and oc['SELECTOR']==proposal['selector_ref'] and oc['POLICY']==proposal['policy_ref'] and oc['REVOCATION']==proposal['permanent_revocation_ref'],'All literal orchestration control pins equal concrete proposal')
for path in [*proposal['fresh_metadata_paths'].values(),*proposal['outcome_metadata_paths'].values(),proposal['actual_intent_path'],oc['INSTRUCTION'],oc['REVIEW']]:
 check(str(Path(path).parent)==oc['DIRECTORY'],'Every exact current/future control/outcome is in sole fixed ROOT numerical staging')
check(len(set(proposal['fresh_metadata_paths'].values()))==3 and len(set(proposal['outcome_metadata_paths'].values()))==4,'Fresh helper and outcome names are distinct')
# Cardinality is exact for these distinct fixed paths, without fabricating any future SHA.
known=list(proposal['expected_guard_roles'].values())+[proposal['source_manifest_ref'],proposal['source_ref'],proposal['memo_ref'],proposal['bridge_ref'],proposal['selector_ref'],proposal['policy_ref'],proposal['callback_source_ref'],proposal['permanent_revocation_ref']]
template_path='research/reviews/Census_numerical_inline_ROOT_orchestration_source_20261017_v1/ROOT_instruction.INACTIVE.template.v1.json'
template=json.loads((ROOT/template_path).read_bytes())
known.append(template['memo_review_ref']);raw(template['memo_review_ref'])
by_path={}
for r in known:
 check(r['path']not in by_path or by_path[r['path']]==r['sha256'],'No conflicting known digest for a bound path')
 by_path[r['path']]=r['sha256']
unknown_paths={oc['INSTRUCTION'],oc['REVIEW']}
check(not unknown_paths.intersection(by_path),'Only instruction and whole-review digests are future ROOT controls in prewrite union')
union_paths=set(by_path)|unknown_paths
fresh_paths=set(proposal['fresh_metadata_paths'].values())
check(not union_paths.intersection(fresh_paths) and len(union_paths)+len(fresh_paths)<=64,'Exact existing-ref union plus all3 future records remains within original64 without fabricated hashes')
# Execute only the exact original pure boolean review checks, with RAM documents.
review=copy.deepcopy(proposal)
review.update(independent=True,verdict='SOURCE_PASS',blockers=[],actual_callback_complete_source_review=True,orchestration_complete_source_review=True)
check(set(review)==set(proposal),'Passing whole review retains the exact proposed original schema/key set')
def need(ok,message):
 if not ok:raise ValueError(message)
def statement(tree,message):
 matches=[n for n in ast.walk(tree) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='need' and n.value.args and isinstance(n.value.args[-1],ast.Constant) and n.value.args[-1].value==message]
 check(len(matches)==1,'One exact original pure review predicate: '+message)
 return matches[0]
helper_messages=['Genuine new whole-join review includes actual ROOT callback; source-template review never suffices','Whole review binds exact target and expected state/47roles']
helper_statements=[statement(helper_tree,m) for m in helper_messages]
orchestration_statement=statement(orchestration_tree,'Independent whole review includes actual orchestration and immutable names')
self=types.SimpleNamespace(helper_ref=review['source_ref'],manifest_ref=review['source_manifest_ref'],callback_ref=review['callback_source_ref'],callback_name=review['callback_name'],paths=review['fresh_metadata_paths'],revocation_ref=review['permanent_revocation_ref'],key=review['key'])
namespace=dict(need=need,self=self,review=review,GUARD_REF=review['guard_ref'],MEMO_REF=review['memo_ref'],SELECTOR_REF=review['selector_ref'],POLICY_REF=review['policy_ref'],BRIDGE_REF=review['bridge_ref'],instruction={'expected_before':proposal['expected_before'],'expected_guard_roles':proposal['expected_guard_roles']},source_ref=review['orchestration_source_ref'],NAMES=oc['NAMES'],INTENT=oc['INTENT'])
code=compile(ast.fix_missing_locations(ast.Module(body=helper_statements+[orchestration_statement],type_ignores=[])),'exact-original-pure-review-predicates','exec')
exec(code,namespace)
check(True,'Concrete passing record satisfies every exact original helper/orchestrator pure review predicate')
negative_count=0
for field,value in [('independent',False),('verdict','UNREVIEWED_INACTIVE'),('blockers',['blocking']),('actual_callback_complete_source_review',False),('orchestration_complete_source_review',False),('orchestration_source_ref',{}),('outcome_metadata_paths',{}),('actual_intent_path','wrong')]:
 bad=copy.deepcopy(review);bad[field]=value;bad_scope=dict(namespace,review=bad)
 try:exec(code,bad_scope)
 except ValueError:negative_count+=1
 else:raise AssertionError('Original predicates failed to reject mutation: '+field)
check(negative_count==8,'All8 negative whole-review mutations rejected by original predicates')
controls=next(n for n in orchestration_tree.body if isinstance(n,ast.FunctionDef)and n.name=='controls')
intent_expected=next(n.value for n in controls.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='expected'for t in n.targets))
check(isinstance(intent_expected,ast.Call) and isinstance(intent_expected.func,ast.Name)and intent_expected.func.id=='dict','Original intent equality uses complete exact dictionary')
intent_fields={n.arg:ast.unparse(n.value)for n in intent_expected.keywords}
check(intent_fields['instruction_path']=='INSTRUCTION' and 'instruction_ref'not in intent_fields,'Intent binds instruction PATH only, so review→intent→instruction digest graph is finite')
check(intent_fields['created_at_utc']=="instruction['created_at_utc']" and intent_fields['expires_at_utc']=="instruction['expires_at_utc']",'Original controls require identical ROOT-created finite interval in instruction and intent')
control_text=ast.unparse(controls)
check("stamp(instruction['created_at_utc']) <= now < stamp(instruction['expires_at_utc'])"in control_text,'Original orchestration requires timezone-aware nonfuture created and unexpired expires')
check("intent_ref in instruction['uncached_current_refs']"in control_text,'Actual intent digest must be an uncached current ref')
callback_text=ast.unparse(callback_tree)
check("authority.update(active=True, expires_at_utc=instruction['expires_at_utc'])"in callback_text,'Actual original-schema authority copies ROOT expiry unchanged')
helper_text=ast.unparse(helper_tree)
for fragment in ["created <= datetime.now(timezone.utc)","self.guard._expire(instruction)","records = [self.manifest_ref, self.review_ref, self.memo_review_ref, self.instruction_ref, self.callback_ref, self.revocation_ref, *self.current_refs]","list(self.paths.values())","tx.finish()","tx.close()"]:
 check(fragment in helper_text,'Exact previously passed finite/uncached/final-control invariant retained: '+fragment)
check(template['created_at_utc']is None and template['expires_at_utc']is None and template['active']is False,'No inactive template clock or authority relabelled as actual')
check(template['uncached_current_refs'][0]['path']==oc['INTENT'] and template['uncached_current_refs'][0]['sha256']is None,'ROOT must fill genuine later intent digest; reviewer fabricates none')
result=dict(schema='Census_numerical_concrete_whole_join_source_checks_v1',passed=True,checks=len(checks),negative_RAM_cases=negative_count,
 source_or_immutable_metadata_only=True,guard_modules_imported=0,callbacks_or_operations_executed=0,current_state_or_process_or_resource_probes=0,journal_mutations=0,
 map_entries_authenticated=218,mapped_python_sources_compiled_without_import=source_count,exact47roles_authenticated=True,
 prewrite_known_unique_refs=len(by_path),prewrite_future_control_names=len(unknown_paths),prewrite_existing_union=len(union_paths),future_helper_names=len(fresh_paths),prewrite_final_union=len(union_paths)+len(fresh_paths),original_direct_binding_limit=64,
 actual_clocks_supplied=False,actual_absence_or_retained_measurement_or_credit_established=False,
 pinned_inputs=PINNED,checks_log=checks)
(HERE/'checks.result.v1.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k!='checks_log'},sort_keys=True,indent=2))
