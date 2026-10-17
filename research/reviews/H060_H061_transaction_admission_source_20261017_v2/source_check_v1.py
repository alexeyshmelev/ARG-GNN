"""Static AST/pinned metadata and pure handwritten role/admission checks only."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,json,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE='research/reviews/H060_H061_transaction_admission_source_20261017_v2'
PRIOR='research/reviews/H060_H061_transaction_admission_source_20261017_v1'
def need(ok,msg):
 if not ok:raise ValueError(msg)
def ref(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def raw(record):need(ref(record['path'])==record,'exactsourceSHA');return (ROOT/record['path']).read_bytes()
def read(record):return json.loads(raw(record))
def tree(path):return ast.parse((ROOT/path).read_bytes())
def fn(t,n):return next(x for x in t.body if isinstance(x,ast.FunctionDef)and x.name==n)
def iso(nodes,scope):exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),'<pure-source-metadata-fixture>','exec'),scope);return scope
def fake_ref(path):return dict(path=path,sha256=hashlib.sha256(path.encode()).hexdigest())
def same(a,b):return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
def rejects(action):
 try:action()
 except (ValueError,KeyError,TypeError):return True
 return False
def check():
 base=ROOT/HERE;b=json.loads((base/'source_bindings.v1.json').read_text());prior=json.loads((ROOT/PRIOR/'source_bindings.v1.json').read_text())
 manifest=base/'source_manifest.v1.json'
 if manifest.exists():
  for path,sha in json.loads(manifest.read_text())['source_sha256'].items():need(ref(path)['sha256']==sha,'manifest '+path)
 current=tree(HERE+'/root_driver_v1.py');old=tree(PRIOR+'/root_driver_v1.py')
 restored=copy.deepcopy(current)
 restored.body=[n for n in restored.body if not(isinstance(n,ast.FunctionDef)and n.name=='_install_H061_prior_freeze_join')]
 for n in restored.body:
  if isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id=='HERE'for x in n.targets):n.value.value=PRIOR
 contract_node=fn(restored,'contract')
 contract_node.body=[n for n in contract_node.body if not(isinstance(n,ast.If)and len(n.body)==1 and isinstance(n.body[0],ast.Expr)and isinstance(n.body[0].value,ast.Call)and ast.unparse(n.body[0].value.func)=='_install_H061_prior_freeze_join')]
 validator=next(n for n in contract_node.body if isinstance(n,ast.FunctionDef)and n.name=='validate_request')
 validator.body=[n for n in validator.body if not(isinstance(n,ast.If)and ast.unparse(n.test)=="consumer == 'H061'")]
 need(same(restored,old),'onlynamespaceandexplicitpriorfreezejoin')
 oldb=copy.deepcopy(prior);newb=copy.deepcopy(b)
 for value in(oldb,newb):
  for key in('schema','transaction_memo','prior_integration_component','H061_prior_freeze'):value.pop(key,None)
 need(oldb==newb,'originalconsumerbindingsunchanged')
 for name in('worker_entry_v1.py','scientific_config.INACTIVE.template.v1.json','storage_profiles.v1.json','handwritten_role_fixture.v1.json'):
  need((base/name).read_bytes()==(ROOT/PRIOR/name).read_bytes(),'unchangedasset '+name)
 for key,record in b['transaction_memo'].items():raw(record)
 review=read(b['transaction_memo']['independent_review_ref']);need(review['verdict']=='SOURCE_PASS'and review['independent']is True and review['source_ref']==b['transaction_memo']['source_ref']and review['blockers']==[],'exactmemo2review')
 need(b['transaction_memo']['source_ref']['sha256']=='487fcd93b73f6668126522b67720041abbbaa8b95ee54c3bc69adffba1a765b2','exactmemo2')
 need(b['transaction_memo']['independent_review_ref']['sha256']=='5f57cd4b46c3575901f660f47ab1c556aabdb2f30b8c15d7ad339b2cf0a29978','exactindependentreview')
 # Derive each exact original admission in isolated AST; reverse only3 substitutions.
 builder=iso([fn(current,'_memo_admission_node')],dict(ast=ast,copy=copy,need=need))['_memo_admission_node'];admissions={}
 for consumer,spec in b['consumers'].items():
  original=fn(tree(spec['driver_ref']['path']),'admit');derived=builder(original,consumer);reverted=copy.deepcopy(derived)
  reverted.body=[n for n in reverted.body if not(isinstance(n,ast.Assign)and isinstance(n.value,ast.Call)and ast.unparse(n.value.func)=='_make_root_mutation_memo')]
  for i,n in enumerate(reverted.body):
   if isinstance(n,ast.Assign)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Attribute):
    if ast.unparse(n.value.func)=='memo.read_state_v1':reverted.body[i]=copy.deepcopy(next(x for x in original.body if isinstance(x,ast.With)))
    if ast.unparse(n.value.func)=='memo.reserve_v1':n.value.func=ast.Attribute(value=ast.Name(id='g',ctx=ast.Load()),attr='apply_reserve_v1',ctx=ast.Load())
  need(same(reverted,original)and not any(isinstance(n,ast.With)for n in ast.walk(derived)),'exact3changesnoouterlock')
  gi=next(i for i,n in enumerate(derived.body)if isinstance(n,ast.Assign)and isinstance(n.value,ast.Call)and ast.unparse(n.value.func)in('guard','c.guard'))
  need(ast.unparse(derived.body[gi+1])=='memo = _make_root_mutation_memo(g, q)','constructorimmediatelyafterguard')
  firstput=next(i for i,n in enumerate(derived.body)if any(isinstance(x,ast.Call)and ast.unparse(x.func)in('put','c.put')for x in ast.walk(n)))
  positions=[next(i for i,n in enumerate(derived.body)if token in ast.unparse(n))for token in('consumer_bindings_v1','scope_spec','operation_draft_v1','memo.reserve_v1')]
  need(gi+1<firstput and positions==sorted(positions),'beforewriteandoriginalpreflight')
  admissions[consumer]=dict(exact_three_substitutions=True,no_outer_mutex=True,memo_before_first_put=True,union_scope_before_authority_reserve=True)
 # Evaluate actual original direct-role functions against pinned actual metadata only.
 fixture=json.loads((base/'handwritten_role_fixture.v1.json').read_text());guards=list(fixture['actual_guard_role_refs_by_name'].values());selected=b['selected_child_source_ref']
 need(selected==fixture['actual_selected_source_ref'],'selectedchild')
 union=iso([fn(tree(selected['path']),'consumer_bindings_v1')],dict(require=need,copy=copy,_raw=lambda r:need(isinstance(r,dict)and set(r)=={'path','sha256'},'exactfixture')) )['consumer_bindings_v1']
 append=iso([fn(current,'append_selected_source')],dict(need=need,copy=copy))['append_selected_source'];cases=[]
 for case in fixture['cases']:
  consumer=case['consumer'];stage=case['stage'];spec=b['consumers'][consumer];deps=read(ref(spec['base_directory']+'/dependencies.v1.json'));binding=read(spec['scientific_binding_ref'])
  scope={k:deps[k]for k in('NAMESPACE','PIDFD','OWNED','SUPERVISOR')};scope.update(ref=fake_ref,binding=lambda binding=binding:binding,MANIFEST=HERE+'/source_manifest.v1.json',SOURCE=HERE+'/root_driver_v1.py',DRIVER=HERE+'/root_driver_v1.py',ENTRY=HERE+'/worker_entry_v1.py',HANDOFF=HERE+'/source_handoff.v1.json',DEPS=deps)
  direct=iso([fn(tree(spec['contract_ref']['path']),'direct_roles')],scope)['direct_roles'];q={key:fake_ref('fixture/'+key+'.json')for key in('source_review_ref','scientific_review_ref','runtime_ref','runtime_qualification_ref','root_directive_ref','registration_ref','registration_decisions_ref','training_context_ref','data_context_ref','train_acceptance_ref')}
  q['storage_policy_selector_ref']=fixture['actual_selector_metadata_ref'];q['freeze_acceptance_ref']=fake_ref('fixture/freeze.json')if stage in('predict','score')else None;q['prediction_acceptance_ref']=fake_ref('fixture/predict.json')if stage=='score'else None
  before=direct(fake_ref('fixture/request.json'),q,fake_ref('fixture/config.json'),fake_ref('fixture/closure.json'));roles=append(before,selected,selected);combined=union(guards,roles)
  need(len(combined)==case['expected_distinct_union']and all(r in combined for r in guards+before+[selected]),'fullrolebaseline')
  need(roles[:len(before)]==before and append(roles,selected,selected)==roles,'orderedidempotent')
  need(b['transaction_memo']['source_ref']not in combined,'noextramemorole')
  overflow=combined+[fake_ref('fixture/extra'+str(i)+'.json')for i in range(65-len(combined))];need(rejects(lambda:union(overflow)),'unchanged65reject')
  cases.append(dict(consumer=consumer,stage=stage,distinct_union=len(combined),all_guards_and_direct_roles_preserved=True,exact_child_direct=True,memo_direct_role=False,role65_rejected=True))
 need([x['distinct_union']for x in cases]==[61,62,63,64],'allfourbaselines')
 # Exact original stage predicate with real sealed v1 freeze metadata and pure negative views.
 priorfreeze=b['H061_prior_freeze']
 for rr in priorfreeze.values():raw(rr)
 acceptance=read(priorfreeze['acceptance_ref']);envelope=read(acceptance['completed_ref'])
 q=dict(freeze_acceptance_ref=priorfreeze['acceptance_ref'],train_acceptance_ref=envelope['upstream']['train_acceptance_ref'],scientific_config=envelope['scientific_config'])
 overrides={};auth_reads=[]
 def reader(record):return overrides[record['path']]if record['path']in overrides else read(record)
 def in_output(record,directory,name):need(record['path']==directory+'/'+name,'exactstagepath');return reader(record)
 def stage_ref(path):return ref(path)if path in(priorfreeze['source_ref']['path'],priorfreeze['source_manifest_ref']['path'])else fake_ref(path)
 spec=b['consumers']['H061'];scope=dict(read=reader,require=need,ref=stage_ref,MANIFEST=HERE+'/source_manifest.v1.json',DRIVER=HERE+'/root_driver_v1.py',STATUSES={'postselection':'completed_H061_postselection_CPU_stage'},PHASE='postselection',_in_output=in_output)
 original_stage=iso([fn(tree(spec['contract_ref']['path']),'accepted_stage')],scope)['accepted_stage'];c=SimpleNamespace(accepted_stage=original_stage,require=need,raw=lambda r:auth_reads.append(r)or raw(r))
 hook=iso([fn(current,'_install_H061_prior_freeze_join')],dict(bindings=lambda:b,types=__import__('types'),copy=copy))['_install_H061_prior_freeze_join'];hook(c)
 cells=dict(zip(c.accepted_stage.__code__.co_freevars,c.accepted_stage.__closure__));frozen=cells['frozen'].cell_contents
 need(frozen.__code__ is original_stage.__code__ and {k for k in scope if frozen.__globals__[k] is not scope[k]}=={'MANIFEST','DRIVER'},'onlytwooriginalglobalschanged')
 need(c.accepted_stage(q,'freeze')==read(envelope['result_ref']),'actualv1freezemetadatajoin')
 need(auth_reads==[priorfreeze[k]for k in('acceptance_ref','source_ref','source_manifest_ref')],'allpriorpinsuncached')
 negatives=[]
 for target,key,value in [('acceptance','source_manifest_ref',fake_ref('wrong/manifest.json')),('acceptance','outer_terminal_exit_code',1),('acceptance','outer_terminal_consumed',False),('wait','returncode',1),('wait','all_owned_processes_and_groups_drained',False),('envelope','root_driver_ref',fake_ref('wrong/driver.py')),('envelope','scientific_config',{}),('envelope','upstream',{}),('envelope','input_ref',fake_ref('wrong/input.json'))]:
  rr={'acceptance':priorfreeze['acceptance_ref'],'wait':acceptance['actual_wait_ref'],'envelope':acceptance['completed_ref']}[target];changed=copy.deepcopy(read(rr));changed[key]=value;overrides[rr['path']]=changed
  need(rejects(lambda:c.accepted_stage(q,'freeze')),'priorpredicatepreserved '+key);overrides.clear();negatives.append(target+':'+key)
 badq=dict(q,freeze_acceptance_ref=fake_ref('other/freeze_acceptance.json'));need(rejects(lambda:c.accepted_stage(badq,'freeze')),'exactacceptanceonly')
 # New prediction must retain current v2 identities; no predecessor fallback.
 pred_accept=fake_ref('fixture/pred_acceptance.json');pred_wait=fake_ref('fixture/pred_wait.json');pred_env=fake_ref('fixture/pred/completed.json');pred_result=fake_ref('fixture/pred/completed_predict.json');pred_input=fake_ref('fixture/pred_input.json')
 overrides.update({pred_accept['path']:dict(schema='H061_postselection_outer_acceptance_v1',root_authorizer='root_agent',accepted=True,stage='predict',source_manifest_ref=stage_ref(HERE+'/source_manifest.v1.json'),outer_terminal_consumed=True,outer_terminal_exit_code=0,actual_wait_ref=pred_wait,completed_ref=pred_env,output_directory='fixture/pred'),pred_wait['path']:dict(returncode=0,actual_direct_worker_wait=True,all_owned_processes_and_groups_drained=True,operational_failure=None,completed_ref=pred_env,input_ref=pred_input),pred_env['path']:dict(status='completed_H061_postselection_CPU_stage',stage='predict',root_driver_ref=stage_ref(HERE+'/root_driver_v1.py'),input_ref=pred_input,upstream=dict(train_acceptance_ref=q['train_acceptance_ref'],freeze_acceptance_ref=q['freeze_acceptance_ref']),scientific_config=q['scientific_config'],result_ref=pred_result),pred_result['path']:dict(stage='predict',status='completed_H061_predict',new_core_fits=0,new_optimizer_updates=0)})
 qp=dict(q,prediction_acceptance_ref=pred_accept);need(c.accepted_stage(qp,'predict')==overrides[pred_result['path']],'v2predictionidentity')
 overrides[pred_accept['path']]['source_manifest_ref']=priorfreeze['source_manifest_ref'];need(rejects(lambda:c.accepted_stage(qp,'predict')),'nooldpredictfallback')
 overrides[pred_accept['path']]['source_manifest_ref']=stage_ref(HERE+'/source_manifest.v1.json');overrides[pred_env['path']]['root_driver_ref']=priorfreeze['source_ref'];need(rejects(lambda:c.accepted_stage(qp,'predict')),'nooldpredictdriverfallback');overrides.clear()
 # Pure exact installed H061 stage guard rejects new freeze and accepts only predict/score.
 validator=next(n for n in fn(current,'contract').body if isinstance(n,ast.FunctionDef)and n.name=='validate_request');stage_if=next(n for n in validator.body if isinstance(n,ast.If)and ast.unparse(n.test)=="consumer == 'H061'")
 for stage in('freeze','predict','score'):
  def guard_stage():iso([stage_if],dict(consumer='H061',q=dict(stage=stage),c=SimpleNamespace(require=need)))
  need(rejects(guard_stage)==(stage=='freeze'),'predictscoreonly')
 # Static default inertness only; do not import driver/worker/memo/guard or run entries.
 main=fn(current,'main');firstif=next(n for n in main.body if isinstance(n,ast.If));need(any(isinstance(n,ast.Return)and isinstance(n.value,ast.Constant)and n.value.value==0 for n in firstif.body),'inactiveearlyreturn')
 worker=tree(HERE+'/worker_entry_v1.py');need(any(isinstance(n,ast.ImportFrom)and n.level==1 and n.module is None and n.names[0].name=='root_driver_v1'for n in worker.body),'relativeworker')
 return dict(schema='H060_H061_transaction_admission_v2_source_checks_v1',source_only=True,source_ref=ref(base/'root_driver_v1.py'),only_namespace_memo_dependency_and_exact_prior_freeze_join=True,actual_v1_freeze_identity_join=True,prior_stage_negative_cases=negatives,H061_predict_score_only=True,prediction_current_v2_identity_required=True,accepted_memo2_and_independent_review_exact=True,original_consumer_bindings_science_caps_selector_and_barriers_unchanged=True,worker_profiles_config_rolefixture_bytes_unchanged=True,admission_AST_checks=admissions,handwritten_role_cases=cases,static_inactive_defaults_only=True,actual_entry_or_guard_or_memo_import=False,actual_state_mutex_resource_numerical_qualification_job_or_Git=False,execution_authorized=False)
if __name__=='__main__':print(json.dumps(check(),sort_keys=True,indent=2))
