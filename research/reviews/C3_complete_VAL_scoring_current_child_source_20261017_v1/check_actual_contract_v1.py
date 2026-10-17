"""Actual imported successor contract tests using source/JSON and authenticated RAM fixtures.

No guard/scientific import, payload read/hash, runtime inspection, state operation,
resource admission, supervisor execution, numerical probe or activation occurs.
Only IO boundaries are adapted: RAM fixture documents and cached source/JSON reads;
two original absent-envelope predicates receive explicit RAM-absence fixtures.
All business predicates (including training, five closed plans, source routes,
selector, manifest, evidence, complete request and input) execute from the real module.
"""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,importlib.util,json,sys,traceback
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE='research/reviews/C3_complete_VAL_scoring_current_child_source_20261017_v1'
P=ROOT/HERE
TESTS=[];REAL_REFS={};READS=set();RAM={};RAM_LATEST={};ABSENCE_CALLS=[]
def need(ok,message):
 if not ok:raise AssertionError(message)
def good(name,action):
 action();TESTS.append(dict(name=name,status='PASS'))
def reject(name,action,part=None):
 try:action()
 except (ValueError,KeyError,TypeError) as e:
  need(part is None or part in str(e),'Wrong rejection for '+name+': '+str(e))
  TESTS.append(dict(name=name,status='PASS_REJECTED',message=str(e)));return
 raise AssertionError('Not rejected: '+name)
def imported(path,name):
 sp=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(sp)
 sys.modules[name]=m;sp.loader.exec_module(m);return m
c=imported(HERE+'/acceptance_contract_v1.py','_C3_actual_imported_current_child_contract')
builder=imported(c.BUILDER,'_C3_actual_imported_current_child_builder')
ORIGINAL_REF=c.ref;ORIGINAL_READ=c.read;ORIGINAL_LOCAL=c.legacy.local
ABSENT={c.legacy.TRAIN_OUTPUT+'/completed.json',c.legacy.TRAIN_OUTPUT+'/observation_prefetch_all_worker_drains.json'}
def local(path):
 if str(path) in ABSENT:
  def absent():ABSENCE_CALLS.append(str(path));return False
  return SimpleNamespace(exists=absent)
 return ORIGINAL_LOCAL(path)
def ref(path):
 path=str(path);path=str(Path(path).relative_to(ROOT)) if Path(path).is_absolute() else path
 if path in RAM_LATEST:return copy.deepcopy(RAM_LATEST[path])
 need(Path(path).suffix in ('.py','.json','.sh'),'No scientific payload reference hash: '+path)
 if path not in REAL_REFS:REAL_REFS[path]=ORIGINAL_REF(path)
 return copy.deepcopy(REAL_REFS[path])
CACHE={}
def read(record,cap=33554432):
 need(isinstance(record,dict) and set(record)=={'path','sha256'},'Exact reference schema')
 key=(record['path'],record['sha256'])
 if key in RAM:return copy.deepcopy(RAM[key])
 need(Path(record['path']).suffix=='.json','JSON metadata only: '+record['path'])
 need(ref(record['path'])==record,'Actual source/metadata digest: '+record['path'])
 if key not in CACHE:CACHE[key]=ORIGINAL_READ(record,cap)
 READS.add(record['path']);return copy.deepcopy(CACHE[key])
def put(label,value,path=None):
 path=path or 'RAM_FIXTURE_ONLY/'+label+'.json'
 raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode();record=dict(path=path,sha256=hashlib.sha256(raw).hexdigest())
 RAM[(path,record['sha256'])]=copy.deepcopy(value);RAM_LATEST[path]=record;return record
def opaque(label,suffix):return dict(path='RAM_FIXTURE_ONLY/'+label+suffix,sha256=hashlib.sha256(('NO_PAYLOAD:'+label).encode()).hexdigest())
c.read=read;c.ref=ref
c.legacy.read=read;c.legacy.ref=ref;c.legacy.local=local
# The real c.local is unchanged and never probes original dynamic scopes.
manifest=c.manifest_v1()
def check_manifest():
 old=read(c.SCHEMA2_MANIFEST_REF)
 need(all(manifest['source_sha256'][p]==h for p,h in old['source_sha256'].items()),'Every115 schema-v2 source pin preserved')
 need(len(old['source_sha256'])==115 and len(manifest['source_sha256'])==134,'Complete source counts')
 need(c.PHASES==builder.PHASES==('matrix','score') and c.DIRECTIVE==builder.DIRECTIVE,'One exact matrix/score directive')
 need(c._GUARD is None and c._SELECTOR_MODULE is None,'No guard imported')
good('Actual imported contract manifest preserves entire115-source predecessor and exact134-source closure',check_manifest)
good('Actual exact current Relate64 selector/source/policy/activation metadata accepted',lambda:c.select_storage_metadata_v1(c.ACTUAL_SELECTOR_REF))
wrong=copy.deepcopy(c.ACTUAL_SELECTOR_REF);wrong['sha256']='0'*64
reject('Wrong selector digest rejected before guard import',lambda:c.select_storage_metadata_v1(wrong),'Only the exact current')
wrong=copy.deepcopy(c.ACTUAL_SELECTOR_REF);wrong['path']=c.OLD_INPUT_REFS['predict_val_000']['path']
reject('No old/native7 or arbitrary selector fallback',lambda:c.select_storage_metadata_v1(wrong),'Only the exact current')
# Exact metadata body predicates are exercised separately at the IO fixture boundary.
selector=read(c.ACTUAL_SELECTOR_REF)
for key,value in [('selector_source_ref',c.LEGACY_CONTRACT_REF),('source_ref',c.LEGACY_DRIVER_REF),('policy_ref',c.LEGACY_MANIFEST_REF),('no_fallback_discovery_upgrade_or_in_run_swap',False),('scheduling_ceiling_bytes',980000000001)]:
 original=c.read
 def changed(r,*a,key=key,value=value):
  v=read(r,*a)
  if r==c.ACTUAL_SELECTOR_REF:v[key]=value
  return v
 c.read=changed
 reject('Selector metadata body rejects altered '+key,lambda:c.select_storage_metadata_v1(c.ACTUAL_SELECTOR_REF),'Exact current Relate64')
 c.read=original
b,training=c.training_ready_v1(c.SAVED_ACCEPTANCE_REF);training_ref=b['training_completion_ref']
need(training['protocol_sha256']=='97cc3dca1039d760bd963f9845fadcb41bb520f0fa6fc3ea039b66c59c9d4cdd','Exact protocol')
good('Real saved600 accepted metadata passes full original training/validator/review/source contract with only explicit RAM absence boundary',lambda:need(training['complete_parent_updates']==600 and training['focal_exposures']==6000000,'Original family'))
inputs={};plan_evidence={};plans={}
for phase,ir in {**c.OLD_INPUT_REFS,**c.SCHEMA2_INPUT_REFS}.items():
 parent=c.VAL[int(phase[-3:])];spec=read(ir);req=read(spec['root_stage_request_ref']);inputs[phase]=ir
 plan_evidence[parent]=req['plan_evidence'];plans[parent]=c.plan_ready_v1(req['plan_evidence'],parent)
 good('Actual imported mixed closed-plan validator accepts genuine '+parent,lambda parent=parent:need(plans[parent]['focal_records']==10000 and plans[parent]['sequence_length_bp']==30000000,'Full original support'))
 env=dict(input_ref=ir,root_driver_ref=spec['root_driver_ref'])
 good('Frozen actual input/request/source map accepted for '+phase,lambda phase=phase,env=env:c.stage_source_v1(env,phase))
# Load only known already audited completed VAL004 metadata as schema templates.
real4=read(ref('experiments/results/C3_full30Mb_eval_actual_plan_lineage_checkpoint_schema_predict_val_004_root01_oct17/completed.json'))
six_template=read(real4['result_ref']);seal_template=read(six_template['prediction_seal_refs'][0])
wait_template=read(ref('research/reviews/C3_full30Mb_eval_ROOT_actual_plan_lineage_checkpoint_schema_predict_val_004_root01_oct17/actual_wait.json'))
epochrefs={x['epoch']:x['checkpoint_ref'] for x in training['checkpoint_candidates']}
review=read(ref(HERE+'/ROOT_review.INACTIVE.template.v1.json'))
reject('Saved inactive review template cannot authorize integration',lambda:c.integration_review_v1(ref(HERE+'/ROOT_review.INACTIVE.template.v1.json')),'Independent complete')
review.update(independent=True,verdict='SOURCE_PASS',SYNTHETIC_RAM_ONLY=True)
review_ref=put('independent_review_never_persisted',review)
good('Actual builder and contract both accept full genuine-shaped RAM independent review schema',lambda:(builder.builder_review_v1(review),c.integration_review_v1(review_ref)))
def stage(phase,result_ref,input_ref,driver=None):
 spec=read(input_ref);driver=driver or spec['root_driver_ref']
 env=copy.deepcopy(real4);env.update(stage=phase,status=c.legacy.STATUSES[phase],input_ref=input_ref,root_driver_ref=driver,result_ref=result_ref,training_completion_ref=training_ref)
 cr=put(phase+'_completion',env)
 wait=copy.deepcopy(wait_template);wait.update(driver_ref=driver,input_ref=input_ref,completed_ref=cr,returncode=0,actual_OS_wait_status=0,actual_direct_launcher_wait=True,actual_direct_worker_wait=True,all_owned_processes_and_groups_drained=True,operational_failure=None)
 wr=put(phase+'_direct_wait',wait)
 outer=dict(schema='ROOT_actual_consumed_stage_terminal_v1',root_authorizer='root_agent',actual_terminal_consumed=True,actual_exit_code=0,completed_ref=cr,actual_wait_ref=wr,input_ref=input_ref,driver_ref=driver,SYNTHETIC_RAM_ONLY=True)
 tr=put(phase+'_outer',outer)
 return dict(completion_ref=cr,actual_wait_ref=wr,outer_terminal_ref=tr)
predictions={};cells=[];seals_by_parent={}
for parent in c.VAL:
 refs=[]
 for epoch in c.EPOCHS:
  seal=copy.deepcopy(seal_template);pred=opaque(parent+'_'+str(epoch),'_predictions.npz')
  seal.update(epoch=epoch,parent_id=parent,checkpoint_ref=epochrefs[epoch],training_completion_ref=training_ref,prediction_ref=pred,focal_registry_ref=plans[parent]['focal_registry_ref'],records=10000,private_truth_opened=False,probabilities_modified=False)
  sr=put(parent+'_'+str(epoch)+'_seal',seal);refs.append(sr)
  cells.append(dict(epoch=epoch,parent_id=parent,checkpoint_ref=epochrefs[epoch],prediction_ref=pred,prediction_seal_ref=sr,focal_registry_ref=plans[parent]['focal_registry_ref']))
 six=copy.deepcopy(six_template);six.update(parent_id=parent,training_completion_ref=training_ref,prediction_seal_refs=refs,private_truth_opened=False)
 rr=put(parent+'_all6',six);phase='predict_val_'+parent[-3:]
 predictions[parent]=stage(phase,rr,inputs[phase]);seals_by_parent[parent]=refs
matrix=dict(status='sealed_C3_all30_original_VAL_predictions',training_completion_ref=training_ref,all_predictions_sealed_before_VAL_truth=True,TEST_deferred=True,checkpoint_epochs=list(c.EPOCHS),validation_parent_ids=list(c.VAL),cells=cells)
matrix_ref=put('all30_matrix',matrix)
simulations={}
# The full simulation JSON schema is copied from the exact same-parent producer;
# its private_truth_ref is metadata only and is never resolved or hashed.
for parent in c.VAL:
 obs=read(plans[parent]['observation_ref']);simdir=Path(obs['source_observation_ref']['path']).parent.parent
 simulations[parent]=ref(str(simdir/'completed.json'))
 sim=read(simulations[parent]);need(sim['parent_id']==parent,'Genuine same-parent simulation metadata')
def evidence(phase):
 v=dict(schema='C3_full30Mb_eval_actual_stage_evidence_v1',phase=phase,ROOT_artifact_acceptance_ref=c.SAVED_ACCEPTANCE_REF,closed_plan_evidence_by_parent=copy.deepcopy(plan_evidence))
 if phase=='matrix':v['VAL_predictions']=copy.deepcopy(predictions)
 else:v.update(matrix=copy.deepcopy(matrix_stage),VAL_simulation_completion_refs=copy.deepcopy(simulations))
 return v
expiry='2099-01-01T00:00:00+00:00'
def request(phase,ev):
 fields,_,_=c.evidence_request_fields_v1(SimpleNamespace(**vars(c)),ev,phase)
 er=put(phase+'_actual_stage_evidence',ev);caps=c.caps_v1(phase);capref=put(phase+'_caps',caps)
 packet='research/reviews/C3_full30Mb_eval_ROOT_current_child_matrix_score_'+phase+'_SYNTHETIC_RAM_ONLY'
 output='experiments/results/C3_full30Mb_eval_current_child_matrix_score_'+phase+'_SYNTHETIC_RAM_ONLY'
 directive=dict(schema='ROOT_C3_full30Mb_one_CPU_evaluation_stage_directive_v1',root_authorizer='root_agent',active=True,execution_authorized=True,expires_at_utc=expiry,phase=phase,directive=c.DIRECTIVE,actual_recorded_utc='2026-10-17T00:00:00+00:00',actual_stage_evidence_ref=er,actual_ROOT_artifact_acceptance_ref=c.SAVED_ACCEPTANCE_REF,main_review_ref=c.LEGACY_MAIN_REVIEW_REF,builder_review_ref=review_ref,original_history=c.ORIGINAL_HISTORY,all5VALx6checkpoint_barrier_preserved=True,storage_policy_selector_ref=c.ACTUAL_SELECTOR_REF,original_producer_success_or_outer_wait_claimed=False,controller_shutdown_or_edit_authorized=False,close_credit_retire_retraining_or_automatic_retry_authorized=False)
 dr=put(phase+'_ROOT_directive',directive)
 req=dict(schema='C3_full30Mb_evaluation_ROOT_request_v1',active=True,root_authorizer='root_agent',execution_authorized=True,created_at_utc='2026-10-17T00:00:00+00:00',expires_at_utc=expiry,phase=phase,device='cpu',GPU_authorized=False,physical_gpu=None,runtime_ref=c.RUNTIME_REF,source_manifest_ref=ref(c.MANIFEST),packet_directory=packet,output_directory=output,caps_ref=capref,integration_review_ref=review_ref,ROOT_directive_ref=dr,actual_stage_evidence_ref=er,storage_policy_selector_ref=c.ACTUAL_SELECTOR_REF,**fields)
 return put(phase+'_request',req),req
input_template=read(ref(c.LEGACY_PACKAGE+'/input.INACTIVE.template.v1.json'))
storage_template=read(read(inputs['predict_val_004'])['refs']['storage'])
def input_for(phase,rr,req):
 spec=copy.deepcopy(input_template);spec.pop('INACTIVE',None)
 spec.update(status='active',active=True,root_authorizer='root_agent',execution_authorized=True,expires_at_utc=expiry,created_at_utc=req['created_at_utc'],job_id='C3-RAM-ONLY-'+phase,phase=phase,output_directory=req['output_directory'],packet_directory=req['packet_directory'],root_stage_request_ref=rr,caps_ref=req['caps_ref'],parent_id=c.PARENT,source_sha256=manifest['source_sha256'],main_source_ref=ref(c.ENTRY),root_driver_ref=ref(c.DRIVER),refs=dict(config=put(phase+'_config',c.make_config(req,c.caps_v1(phase))),runtime=c.RUNTIME_REF,storage=None),max_cpu_rss_bytes=137438953472,max_cuda_reserved_bytes=0,root_auxiliary_directory=req['packet_directory']+'/worker_aux',completed_overlay_policy_ref=c.POLICY_REF,storage_policy_selector_ref=c.ACTUAL_SELECTOR_REF)
 required=c.required_direct_refs_v1(spec,req)
 storage=copy.deepcopy(storage_template);storage.update(accounting_mode=c.SELECTED_MODE,completed_overlay_policy_ref=c.POLICY_REF,storage_policy_selector_ref=c.ACTUAL_SELECTOR_REF,job_id=spec['job_id'],output_directory=spec['output_directory'],reserved_new_bytes=33554432)
 storage['reservation'].update(output=str(ROOT/spec['output_directory']),auxiliary_scopes=[str(ROOT/(req['packet_directory']+'/worker_aux')),str(ROOT/(req['packet_directory']+'/worker_tmp'))],consumer_bindings=required)
 spec['refs']['storage']=put(phase+'_storage',storage)
 return put(phase+'_input',spec),spec,len(required)
requests={};fixture_inputs={};counts={}
rr,req=request('matrix',evidence('matrix'));requests['matrix']=(rr,req)
ir,spec,counts['matrix']=input_for('matrix',rr,req);fixture_inputs['matrix']=(ir,spec)
good('Actual complete matrix validate_request accepts all5 frozen producers/all30 synthetic seals with real training and five real plans',lambda:need(c.validate_request(rr)[0]==req,'Whole matrix request'))
good('Actual complete matrix validate_input accepts exact source map/current-child storage/direct roles/config',lambda:need(c.validate_input_v1(ir)[0]==spec,'Whole matrix input'))
matrix_stage=stage('matrix',matrix_ref,ir)
rr,req=request('score',evidence('score'));requests['score']=(rr,req)
ir,spec,counts['score']=input_for('score',rr,req);fixture_inputs['score']=(ir,spec)
good('Actual complete score validate_request accepts all30 matrix plus its own real-schema wait0/drain/consumed outer0 and real same-parent simulation metadata',lambda:need(c.validate_request(rr)[0]==req,'Whole score request'))
good('Actual complete score validate_input accepts exact current-child storage/direct-role/schema joins without opening truth',lambda:need(c.validate_input_v1(ir)[0]==spec,'Whole score input'))
def changed_request(phase,update):
 rr,req=requests[phase];q=copy.deepcopy(req);update(q);return c.validate_request(put('negative_request_'+phase,q))
def validate_evidence_request(phase,ev):
 # Carry the altered evidence through the full real request path, with its
 # directive/evidence identity join intact, to test the intended inner barrier.
 _,original=requests[phase];req=copy.deepcopy(original)
 er=put('negative_evidence_'+phase,ev);directive=read(req['ROOT_directive_ref'])
 directive['actual_stage_evidence_ref']=er
 req.update(actual_stage_evidence_ref=er,ROOT_directive_ref=put('negative_directive_'+phase,directive))
 return c.validate_request(put('negative_whole_request_'+phase,req))
def changed_evidence(phase,update):
 ev=evidence(phase);update(ev)
 return validate_evidence_request(phase,ev)
for phase in c.PHASES:
 reject(phase+' rejects prediction admission phase',lambda phase=phase:changed_request(phase,lambda q:q.update(phase='predict_val_004')),'New matrix/score admissions only')
 reject(phase+' rejects TEST/extra truth capability',lambda phase=phase:changed_request(phase,lambda q:q.update(TEST_truth_capability=opaque('forbidden','.npz'))),'Exact phase capabilities')
 reject(phase+' rejects wrong current source manifest',lambda phase=phase:changed_request(phase,lambda q:q.update(source_manifest_ref=c.SCHEMA2_MANIFEST_REF)),'Exact current CPU runtime')
 reject(phase+' rejects old namespace',lambda phase=phase:changed_request(phase,lambda q:q.update(packet_directory='research/reviews/C3_full30Mb_eval_ROOT_actual_plan_lineage_'+phase)),'Fresh separated')
 reject(phase+' rejects missing closed parent plan',lambda phase=phase:changed_evidence(phase,lambda ev:ev['closed_plan_evidence_by_parent'].pop(c.VAL[-1])),'matrix/score require all5')
 reject(phase+' rejects swapped selector',lambda phase=phase:changed_request(phase,lambda q:q.update(storage_policy_selector_ref=c.LEGACY_MANIFEST_REF)),'Only the exact current')
 reject(phase+' rejects increased budget',lambda phase=phase:changed_request(phase,lambda q:q.update(caps_ref=put('bad_caps',dict(c.caps_v1(phase),reservation_cap_bytes=33554433)))),'Every original cap')
 reject(phase+' rejects expired authority',lambda phase=phase:changed_request(phase,lambda q:q.update(expires_at_utc='2000-01-01T00:00:00+00:00')),'Unexpired actual ROOT')
reject('Matrix rejects24/30 six-seal evidence',lambda:changed_evidence('matrix',lambda ev:ev['VAL_predictions'].update({c.VAL[-1]:stage('predict_val_004',put('only0_six',dict(six_template,parent_id=c.VAL[-1],prediction_seal_refs=[])),inputs['predict_val_004'])})),'One actual original VAL parent/all6')
reject('Matrix rejects one missing producer envelope despite30 leaf seals',lambda:changed_evidence('matrix',lambda ev:ev['VAL_predictions'].pop(c.VAL[-1])),'All5 actually completed prediction stages')
def terminal_case(phase,kind,key,value):
 ev=evidence(phase);st=copy.deepcopy(ev['VAL_predictions'][c.VAL[0]] if phase=='matrix' else ev['matrix'])
 obj=read(st[kind]);obj[key]=value;st[kind]=put('bad_'+phase+'_'+kind+'_'+key,obj)
 if kind=='actual_wait_ref':
  outer=read(st['outer_terminal_ref']);outer['actual_wait_ref']=st[kind];st['outer_terminal_ref']=put('consistent_negative_outer',outer)
 if phase=='matrix':ev['VAL_predictions'][c.VAL[0]]=st
 else:ev['matrix']=st
 return validate_evidence_request(phase,ev)
for phase in c.PHASES:
 for kind,key,value,part in [('actual_wait_ref','returncode',1,'actual direct wait0'),('actual_wait_ref','returncode',False,'Genuine integer stage'),('actual_wait_ref','all_owned_processes_and_groups_drained',False,'actual direct wait0'),('outer_terminal_ref','actual_terminal_consumed',False,'Actual root-consumed stage'),('outer_terminal_ref','actual_exit_code',1,'Actual root-consumed stage')]:
  reject(phase+' rejects '+kind+'/'+key+'='+str(value),lambda phase=phase,kind=kind,key=key,value=value:terminal_case(phase,kind,key,value),part)
 def noouter(ev,phase=phase):
  if phase=='matrix':ev['VAL_predictions'][c.VAL[0]].pop('outer_terminal_ref')
  else:ev['matrix'].pop('outer_terminal_ref')
 reject(phase+' rejects absent consumed outer envelope',lambda phase=phase:changed_evidence(phase,noouter),'Actual stage completion/direct wait/consumed outer')
def bad_stage_input():
 ev=evidence('matrix');st=ev['VAL_predictions'][c.VAL[3]];env=read(st['completion_ref'])
 env['input_ref']=inputs['predict_val_004'];st2=stage('predict_val_003',env['result_ref'],env['input_ref'])
 ev['VAL_predictions'][c.VAL[3]]=st2
 return validate_evidence_request('matrix',ev)
reject('Matrix rejects schema-v2 input substitution between parents',bad_stage_input,'Exact already-admitted schema-v2')
def bad_matrix(change):
 m=copy.deepcopy(matrix);change(m)
 ev=evidence('score');ev['matrix']=stage('matrix',put('bad_matrix',m),fixture_inputs['matrix'][0])
 return validate_evidence_request('score',ev)
reject('Score rejects29/30 cells before truth capability',lambda:bad_matrix(lambda m:m['cells'].pop()),'Actual complete original5VALx6')
reject('Score rejects duplicate30-cell identities',lambda:bad_matrix(lambda m:m['cells'].__setitem__(-1,copy.deepcopy(m['cells'][0]))),'Unique original30 Cartesian')
reject('Score rejects wrong checkpoint lineage',lambda:bad_matrix(lambda m:m['cells'][0].update(checkpoint_ref=epochrefs[30])),'Unique original30 Cartesian')
reject('Score rejects early truth marker',lambda:bad_matrix(lambda m:m.update(all_predictions_sealed_before_VAL_truth=False)),'Actual complete original5VALx6')
reject('Score rejects unrelated parent truth metadata',lambda:changed_evidence('score',lambda ev:ev['VAL_simulation_completion_refs'].update({c.VAL[0]:simulations[c.VAL[1]]})),'Actual same-parent simulation')
reject('Score rejects missing truth metadata capability',lambda:changed_evidence('score',lambda ev:ev['VAL_simulation_completion_refs'].pop(c.VAL[0])),'Only5 actual same-parent')
def changed_input(phase,update):
 ir,spec=fixture_inputs[phase];v=copy.deepcopy(spec);update(v);return c.validate_input_v1(put('negative_input',v))
for phase in c.PHASES:
 reject(phase+' input rejects old producer source map',lambda phase=phase:changed_input(phase,lambda s:s.update(source_sha256=read(c.SCHEMA2_MANIFEST_REF)['source_sha256'])),'Actual reviewed CPU producer')
 def changed_storage(s,update):
  st=read(s['refs']['storage']);update(st);s['refs']['storage']=put('negative_storage',st)
 reject(phase+' input rejects missing required direct role',lambda phase=phase:changed_input(phase,lambda s:changed_storage(s,lambda st:st['reservation']['consumer_bindings'].pop())),'every required direct role')
 reject(phase+' input rejects65 direct roles',lambda phase=phase:changed_input(phase,lambda s:changed_storage(s,lambda st:st['reservation'].update(consumer_bindings=[opaque('role_'+str(i),'.json')for i in range(65)]))),'every required direct role')
 reject(phase+' input rejects old storage policy',lambda phase=phase:changed_input(phase,lambda s:changed_storage(s,lambda st:st.update(completed_overlay_policy_ref=c.LEGACY_MANIFEST_REF))),'every required direct role')
 def altered_config(s):
  oldref=s['refs']['config'];newref=put('bad_config',dict(execution_enabled=False));s['refs']['config']=newref
  changed_storage(s,lambda st:st['reservation'].update(consumer_bindings=[newref if r==oldref else r for r in st['reservation']['consumer_bindings']]))
 reject(phase+' input rejects altered operational config',lambda phase=phase:changed_input(phase,altered_config),'Exact operational-only config')
reject('Matrix full request rejects early VAL truth capability',lambda:changed_evidence('matrix',lambda ev:ev.update(VAL_truth_capability=simulations)),'Exact phase-specific actual evidence')
reject('Score full request rejects request-only truth capability absent evidence',lambda:changed_request('score',lambda q:q['VAL_truth_capability'].update({c.VAL[0]:simulations[c.VAL[1]]})),'Request is exactly')
# Import the exact preparation helper safely, and exercise its role union with
# a pure RAM consumer interface. No real guard or actual returned-role query.
helper=imported(c.PREP_HELPER_REF['path'],'_C3_exact_source_prep_helper_RAM_tests')
def ram_union(*groups):
 rows=[]
 for group in groups:
  for r in group:
   if r not in rows:rows.append(copy.deepcopy(r))
 return rows
def union_test(mode):
 rr,req=requests['matrix'];_,spec=fixture_inputs['matrix']
 returned={'extra_current_capsule':c.ACTUAL_SELECTOR_REF,'existing_role':c.REPAIR_REF}
 guard=SimpleNamespace(consumer_bindings_v1=ram_union)
 if mode=='drop':guard.consumer_bindings_v1=lambda *groups:ram_union(*groups)[1:]
 if mode=='overflow':returned.update({'RAM_EXTRA_'+str(i):opaque('RAM_UNION_'+str(i),'.json') for i in range(65)})
 roles=helper.role_union_v1(c,guard,dict(bindings=returned),spec,req,[ref(c.HANDOFF)])
 need(all(r in roles for r in returned.values()),'Every returned RAM role retained')
good('Actual sealed preparation helper role_union retains all RAM-returned/direct/extra roles',lambda:union_test('keep'))
reject('Actual preparation helper role_union rejects dropped returned RAM role',lambda:union_test('drop'),'Never drop a returned')
reject('Actual preparation helper role_union rejects over64 union without pruning',lambda:union_test('overflow'),'Every actual returned/current/operational role')
def nodes(path):return {n.name:n for n in ast.parse((ROOT/path).read_text()).body if isinstance(n,ast.FunctionDef)}
old='research/reviews/C3_CPU_checkpoint_schema_repair_source_20261017_v2'
a=nodes(c.ENTRY);z=nodes(old+'/worker_entry_v1.py')
for name in ('matrix_v1','score_v1'):
 good('Original worker '+name+' body AST unchanged',lambda name=name:need(ast.dump(a[name],include_attributes=False)==ast.dump(z[name],include_attributes=False),'Same scientific worker body'))
oldc=nodes(old+'/acceptance_contract_v1.py');newc=nodes(c.SOURCE)
for name in ('plan_producer_lineage_v1','plan_ready_v1','training_ready_v1','actual_stage_v1','prediction_metadata_v1','matrix_metadata_v1','evidence_request_fields_v1','validate_input_v1','required_direct_refs_v1'):
 good('Original metadata/barrier function '+name+' AST unchanged',lambda name=name:need(ast.dump(newc[name],include_attributes=False)==ast.dump(oldc[name],include_attributes=False),'Same preserved predicate'))
good('Root supervisor adapter unchanged except package namespace',lambda:need((ROOT/c.DRIVER).read_text().replace(HERE,old)==(ROOT/old/'root_driver_v1.py').read_text(),'Same supervisor adapter'))
def final_boundary():
 need(not any(k=='torch' or k.startswith(('arggnn','numpy','torch_geometric')) for k in sys.modules),'No scientific import')
 need(c._GUARD is None and c._SELECTOR_MODULE is None,'No guard load/configure')
 need(ABSENCE_CALLS and set(ABSENCE_CALLS)==ABSENT,'Explicit absence IO fixtures only')
 need(all(Path(p).suffix in ('.py','.json','.sh')for p in REAL_REFS),'No payload hashes')
 for path,record in REAL_REFS.items():need(ORIGINAL_REF(path)==record,'Original source/metadata changed: '+path)
good('All fixtures remain RAM-only; real source/metadata pins reverified; no guard/scientific/payload/state/resource/admission/launch actions',final_boundary)
result=dict(schema='C3_actual_current_child_complete_contract_RAM_checks_v2',status='PASS_SOURCE_ONLY',source_only=True,active=False,execution_authorized=False,test_count=len(TESTS),checks=TESTS,source_ref=ref(c.SOURCE),test_script_ref=ref(HERE+'/check_actual_contract_v1.py'),manifest_ref=ref(c.MANIFEST),current_selector_ref=c.ACTUAL_SELECTOR_REF,real_source_metadata_refs=sorted(REAL_REFS.values(),key=lambda r:r['path']),real_JSON_metadata_reads=sorted(READS),RAM_document_count=len(RAM),mandatory_direct_role_counts=counts,original_absence_IO_boundary_fixture_paths=sorted(ABSENT),business_functions_stubbed=[],whole_request_phases=list(c.PHASES),whole_input_phases=list(c.PHASES),real_all30_or_genuine_new_producer_success_claimed=False,guard_state_resource_process_GPU_array_tensor_numerical_or_Git_actions=False)
with (P/'checks.result.v2.json').open('x')as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(dict(status=result['status'],test_count=len(TESTS),mandatory_direct_role_counts=counts,result_ref=ref(HERE+'/checks.result.v2.json')),sort_keys=True),flush=True)
