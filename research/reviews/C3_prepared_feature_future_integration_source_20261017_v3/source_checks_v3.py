"""Source/metadata and RAM bridge checks only. No operational module imports."""
from pathlib import Path
import ast,copy,hashlib,json,types
R=Path('/workspace-SR003.nfs2/ARG-GNN')
D=R/'research/reviews/C3_prepared_feature_future_integration_source_20261017_v3'
OLD=R/'research/reviews/C3_remaining_cells_GPU0_source_20261017_v5'
positive=[];negative=[]
def check(ok,label):
    if not ok:raise AssertionError(label)
    positive.append(label)
def require(ok,label):
    if not ok:raise ValueError(label)
def rejected(fn,label):
    try:fn()
    except (ValueError,KeyError,AssertionError,TypeError):negative.append(label);return
    raise AssertionError('Accepted '+label)
def ref(path):
    p=Path(path);p=p if p.is_absolute()else R/p
    return dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
pins=json.loads((D/'pins.v1.json').read_text())
source=(D/'contract_v1.py').read_text();tree=ast.parse(source)
worker=ast.parse((D/'worker_v1.py').read_text())
execution=ast.parse((D/'execution_v1.py').read_text())
for p in D.glob('*.py'):ast.parse(p.read_text())
check(True,'All owned Python sources parse without import')
manifest=json.loads((D/'source_manifest.v1.json').read_text())
check(len(manifest['executed_source_refs'])==5,'Five real executed source roles including accepted v3')
for row in manifest['executed_source_refs']+[ref(D/'pins.v1.json'),pins['feature_review']]:
    check(ref(row['path'])==row,'Selected exact source/review pin '+row['path'])
review=json.loads((R/pins['feature_review']['path']).read_text())
check(review['verdict']=='SOURCE_PASS' and review['independent']is True and review['blockers']==[],
      'Component SOURCE_PASS consumed without whole integration acceptance')
get=lambda t,name:next(n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
env=dict(require=require,VAL=tuple('american_c3_val_%03d'%i for i in range(5)),EPOCHS=(5,10,15,20,25,30),
         PARENT='American_C3_short20_TRAIN_original5VAL',MANIFEST=str((D/'source_manifest.v1.json').relative_to(R)),
         DRIVER=str((D/'execution_v1.py').relative_to(R)),CLASSES=('EUR','EAS','AFR'),ref=ref,pins_v1=lambda:copy.deepcopy(pins))
for name in ['new_stage_values_v1','_complete_plan_metadata_v6','validate_all_original_plan_joins_v6','per_parent_source_policy_v6']:
    exec(compile(ast.Module(body=[get(tree,name)],type_ignores=[]),'<pure_RAM_'+name+'>','exec'),env)
training=dict(path='RAM/training_completion.json',sha256='a'*64)
env['tref_v1']=lambda:training
ram={}
def put(name,value):
    row=dict(path='RAM/'+name+'.json',sha256=hashlib.sha256(name.encode()).hexdigest())
    ram[json.dumps(row,sort_keys=True)]=value
    return row
env['read']=lambda row:copy.deepcopy(ram[json.dumps(row,sort_keys=True)])
# Keep original drain predicates unchanged except the expected exact source parameter.
drain=get(tree,'owned_drain_values_v2')
old_drain=get(ast.parse((OLD/'contract_v1.py').read_text()),'owned_drain_values_v2')
normal=copy.deepcopy(drain);normal.args.args=normal.args.args[:1];normal.args.defaults=[]
normal.body=normal.body[1:]
class ReplaceDriver(ast.NodeTransformer):
    def visit_Name(self,node):
        if node.id=='expected_driver':return ast.Call(func=ast.Name(id='ref',ctx=ast.Load()),args=[ast.Name(id='DRIVER',ctx=ast.Load())],keywords=[])
        return node
normal=ReplaceDriver().visit(normal)
check(ast.dump(normal,include_attributes=False)==ast.dump(old_drain,include_attributes=False),
      'Whole bounded wait/drain policy AST unchanged except explicit per-source ref')
env['owned_drain_values_v2']=lambda value,driver:require(value['owned_roster_policy_ref']==driver,'RAM exact drain source')
env['prior_v5_failure_resolution_values_v3']=lambda q,**kwargs:None
election=dict(path='RAM/original_election.json',sha256='e'*64)
created=[]
for i,parent in enumerate(env['VAL'][:3]):
    expected_manifest=ref(env['MANIFEST'])
    expected_driver=ref(env['DRIVER'])
    inp=dict(path='RAM/v7_input_'+str(i)+'.json',sha256=str(i)*64)
    out='RAM/output_'+str(i)
    seal=put('output_'+str(i)+'/seal',dict(epoch=30,parent_id=parent,training_completion_ref=training,private_truth_opened=False,
       prediction_ref=dict(path=out+'/predictions.npz',sha256='2'*64)))
    result=put('result_'+str(i),dict(status='sealed_C3_original_remaining_cells',parent_id=parent,epochs=[30],
       training_completion_ref=training,cell_election_ref=election,private_truth_opened=False,prediction_seal_refs=[seal]))
    source_review=put('review_'+str(i),dict(independent=True,verdict='SOURCE_PASS',blockers=[],source_manifest_ref=expected_manifest))
    q=put('request_'+str(i),dict(source_manifest_ref=expected_manifest,phase='predict_val_'+parent[-3:],
       cell_election_ref=election,parent_id=parent,pending_epochs=[30],source_review_ref=source_review))
    ram[json.dumps(inp,sort_keys=True)]=dict(root_stage_request_ref=q,source_manifest_ref=expected_manifest,
       phase='predict_val_'+parent[-3:],output_directory=out)
    done=put('done_'+str(i),dict(status='completed_C3_remaining_cells',parent_id=env['PARENT'],original_VAL_parent_id=parent,
       child_input_ref=inp,actual_execution_source_manifest_ref=expected_manifest,cell_election_ref=election,epochs=[30],
       training_completion_ref=training,query_ancestry_truth_opened=False,result_ref=result))
    wait=put('wait_'+str(i),dict(owned_roster_policy_ref=expected_driver,returncode=0,actual_direct_worker_wait=True,
       actual_OS_wait_status=0,all_owned_processes_and_groups_drained=True,complete_owned_roster_known=True,
       completed_ref=done,input_ref=inp,adapter_manifest_ref=expected_manifest,operational_failure=None,group_observations=[dict(absent=True)]))
    outer=put('outer_'+str(i),dict(root_authorizer='root_agent',actual_exit_code=0,actual_terminal_consumed=True,
       actual_wait_ref=wait,actual_direct_worker_returncode=0,actual_tool_session_id=100+i))
    evidence=dict(input_ref=inp,completion_ref=done,wait_ref=wait,outer_ref=outer)
    got=env['new_stage_values_v1'](evidence,parent,election,[30])
    check(got=={30:seal},'Pure bridge accepts exact source for '+parent)
    created.append((parent,evidence,inp,expected_manifest))
for parent,evidence,inp,expected in created:
    key=json.dumps(inp,sort_keys=True);before=copy.deepcopy(ram[key])
    ram[key]['source_manifest_ref']=dict(path='RAM/wrong_manifest',sha256='0'*64)
    rejected(lambda parent=parent,evidence=evidence:env['new_stage_values_v1'](evidence,parent,election,[30]),
       'Wrong per-parent source manifest rejected '+parent)
    ram[key]=before
parent,evidence,inp,_=created[0]
wrong=copy.deepcopy(evidence);wrong['input_ref']=pins['preserved_v5_VAL000_input']
ram[json.dumps(wrong['input_ref'],sort_keys=True)]=copy.deepcopy(ram[json.dumps(inp,sort_keys=True)])
rejected(lambda:env['new_stage_values_v1'](wrong,parent,election,[30]),'Stopped v5 input cannot represent a fresh successful VAL000 cell')
rejected(lambda:env['new_stage_values_v1'](created[1][1],env['VAL'][1],election,[5,30]),'Rerun of retained epoch rejected')
chosen={}
for parent in env['VAL']:
    row=pins['original_plan_refs'][parent];plan=json.loads((R/row['path']).read_text())
    ram[json.dumps(row,sort_keys=True)]=plan
    chosen[parent]={}
    for epoch in env['EPOCHS']:
        chosen[parent][epoch]=put(parent+'_'+str(epoch),dict(parent_id=parent,epoch=epoch,block_index_ref=row,
           observation_ref=plan['observation_ref'],focal_registry_ref=plan['focal_registry_ref'],records=10000,private_truth_opened=False))
check(env['validate_all_original_plan_joins_v6'](chosen)==pins['original_plan_refs'],'Exact all5plan/all30seal RAM barrier')
bad=copy.deepcopy(chosen);del bad[env['VAL'][4]][30]
rejected(lambda:env['validate_all_original_plan_joins_v6'](bad),'Missing original cell rejected before original barrier')
key=json.dumps(chosen[env['VAL'][2]][30],sort_keys=True);before=copy.deepcopy(ram[key])
ram[key]['block_index_ref']=pins['original_plan_refs'][env['VAL'][1]]
rejected(lambda:env['validate_all_original_plan_joins_v6'](chosen),'Cross-parent plan rejected')
ram[key]=before
# Direct and scientific source preservation.
texts=(D/'execution_v1.py').read_text()
embedded=next(n for n in execution.body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='_DIRECT_SOURCE_V6')
normalized=(OLD/'direct_gpu0_v1.py').read_text().replace(str(OLD.relative_to(R)),str(D.relative_to(R))).replace(str(OLD.relative_to(R)).replace('/','.'),str(D.relative_to(R)).replace('/','.'))
check(ast.literal_eval(embedded.value)==normalized,'Embedded direct attachment role exact after namespace substitution')
calls=lambda t,name:[ast.dump(n,include_attributes=False)for n in ast.walk(t)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id==name]
oldworker=ast.parse((OLD/'worker_v1.py').read_text())
for name in ['AmericanC3WholeTreeGNNV1','seal_shared_val_prediction_v1','seal_validation_matrix_v1']:
    check(calls(worker,name)==calls(oldworker,name),'Original scientific call AST unchanged '+name)
policy=env['per_parent_source_policy_v6']()
check(policy[env['VAL'][0]]['source_manifest_ref']==ref(env['MANIFEST'])
      and policy[env['VAL'][1]]['source_manifest_ref']==ref(env['MANIFEST']),'Fresh same-source all3parent provenance is explicit')
check('for i in (0,1,2)'in source and 'dataloader_workers=0'in source
      and 'max_cuda_reserved_bytes=8*2**30'in source,'Only fresh VAL000/001/002 GPU stages, original0workers/8GiB')
own=json.loads((R/'research/reviews/C3_remaining_cells_GPU0_ROOT_v5_predict_val_000_root01_oct17/storage.json').read_text())['reservation']
oldm=json.loads((OLD/'source_manifest.v1.json').read_text())
replaced=[v for v in own['consumer_bindings']if v not in oldm['executed_source_refs']]
replaced+=manifest['executed_source_refs']
check(len({(v['path'],v['sha256'])for v in replaced})==64,
      'Static exact5to5executed-role replacement preserves64direct slots, actual all47union remains ROOT check')
result=dict(schema='C3_prepared_feature_future_integration_author_source_checks_v3',status='AUTHOR_SOURCE_CHECKS_PENDING_INDEPENDENT_REVIEW',
    positive=len(positive),negative=len(negative),checks=positive,negative_cases=negative,
    operational_module_imports=0,guard_imports=0,actual_state_or_resource_probes=0,scientific_arrays_opened=0,
    prediction_or_training_jobs=0,original_payload_reads=0,truth_reads=0,GPU_operations=0,Git_actions=0,
    source_bridge_metadata_RAM_only=True,whole_route_accepted=False)
(D/'source_checks.result.v3.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(status=result['status'],positive=len(positive),negative=len(negative))))
