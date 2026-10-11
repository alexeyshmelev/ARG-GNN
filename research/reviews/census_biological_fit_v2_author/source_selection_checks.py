"""Bounded source/selection author checks. No module/model/optimizer or actual arrays."""
from __future__ import annotations
import ast, copy, difflib, hashlib, json, math, os, sys
from pathlib import Path
import numpy as np
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=ROOT/'research/reviews/census_biological_fit_v2_author'
sys.dont_write_bytecode=True
assert os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
checks=[]
def check(name,value):
    if not value:raise AssertionError(name)
    checks.append(name)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ref(path):return {'path':str(path.relative_to(ROOT)),'sha256':sha(path)}
def dump(node):return ast.dump(node,include_attributes=False)
def write(name,value):
    with (OUT/name).open('x') as stream:stream.write(value)
pins=json.loads((OUT/'baseline_pins.json').read_text())
for p,h in pins.items():check('sealed_baseline_'+p,sha(ROOT/p)==h)
p1=ROOT/'src/arggnn/census_biological_fit_v1.py';p2=ROOT/'src/arggnn/census_biological_fit_v2.py'
c1=ROOT/'configs/census_biological_fit_v1.json';c2=ROOT/'configs/census_biological_fit_v2.json'
s1=p1.read_text();s2=p2.read_text();j1=json.loads(c1.read_text());j2=json.loads(c2.read_text())
versions=[
('census_biological_fit_v1','census_biological_fit_v2'),
('census_biological_fit_admission_v1','census_biological_fit_admission_v2'),
('census_biological_fit_runtime_v1','census_biological_fit_runtime_v2'),
('census_biological_fit_storage_v1','census_biological_fit_storage_v2'),
('census_biological_fit_launcher_v1','census_biological_fit_launcher_v2'),
('census_biological_fit_profile_gate_v1','census_biological_fit_profile_gate_v2'),
('census_biological_fit_evidence_v1','census_biological_fit_evidence_v2')]
predicate=("if nll<state['best']:","if state['best_epoch'] is None or nll<state['best']:")
expected=s1
for a,b in versions:expected=expected.replace(a,b)
expected=expected.replace(*predicate)
check('complete_source_byte_delta_exactly_versions_plus_one_predicate',s2==expected and s1.count(predicate[0])==1)
check('complete_config_byte_delta_exactly_two_fit_bindings',c2.read_text()==c1.read_text().replace('census_biological_fit_v1','census_biological_fit_v2') and c1.read_text().count('census_biological_fit_v1')==2)
expected_config=copy.deepcopy(j1)
expected_config['kind']='census_biological_fit_v2'
expected_config['gpu_runtime']['launcher_module']='arggnn.census_biological_fit_v2'
check('all_other_config_data_deep_equal',j2==expected_config)
for p,h in j1['source_sha256'].items():check('unchanged_dependency_'+p,sha(ROOT/p)==h)
check('all35_dependency_bindings_unchanged',len(j2['source_sha256'])==35 and j2['source_sha256']==j1['source_sha256'])
for receipt_path in ('research/reviews/census_biological_fit_v1_author/author_receipt.json','research/reviews/census_biological_fit_v1_independent/review_receipt.json'):
    receipt=json.loads((ROOT/receipt_path).read_text())
    for p,h in receipt['artifact_sha256'].items():check('sealed_receipt_artifact_'+p,sha(ROOT/p)==h)
check('v1_independent_REJECT_preserved',json.loads((ROOT/'research/reviews/census_biological_fit_v1_independent/review_receipt.json').read_text())['status'].startswith('rejected'))
t1=ast.parse(s1);t2=ast.parse(s2)
check('successor_python_syntax_compiles',isinstance(compile(t2,str(p2),'exec'),type(compile('','<empty>','exec'))))
class UndoVersionLiterals(ast.NodeTransformer):
    def visit_Constant(self,node):
        if isinstance(node.value,str):
            value=node.value
            for a,b in versions:value=value.replace(b,a)
            node.value=value
        return node
n2=UndoVersionLiterals().visit(copy.deepcopy(t2))
def definition(tree,name):return next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
def selection_loop(tree):
    run=definition(tree,'run_fit_v1')
    return next(n for n in ast.walk(run) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='control' and isinstance(n.iter,ast.Name) and n.iter.id=='CONTROLS' and any(isinstance(c,ast.If) and 'nll' in ast.unparse(c.test) and "state['best']" in ast.unparse(c.test) for c in n.body))
def checkpoint_if(tree):return next(n for n in selection_loop(tree).body if isinstance(n,ast.If))
def completion_guard(tree):return next(n for n in ast.walk(definition(tree,'run_fit_v1')) if isinstance(n,ast.If) and "state['steps']" in ast.unparse(n.test) and "state['best_epoch'] is None" in ast.unparse(n.test))
old_if=checkpoint_if(t1);new_if=checkpoint_if(t2)
check('exact_new_predicate_first_epoch_or_strict_later_improvement',ast.unparse(new_if.test)=="state['best_epoch'] is None or nll < state['best']")
check('checkpoint_body_and_else_unchanged',dump(ast.Module(body=old_if.body,type_ignores=[]))==dump(ast.Module(body=new_if.body,type_ignores=[])) and old_if.orelse==new_if.orelse==[])
check('terminal_fixed_budget_guard_unchanged',dump(completion_guard(t1))==dump(completion_guard(t2)))
check('normalized_ast_only_run_fit_changed',all(dump(a)==dump(b) for a,b in zip(t1.body,n2.body) if not (isinstance(a,ast.FunctionDef) and a.name=='run_fit_v1')) and len(t1.body)==len(n2.body))
checkpoint_if(n2).test=copy.deepcopy(old_if.test)
check('entire_scientific_module_AST_unchanged_after_authorized_inverse_delta',dump(t1)==dump(n2))
scientific_ast=[]
for node in t1.body:
    if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
        peer=definition(n2,node.name)
        check('normalized_definition_AST_unchanged_'+node.name,dump(node)==dump(peer))
        scientific_ast.append({'name':node.name,'unchanged_after_allowed_inverse_delta':True,'AST_sha256':hashlib.sha256(dump(node).encode()).hexdigest()})
source_diff=''.join(difflib.unified_diff(s1.splitlines(True),s2.splitlines(True),fromfile=str(p1.relative_to(ROOT)),tofile=str(p2.relative_to(ROOT))))
config_diff=''.join(difflib.unified_diff(c1.read_text().splitlines(True),c2.read_text().splitlines(True),fromfile=str(c1.relative_to(ROOT)),tofile=str(c2.relative_to(ROOT))))
write('source_complete_delta.diff',source_diff);write('config_complete_delta.diff',config_diff)

# Execute exact AST scoring/history/checkpoint loop, not a reimplementation of its predicate.
# Only explicit in-memory state/score/seal/output stand-ins are supplied.
class ScalarStateValue:
    def __init__(self,tag):self.tag=tag
    def detach(self):return self
    def cpu(self):return self.tag
class ModelStateStandIn:
    def __init__(self,epoch):self.epoch=epoch
    def state_dict(self):return {'explicit_handmade_epoch_token':ScalarStateValue(self.epoch)}
class OptimizerStateStandIn:
    def __init__(self,steps):self.steps=steps
    def state_dict(self):return {'explicit_handmade_step_counter':self.steps}
class MemoryOutput:
    def __init__(self):self.files={};self.calls=[];self.bytes=0
    def save(self,name,value,kind='json'):
        assert name not in self.files
        data=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
        self.bytes+=len(data)
        assert self.bytes<1000000
        self.files[name]=copy.deepcopy(value);self.calls.append({'name':name,'kind':kind,'bytes':len(data)})
        return {'path':'explicit_handmade_memory/'+name,'sha256':hashlib.sha256(data).hexdigest()}
def compiled_path(tree):
    constants=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='CONTROLS' for x in n.targets))
    nll=definition(tree,'_nll_v1')
    namespace={'np':np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(constants),copy.deepcopy(nll)],type_ignores=[])),'<exact_source_constants_and_raw_NLL>','exec'),namespace)
    loop=compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(selection_loop(tree))],type_ignores=[])),'<exact_source_selection_loop>','exec')
    guard=compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(completion_guard(tree))],type_ignores=[])),'<exact_source_completion_guard>','exec')
    return namespace,loop,guard

def exercise(tree,sequence,expected_epoch,expected_checkpoint_epochs,label):
    env,loop,guard=compiled_path(tree);controls=env['CONTROLS'];output=MemoryOutput();resource_calls=[];score_calls=[]
    states={c:{'best':float('inf'),'best_epoch':None,'steps':0,'history':[]} for c in controls}
    admitted={'seed':101};cfg=copy.deepcopy(j2);seal_ids=set();current={'nll':None}
    def resource_check(*args):resource_calls.append((current['epoch'],args))
    def score(seal,parent,binding,reader,helpers,config):
        assert (seal['path'],seal['sha256']) in seal_ids
        # Ten prior seals must exist before this epoch's first score call.
        assert len(seal_ids)==10
        score_calls.append({'epoch':current['epoch'],'parent_id':parent['parent_id'],'sealed_first':True})
        value=current['nll'];infinite=math.isinf(value)
        return {'parent_id':parent['parent_id'],'groups':{'primary_present_day_64':{'nll':None if infinite else value,'infinite_nll_query_bp':1.0 if infinite else 0.0}},'explicit_handmade_metric_only':True}
    env.update({'states':states,'admission':admitted,'admission_ref':{'path':'explicit_handmade_admission','sha256':'0'*64},'cfg':cfg,'output':output,'storage':{},'lease':None,'_check_resources_v1':resource_check,'score_sealed_v1':score,'reader':None,'helpers':None})
    for epoch,value in enumerate(sequence,1):
        current.update(epoch=epoch,nll=value);seal_ids.clear()
        seals={c:[] for c in controls};parents={p:{'parent_id':p,'split':'val'} for p in ('handmade_val_A','handmade_val_B')}
        for c in controls:
            states[c].update(steps=16*epoch,model=ModelStateStandIn(epoch),optimizer=OptimizerStateStandIn(16*epoch))
            for p in parents:
                record={'path':f'explicit_handmade_memory/{c}.{epoch}.{p}.npz','sha256':hashlib.sha256(f'{c}:{epoch}:{p}'.encode()).hexdigest()}
                seals[c].append((p,record));seal_ids.add((record['path'],record['sha256']))
        env.update(epoch=epoch,seals=seals,val_parents=parents,bindings={'parents':{p:{} for p in parents}},losses={c:[1.0] for c in controls})
        exec(loop,env)
    checkpoint_counts={};terminal={}
    for c,state in states.items():
        check(label+'_'+c+'_selected_epoch',state['best_epoch']==expected_epoch)
        check(label+'_'+c+'_history30_and_scalar_steps480',len(state['history'])==30 and state['steps']==480)
        epochs=[value['epoch'] for name,value in output.files.items() if name.startswith(c+'.best_epoch_')]
        check(label+'_'+c+'_checkpoint_write_epochs',epochs==expected_checkpoint_epochs)
        checkpoint_counts[c]=epochs
        selected=output.files[Path(state['checkpoint']['path']).name] if state.get('checkpoint') else None
        if selected:
            check(label+'_'+c+'_checkpoint_real_body_state_and_optimizer_payload',selected['model_state']['explicit_handmade_epoch_token']==expected_epoch and selected['optimizer_state']['explicit_handmade_step_counter']==expected_epoch*16 and selected['control']==c)
            row=state['history'][expected_epoch-1]
            check(label+'_'+c+'_selected_history_raw_infinite_flag',row['VAL_NLL_is_infinite']==math.isinf(sequence[expected_epoch-1]) and (row['VAL_equal_parent_present64_NLL'] is None)==math.isinf(sequence[expected_epoch-1]))
        env['state']=state
        try:exec(guard,env);terminal[c]='accepted_scalar_fixed_budget_guard'
        except ValueError:terminal[c]='rejected_scalar_fixed_budget_guard'
    check(label+'_all_seals_precede_scores_and_parent_count',len(score_calls)==30*10 and all(x['sealed_first'] for x in score_calls))
    check(label+'_all_control_resource_stub_calls',len(resource_calls)==30*5)
    expected_terminal='rejected_scalar_fixed_budget_guard' if expected_epoch is None else 'accepted_scalar_fixed_budget_guard'
    check(label+'_terminal_expected',set(terminal.values())=={expected_terminal})
    return {'label':label,'epochs':30,'controls':list(controls),'final_scalar_step_count_per_control':480,'expected_selected_epoch':expected_epoch,'selected_epoch_by_control':{c:s['best_epoch'] for c,s in states.items()},'checkpoint_epochs_by_control':checkpoint_counts,'terminal_by_control':terminal,'all_epoch_raw_NLL_inputs':['infinity' if math.isinf(x) else x for x in sequence],'history_row_count_per_control':30,'score_calls':len(score_calls),'memory_output_bytes':output.bytes,'actual_scientific_model_or_optimizer':False,'actual_steps_performed':0,'actual_checkpoint_serialization_or_writer_resource_guards_exercised':False,'selection_loop_AST_sha256':hashlib.sha256(dump(selection_loop(tree)).encode()).hexdigest(),'completion_guard_AST_sha256':hashlib.sha256(dump(completion_guard(tree)).encode()).hexdigest()}
fixtures=[]
fixtures.append(exercise(t2,[float('inf')]*30,1,[1],'v2_all_infinite30epochs'))
fixtures.append(exercise(t2,[float('inf')]*4+[2.0]*26,5,[1,5],'v2_infinity_to_first_finite_then_ties'))
fixtures.append(exercise(t2,[2.0]*30,1,[1],'v2_equal_finite30epoch_ties'))
fixtures.append(exercise(t2,[float('inf'),2.0,float('inf')]+[2.0]*27,2,[1,2],'v2_later_infinite_never_overwrites_finite'))
fixtures.append(exercise(t1,[float('inf')]*30,None,[],'preserved_v1_all_infinite_counterexample'))
fixtures.append(exercise(t1,[float('inf')]*4+[2.0]*26,5,[5],'preserved_v1_infinity_to_finite'))
fixtures.append(exercise(t1,[2.0]*30,1,[1],'preserved_v1_equal_finite_ties'))
env,_,guard=compiled_path(t2)
env.update(state={'steps':479,'best_epoch':1},cfg=j2)
try:exec(guard,env)
except ValueError:checks.append('v2_terminal_guard_still_rejects479_synthetic_steps')
else:raise AssertionError('479-step completion accepted')
namespace,_,_=compiled_path(t2)
check('exact_raw_NLL_function_keeps_infinity',namespace['_nll_v1']({'nll':None,'infinite_nll_query_bp':1.0})==float('inf'))
check('finite_NLL_function_keeps_value',namespace['_nll_v1']({'nll':2.0,'infinite_nll_query_bp':0.0})==2.0)
poolpath=ROOT/'configs/budgeted_shared_pool_v2.json';pool=json.loads(poolpath.read_text());poolsource=ROOT/'src/arggnn/budgeted_shared_pool_v2.py'
check('current_poolv2_allowlist_lacks_fit2','arggnn.census_biological_fit_v2' not in pool['allowed_modules'] and 'arggnn.census_biological_fit_v2' not in pool['additional_worker_contracts'])
check('current_poolv2_config_still_pins_rejected_fit1',pool['additional_worker_contracts']['arggnn.census_biological_fit_v1']['source']=={'path':str(p1.relative_to(ROOT)),'sha256':pins[str(p1.relative_to(ROOT))]})
check('same_tsinfer_consumer_not_added_or_admitted',j2['data_provenance']==j1['data_provenance'] and j2['source_sha256']==j1['source_sha256'])
for p,h in pins.items():check('final_sealed_baseline_'+p,sha(ROOT/p)==h)
for p,h in j2['source_sha256'].items():check('final_unchanged_dependency_'+p,sha(ROOT/p)==h)
delta={'schema':'census_biological_fit_v2_exact_delta_v1','complete_source_expected_transform_equal':True,'source_token_replacements':[{'old':a,'new':b,'count':s1.count(a)} for a,b in versions]+[{'old':predicate[0],'new':predicate[1],'count':1}],'complete_config_expected_transform_equal':True,'config_changed_leaf_paths':['gpu_runtime.launcher_module','kind'],'all_other_config_deep_equal':True,'all_scientific_ASTs_equal_after_only_authorized_inverse_delta':True,'normalized_definition_ASTs':scientific_ast,'baseline_refs':pins,'successor_refs':{str(p.relative_to(ROOT)):sha(p) for p in (p2,c2)},'source_diff':ref(OUT/'source_complete_delta.diff'),'config_diff':ref(OUT/'config_complete_delta.diff'),'separate_inherited_helper_function_class_names_retained':True}
report={'schema':'census_biological_fit_v2_author_checks_v1','status':'passed_bounded_exact_selection_path_and_complete_delta','check_count':len(checks),'checks':checks,'fixtures':fixtures,'baseline_v1_REJECT_preserved':True,'actual_admission':False,'fixture_mode':True,'actual_targets_opened':False,'actual_arrays_read':False,'scientific_model_instantiated':False,'optimizer_instantiated':False,'optimizer_or_fitting_steps_performed':0,'GPU_or_device_query_or_context':False,'runtime_capture_or_install_or_Git':False,'live_ledger_or_lock_access':False,'same_tsinfer_consumer':'separate_unqualified_not_added','poolv2_eligibility':'fit2_absent_from_finite_allowlist_requires_separate_successor','current_poolv2_refs':{'source':ref(poolsource),'config':ref(poolpath)},'delta_proof_ref':{'path':str((OUT/'exact_delta_proof.json').relative_to(ROOT))},'fixture_limits':['Exact checkpoint block uses in-memory stand-ins for model/optimizer state, score records, seals, resource check and output writer.','No optimizer steps, full fit wrapper, actual checkpoint serialization, target/scientific arrays or resource admission are executed.','Synthetic scalar steps equal the fixed budget solely to exercise the unchanged terminal guard.']}
write('exact_delta_proof.json',json.dumps(delta,sort_keys=True,indent=2)+'\n')
report['delta_proof_ref']['sha256']=sha(OUT/'exact_delta_proof.json')
write('selection_checks.json',json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'check_count':len(checks),'fixture_count':len(fixtures),'source_sha256':sha(p2),'config_sha256':sha(c2),'selection_checks_sha256':sha(OUT/'selection_checks.json'),'exact_delta_proof_sha256':sha(OUT/'exact_delta_proof.json')},indent=2))
