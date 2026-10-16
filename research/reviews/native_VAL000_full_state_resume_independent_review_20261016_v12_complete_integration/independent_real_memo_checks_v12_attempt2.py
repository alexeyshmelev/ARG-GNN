"""Additional independent tests on real memo + exact factories using tiny synthetic modules."""
from pathlib import Path
import ast,copy,hashlib,json,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
CAND='research/reviews/native_VAL000_full_state_resume_source_20261016_v12'
REVIEW='research/reviews/native_VAL000_full_state_resume_independent_review_20261016_v12_complete_integration'
checks=[]
def require(ok,label):
    if not ok:raise AssertionError(label)
def rejection(label,call):
    try:call()
    except (ValueError,RuntimeError):checks.append(dict(check=label,passed=True));return
    raise AssertionError('Expected reject: '+label)
def extra(L):
    order=L['order'];v4=L['v4'];budget=L['budget'];handle=L['handle'];g=L['g'];scratch=L['scratch'];payload=L['payload'];reservation=L['reservation']
    def file_churn():
        with v4.mutex(scratch):
            a=v4.file_bytes(scratch,'payload.json');b=v4.file_bytes(scratch,'payload.json');require(a==b,'same-operation read')
            payload.write_text('{"independent":true}')
    rejection(order+'_actual_memo_transaction_end_source_identity_churn',file_churn)
    require(handle._state() is None and handle.active_count==0 and handle.last_operation_counts()['success'] is False,'failed operation cleared')
    payload.write_text('{"value":1}')
    link=scratch/'scope_out';target=scratch/'owned_target';target.mkdir()
    def namespace_churn():
        with v4.mutex(scratch):
            budget.successor_paths(scratch,reservation)
            budget.successor_paths(scratch,reservation)
            link.symlink_to(target)
    rejection(order+'_actual_memo_transaction_end_namespace_churn',namespace_churn)
    link.unlink();target.rmdir()
    require(handle._state() is None and handle.active_count==0,'namespace failure cleared')
    def nested():
        with v4.mutex(scratch):
            with v4.mutex(scratch):pass
    rejection(order+'_actual_memo_nested_operation',nested)
    for field,value in [('consumer_role','root_mutation'),('cross_operation_cache',True),('namespace_extension_ref',{'changed':True}),('original_memo2_ref',{'changed':True}),('original_budget_v9_ref',{'changed':True}),('installed',False)]:
        saved=copy.deepcopy(handle.receipt[field]);handle.receipt[field]=value
        rejection(order+'_memo_receipt_drift_'+field,g['_consumer_guards_v12']);handle.receipt[field]=saved
    old_budget=L['freshguard'].budget;L['freshguard'].budget=type('DifferentBudget',(),{'v4':v4})()
    rejection(order+'_cached_budget_singleton_identity_drift',g['_consumer_guards_v12']);L['freshguard'].budget=old_budget
    saved=handle.original['canonical'];handle.original['canonical']=lambda *a:None
    rejection(order+'_cached_original_function_identity_drift',g['_consumer_guards_v12']);handle.original['canonical']=saved
    g['_consumer_guards_v12']()
    with v4.mutex(scratch):
        require(v4.file_bytes(scratch,'payload.json')[0]==b'{"value":1}','new independent operation rereads restored tiny file')
    require(handle._state() is None and handle.active_count==0,'success clears state')
    checks.append(dict(check=order+'_all_failed_and_successful_memo_transactions_clear_state',passed=True))
source=(ROOT/CAND/'dual_guard_memo_inert_v12.py').read_text();tree=ast.parse(source)
scenario=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='scenario')
class AddExtra(ast.NodeTransformer):
    count=0
    def visit_Assign(self,node):
        if any(isinstance(t,ast.Name) and t.id=='handle' for t in node.targets):
            self.count+=1
            return [node,ast.parse('INDEPENDENT_EXTRA(locals())').body[0]]
        return node
inject=AddExtra();inject.visit(scenario)
assert inject.count==1,'Exactly one memo handle capture point'
ast.fix_missing_locations(tree)
ns={'__name__':'_independent_exact_v12_real_memo_extra','INDEPENDENT_EXTRA':extra}
exec(compile(tree,'<sealed_v12_real_factory_fixture_plus_independent_local_tests>','exec'),ns)
ns['HERE']=REVIEW
base=[]
for order in ('ROOT_prepare','execute_initial','worker_entry'):ns['scenario'](order,base)
require(not any(n in sys.modules for n in ('torch','numpy','composer','torch_geometric')),'no scientific imports')
value=dict(status='PASS_SOURCE_ONLY_SOFTWARE_FIXTURE',independent_check_count=len(checks),checks=checks,retained_author_scenario_checks=len(base),exact_pinned_candidate_factories_used=True,exact_pinned_memo_and_namespace_extension_used=True,all_guard_and_budget_modules_synthetic=True,all_operation_paths_owned_tiny_fixtures=True,no_actual_policy_guard_census_lease_capacity_process_accounting_admission_or_job=True)
with (ROOT/REVIEW/'independent_real_memo_checks.v12.json').open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n')
