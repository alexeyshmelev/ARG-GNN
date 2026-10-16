"""Independent exact v12 VFS boundary cases; no actual resource or guard operation."""
from pathlib import Path
import ast,copy,hashlib,json,sys,types
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
CAND='research/reviews/native_VAL000_full_state_resume_source_20261016_v12'
REVIEW='research/reviews/native_VAL000_full_state_resume_independent_review_20261016_v12_complete_integration'
def save(name,value):
    with (ROOT/REVIEW/name).open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n')
def selected(path,name,namespace):
    tree=ast.parse((ROOT/path).read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name]
    assert len(nodes)==1
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROOT/path),'exec'),namespace)
    return namespace[name]
def rejects(fn):
    try:fn()
    except (ValueError,KeyError):return
    raise AssertionError('Expected rejection')
text=(ROOT/CAND/'fresh_scope_integration_inert_v12.py').read_text();tree=ast.parse(text)
run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
run.body.insert(len(run.body)-1,ast.parse('globals()["_VFS_locals"]=locals()').body[0]);ast.fix_missing_locations(tree)
ns={'__name__':'_independent_native12_VFS_local_capture'}
exec(compile(tree,'<sealed_v12_VFS_with_only_local_capture>','exec'),ns)
reproduction=ns['run']();assert reproduction['check_count']==60
L=ns['_VFS_locals'];c=L['c'];pp=L['pp'];g=L['g'];args=L['args'];checks=[]
def ok(name):checks.append(dict(name=name,status='PASS_SOFTWARE_ONLY'))
guardns=dict(copy=copy,require=ns['require'],_raw=lambda record:None)
bindings=selected(c.FRESH_GUARD_REF['path'],'consumer_bindings_v1',guardns)
view=dict(bindings=copy.deepcopy(L['current']));guard=types.SimpleNamespace(consumer_bindings_v1=bindings)
fresh=c.FRESH_STATE
base=pp.role_union_v11(c,guard,view,fresh[0]['root_stage_request_ref'],fresh[1],fresh[0]['consumer_manifest_ref']);assert len(base)==51
for i in range(13):view['bindings']['extra'+str(i)]=dict(path='VIRTUAL/extra'+str(i)+'.json',sha256='0'*64)
union=pp.role_union_v11(c,guard,view,fresh[0]['root_stage_request_ref'],fresh[1],fresh[0]['consumer_manifest_ref']);assert len(union)==64 and all(r in union for r in view['bindings'].values())
view['bindings']['extra65']=dict(path='VIRTUAL/extra65.json',sha256='1'*64)
rejects(lambda:pp.role_union_v11(c,guard,view,fresh[0]['root_stage_request_ref'],fresh[1],fresh[0]['consumer_manifest_ref']))
ok('actual_native14_binding_body_union51_preserves_extra_roles_to64_rejects65')
rosters=L['rosters'];rosters.clear();rosters.update(copy.deepcopy(L['baseline_rosters']));down=L['downspec']
rosters[down['output_directory']]=[]
prospective=2*(2*2**30+4096)+256*1024
rosters[down['worker_auxiliary_scope']]=[(("cache",),17*2**28-prospective,'regular')]
c.charged_census_v1(down);rejects(lambda:c.charged_census_v1(down,required_write_bytes=1))
assert c.CAPS['reservation_bytes']==5*2**30 and c.CAPS['auxiliary_bytes']==768*2**20
ok('lower_selected_cap_exact_prospective_two_products_actual_AUX_boundary_next_byte_reject')
old=L['old'];oldstorage=L['oldstorage'];storage=c.FRESH_STATE[2];bridge=c.guard_v1({})
events=L['events'];events.clear()
for amount in (0,1,500,8192,65536):
    bridge.budget_check_v9(storage,boundary=True,required_write_bytes=amount)
    assert events[-2:]==[('old_check',dict(boundary=True,required_write_bytes=0)),('fresh_check',dict(boundary=True,required_write_bytes=amount))]
ok('every_reused_bridge_call_rechecks_old_zero_growth_and_exact_fresh_prospective_growth')
old_guard=L['oldguard'];old_check=old_guard.budget_check_v9;fresh_check=g.budget_check_v9
calls=[]
def old_fail(*a,**k):calls.append('old');raise ValueError('synthetic old scope fails')
old_guard.budget_check_v9=old_fail
rejects(lambda:bridge.budget_check_v9(storage,boundary=True,required_write_bytes=42));assert calls==['old']
old_guard.budget_check_v9=old_check
calls=[]
def fresh_fail(*a,**k):calls.append('fresh');raise ValueError('synthetic new scope fails')
g.budget_check_v9=fresh_fail
before=len(events);rejects(lambda:bridge.budget_check_v9(storage,boundary=True,required_write_bytes=43));assert calls==['fresh'] and events[before:]==[('old_check',dict(boundary=True,required_write_bytes=0))]
g.budget_check_v9=fresh_check
ok('old_failure_blocks_new_and_new_failure_never_reuses_previous_success')
original_apply=g.apply_reserve_v1;calls=[]
def race(*a,**kw):calls.append(1);raise ValueError('SYNTHETIC prefix race')
g.apply_reserve_v1=race;args.instance='INDEPENDENT_V12_RACE';args.reservation_bytes=5*2**30
rejects(lambda:pp.prepare_v11(args));assert len(calls)==1 and not any(k.endswith('INDEPENDENT_V12_RACE/reserve_receipt.json') for k in L['files'])
g.apply_reserve_v1=original_apply
ok('actual_ROOT_prepare_prefix_race_no_retry_receipt_or_command')
original_inspect=g.inspect_v1;g.inspect_v1=lambda **kw:dict(projected_bytes=980000000000-5*2**30-2*1048576,bindings=copy.deepcopy(L['current']))
args.instance='INDEPENDENT_V12_NO_CREDIT';before=len(L['dirs']);rejects(lambda:pp.prepare_v11(args));assert len(L['dirs'])==before
g.inspect_v1=original_inspect
ok('actual_current_projection_ceiling_rejects_before_packet_no_future_credit')
assert not any(n in sys.modules for n in ('torch','numpy','composer','torch_geometric'))
save('independent_VFS_boundary_checks.v12.json',dict(status='PASS_SOFTWARE_ONLY',checks=checks,independent_check_groups=len(checks),author_check_count=60,all_mutations_in_RAM=True,real_guard_factories_not_used_by_VFS=True,actual_guards_probes_payload_accounting_admissions_or_jobs=False))
