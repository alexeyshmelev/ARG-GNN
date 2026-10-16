"""Independent source/metadata-only tests. Never import/configure a real guard."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from contextlib import contextmanager
import ast,copy,hashlib,json,types
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE='research/reviews/Relate_failed64GiB_retention_successor_independent_review_20261017_v1'
SRC='research/reviews/Relate_failed64GiB_retention_successor_source_20261017_v1'
SOURCE=SRC+'/relate_failed64_retention_v1.py'
def need(ok,message):
    if not ok:raise AssertionError(message)
def encoded(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v):return hashlib.sha256(encoded(v)).hexdigest()
def ref(p):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def raw(r):
    need(set(r)=={'path','sha256'} and ref(r['path'])==r,'Exact source/metadata SHA')
    return (ROOT/r['path']).read_bytes()
def read(r):return json.loads(raw(r))
def main():
    m=read(ref(SRC+'/source_manifest.v1.json'));h=read(ref(SRC+'/source_handoff.v1.json'))
    exact={'relate_failed64_retention_v1.py':'b8b696a0df45c12b931e563600c2b52e89e73dd19c7d160351063687dcfb5164',
        'source_manifest.v1.json':'8c9339104cff95bbada5f8a36c41908ede6b2eccb6aef53640a913de14ad2774',
        'source_handoff.v1.json':'5c493e4a6196b11520a389217197d29520d3492c72b8756cd1f6dc19e1a51b69'}
    for p,sha in exact.items():need(ref(SRC+'/'+p)['sha256']==sha,'Exact assigned seal '+p)
    for p,sha in m['source_sha256'].items():raw(dict(path=p,sha256=sha))
    for r in m['runtime_authenticated_historical_source_and_metadata_refs']:raw(r)
    need(h['source_manifest_ref']==ref(SRC+'/source_manifest.v1.json'),'Handoff joins exact manifest')
    source=(ROOT/SOURCE).read_text();tree=ast.parse(source)
    old=ast.parse(raw(m['predecessor_source_ref']).decode())
    funcs={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    oldfuncs={n.name:n for n in old.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    unchanged=[n for n in sorted(funcs.keys()&oldfuncs.keys()) if ast.dump(funcs[n],include_attributes=False)==ast.dump(oldfuncs[n],include_attributes=False)]
    for n in ('PilotOutputV9','charge','consumer_bindings_v1','check_storage_locked_v1','_proof','_absent','_retained_measurement','_validate_targets','_observe_targets','_recorded_stopped_proof'):
        need(n in unchanged,'Inherited guard unchanged '+n)
    env=dict(copy=copy,datetime=datetime,timezone=timezone,Path=Path,ROOT=ROOT,SOURCE=SOURCE,
        MANIFEST=SRC+'/source_manifest.v1.json',MEASURE_SOURCE=m['measurement_source_ref']['path'],
        MODE='Relate_failed64GiB_retention_v1',JOURNAL='.runtime/storage/Relate_failed64GiB_retention_v1',
        BASE_REF=m['predecessor_source_ref'],BASE_POLICY_REF=m['predecessor_policy_ref'],BASE_SELECTOR_REF=m['predecessor_selector_ref'],
        BASE_MODE='native7_H050_stopped_retention_v1',TARGET_KEYS=tuple(m['target_order']),
        PARENT_STOPPED_KEYS=('native32/native-VAL000-full227990-borrowed32-GPU0-v1','H050/H050-direct-readout-GPU0-TRAIN-v1'),
        MALFORMED_KEY='H039/directed-clade-exchange-train-stage-seed101-v1',CEILING_BYTES=980000000000,
        HUMAN_LIMIT_BYTES=1000000000000,WARNING_BYTES=950000000000,BOOTSTRAP_CHARGE=2097152,META_CHARGE=524288,
        OPERATION_WALL_SECONDS=300,RETIRE_OPERATION='retire_exact_failed_Relate64_retention_v1',
        _ref=ref,_raw=raw,_json=read,_completed_json=read,_POLICY_REF=None,_SELECTOR_REF=None)
    env['ORDINARY_EXCLUSIONS']=env['TARGET_KEYS']+env['PARENT_STOPPED_KEYS']
    def integer(v,name,minimum=0):
        if type(v) is not int or v<minimum:raise ValueError(name)
        return v
    env['budget']=types.SimpleNamespace(SOURCE_REL='scripts/project_budget_v9.py',LAUNCHER_REL='scripts/launch_reserved_gpu_v6.py',
        digest=digest,v4=types.SimpleNamespace(integer=integer,encoded=encoded))
    names=('require','charge','warning_fields','root_activation_draft_v1','activation_policy_draft_v1','_policy',
        'selector_draft_v1','_selected_record','_manifest','_target','_historical_stop','operation_draft_v1','apply_retire_stopped_v1','assert_open')
    exec(compile(ast.Module(body=[funcs[n] for n in names],type_ignores=[]),'<isolated_exact_source_functions>','exec'),env)
    need(env['_manifest']()==m,'Actual sealed manifest validates via metadata-only AST extraction')
    target=m['targets'][env['TARGET_KEYS'][0]]
    need(env['_historical_stop'](target) is True,'Genuine exact failed outer/direct lineage')
    need(target['outer_terminal_ref']['sha256']=='ddd7ca96329de7219a8c49ef660fa05cbf417e74f1d1a6cd498c3ecc10ef63aa','Actual outer41415 digest')
    negative=[]
    def reject(f,label):
        try:f()
        except (ValueError,KeyError,TypeError,AssertionError,OSError):negative.append(label);return
        raise AssertionError('Accepted invalid isolated fixture: '+label)
    # Synthetic JSON store. All fixture refs are metadata identities; no files created.
    fixtures={}
    def synthetic(name,value):
        r=dict(path=HERE+'/SYNTHETIC_NOT_AUTHORITY_'+name+'.json',sha256=digest(value));fixtures[r['path']]=copy.deepcopy(value);return r
    actual_read=read;actual_raw=raw
    def fixture_raw(r):
        if r['path'] in fixtures:
            need(digest(fixtures[r['path']])==r['sha256'],'Synthetic exact fixture SHA');return encoded(fixtures[r['path']])
        return actual_raw(r)
    env['_json']=lambda r:json.loads(fixture_raw(r));env['_raw']=fixture_raw
    repair=read(m['predecessor_policy_ref'])['marker_repair_authority_ref']
    pins=dict(base_pins={'synthetic_only':True},completed_overlay_prefix=dict(events=0,tail_sha256=None));projection=800000000000
    review=synthetic('review',dict(independent=True,verdict='SOURCE_PASS',blockers=[],source_ref=ref(SOURCE),
        source_manifest_ref=ref(env['MANIFEST']),measurement_source_ref=ref(env['MEASURE_SOURCE'])))
    rev=synthetic('parent_revocation',dict(schema='Relate64_parent_future_mutation_revocation_v1',root_authorizer='root_agent',active=True,
        parent_source_ref=env['BASE_REF'],parent_policy_ref=env['BASE_POLICY_REF'],parent_selector_ref=env['BASE_SELECTOR_REF'],
        actual_parent_pins=pins,all_future_parent_reserve_close_retire_mutations_permanently_cooperatively_revoked=True,
        existing_parent_read_check_and_fully_charged_writers_unchanged=True,no_historical_mutation=True))
    activation=env['root_activation_draft_v1'](base_pins=pins,base_projected_bytes=projection,permanent_revocation_ref=rev,source_review_ref=review)
    activation['active']=True;actref=synthetic('activation',activation)
    bp=read(m['predecessor_policy_ref'])
    policy=env['activation_policy_draft_v1'](repair_authority_ref=repair,base_pins=pins,base_projected_bytes=projection,
        implementation_ref=bp['implementation_ref'],launcher_ref=bp['launcher_ref'],root_activation_ref=actref,permanent_revocation_ref=rev,source_review_ref=review)
    policy['active']=True;polref=synthetic('policy',policy)
    selector=env['selector_draft_v1'](policy_ref=polref,source_review_ref=review,repair_authority_ref=repair,
        frozen_parent_lineage=dict(base_ref=env['BASE_REF'],base_policy_ref=env['BASE_POLICY_REF'],base_pins=pins,
            base_projected_bytes=projection,base_accounting_mode=env['BASE_MODE']))
    selector['active']=True;selref=synthetic('selector',selector)
    env.update(_POLICY_REF=polref,_SELECTOR_REF=selref,_REPAIR_AUTHORITY_REF=repair)
    need(env['_policy']()==policy and env['_selected_record'](selref)==selector,'Exact isolated activation+selector joins')
    for field,bad in [('active',False),('base_ref',dict(path='wrong.py',sha256='a'*64)),('scheduling_ceiling_bytes',1000000000001),
        ('no_deletion_or_rename_or_normalization',False),('existing_parent_consumers_never_upgraded_or_rebound',False)]:
        invalid=copy.deepcopy(policy);invalid[field]=bad;r=synthetic('bad_policy_'+field,invalid);env['_POLICY_REF']=r
        reject(env['_policy'],'policy_'+field)
    env['_POLICY_REF']=polref
    for field,bad in [('active',False),('accounting_mode','native7_H050_stopped_retention_v1'),('one_new_direct_guard_selector_capsule',False),
        ('source_review_ref',dict(path='missing.json',sha256='b'*64)),('frozen_parent_lineage',{})]:
        invalid=copy.deepcopy(selector);invalid[field]=bad;r=synthetic('bad_selector_'+field,invalid)
        reject(lambda r=r:env['_selected_record'](r),'selector_'+field)
    # Isolated entrypoint fixture: all state, mutex, targeted observations, helper,
    # and append operations are stubs. No real guard/module/helper is imported.
    log=[];clock=[100.0];fault=[None];auth_expired=[False]
    @contextmanager
    def mutex(root):log.append('synthetic_mutex');yield
    env['budget'].v4.mutex=mutex
    env['time']=types.SimpleNamespace(monotonic=lambda:clock[0])
    own=target['original_reservation'];key=own['key'];binding={}
    state=({key:own},set(),dict(new_reserved_bytes=0,completed_credit_bytes=0),binding,[],[],{})
    env['_state_locked']=lambda *a,**k:state
    env['_authority']=lambda *a,**k:dict(request=dict(key=key,revocation_ref={'synthetic':1},observation_ref={'synthetic':2}),expected_before='pins')
    env['_pins']=lambda v:'pins';env['_policy']=lambda:dict(base_projected_bytes=projection)
    env['consumer_bindings_v1']=lambda *a:[]
    def proof(*a,**k):log.append('fresh_target_check' if k.get('observe') else 'metadata_proof');return {}
    env['_stopped_proof']=proof;env['_target']=lambda k:target;env['_paths']=lambda o:target['scope_paths']
    def measurement(*a,**k):
        log.append('complete_measurement_stub')
        need(k['socket_is_absent']('/SYNTHETIC') is False,'Every socket denied')
        if fault[0]=='measurement':raise ValueError('synthetic changed metadata')
        if fault[0]=='timeout':clock[0]=401.0
        return dict(synthetic_measurement_only=True)
    helper=types.SimpleNamespace(require=env['require'],measure_retained_nodes_v1=measurement)
    env['importlib']=types.SimpleNamespace(util=types.SimpleNamespace(spec_from_file_location=lambda *a:types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda h:None)),module_from_spec=lambda s:helper))
    env['_retained_measurement']=lambda v,o:8192
    def observe(t):
        log.append('fresh_post_target_check')
        if fault[0]=='post_target':raise ValueError('synthetic PID reused')
        return {}
    env['_observe_targets']=observe;env['_utc']=lambda:datetime.now(timezone.utc).isoformat()
    env['_recorded_stopped_proof']=lambda *a:log.append('validate_chronology')
    def expire(a):
        log.append('final_authority_expiry_check')
        if fault[0]=='expired':raise ValueError('synthetic authority expired')
    env['_expire']=expire
    env['_append']=lambda *a:log.append('SYNTHETIC_APPEND_ONLY') or 'd'*64
    result=env['apply_retire_stopped_v1']({'synthetic_authority':True},expected_source_sha256='a',expected_launcher_sha256='b')
    need(result['scientific_success_asserted'] is False and result['artifacts_deleted']==0 and result['old_scope_mutated'] is False,'Distinct accounting-only return')
    need(log.index('fresh_target_check')<log.index('complete_measurement_stub')<log.index('fresh_post_target_check')<log.index('validate_chronology')<log.index('final_authority_expiry_check')<log.index('SYNTHETIC_APPEND_ONLY'),'All mandatory facts precede synthetic append')
    order=copy.deepcopy(log)
    for name in ('measurement','post_target','timeout','expired'):
        fault[0]=name;clock[0]=100.0;log.clear()
        reject(lambda:env['apply_retire_stopped_v1']({},expected_source_sha256='a',expected_launcher_sha256='b'),'no_append_on_'+name)
        need('SYNTHETIC_APPEND_ONLY' not in log,'Failure precedes append '+name)
    return dict(schema='Relate64_independent_source_metadata_fixtures_v1',status='PASS_SOURCE_AND_ISOLATED_METADATA_ONLY',
        exact_source_manifest_handoff_sha256=exact,own_source_refs_authenticated=len(m['source_sha256']),historical_refs_authenticated=len(m['runtime_authenticated_historical_source_and_metadata_refs']),
        all_original_55_direct_refs_authenticated=len(target['all_original_binding_refs'])==55,unchanged_predecessor_functions_and_class=unchanged,
        actual_manifest_validator_and_historical_stop_exercised_via_AST=True,synthetic_activation_policy_selector_positive=True,
        synthetic_entrypoint_order=order,negative_isolated_fixtures=negative,
        real_guard_imports_configurations_factories_mutexes_probes_resource_scans_payloads_signals_mutations_credits_or_Git=0)
if __name__=='__main__':
    value=main();data=json.dumps(value,indent=2,sort_keys=True)+'\n'
    with (ROOT/HERE/'independent_fixtures.result.v1.json').open('x') as f:f.write(data)
    print(data)
