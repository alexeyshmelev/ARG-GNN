"""Source-only typed H039 v1 failure capacity successor.

Root may activate only after every cooperative old-guard consumer/supervisor is
terminal. Preserve all prior sources, journal prefixes and failed artifacts.
No action on import. H039 v2 is deliberately outside this contract.
"""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from arggnn import project_storage_terminal_retired_scopes_v1 as previous

old = previous.old
budget = previous.budget
ROOT = previous.ROOT
SOURCE = 'research/reviews/H039_failed_scope_capacity_v1/failed_scope_capacity_v1.py'
JOURNAL = previous.JOURNAL
MODE = 'journal_H039_failed_scope_capacity_v1'
PREVIOUS_SHA256 = 'bbbdf26206e4e1725c79d2558b4426af848df4cbbd5ac2d0f2bc7051c0606bab'
METADATA_CHARGE = previous.METADATA_CHARGE
V9_CONSUMER = dict(path='src/arggnn/project_budget_v9_consumer.py',
                   sha256='c11a5543746899c8a387508e9877ce6ae2bf80fe4d3076c1dbe1cb8f7a43b859')
KEY = 'H039/directed-clade-exchange-full-panel-seed101'
OUTPUT = 'experiments/results/directed_clade_exchange_full_panel_v1_seed101_GPU2'
FOLDER = 'research/reviews/expanded_full_panel_v1_root/H039_shared_continuation_v1/'
PIDS = [402228, 402352]
ARMS = ['plain_continuation', 'generic_unary', 'directed']
EVIDENCE = {
 'input': (FOLDER+'input.json', 'b6332d88a12ce8caab8b2ffa3876180878b6391e74e781f58d507169cd74c196'),
 'storage': (FOLDER+'storage.json', '357420792bbe873ab1b309fc9c2b12c2c9617aff68a3df81658b9b91dd3b58f9'),
 'runtime': (FOLDER+'runtime.json', 'f61fd7a49fa981c288702dbb03cb752ce15fd5d75e957d74def9915a8e05450e'),
 'launch': (FOLDER+'launch.json', 'e29ba519b96068b55086c98b56f32a582f8bd3b84620d1f05da249aa16633420'),
 'wait': (FOLDER+'root_wait.json', 'd582e9826ca13e9d35f6e3a70054e1bf91c173d9ff53d64df4ce8ca49702891f'),
 'log': (FOLDER+'launch.log', '02f03f7e08821b3435be5b321350e530fbd3cfdb70a85e7204eeca23839913e0'),
 'started': (OUTPUT+'/started.json', '4410facce7a94da305a4dd96c0631dd8d7d5ef58d4952c23c11263e24220382c'),
 'failed': (OUTPUT+'/failed.json', 'de9244840fbfde32688bca594ba2ed6b6d26bc7980bb0d7ce364eb043511fa45'),
 'producer': ('src/arggnn/train_directed_clade_exchange_full_panel_v1.py', 'f387ea88868fed1090190da894ca19bb5b1aeffe80ca9239a9222164e2b86d3a'),
 'continuation': ('src/arggnn/directed_clade_exchange_continuation_v1.py', 'ea0be2a4d9c8e6e750d6f5166038d2197a72fccd2ee74a22cbbfc55a338b8b0b'),
}
_CACHE = None


def _utc(): return datetime.now(timezone.utc).isoformat()
def _ref(path): return budget.v4.reference(ROOT, path)
def _bound(ref): return budget.v4.bound(ROOT, ref, True)
def _pins(view): return previous._pins(view)
def _paths(own): return previous._paths(own)
def _clone(method, **changes): return previous._clone(method, **changes)
def _ceiling(value, view): return previous._ceiling(value, view)


def _frozen():
    budget.v4.bound(ROOT, dict(path=previous.SOURCE, sha256=PREVIOUS_SHA256))
    return previous._frozen()


def retirement_request_v1(observation_ref):
    """Pure request construction. Root must create a genuine observation receipt."""
    return dict(key=KEY, evidence_refs={k:dict(path=p,sha256=h) for k,(p,h) in EVIDENCE.items()},
                observation_ref=copy.deepcopy(observation_ref))


def _barrier(ref, *, observe=False):
    value = _bound(ref)
    fields = {'kind','root_authorizer','all_old_guard_consumers_and_supervisors_terminal',
              'all_old_guard_future_writes_and_relaunch_revoked','root_owned_process_roster_complete',
              'pids','observations','launch_refs'}
    if (set(value) != fields or value['kind'] != 'H039_failed_scope_capacity_activation_barrier_v1'
            or value['root_authorizer'] != 'root_agent'
            or any(value[k] is not True for k in ('all_old_guard_consumers_and_supervisors_terminal',
                'all_old_guard_future_writes_and_relaunch_revoked','root_owned_process_roster_complete'))
            or not isinstance(value['pids'],list) or not value['pids'] or len(value['pids'])>256
            or any(type(p) is not int or p<2 for p in value['pids'])
            or len(set(value['pids'])) != len(value['pids'])
            or not isinstance(value['launch_refs'],list) or not value['launch_refs']
            or len(value['launch_refs'])>256 or len(value['observations']) != len(value['pids'])):
        raise ValueError('Genuine complete root old-guard terminal/revocation barrier required')
    for record in value['launch_refs']:
        if Path(record['path']).suffix != '.json': raise ValueError('Recorded launch metadata only')
        _bound(record)
    for pid,row in zip(value['pids'],value['observations']): old._observation(row,pid)
    return [_frozen()['_absent'](p) for p in value['pids']] if observe else value['observations']


def _authority(ref, operation, bindings, *, fresh):
    value = _bound(ref)
    required = dict(kind='H039_failed_scope_capacity_root_admission_v1',root_authorizer='root_agent',
        active=True,operation=operation,accounting_mode=MODE,bindings=bindings,
        all_budget_writers_use_scheduler_mutex=True,
        all_consumers_and_launchers_use_H039_failed_scope_guard=True,
        all_old_guard_consumers_and_supervisors_terminal_before_activation=True,
        historical_records_mutation_authorized=False,global_rebase_authorized=False,
        scientific_execution_authorized=False,GPU_authorized=False)
    if (operation not in ('reserve','retire_failed_H039_v1')
            or any(budget.v4.encoded(value.get(k)) != budget.v4.encoded(v) for k,v in required.items())):
        raise ValueError('Exact typed root authority and completed old-guard barrier required')
    _barrier(value['activation_barrier_ref'])
    if operation == 'retire_failed_H039_v1' and any(value.get(k) is not True for k in (
            'scope_write_authority_revoked_for_all_cooperative_project_jobs',
            'closed_scope_future_writes_and_relaunch_revoked',
            'all_original_and_resume_capabilities_for_retired_scope_revoked',
            'preserve_all_failed_scope_artifacts','external_NFS_writers_outside_observation')):
        raise ValueError('Explicit failed-scope preservation and write/relaunch revocation required')
    if operation == 'reserve' and any(r not in value['reservation']['consumer_bindings'] for r in [*bindings.values(),V9_CONSUMER]):
        raise ValueError('Every historical/new budget source must be reserved')
    if fresh:
        expiry = datetime.fromisoformat(value['expires_at_utc'].replace('Z','+00:00'))
        if expiry.tzinfo is None or expiry <= datetime.now(timezone.utc): raise ValueError('Expired root authority')
    return value


def _mixed_events(bindings):
    """Validate all four immutable phases in the original terminal journal."""
    directory = budget.v4.canonical(ROOT,JOURNAL)
    if not directory.exists(): return [],[],[],[],None
    if not directory.is_dir(): raise ValueError('Canonical shared journal required')
    before = directory.lstat(); names = sorted(p.name for p in directory.iterdir())
    if len(names)>4096: raise ValueError('Shared event cap exceeded')
    retired = {k:v for k,v in bindings.items() if k!='H039_failed_scope_overlay'}
    success = {k:v for k,v in retired.items() if k!='terminal_retired_overlay'}
    terminal = {k:v for k,v in success.items() if k!='terminal_success_overlay'}
    phases = [terminal,success,retired,bindings]
    operations = [('reserve','close_failed_cpu'),('reserve','close_completed_cpu_observed_absent'),
                  ('reserve','retire_obsolete_scope'),('reserve','retire_failed_H039_v1')]
    authorities = [old._authority,previous.previous._authority,previous._authority,_authority]
    rows = [[],[],[],[]]; phase = 0; tail = None; total = 0
    for number,name in enumerate(names,1):
        if name != f'{number:06d}.json': raise ValueError('Shared journal gap/unknown entry')
        raw,sha = budget.v4.file_bytes(ROOT,directory/name); total += len(raw)
        if len(raw)>METADATA_CHARGE or total>32*1024*1024: raise ValueError('Charged shared metadata cap exceeded')
        event = budget.v4.parse(raw)
        if (set(event) != {'schema','sequence','previous_sha256','bindings','operation','authority','value'}
                or event['schema'] != 'project_budget_terminal_overlay_event_v1'
                or type(event['sequence']) is not int or event['sequence'] != number or event['previous_sha256'] != tail):
            raise ValueError('Immutable shared chain drift')
        index = next((i for i,b in enumerate(phases) if event['bindings']==b),None)
        if index is None or index<phase or event['operation'] not in operations[index]:
            raise ValueError('Exact successor phase required; no old writer after activation')
        phase = index
        authority = authorities[index](event['authority'],event['operation'],phases[index],fresh=False)
        if authority.get('expected_before',{}).get('terminal_overlay_tail_sha256') != tail:
            raise ValueError('Shared authority predecessor drift')
        rows[index].append(event); tail = sha
    after = directory.lstat()
    if (before.st_dev,before.st_ino,before.st_mtime_ns,before.st_ctime_ns) != (
            after.st_dev,after.st_ino,after.st_mtime_ns,after.st_ctime_ns):
        raise ValueError('Shared journal changed during read')
    return *rows,tail


def _failed_H039(request, own, *, observe=False):
    expected = retirement_request_v1(request.get('observation_ref'))
    if (request != expected or request['key'] != own['key'] or own['key'] != KEY
            or own['reserved_bytes'] != 10000000000 or own['output'] != str(ROOT/OUTPUT)
            or own['auxiliary_scopes'] != []):
        raise ValueError('Only exact H039 v1 failed10GB reservation may be retired')
    _paths(own)
    e = {}
    for name,record in request['evidence_refs'].items():
        raw = budget.v4.bound(ROOT,record)
        e[name] = budget.v4.parse(raw) if name not in ('producer','continuation','log') else raw.decode()
    spec,storage,launch,wait,started,failed,runtime = (e[k] for k in ('input','storage','launch','wait','started','failed','runtime'))
    if (storage['reservation'] != own or spec['refs']['storage'] != request['evidence_refs']['storage']
            or spec['refs']['runtime'] != request['evidence_refs']['runtime']
            or spec['output_directory'] != OUTPUT or runtime['output_directory'] != OUTPUT
            or runtime['device'] != 'cuda' or runtime['physical_GPU'] != 2
            or launch['pid'] != PIDS[0] or launch['gpu'] != 2
            or launch['input_ref'] != request['evidence_refs']['input']
            or wait['actual_direct_launcher_wait'] is not True or wait['pid'] != PIDS[0]
            or type(wait['returncode']) is not int or wait['returncode'] != 1 or wait['completed_ref'] is not None):
        raise ValueError('Exact original source/input/runtime/storage/direct exit1 wait required')
    for name in ('producer','continuation'):
        r = request['evidence_refs'][name]
        if spec['source_sha256'].get(r['path']) != r['sha256'] or r not in own['consumer_bindings']:
            raise ValueError('Exact bound original pre-optimizer failure source required')
    if (failed['status'] != 'failed_partial_preserved' or failed['error_type'] != 'TypeError'
            or failed['error'] != "unhashable type: 'dict'" or failed['automatic_replay'] is not False
            or failed['retained_selected_records'] != {a:[] for a in ARMS}
            or any(e[k]['input_ref'] != request['evidence_refs']['input'] or e[k]['hypothesis_id']!='H039'
                   or e[k]['seed']!=101 or e[k]['arms']!=ARMS or e[k]['runtime']['pid']!=PIDS[1]
                   for k in ('started','failed'))
            or failed['runtime']['lease']['launcher_metadata']['launcher_pid'] != PIDS[0]):
        raise ValueError('Genuine preserved pre-optimizer H039 v1 failure required')
    log = e['log']
    if ('train_directed_clade_exchange_full_panel_v1.py", line 182, in run_v1' not in log
            or 'directed_clade_exchange_continuation_v1.py", line 23, in load_selected_warmstart_v1' not in log
            or "TypeError: unhashable type: 'dict'" not in log
            or json.loads(log.splitlines()[-1]) != dict(event='supervisor_wait',worker_pid=PIDS[1],returncode=1,wait_status_known=True)):
        raise ValueError('Actual warmstart failure traceback and direct worker exit1 wait required')
    terminal = _bound(request['observation_ref'])
    fields = {'kind','root_authorizer','key','evidence_refs','pids','observations','all_original_and_resume_capabilities_revoked'}
    if (set(terminal)!=fields or terminal['kind']!='H039_v1_failed_scope_terminal_observation_v1'
            or terminal['root_authorizer']!='root_agent' or terminal['key']!=KEY
            or terminal['evidence_refs']!=request['evidence_refs'] or terminal['pids']!=PIDS
            or len(terminal['observations'])!=len(PIDS) or terminal['all_original_and_resume_capabilities_revoked'] is not True):
        raise ValueError('Exact root H039 launcher/worker absence and revoked-capability receipt required')
    finished = datetime.fromisoformat(wait['finished_utc'].replace('Z','+00:00'))
    if finished.tzinfo is None: raise ValueError('Timezone-aware actual root wait required')
    for pid,row in zip(PIDS,terminal['observations']):
        if old._observation(row,pid)<finished: raise ValueError('Root H039 absence observation predates direct wait')
    # Pinned producer fails at182; family/optimizers are first built at185 and
    # continuation is first called at231. Failure is not inferred from no checkpoint.
    return dict(failure_kind='H039_v1_warmstart_TypeError_before_optimizer_construction',
        actual_direct_launcher_wait=True,launcher_returncode=1,actual_worker_wait=True,worker_returncode=1,
        family_and_optimizers_constructed=False,additional_optimizer_updates_per_arm={a:0 for a in ARMS},
        total_additional_optimizer_updates=0,scientific_completion_claimed=False,preserve_all_artifacts=True,
        PID_observations=[_frozen()['_absent'](p) for p in PIDS] if observe else terminal['observations'])


def _measurement(value, own):
    if (set(value) != {'conservative_scope_bytes','entries','scope_paths','stat_only'} or value['stat_only'] is not True
            or value['scope_paths'] != [str(p) for p in _paths(own)]):
        raise ValueError('Exact one owned-scope retained-artifact stat certificate required')
    actual = budget.v4.integer(value['conservative_scope_bytes'],'H039 retained bytes')
    budget.v4.integer(value['entries'],'H039 retained entries')
    if actual>own['reserved_bytes']: raise ValueError('Retained failed scope exceeds original allowance')
    return actual


def _state_locked(source_sha, launcher_sha, *, extra=(), fresh=False):
    global _CACHE
    cpu = _frozen()
    bindings = dict(implementation=dict(path=budget.SOURCE_REL,sha256=source_sha),
        launcher=dict(path=budget.LAUNCHER_REL,sha256=launcher_sha),
        cpu_overlay=dict(path=old.CPU_SOURCE,sha256=old.CPU_SHA256),
        terminal_overlay=dict(path=old.SOURCE,sha256=previous.previous.OLD_SHA256),
        terminal_success_overlay=dict(path=previous.previous.SOURCE,sha256=previous.PREVIOUS_SHA256),
        terminal_retired_overlay=dict(path=previous.SOURCE,sha256=PREVIOUS_SHA256),H039_failed_scope_overlay=_ref(SOURCE))
    deps = _CACHE[3] if _CACHE is not None else ()
    watched = sorted(set(extra)|set(deps)|{SOURCE,previous.SOURCE,JOURNAL})
    before = cpu['_fingerprint'](watched)
    if not fresh and _CACHE is not None and _CACHE[0]==bindings and _CACHE[1]==before: return copy.deepcopy(_CACHE[2])
    first,second,third,suffix,tail = _mixed_events(bindings); prefix=first+second+third
    prefix_tail = budget.v4.file_bytes(ROOT,ROOT/JOURNAL/f'{len(prefix):06d}.json')[1] if prefix else None
    base_reader = _clone(previous._state_locked,_CACHE=None,
                         _mixed_events=lambda bindings:(first,second,third,prefix_tail))
    reservations,closed,base,_,_ = base_reader(source_sha,launcher_sha,extra=extra,fresh=fresh)
    reservations=copy.deepcopy(reservations);closed=set(closed)
    legacy=budget.v4.read_json(ROOT,budget.v4.LEGACY)[0]
    occupied=[budget.v4.canonical(ROOT,r['output']) for r in legacy['jobs'].values()]
    for own in reservations.values():occupied.extend(_paths(own))
    discovered=set(base_reader.__globals__['_CACHE'][3])|set(extra)|{SOURCE,previous.SOURCE,JOURNAL}
    def collect(value):
        if isinstance(value,dict):
            if set(value)=={'path','sha256'}:discovered.add(value['path'])
            for child in value.values():collect(child)
        elif isinstance(value,(list,tuple)):
            for child in value:collect(child)
    credit=added=0
    for event in suffix:
        authority=_authority(event['authority'],event['operation'],bindings,fresh=False);value=event['value']
        collect(event);collect(authority);collect(_bound(authority['activation_barrier_ref']))
        if event['operation']=='reserve':
            request=authority['reservation'];paths=_paths(value);key=value['key']
            if (value!={**request,'root_admission':event['authority']} or len(reservations)>=4096
                    or key in reservations or key in legacy['jobs']
                    or any(budget.v4.paths_overlap(a,b) for a in paths for b in occupied)):
                raise ValueError('New reservation authority/key/scope collision')
            reservations[key]=value;occupied.extend(paths);added+=value['reserved_bytes']
        else:
            request=authority['retirement'];key=request['key'];own=reservations.get(key)
            if (own is None or key in closed or set(value)!={'request','reservation_sha256','measurement','failure_proof',
                    'retired_utc','future_writes_and_relaunch_revoked'} or value['request']!=request
                    or value['reservation_sha256']!=budget.digest(own) or value['future_writes_and_relaunch_revoked'] is not True):
                raise ValueError('Unknown/duplicate failed-scope retirement or certificate drift')
            proof=_failed_H039(request,own);recorded=value['failure_proof']
            if set(recorded)!=set(proof) or any(recorded[k]!=proof[k] for k in proof if k!='PID_observations'):
                raise ValueError('Honest exact failed-H039 no-update proof drift')
            if len(recorded['PID_observations'])!=len(PIDS): raise ValueError('H039 PID proof count drift')
            for current,root in zip(recorded['PID_observations'],proof['PID_observations']):
                if old._observation(current,root['pid'])<old._observation(root,root['pid']):
                    raise ValueError('Applied PID observation predates root observation')
            credit+=own['reserved_bytes']-_measurement(value['measurement'],own);closed.add(key)
            collect(_bound(request['observation_ref']))
    projected=base['projected_bytes']+added-credit+len(suffix)*METADATA_CHARGE;_ceiling(projected,base)
    view=dict(base,original_terminal_retired_projected_bytes=base['projected_bytes'],projected_bytes=projected,
        H039_failed_scope_allowance_credit_bytes=credit,H039_failed_scope_reserved_bytes=added,
        H039_failed_scope_metadata_charge_bytes=len(suffix)*METADATA_CHARGE,
        terminal_overlay_tail_sha256=tail,terminal_overlay_events=len(prefix)+len(suffix),
        H039_failed_scope_events=len(suffix),accounting_mode=MODE)
    result=reservations,closed,view,bindings,prefix+suffix
    if before!=cpu['_fingerprint'](watched):raise ValueError('Shared accounting namespace changed during replay')
    _CACHE=bindings,cpu['_fingerprint'](sorted(discovered)),copy.deepcopy(result),tuple(sorted(discovered))
    return result


def inspect_v1(*, expected_source_sha256, expected_launcher_sha256):
    with budget.v4.mutex(ROOT):return _state_locked(expected_source_sha256,expected_launcher_sha256,fresh=True)[2]


def _append(operation, authority_ref, value, state):return old._append(operation,authority_ref,value,state)


def apply_retire_failed_H039_v1(authority_ref, *, expected_source_sha256, expected_launcher_sha256):
    with budget.v4.mutex(ROOT):
        state=_state_locked(expected_source_sha256,expected_launcher_sha256,fresh=True)
        reservations,closed,view,bindings,_=state
        authority=_authority(authority_ref,'retire_failed_H039_v1',bindings,fresh=True)
        if authority['expected_before']!=_pins(view):raise ValueError('Fresh immutable shared-journal pins required')
        _barrier(authority['activation_barrier_ref'],observe=True)
        request=authority['retirement'];key=request['key']
        if key not in reservations or key in closed:raise ValueError('Unknown/already retired H039 v1')
        own=reservations[key];proof=_failed_H039(request,own,observe=True)
        measurement=budget.v4.measure_scopes(ROOT,_paths(own))
        actual=_measurement(measurement,own)
        for row in proof['PID_observations']:_frozen()['_absent'](row['pid'])
        _barrier(authority['activation_barrier_ref'],observe=True)
        projected=view['projected_bytes']-(own['reserved_bytes']-actual)+METADATA_CHARGE;_ceiling(projected,view)
        value=dict(request=copy.deepcopy(request),reservation_sha256=budget.digest(own),measurement=measurement,
            failure_proof=proof,retired_utc=_utc(),future_writes_and_relaunch_revoked=True)
        tail=_append('retire_failed_H039_v1',authority_ref,value,state)
        return dict(kind='H039_v1_failed_scope_retirement_receipt_v1',key=key,value=value,retained_bytes=actual,
            allowance_bytes=own['reserved_bytes'],credit_bytes=own['reserved_bytes']-actual,projected_bytes=projected,
            terminal_overlay_tail_sha256=tail,scope_measurements=1,scientific_completion_claimed=False,
            original_records_mutated=False,artifacts_deleted=0,global_process_scan_performed=False)


def _adapt(name):
    return _clone(_frozen()[name],SOURCE=SOURCE,MODE=MODE,JOURNAL=JOURNAL,_state_locked=_state_locked,_pins=_pins,
        _paths=_paths,_authority=_authority,_append=_append,budget_check_v9=budget_check_v9,
        check_storage_locked_v1=check_storage_locked_v1)


def apply_reserve_v1(authority_ref, *, expected_source_sha256, expected_launcher_sha256):
    value=_adapt('apply_reserve_v1')(authority_ref,expected_source_sha256=expected_source_sha256,
                                   expected_launcher_sha256=expected_launcher_sha256)
    value['kind']='H039_failed_scope_capacity_reserve_receipt_v1'
    value['terminal_overlay_tail_sha256']=value.pop('cpu_overlay_tail_sha256')
    return value


def check_storage_locked_v1(storage, *, first=False, boundary=False, required_write_bytes=0):
    return _adapt('check_storage_locked_v1')(storage,first=first,boundary=boundary,required_write_bytes=required_write_bytes)


def budget_check_v9(storage, *, first=False, boundary=False, required_write_bytes=0):
    with budget.v4.mutex(ROOT):
        return check_storage_locked_v1(storage,first=first,boundary=boundary,required_write_bytes=required_write_bytes)


def validate_storage_v9(spec,cfg,storage,*,source_ref,config_ref,runtime_ref):
    return _adapt('validate_storage_v9')(spec,cfg,storage,source_ref=source_ref,config_ref=config_ref,runtime_ref=runtime_ref)
