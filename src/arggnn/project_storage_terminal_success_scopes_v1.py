"""Narrow six-orphan-VAL CPU success credits in the existing terminal journal.

Old terminal events are an immutable validated prefix. Root must finish every
v1 terminal writer before appending successor events. Launcher exits remain
explicitly unknown. No action at import or global process/FD scan.
"""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from types import FunctionType
from . import project_storage_terminal_scopes_v1 as old

ROOT = old.ROOT
SOURCE = 'src/arggnn/project_storage_terminal_success_scopes_v1.py'
JOURNAL = old.JOURNAL
MODE = 'journal_terminal_success_scopes_credit_v1'
OLD_SHA256 = 'f06cf7028af7e86deb54caa1e8a01f0110db47b0d992bfe8aec8ee013ab7d79c'
METADATA_CHARGE = old.METADATA_CHARGE
budget = old.budget
_CACHE = None


def _utc(): return datetime.now(timezone.utc).isoformat()
def _ref(path): return budget.v4.reference(ROOT, path)
def _bound(ref): return budget.v4.bound(ROOT, ref, True)
def _pins(view): return old._pins(view)
def _paths(own): return old._paths(own)
def _ceiling(value, view): return old._ceiling(value, view)


def _frozen():
    budget.v4.bound(ROOT, {'path': old.SOURCE, 'sha256': OLD_SHA256})
    return old._cpu_reader()


def _clone(method, **changes):
    value = FunctionType(method.__code__, dict(method.__globals__, **changes), method.__name__,
                         method.__defaults__, method.__closure__)
    value.__kwdefaults__ = method.__kwdefaults__
    return value


def _authority(ref, operation, bindings, *, fresh):
    value = _bound(ref)
    required = dict(kind='project_budget_terminal_success_root_admission_v1', root_authorizer='root_agent',
        active=True, operation=operation, accounting_mode=MODE, bindings=bindings,
        all_budget_writers_use_scheduler_mutex=True, all_consumers_and_launchers_use_terminal_success_guard=True,
        all_terminal_v1_writers_finished_before_successor_activation=True,
        historical_records_mutation_authorized=False, global_rebase_authorized=False,
        scientific_execution_authorized=False, GPU_authorized=False)
    if any(budget.v4.encoded(value.get(k)) != budget.v4.encoded(v) for k, v in required.items()):
        raise ValueError('Exact root successor authority and v1 writer barrier required')
    if operation == 'close_completed_cpu_observed_absent' and any(value.get(k) is not True for k in (
            'scope_write_authority_revoked_for_all_cooperative_project_jobs',
            'closed_scope_future_writes_and_relaunch_revoked', 'external_NFS_writers_outside_observation')):
        raise ValueError('Successor closure requires cooperative write/relaunch revocation')
    if operation == 'reserve' and any(ref not in value['reservation']['consumer_bindings'] for ref in bindings.values()):
        raise ValueError('Every original and successor source must be reserved')
    if fresh:
        expiry = datetime.fromisoformat(value['expires_at_utc'].replace('Z', '+00:00'))
        if expiry.tzinfo is None or expiry <= datetime.now(timezone.utc): raise ValueError('Expired successor authority')
    return value


def _mixed_events(bindings):
    directory = budget.v4.canonical(ROOT, JOURNAL)
    if not directory.exists(): return [], [], None
    if not directory.is_dir(): raise ValueError('Canonical existing terminal journal required')
    before = directory.lstat(); names = sorted(p.name for p in directory.iterdir())
    if len(names) > 4096: raise ValueError('Shared terminal event count cap exceeded')
    original = {k: v for k, v in bindings.items() if k != 'terminal_success_overlay'}
    prefix, suffix, tail, total = [], [], None, 0
    for number, name in enumerate(names, 1):
        if name != f'{number:06d}.json': raise ValueError('Shared terminal journal gap/unknown entry')
        raw, sha = budget.v4.file_bytes(ROOT, directory / name); total += len(raw)
        if len(raw) > METADATA_CHARGE or total > 32 * 1024 * 1024: raise ValueError('Shared metadata charge/cap exceeded')
        event = budget.v4.parse(raw)
        if (set(event) != {'schema','sequence','previous_sha256','bindings','operation','authority','value'}
                or event['schema'] != 'project_budget_terminal_overlay_event_v1'
                or type(event['sequence']) is not int or event['sequence'] != number or event['previous_sha256'] != tail):
            raise ValueError('Immutable shared terminal chain drift')
        if event['bindings'] == original:
            if suffix or event['operation'] not in ('reserve', 'close_failed_cpu'):
                raise ValueError('Old terminal writer appeared after successor activation')
            authority = old._authority(event['authority'], event['operation'], original, fresh=False)
            prefix.append(event)
        else:
            if event['bindings'] != bindings or event['operation'] not in ('reserve', 'close_completed_cpu_observed_absent'):
                raise ValueError('Exact typed successor event/bindings required')
            authority = _authority(event['authority'], event['operation'], bindings, fresh=False)
            suffix.append(event)
        if authority.get('expected_before', {}).get('terminal_overlay_tail_sha256') != tail:
            raise ValueError('Shared terminal authority predecessor drift')
        tail = sha
    after = directory.lstat()
    if (before.st_dev,before.st_ino,before.st_mtime_ns,before.st_ctime_ns) != (
            after.st_dev,after.st_ino,after.st_mtime_ns,after.st_ctime_ns): raise ValueError('Shared journal changed during replay')
    return prefix, suffix, tail


def _completed_cpu(request, reservation, *, observe=False):
    if set(request) != {'key','parent_id','storage_ref','completion_ref','input_ref','launch_ref','observation_ref'}:
        raise ValueError('Exact narrow observed-absence CPU success request required')
    parent = request['parent_id']
    allowed = {f'ap_expanded_val_{index:03d}': 4163579 + index for index in range(6)}
    if parent not in allowed or request['key'] != 'H038/' + parent or request['key'] != reservation['key']:
        raise ValueError('Only original orphan VAL000 through VAL005 may use unknown-exit success credits')
    paths = _paths(reservation)
    if paths != [ROOT / 'data/expanded_full_panel_v1' / parent] or reservation['reserved_bytes'] != 3000000000:
        raise ValueError('Exact original six VAL scopes/allowances required')
    folder = f'research/reviews/expanded_full_panel_v1_root/{parent}'
    if (request['input_ref']['path'] != folder + '/input.json' or request['launch_ref']['path'] != folder + '/launch.json'
            or request['storage_ref']['path'] != folder + '/storage.json'
            or request['completion_ref']['path'] != str((paths[0]/'expanded_parent_completed.json').relative_to(ROOT))):
        raise ValueError('Exact original own-scope launch/admission/completion locations required')
    storage, completion, admission, launch, terminal = (_bound(request[k]) for k in (
        'storage_ref','completion_ref','input_ref','launch_ref','observation_ref'))
    source = {'path':'src/arggnn/prepare_expanded_full_panel_v1.py','sha256':'2ff091677986d5cfec2936c2acf53eeed71c957192f9f048cb47f59d3351913e'}
    producer = {'path':'src/arggnn/prepare_ancestralpaths_full_panel_discovery_v1.py','sha256':'ead823158a001dabdbf2224d812e7ef8079b6af4382f6e8e10ca0e0894760e71'}
    config = {'path':'configs/expanded_full_panel_v1.json','sha256':'662ba8e4613ce85b9aaec9909e873c9cddf9c62672b895e7d9e02f851d518d08'}
    if (storage.get('reservation') != reservation or admission.get('storage') != storage
            or storage.get('kind') != 'project_budget_v9_storage_record'
            or storage.get('accounting_mode') != 'journal_cpu_completed_scopes_credit_v1'
            or storage.get('output_directory') != str(paths[0].relative_to(ROOT))
            or completion.get('schema') != 'expanded_full_panel_v1_parent_completion'
            or completion.get('parent_id') != parent or completion.get('split') != 'val'
            or completion.get('generation_completed') is not True or completion.get('Relate_completed') is not True
            or completion.get('GPU_used') is not False or completion.get('outer_truth_evaluated') is not False
            or completion.get('source_ref') != source or completion.get('original_producer_ref') != producer
            or admission.get('schema') != 'expanded_full_panel_v1_root_admission'
            or admission.get('root_authorizer') != 'root_agent' or admission.get('execution_authorized') is not True
            or admission.get('parent_id') != parent or admission.get('source_ref') != source
            or admission.get('config_ref') != config or admission.get('outer_evaluation_authorized') is not False
            or admission.get('output_directory') != storage['output_directory']
            or admission.get('config_content_sha256') != completion.get('config_content_sha256')
            or admission.get('parent_registry_sha256') != completion.get('parent_registry_sha256')
            or launch.get('parent_id') != parent or type(launch.get('pid')) is not int or launch['pid'] != allowed[parent]
            or launch.get('input_ref') != request['input_ref'] or launch.get('device') != 'cpu'
            or launch.get('CUDA_VISIBLE_DEVICES') != '' or 'returncode' not in launch or launch['returncode'] is not None):
        raise ValueError('Genuine original scientific completion and honest unknown CPU launcher exit required')
    for ref in (source,producer,config,admission['runtime_ref'],admission['storage_adapter_ref']):
        if ref not in reservation['consumer_bindings']: raise ValueError('Original admitted source/config/runtime pins required')
        budget.v4.bound(ROOT, ref)
    runtime = _bound(admission['runtime_ref'])
    if runtime.get('device') != 'cpu' or runtime.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('Original CPU-only runtime required')
    expected_command = [str(ROOT/'.runtime/env/bin/python'),'-B','-m','arggnn.project_storage_cpu_closed_scopes_v1',
        '--prepare-expanded','--admission',request['input_ref']['path'],'--expected-admission-sha256',request['input_ref']['sha256'],
        '--config',config['path'],'--expected-config-sha256',config['sha256'],'--expected-producer-sha256',source['sha256'],
        '--parent',parent,'--stage','all']
    if launch.get('command') != expected_command: raise ValueError('Exact recorded original CPU command required')
    for key, name in [('generation_ref','generation.json'),('inference_ref','relate/inference.json'),
                      ('observation_entry_ref','full_panel_observed_entry.json')]:
        if completion[key]['path'] != str((paths[0]/name).relative_to(ROOT)): raise ValueError('Same-parent scientific receipt paths required')
    generation = _bound(completion['generation_ref']); inference = _bound(completion['inference_ref'])
    entry = _bound(completion['observation_entry_ref'])
    codes = inference.get('outer_command_returncodes')
    if (generation.get('parent_id') != parent or inference.get('parent_id') != parent
            or inference.get('status') != 'post_five_iterations_inferred' or inference.get('truth_or_oracle_read') is not False
            or not isinstance(codes,list) or len(codes) != 5 or any(type(code) is not int or code != 0 for code in codes)
            or entry.get('parent_id') != parent):
        raise ValueError('Bound same-parent generation/observed completion and all five zero native helper waits required')
    fields = {'kind','root_authorizer','status','parent_id','launch_ref','completion_ref','worker_pid','observation',
              'launcher_exit_status_known','launcher_returncode'}
    if (set(terminal) != fields or terminal['kind'] != 'project_budget_CPU_completed_observed_absent_v1'
            or terminal['root_authorizer'] != 'root_agent'
            or terminal['status'] != 'completed_CPU_scope_observed_absent_launcher_exit_unknown'
            or terminal['parent_id'] != parent or terminal['launch_ref'] != request['launch_ref']
            or terminal['completion_ref'] != request['completion_ref'] or terminal['worker_pid'] != launch['pid']
            or terminal['launcher_exit_status_known'] is not False or terminal['launcher_returncode'] is not None):
        raise ValueError('Bound root process absence must preserve unknown launcher exit status')
    old._observation(terminal['observation'], launch['pid'])
    observation = _frozen()['_absent'](launch['pid']) if observe else terminal['observation']
    return dict(status='completed_CPU_scope_observed_absent_launcher_exit_unknown', scientific_completion_verified=True,
        generation_completed=True, native_outer_command_returncodes=codes, native_child_statuses_independently_asserted=False,
        launcher_exit_status_known=False, launcher_returncode=None, worker_observation=observation, GPU_used=False)


def _state_locked(source_sha, launcher_sha, *, extra=(), fresh=False):
    global _CACHE
    cpu = _frozen()
    bindings = dict(implementation={'path':budget.SOURCE_REL,'sha256':source_sha},launcher={'path':budget.LAUNCHER_REL,'sha256':launcher_sha},
        cpu_overlay={'path':old.CPU_SOURCE,'sha256':old.CPU_SHA256},terminal_overlay={'path':old.SOURCE,'sha256':OLD_SHA256},terminal_success_overlay=_ref(SOURCE))
    deps = _CACHE[3] if _CACHE is not None else ()
    watched = sorted(set(extra)|set(deps)|{SOURCE,old.SOURCE,JOURNAL})
    before = cpu['_fingerprint'](watched)
    if not fresh and _CACHE is not None and _CACHE[0] == bindings and _CACHE[1] == before: return copy.deepcopy(_CACHE[2])
    prefix, suffix, tail = _mixed_events(bindings)
    old_tail = budget.v4.file_bytes(ROOT, ROOT/JOURNAL/f'{len(prefix):06d}.json')[1] if prefix else None
    base_reader = _clone(old._state_locked, _CACHE=None, _load_events=lambda bindings: (prefix,old_tail))
    reservations, closed, base, _, _ = base_reader(source_sha,launcher_sha,extra=extra,fresh=fresh)
    reservations=copy.deepcopy(reservations);closed=set(closed)
    legacy=budget.v4.read_json(ROOT,budget.v4.LEGACY)[0]
    occupied=[budget.v4.canonical(ROOT,row['output']) for row in legacy['jobs'].values()]
    for own in reservations.values(): occupied.extend(_paths(own))
    discovered=set(base_reader.__globals__['_CACHE'][3])|set(extra)|{SOURCE,old.SOURCE,JOURNAL}
    def collect(value):
        if isinstance(value,dict):
            if set(value)=={'path','sha256'}: discovered.add(value['path'])
            for child in value.values():collect(child)
        elif isinstance(value,(list,tuple)):
            for child in value:collect(child)
    credit=added=0
    for event in suffix:
        authority=_authority(event['authority'],event['operation'],bindings,fresh=False);value=event['value']
        collect(event);collect(authority)
        if event['operation']=='reserve':
            request=authority['reservation'];paths=_paths(value);key=value['key']
            if (value!={**request,'root_admission':event['authority']} or len(reservations)>=4096
                    or key in reservations or key in legacy['jobs']
                    or any(budget.v4.paths_overlap(a,b) for a in paths for b in occupied)):
                raise ValueError('Successor reservation authority/key/scope collision')
            reservations[key]=value;occupied.extend(paths);added+=value['reserved_bytes']
        else:
            request=authority['closure'];key=request['key'];own=reservations.get(key)
            if (own is None or key in closed or set(value)!={'request','reservation_sha256','measurement','completion_proof',
                    'closed_utc','future_writes_and_relaunch_revoked'} or value['request']!=request
                    or value['reservation_sha256']!=budget.digest(own) or value['future_writes_and_relaunch_revoked'] is not True):
                raise ValueError('Unknown/duplicate successor closure or certificate drift')
            proof=_completed_cpu(request,own);recorded=value['completion_proof']
            if set(recorded)!=set(proof) or any(recorded[k]!=proof[k] for k in proof if k!='worker_observation'):
                raise ValueError('Scientific completion or honest unknown-exit certificate drift')
            if old._observation(recorded['worker_observation'],proof['worker_observation']['pid']) < old._observation(proof['worker_observation'],proof['worker_observation']['pid']):
                raise ValueError('Applied PID proof predates root observation')
            measurement=value['measurement']
            if (set(measurement)!={'conservative_scope_bytes','entries','scope_paths','stat_only'} or measurement['stat_only'] is not True
                    or measurement['scope_paths']!=[str(p) for p in _paths(own)]): raise ValueError('Exact owned-scope stat certificate required')
            actual=budget.v4.integer(measurement['conservative_scope_bytes'],'retained CPU bytes')
            budget.v4.integer(measurement['entries'],'retained CPU entries')
            if actual>own['reserved_bytes']:raise ValueError('Completed CPU scope exceeds allowance')
            credit+=own['reserved_bytes']-actual;closed.add(key)
            for name in request:
                if name.endswith('_ref'): collect(_bound(request[name]))
    projected=base['projected_bytes']+added-credit+len(suffix)*METADATA_CHARGE;_ceiling(projected,base)
    view=dict(base,original_terminal_projected_bytes=base['projected_bytes'],projected_bytes=projected,
        terminal_success_closed_allowance_credit_bytes=credit,terminal_success_reserved_bytes=added,
        terminal_success_metadata_charge_bytes=len(suffix)*METADATA_CHARGE,terminal_overlay_tail_sha256=tail,
        terminal_overlay_events=len(prefix)+len(suffix),terminal_success_events=len(suffix),accounting_mode=MODE)
    result=reservations,closed,view,bindings,prefix+suffix
    if before!=cpu['_fingerprint'](watched):raise ValueError('Shared accounting namespace changed during replay')
    _CACHE=bindings,cpu['_fingerprint'](sorted(discovered)),copy.deepcopy(result),tuple(sorted(discovered))
    return result


def inspect_v1(*,expected_source_sha256,expected_launcher_sha256):
    with budget.v4.mutex(ROOT):return _state_locked(expected_source_sha256,expected_launcher_sha256,fresh=True)[2]


def _append(operation,authority_ref,value,state):
    return old._append(operation,authority_ref,value,state)


def apply_close_completed_cpu_observed_absent_v1(authority_ref,*,expected_source_sha256,expected_launcher_sha256):
    with budget.v4.mutex(ROOT):
        state=_state_locked(expected_source_sha256,expected_launcher_sha256,fresh=True)
        reservations,closed,view,bindings,_=state
        authority=_authority(authority_ref,'close_completed_cpu_observed_absent',bindings,fresh=True)
        if authority['expected_before']!=_pins(view):raise ValueError('Fresh shared-journal pins required')
        request=authority['closure'];key=request['key']
        if key not in reservations or key in closed:raise ValueError('Unknown/already closed orphan CPU scope')
        own=reservations[key];proof=_completed_cpu(request,own,observe=True)
        measurement=budget.v4.measure_scopes(ROOT,_paths(own))  # Exactly one actual owned-scope inventory.
        _frozen()['_absent'](proof['worker_observation']['pid'])
        actual=measurement['conservative_scope_bytes']
        if actual>own['reserved_bytes']:raise ValueError('Completed CPU output exceeds allowance')
        projected=view['projected_bytes']-(own['reserved_bytes']-actual)+METADATA_CHARGE;_ceiling(projected,view)
        value=dict(request=copy.deepcopy(request),reservation_sha256=budget.digest(own),measurement=measurement,
            completion_proof=proof,closed_utc=_utc(),future_writes_and_relaunch_revoked=True)
        tail=_append('close_completed_cpu_observed_absent',authority_ref,value,state)
        return dict(kind='project_budget_terminal_success_CPU_close_receipt_v1',key=key,value=value,retained_bytes=actual,
            allowance_bytes=own['reserved_bytes'],credit_bytes=own['reserved_bytes']-actual,projected_bytes=projected,
            terminal_overlay_tail_sha256=tail,scope_measurements=1,launcher_exit_status_known=False,launcher_returncode=None,
            GPU_used=False,original_records_mutated=False,global_process_scan_performed=False)


def _adapt(name):
    method=_frozen()[name]
    return _clone(method,SOURCE=SOURCE,MODE=MODE,JOURNAL=JOURNAL,_state_locked=_state_locked,_pins=_pins,
        _paths=_paths,_authority=_authority,_append=_append,budget_check_v9=budget_check_v9,check_storage_locked_v1=check_storage_locked_v1)


def apply_reserve_v1(authority_ref,*,expected_source_sha256,expected_launcher_sha256):
    value=_adapt('apply_reserve_v1')(authority_ref,expected_source_sha256=expected_source_sha256,expected_launcher_sha256=expected_launcher_sha256)
    value['kind']='project_budget_terminal_success_reserve_receipt_v1'
    value['terminal_overlay_tail_sha256']=value.pop('cpu_overlay_tail_sha256')
    return value


def check_storage_locked_v1(storage,*,first=False,boundary=False,required_write_bytes=0):
    return _adapt('check_storage_locked_v1')(storage,first=first,boundary=boundary,required_write_bytes=required_write_bytes)


def budget_check_v9(storage,*,first=False,boundary=False,required_write_bytes=0):
    with budget.v4.mutex(ROOT):return check_storage_locked_v1(storage,first=first,boundary=boundary,required_write_bytes=required_write_bytes)


def validate_storage_v9(spec,cfg,storage,*,source_ref,config_ref,runtime_ref):
    return _adapt('validate_storage_v9')(spec,cfg,storage,source_ref=source_ref,config_ref=config_ref,runtime_ref=runtime_ref)


class PilotOutputV9(old.PilotOutputV9):
    __init__=_clone(old.PilotOutputV9.__init__,budget_check_v9=budget_check_v9)
    save=_clone(old.PilotOutputV9.save,budget_check_v9=budget_check_v9)
    discard=_clone(old.PilotOutputV9.discard,budget_check_v9=budget_check_v9)


if __name__=='__main__':
    print(json.dumps(dict(status='source_only_successor_not_applied',source=_ref(SOURCE),accounting_mode=MODE,journal=JOURNAL),indent=2))
