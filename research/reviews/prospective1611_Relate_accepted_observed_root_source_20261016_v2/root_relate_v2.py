"""One accepted prospective1611 TRAIN000 Relate pipeline. Default inert.

Original numerical adapter is unchanged. ROOT separately admits this fixed
CPU-only scope. Existing H020v3 guard and owned pidfd helpers remain unchanged.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ctypes, hashlib, importlib.util, json, os, resource, selectors, signal, stat, subprocess, sys, tempfile, time
ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
HERE = 'research/reviews/prospective1611_Relate_accepted_observed_root_source_20261016_v2'
SOURCE = HERE + '/root_relate_v2.py'
MANIFEST = HERE + '/source_manifest.v2.json'
PACKET = HERE + '/admitted_train000_v2'
OUTPUT = 'data/ARGMix_prospective1611_Relate_accepted_train000_v2'
JOB = 'prospective1611-accepted-TRAIN000-Relate-v2'
PARENT = 'published_recon_train_000'
ACCEPTANCE = dict(path='research/reviews/prospective1611_observed_data_acceptance_root_20261016_v1/acceptance.root_v1.json', sha256='4ae9614b201f5866be3a23874941404056b0609608b7125bbdae0f3816208e21')
ADAPTER = dict(path='research/reviews/ARGMix_prospective1611_Relate_adapter_20261016_v1/relate_adapter_v1.py', sha256='66b2f0f56746530af0c7473dbf014f0f8483cdd0c101cb777e2340901f07ca1f')
PINS = dict(path='research/reviews/ARGMix_prospective1611_Relate_adapter_20261016_v1/source_pins.v1.json', sha256='df6c9282ed201aa10c7cf11cfaa9d76451692c0980d17e3d9c331f8744b6bdfd')
MAP = dict(path='data/ARGMix_prospective1611_HapMapII_GRCh37_chr22_v1/derivation_receipt.json', sha256='72453ecd3fc5940d4d3fd560f7897342b10e608cc81d6155b24f21f4d57e88e6')
RUNTIME = dict(path='research/reviews/ARGMix_prospective1611_runtime_root_20261016_v1/runtime_binding.root_active_v1.json', sha256='b3b8f3e037b1ae24bd650894483fe84f7fc6fbcb51c50ce64957418c0e118e46')
GUARD_FACTORY = dict(path='research/reviews/H058_one_hold_ROOT_execution_source_20261016_v3/root_qualify_v3.py', sha256='0e81b1067b38296982db34fcc7eda7d9b85a07a1aea56dd7cf9a9dcbd0291f9e')
PIDFD = dict(path='research/reviews/American_C3_native_MLM_full_source_v5/pidfd_backend_v1.py', sha256='61be5d89357d1b95611e0ba26dd39d381ececf3ce6a36c347507cdc062373eca')
OWNED = dict(path='research/reviews/American_C3_native_MLM_amendment_v12/owned_lifecycle_v2.py', sha256='93d71acc40f1037d0b37acb2ca0859300ea4bd9741409e9bd00ca21bf100d633')
CAPS = dict(schema='prospective1611_Relate_operational_caps_v1', active=True, cpu_threads=1,
    rss_bytes=32*2**30, wall_seconds=21600, scope_bytes=64*2**30,
    input_bytes=8*2**30, native_bytes=40*2**30, export_bytes=12*2**30,
    receipt_bytes=64*2**20, log_bytes=16*2**20, automatic_retry=False, automatic_downscale=False)

def need(ok, message):
    if not ok:
        raise ValueError(message)

def local(path):
    need(isinstance(path, str) and not Path(path).is_absolute() and '..' not in Path(path).parts, 'Relative contained path')
    p = ROOT/path
    need(p.resolve() == p and p.is_relative_to(ROOT), 'Canonical project path')
    return p

def ref(path):
    p = local(path)
    need(p.is_file() and not p.is_symlink() and p.stat().st_nlink == 1 and p.stat().st_size <= 32*2**20, 'Bounded singly linked source/metadata')
    return dict(path=path, sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def read(record):
    need(isinstance(record, dict) and set(record) == {'path', 'sha256'} and ref(record['path']) == record, 'Exact metadata/source SHA')
    return json.loads(local(record['path']).read_text())

def load(record, name):
    need(ref(record['path']) == record, 'Exact source before import')
    sp = importlib.util.spec_from_file_location(name, local(record['path']))
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m

def active(value):
    now = datetime.now(timezone.utc)
    a = datetime.fromisoformat(value['created_at_utc'])
    b = datetime.fromisoformat(value['expires_at_utc'])
    need(a.tzinfo and b.tzinfo and a <= now < b, 'Current ROOT authority')

def unique(rows):
    result = {}
    for row in rows:
        need(row['path'] not in result or result[row['path']] == row, 'No conflicting direct SHA')
        result[row['path']] = row
    need(len(result) <= 64, 'Original64 direct-ref limit, never omit roles')
    return list(result.values())

def sources():
    manifest = read(ref(MANIFEST))
    rows = manifest['refs']
    need(ref(SOURCE) in rows and all(ref(r['path']) == r for r in rows), 'Full fixed source/metadata closure')
    return rows

def acceptance():
    a = read(ACCEPTANCE)
    need(a.get('schema') == 'prospective1611_saved_observed_data_acceptance_root_v1'
        and a.get('status') == 'accepted_observed_data_for_downstream_inference'
        and a.get('root_authorizer') == 'root_agent'
        and a.get('accepted_for_observed_inference_only') is True
        and a.get('parent_outer_returncode') == 1 and a.get('numerical_child_returncode') == 0
        and a.get('full_original_charge_retained') is True
        and a.get('private_truth_payload_opened') is False
        and a.get('published_table_reproduction') is False
        and a.get('downstream_execution_requires_separate_admission') is True
        and (a['diploids'], a['haplotypes'], a['length_bp']) == (1611, 3222, 35178458), 'Exact truthful accepted observed data, failed monitor preserved')
    observed = read(a['observed_receipt_ref'])
    need(observed['complete'] is True and observed['parent_id'] == PARENT
        and observed['observed_sites'] == 311599 and observed['contains_genealogy_or_founder_truth'] is False,
        'Exact accepted observed TRAIN000 receipt')
    # Metadata only, deliberately do not follow private-generation metadata.
    proofs = {key: read(a[key]) for key in ('byte_integrity_ref', 'strict_postdrain_guard_ref', 'actual_child_wait_ref', 'actual_outer_terminal_ref', 'future_write_revocation_ref')}
    integrity = proofs['byte_integrity_ref']
    need(integrity['observed_receipt_ref'] == a['observed_receipt_ref']
        and integrity['observed_bytes_stream_hashed'] == 4052787961
        and len(integrity['observed_products_independently_stream_hashed']) == 7,
        'Actual independent all-seven observed byte verification')
    need(proofs['strict_postdrain_guard_ref']['status'] == 'PASS'
        and proofs['actual_child_wait_ref']['returncode'] == 0
        and proofs['actual_child_wait_ref']['actual_direct_worker_wait'] is True
        and proofs['actual_child_wait_ref']['all_owned_processes_and_groups_drained'] is True
        and proofs['actual_outer_terminal_ref']['actual_exit_code'] == 1
        and proofs['actual_outer_terminal_ref']['actual_terminal_consumed'] is True
        and proofs['future_write_revocation_ref']['all_future_launch_resume_and_write_capabilities_revoked'] is True,
        'Actual accepted data provenance, child0/drain, outer1 and revoked old writer')
    return a

def guard(consumer):
    return load(GUARD_FACTORY, '_Relate_reused_guard_factory').guard_v1(consumer=consumer)

def put(path, value, g=None, storage=None, cap=2**20):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    need(len(raw) <= cap, 'Compact bounded metadata')
    if g is not None:
        g.budget_check_v9(storage, boundary=True, required_write_bytes=len(raw)+8192)
    with local(path).open('xb') as stream:
        stream.write(raw)
    return ref(path)

def request(record):
    q = read(record)
    active(q)
    need(set(q) == {'schema','status','root_authorizer','execution_authorized','GPU_authorized','acceptance_ref','driver_ref','source_manifest_ref','created_at_utc','expires_at_utc','cpu_id'}
        and q['schema'] == 'prospective1611_accepted_Relate_ROOT_request_v1'
        and q['status'] == 'active' and q['root_authorizer'] == 'root_agent'
        and q['execution_authorized'] is True and q['GPU_authorized'] is False
        and q['acceptance_ref'] == ACCEPTANCE and q['driver_ref'] == ref(SOURCE)
        and q['source_manifest_ref'] == ref(MANIFEST)
        and type(q['cpu_id']) is int and q['cpu_id'] in os.sched_getaffinity(0), 'Exact fixed CPU-only ROOT request')
    sources()
    return q, acceptance()

def admit(record):
    need(Path.cwd() == ROOT and not local(PACKET).exists() and not local(OUTPUT).exists(), 'Fresh fixed packet/output, no retry')
    q, a = request(record)
    g = guard(False)
    local(PACKET).mkdir()
    caps_ref = put(PACKET+'/caps.json', CAPS)
    config_ref = put(PACKET+'/config.json', dict(schema='prospective1611_accepted_Relate_config_v1', output_cap_bytes=CAPS['scope_bytes']))
    implementation, launcher = ref(g.budget.SOURCE_REL), ref(g.budget.LAUNCHER_REL)
    with g.budget.v4.mutex(ROOT):
        state = g._state_locked(implementation['sha256'], launcher['sha256'], fresh=True)
    bindings = unique([*state[3].values(), *sources(), ref(MANIFEST), record, caps_ref, config_ref])
    authority = g.operation_draft_v1('reserve', dict(key='ARGMix/'+JOB, output=str(local(OUTPUT)),
        auxiliary_scopes=[str(local(PACKET+'/worker_aux'))], reserved_bytes=CAPS['scope_bytes'], consumer_bindings=bindings),
        expected_before=g._pins(state[2]), bindings=state[3])
    authority.update(active=True, expires_at_utc=q['expires_at_utc'])
    auth_ref = put(PACKET+'/reserve_authority.json', authority)
    receipt = g.apply_reserve_v1(auth_ref, expected_source_sha256=implementation['sha256'], expected_launcher_sha256=launcher['sha256'])
    receipt_ref = put(PACKET+'/reserve_receipt.json', receipt)
    factory = load(GUARD_FACTORY, '_Relate_original_guard_constants')
    storage = dict(kind='project_budget_v9_storage_record', active=True, consumer_uses_v9_guard=True, accounting_mode=g.MODE,
        job_id=JOB, output_directory=OUTPUT, reserved_new_bytes=CAPS['scope_bytes'], reservation=receipt['value'],
        initial_ledger_pins=receipt['initial_ledger_pins'], expires_at_utc=q['expires_at_utc'], completed_overlay_policy_ref=factory.POLICY,
        budget_components=dict(implementation=implementation, launcher=launcher, consumer_adapter=factory.GUARD))
    storage_ref = put(PACKET+'/storage.json', storage)
    resource_ref = put(PACKET+'/resource_binding.json', dict(schema='prospective1611_Relate_root_resource_binding_v1', root_authorizer='root_agent',
        root_enforced=True, outer_process_limits_enforced=True, output_directory=str(local(OUTPUT)), caps_ref=caps_ref,
        runner_factory_ref=ref(SOURCE), adapter_source_ref=ADAPTER, admission_receipt_ref=receipt_ref))
    original_request = dict(schema='prospective1611_Relate_root_request_v1', root_approved=True, root_authorizer='root_agent',
        registered_before_execution=True, parent_id=PARENT, source_ref=ADAPTER, source_pins_ref=PINS,
        observed_receipt_ref=a['observed_receipt_ref'], map_derivation_receipt_ref=MAP, runtime_binding_ref=RUNTIME,
        caps_ref=caps_ref, resource_binding_ref=resource_ref, runner_factory_ref=ref(SOURCE), output_directory=str(local(OUTPUT)))
    original_ref = put(PACKET+'/original_infer_request.json', original_request)
    spec = dict(schema='prospective1611_accepted_Relate_ROOT_input_v1', status='active', root_authorizer='root_agent',
        job_id=JOB, output_directory=OUTPUT, root_request_ref=record, acceptance_ref=ACCEPTANCE,
        original_infer_request_ref=original_ref, caps_ref=caps_ref,
        refs=dict(storage=storage_ref, config=config_ref, runtime=RUNTIME), driver_ref=ref(SOURCE),
        created_at_utc=q['created_at_utc'], expires_at_utc=q['expires_at_utc'])
    input_ref = put(PACKET+'/input.json', spec)
    put(PACKET+'/admitted.json', dict(input_ref=input_ref, reserve_receipt_ref=receipt_ref, consumer_binding_count=len(bindings), execution_started=False))
    return input_ref

def context(record):
    s = read(record)
    active(s)
    need(record['path'] == PACKET+'/input.json' and s['schema'] == 'prospective1611_accepted_Relate_ROOT_input_v1'
        and s['driver_ref'] == ref(SOURCE) and s['acceptance_ref'] == ACCEPTANCE and s['job_id'] == JOB, 'Exact admitted fixed input')
    q, a = request(s['root_request_ref'])
    storage, config = read(s['refs']['storage']), read(s['refs']['config'])
    need(storage['job_id'] == JOB and storage['output_directory'] == OUTPUT and storage['reserved_new_bytes'] == CAPS['scope_bytes']
        and storage['reservation']['auxiliary_scopes'] == [str(local(PACKET+'/worker_aux'))]
        and read(s['caps_ref']) == CAPS and config['output_cap_bytes'] == CAPS['scope_bytes'], 'Exact fresh complete resource scope')
    needed = unique([*sources(), ref(MANIFEST), s['root_request_ref'], s['caps_ref'], s['refs']['config']])
    need(all(r in storage['reservation']['consumer_bindings'] for r in needed), 'Every exact role directly bound')
    need(read(s['original_infer_request_ref'])['observed_receipt_ref'] == a['observed_receipt_ref'], 'Same accepted observed receipt, no wrapper success fiction')
    return s, q, storage

def census():
    total = 0
    categories = {'inputs': 0, 'native': 0, 'observed': 0, 'other': 0, 'aux': 0}
    for relative, kind in ((OUTPUT, None), (PACKET+'/worker_aux', 'aux')):
        top = local(relative)
        if not top.exists():
            continue
        stack = [top]
        visited = 0
        while stack:
            p = stack.pop()
            try:
                st = p.lstat()
            except FileNotFoundError:
                continue
            visited += 1
            need(visited <= 100000 and not stat.S_ISLNK(st.st_mode), 'Bounded no-follow owned scope')
            need(stat.S_ISDIR(st.st_mode) or (stat.S_ISREG(st.st_mode) and st.st_nlink == 1), 'No special or linked output')
            charge = max(st.st_size, st.st_blocks*512)
            total += charge
            parts = p.relative_to(top).parts
            bucket = kind or (parts[0] if parts and parts[0] in ('inputs','native','observed') else 'other')
            categories[bucket] += charge
            if stat.S_ISDIR(st.st_mode):
                try:
                    stack.extend(p.iterdir())
                except FileNotFoundError:
                    pass
    need(total <= CAPS['scope_bytes'] and categories['inputs'] <= CAPS['input_bytes']
        and categories['native'] <= CAPS['native_bytes'] and categories['observed'] <= CAPS['export_bytes']
        and categories['other'] <= CAPS['receipt_bytes'] and categories['aux'] <= 64*2**20, 'Sampled aggregate/category storage caps')
    return dict(total_bytes=total, categories=categories, sampled_live_inventory=True)

def lifecycle():
    backend = load(PIDFD, 'pidfd_backend_v1')
    owned = load(OWNED, '_Relate_exact_owned_lifecycle')
    libc = ctypes.CDLL(None, use_errno=True)
    need(libc.prctl(36, 1, 0, 0, 0) == 0, 'Linux subreaper')
    return backend, owned

def current_rss(owned, rows):
    total = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    for row in rows:
        actual = owned._stat(row['pid'])
        if not owned._same(actual, row) or actual.get('state') in ('Z','X','x'):
            continue
        try:
            text = Path('/proc/%d/status'%row['pid']).read_text()
        except (FileNotFoundError, ProcessLookupError):
            continue
        values = [int(x.split()[1])*1024 for x in text.splitlines() if x.startswith('VmRSS:')]
        if not values:
            after = owned._stat(row['pid'])
            if not owned._same(after, row) or after.get('state') in ('Z','X','x'):
                continue
            need(False, 'Live owned identity has no RSS after identity/state recheck')
        total += values[0]
    return total

def supervise(proc, started, expires, owned, check, gate=None, log_remaining=None):
    tree = owned.OwnedTree(proc.pid)
    selector = selectors.DefaultSelector()
    os.set_blocking(proc.stdout.fileno(), False)
    selector.register(proc.stdout, selectors.EVENT_READ)
    captured = bytearray()
    total = peak = 0
    failure = None
    status = None
    limit = CAPS['log_bytes'] if log_remaining is None else log_remaining
    try:
        if gate is not None:
            os.write(gate, b'GO\n')
            os.close(gate)
            gate = None
        terminal = None
        eof = False
        while terminal is None or not eof:
            need(_CANCELLED is None, 'ROOT cancellation signal '+str(_CANCELLED))
            # Observe direct exit before RSS. Recheck identity/state for /proc exit races.
            terminal = os.waitid(os.P_PID, proc.pid, os.WEXITED|os.WNOHANG|os.WNOWAIT)
            tree.refresh()
            need(len(tree.known) <= 16, 'Only owned Relate pipeline descendants')
            peak = max(peak, current_rss(owned, tree.known.values()))
            need(peak <= CAPS['rss_bytes'] and time.monotonic()-started <= CAPS['wall_seconds']
                and datetime.now(timezone.utc) < datetime.fromisoformat(expires), 'Aggregate RSS/wall/authority cap')
            check()
            for key, _ in selector.select(.25):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if chunk:
                    total += len(chunk)
                    captured.extend(chunk[:max(0, limit-len(captured))])
                    need(total <= limit, 'Aggregate bounded pipeline log')
                else:
                    eof = True
                    selector.unregister(key.fileobj)
            if terminal is not None and tree.live():
                # Zombie leader is not live. Any remaining descendant is a failure.
                need(not any(pid != proc.pid for pid in tree.live()), 'No live owned descendant after direct exit')
    except BaseException as exc:
        failure = dict(type=type(exc).__name__, message=str(exc)[:2000])
    finally:
        if gate is not None:
            os.close(gate)
        # One finite emergency-cleanup allowance covers direct wait AND all
        # descendant drain. It never becomes another scientific stage budget.
        cleanup_started = time.monotonic()
        cleanup_deadline = cleanup_started + 30
        adopted = []
        def retain_cleanup_pipe():
            nonlocal total, failure
            for _ in range(4):
                try:
                    chunk = os.read(proc.stdout.fileno(), 65536)
                except BlockingIOError:
                    break
                if not chunk:
                    break
                total += len(chunk)
                captured.extend(chunk[:max(0, limit-len(captured))])
            if total > limit:
                failure = failure or dict(type='LogLimit', message='Aggregate log cap exceeded during cleanup')
        while status is None and time.monotonic() < cleanup_deadline:
            try:
                terminal = os.waitid(os.P_PID, proc.pid, os.WEXITED|os.WNOHANG|os.WNOWAIT)
                if terminal is not None:
                    pid, observed_status = os.waitpid(proc.pid, os.WNOHANG)
                    if pid == proc.pid:
                        status = observed_status
                        proc.returncode = os.waitstatus_to_exitcode(status)
                        break
                    need(pid == 0, 'No unrelated direct wait during cleanup')
                tree.signal_all(signal.SIGTERM if time.monotonic()-cleanup_started < 10 else signal.SIGKILL)
                retain_cleanup_pipe()
                try:
                    # No writes or new contended guard transaction during cleanup.
                    census()
                    peak = max(peak, current_rss(owned, tree.known.values()))
                    if peak > CAPS['rss_bytes']:
                        failure = failure or dict(type='RSSLimit', message='Owned RSS cap exceeded during cleanup')
                except BaseException as exc:
                    failure = failure or dict(type=type(exc).__name__, message=str(exc)[:2000])
            except ChildProcessError as exc:
                failure = failure or dict(type='ActualWaitUnavailable', message=str(exc)[:2000])
                break
            except BaseException as exc:
                failure = failure or dict(type=type(exc).__name__, message=str(exc)[:2000])
            time.sleep(min(.05, max(0., cleanup_deadline-time.monotonic())))
        if status is None:
            failure = failure or dict(type='DirectLeaderCleanupDeadline',
                message='Direct leader not genuinely reaped within aggregate30s cleanup, OS wait unknown')
        while status is not None and time.monotonic() < cleanup_deadline:
            try:
                tree.refresh()
                if tree.live():
                    tree.signal_all(signal.SIGKILL)
                while time.monotonic() < cleanup_deadline:
                    try:
                        pid, extra = os.waitpid(-1, os.WNOHANG)
                    except ChildProcessError:
                        break
                    if not pid:
                        break
                    adopted.append(dict(pid=pid, actual_OS_wait_status=extra, returncode=os.waitstatus_to_exitcode(extra)))
                retain_cleanup_pipe()
                if tree.identities_absent() and not owned._children(os.getpid()):
                    break
            except BaseException as exc:
                failure = failure or dict(type=type(exc).__name__, message=str(exc)[:2000])
            time.sleep(min(.05, max(0., cleanup_deadline-time.monotonic())))
        drained = status is not None and tree.identities_absent() and not owned._children(os.getpid())
        if not drained:
            failure = failure or dict(type='OwnedCleanupIncomplete',
                message='Aggregate30s cleanup ended without genuine direct wait and full owned absence')
        try:
            retain_cleanup_pipe()
        except BaseException as exc:
            failure = failure or dict(type=type(exc).__name__, message=str(exc)[:2000])
        selector.close()
        proc.stdout.close()
    group_observations = []
    if drained:
        for group in sorted({r['group'] for r in tree.known.values()}):
            try:
                os.killpg(group, 0)
            except ProcessLookupError:
                absent = True
            else:
                absent = False
            group_observations.append(dict(pgid=group, absent=absent, method='targeted ESRCH after identity drain'))
        if not all(x['absent'] for x in group_observations):
            drained = False
            failure = failure or dict(type='OwnedGroupAbsenceUnknown', message='Group ESRCH not established after drain')
    else:
        group_observations = [dict(pgid=group, absent=None, method='not asserted, owned cleanup incomplete')
            for group in sorted({r['group'] for r in tree.known.values()})]
    if any(row['returncode'] != 0 for row in adopted):
        failure = failure or dict(type='AdoptedChildFailure', message='A genuinely reaped descendant returned nonzero')
    observation = dict(group_observations=group_observations,
        actual_direct_native_wait=status is not None, actual_direct_worker_wait=status is not None,
        actual_OS_wait_status=status, returncode=os.waitstatus_to_exitcode(status) if status is not None else None,
        all_owned_processes_and_groups_drained=drained, operational_failure=failure,
        ownership=tree.evidence(), adopted_waits=adopted, peak_combined_RSS_bytes=peak,
        elapsed_seconds=time.monotonic()-started, log_total_bytes=total,
        cleanup_deadline_seconds=30, cleanup_seconds=time.monotonic()-cleanup_started,
        unresolved_owned_identities=not drained, no_future_stage_authorized=not drained or status is None)
    if failure is not None or not drained or status is None:
        # A strict filesystem writer cannot assume a quiescent namespace while
        # owned writers remain. Preserve truthful partial evidence on stdout.
        print(json.dumps(dict(partial_owned_cleanup_observation=observation)), flush=True)
    return observation, bytes(captured)

class RelateRunner:
    def __init__(self, s, g, storage, owned, started):
        self.s, self.g, self.storage, self.owned, self.started = s, g, storage, owned, started
        self.index = 0
        self.log_bytes = 0
    def run_stage_v1(self, *, label, argv, cwd, caps, resource_binding_ref):
        need(label == ('convert_from_vcf','annotate','infer_all','convert_to_trees')[self.index]
            and cwd == str(local(OUTPUT+'/native')) and caps == CAPS
            and resource_binding_ref == read(self.s['original_infer_request_ref'])['resource_binding_ref'], 'One exact ordered native stage')
        need(not self.owned._children(os.getpid()), 'No unrelated child before native stage')
        proc = subprocess.Popen(argv, cwd=cwd, env=os.environ.copy(), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        observation, log = supervise(proc, self.started, self.s['expires_at_utc'], self.owned,
            census, log_remaining=CAPS['log_bytes']-self.log_bytes)
        self.log_bytes += observation['log_total_bytes']
        need(observation['actual_direct_native_wait'] is True
            and observation['all_owned_processes_and_groups_drained'] is True,
            'Incomplete native cleanup, partial stdout evidence retained, no metadata write or later stage')
        self.index += 1
        observation.update(stage=label, argv=argv, cwd=cwd, resource_binding_ref=resource_binding_ref, fixture_only=False)
        put(PACKET+'/worker_aux/stage_%d_wait.json'%self.index, observation, self.g, self.storage)
        put(PACKET+'/worker_aux/stage_%d_log.json'%self.index,
            dict(text=log.decode('utf-8','replace'), raw_retained_bytes=len(log)), self.g, self.storage, cap=CAPS['log_bytes']+2**20)
        need(observation['returncode'] == 0 and observation['operational_failure'] is None, 'Native failure preserved, no retry')
        return observation

def create_runner_v1(original_request):
    main_module = sys.modules.get('__main__')
    need(main_module is not None and Path(main_module.__file__).resolve() == local(SOURCE), 'Exact admitted pipeline main source')
    runner = getattr(main_module, '_RELATE_ACTIVE_RUNNER', None)
    need(runner is not None and read(runner.s['original_infer_request_ref']) == original_request,
        'Runner only inside actually admitted owned pipeline worker')
    return runner

def environment(aux):
    paths = {k: str(aux/k.lower()) for k in ('TMPDIR','XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME','XDG_STATE_HOME','TORCH_HOME','TORCH_EXTENSIONS_DIR','TORCHINDUCTOR_CACHE_DIR','MPLCONFIGDIR')}
    os.environ.update(paths)
    os.environ.update(TMP=paths['TMPDIR'], TEMP=paths['TMPDIR'], CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
        PYTHONNOUSERSITE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]), LC_ALL='C', LANG='C')
    tempfile.tempdir = paths['TMPDIR']
    return paths

_CANCELLED = None

def cancelled(signum, frame):
    global _CANCELLED
    _CANCELLED = signum

def install_cancellation():
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, cancelled)

def worker(record, gate):
    need(os.read(gate, 3) == b'GO\n', 'No scientific work before recorded ownership')
    os.close(gate)
    install_cancellation()
    s, q, storage = context(record)
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.getpid() == os.getsid(0) == os.getpgrp(), 'Dedicated CPU-only worker')
    os.sched_setaffinity(0, {q['cpu_id']})
    resource.setrlimit(resource.RLIMIT_CORE, (0,0))
    resource.setrlimit(resource.RLIMIT_CPU, (CAPS['wall_seconds'],CAPS['wall_seconds']+10))
    resource.setrlimit(resource.RLIMIT_FSIZE, (CAPS['native_bytes'],)*2)
    environment(local(PACKET+'/worker_aux'))
    import setproctitle
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    g = guard(True)
    _, owned = lifecycle()
    global _RELATE_ACTIVE_RUNNER
    _RELATE_ACTIVE_RUNNER = RelateRunner(s, g, storage, owned, time.monotonic())
    adapter = load(ADAPTER, '_Relate_original_scientific_adapter')
    need(_CANCELLED is None, 'Cancelled before original inference')
    result = adapter.infer(s['original_infer_request_ref'])
    need(_CANCELLED is None and _RELATE_ACTIVE_RUNNER.index == 4 and result['complete'] is True, 'All four original native stages and complete bank')
    return 0

def execute(record):
    need(Path.cwd() == ROOT and not any(n in sys.modules for n in ('torch','numpy','tskit')), 'Fresh metadata-only ROOT interpreter')
    install_cancellation()
    started = time.monotonic()
    s, q, storage = context(record)
    need(not local(OUTPUT).exists() and not local(PACKET+'/worker_aux').exists(), 'Fresh pipeline, no automatic retry')
    g = guard(True)
    factory = load(GUARD_FACTORY, '_Relate_guard_source_constants')
    g.validate_storage_v9(s, dict(output_cap_bytes=CAPS['scope_bytes']), storage,
        source_ref=ref(SOURCE), config_ref=s['refs']['config'], runtime_ref=RUNTIME)
    g.budget_check_v9(storage, boundary=True, required_write_bytes=32*8192)
    aux = local(PACKET+'/worker_aux')
    aux.mkdir()
    paths = environment(aux)
    for p in paths.values():
        Path(p).mkdir()
    os.sched_setaffinity(0, {q['cpu_id']})
    _, owned = lifecycle()
    need(not owned._children(os.getpid()), 'Fresh ROOT with no unrelated children')
    runtime = read(RUNTIME)
    gate_read, gate_write = os.pipe()
    command = [runtime['python_executable'], '-B', str(local(SOURCE)), '--worker-input-ref', json.dumps(record), '--gate-fd', str(gate_read)]
    need(_CANCELLED is None, 'Cancelled before owned worker launch')
    proc = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(gate_read,))
    os.close(gate_read)
    # Owned direct child remains unreaped and gated until its launch is persisted.
    try:
        initial = owned.OwnedTree(proc.pid)
        put(PACKET+'/worker_aux/launch.json', dict(command=command, input_ref=record, ownership=initial.evidence(), gate_closed=True), g, storage)
    except BaseException:
        # EOF makes this exact gated child fail before context/scientific imports.
        os.close(gate_write)
        supervise(proc, started, s['expires_at_utc'], owned, census)
        raise
    last_guard = [time.monotonic()]
    def check():
        census()
        if time.monotonic()-last_guard[0] >= 60:
            g.budget_check_v9(storage, boundary=False)
            last_guard[0] = time.monotonic()
    observation, log = supervise(proc, started, s['expires_at_utc'], owned, check, gate=gate_write)
    need(observation['actual_direct_worker_wait'] is True
        and observation['all_owned_processes_and_groups_drained'] is True,
        'Incomplete worker cleanup, partial stdout evidence retained, scope remains unresolved and no completion')
    wait_ref = put(PACKET+'/worker_aux/actual_wait.json', observation, g, storage)
    put(PACKET+'/worker_aux/worker_log.json', dict(text=log.decode('utf-8','replace'), raw_retained_bytes=len(log)), g, storage, cap=CAPS['log_bytes']+2**20)
    need(observation['returncode'] == 0 and observation['operational_failure'] is None, 'Actual pipeline failed, preserve all charges/artifacts')
    bank_ref = ref(OUTPUT+'/bank_receipt.json')
    bank = read(bank_ref)
    need(bank['complete'] is True and bank['parent_id'] == PARENT and bank['contains_simulation_truth'] is False, 'Genuine original Relate bank')
    final = census()
    completed = put(OUTPUT+'/root_completed.json', dict(schema='prospective1611_accepted_Relate_ROOT_completion_v1',
        status='completed_original_observed_Relate_pipeline', acceptance_ref=ACCEPTANCE, input_ref=record,
        actual_wait_ref=wait_ref, bank_receipt_ref=bank_ref, measured_scope=final, original_monitor_success_relabelled=False,
        original_failed_charge_retained=True, ancestry_score=None, private_truth_opened=False), g, storage)
    g.budget_check_v9(storage, boundary=True)
    return dict(completed_ref=completed, actual_wait_ref=wait_ref)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--admit-request-ref')
    modes.add_argument('--execute-input-ref')
    modes.add_argument('--worker-input-ref')
    p.add_argument('--gate-fd', type=int)
    a = p.parse_args()
    sys.dont_write_bytecode = True
    if a.admit_request_ref:
        print(json.dumps(admit(json.loads(a.admit_request_ref))), flush=True)
    elif a.execute_input_ref:
        print(json.dumps(execute(json.loads(a.execute_input_ref))), flush=True)
    elif a.worker_input_ref:
        return worker(json.loads(a.worker_input_ref), a.gate_fd)
    else:
        print(json.dumps(dict(status='SOURCE_ONLY_INACTIVE', native_work=False)))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
