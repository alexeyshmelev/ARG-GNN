"""Concrete ROOT execution-only native same-charge amendment. Default inert.
Original storage guard binding and new ROOT adapter authentication are distinct.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, collections, ctypes, fcntl, hashlib, importlib, io, itertools, json, os, resource, selectors, signal, stat, subprocess, sys, tempfile, time, types
from research.reviews.C3_prepared_feature_future_integration_source_20261017_v3.contract_v1 import *
from research.reviews.C3_prepared_feature_future_integration_source_20261017_v3 import contract_v1 as c


_DIRECT_SOURCE_V6 = '"""Fresh8GiB evaluator borrows the held GPU0 inode; existing H058/controller remain unchanged."""\nfrom pathlib import Path\nimport fcntl,hashlib,json,os,stat,sys,time\nfrom .contract_v1 import *\nfrom . import contract_v1 as c\nSOURCE=DIRECT;CONTROLLER_PID=3349405;CONTROLLER_START=481281638\n_PEER=None;CANCEL_CHECK=None\ndef install_cancellation_check_v1(callback):\n global CANCEL_CHECK\n require(callable(callback),"Finite actual caller cancellation callback");CANCEL_CHECK=callback\ndef base_v1():return load(BASE_POOL,"_C3_remaining_original_pool")\ndef peer_module_v1():\n global _PEER\n if _PEER is None:_PEER=load(c.pins_v1()["pins"]["H058_DIRECT"],"_C3_remaining_unchanged_H058_guard")\n return _PEER\ndef controller_v1(q,base):\n if CANCEL_CHECK is not None:CANCEL_CHECK()\n e,hq=c.peer_values_v1(q["peer_launch_capsule_ref"])\n try:row=base.proc_identity(e["worker_identity"]["pid"])\n except (FileNotFoundError,ProcessLookupError):row=None\n require(e.get("terminal_peer_absence_preserves_full30GiB")is True,"Peer disappearance never releases its full reserved envelope")\n if row is not None:require(row["uid"]==os.getuid()and row["start_ticks"]==e["worker_identity"]["start"]and row["state"]not in("Z","X"),"Exact live peer PID/start/UID; positive reuse fails closed")\n a=read(q["direct_attachment_authority_ref"]);active(a)\n require(a.get("schema")=="C3_remaining_ROOT_direct_attachment_v1"and a.get("job_id")==q["job_id"]and a.get("peer_launch_capsule_ref")==q["peer_launch_capsule_ref"]and a.get("borrowed_physical_inode_attachment_authorized")is True and a.get("old_controller_and_H058_preserved")is True and a.get("old_controller_submissions_remain_revoked")is True and a.get("evaluator_cuda_bytes")==8*2**30 and a.get("evaluator_margin_bytes")==2*2**30 and a.get("C3_reserved_bytes")==26*2**30 and a.get("H058_reserved_bytes")==30*2**30 and a.get("headroom_bytes")==4*2**30,"Separate exact direct-borrow authority, never reuse/relabel C3 slot")\n cap,registry,c3,oldauth=peer_module_v1().controller_v1(hq,base)\n require(datetime.fromisoformat(q["expires_at_utc"])<=datetime.fromisoformat(hq["expires_at_utc"]),"New borrower cannot outlive peer/controller authority")\n return cap,registry,c3,a\ndef capacity_v1(q,base,*,worker_pid=None):\n controller_v1(q,base);e,hq=c.peer_values_v1(q["peer_launch_capsule_ref"])\n sample=peer_module_v1().capacity_v1(hq,base,worker_pid=e["worker_identity"]["pid"])\n gpu=sample["GPU0"]\n try:peer_now=base.proc_identity(e["worker_identity"]["pid"])\n except (FileNotFoundError,ProcessLookupError):peer_now=None\n if peer_now is None:require(not any(r["pid"]==e["worker_identity"]["pid"]for r in gpu["compute"]),"Absent peer PID cannot retain an unexplained CUDA context")\n else:require(peer_now["uid"]==os.getuid()and peer_now["start_ticks"]==e["worker_identity"]["start"],"Peer identity remains exact after capacity sample")\n mine=[r for r in gpu["compute"]if worker_pid is not None and r["pid"]==worker_pid]\n require(len(mine)<=1 and all(r["name"]=="AI Auto Research Pipeline Test"for r in mine),"Own exact titled evaluator CUDA identity")\n own=sum(r["used_mib"]for r in mine);limit=10*1024\n require(own<=limit,"8GiB allocator plus2GiB nonTorch evaluator envelope")\n unfilled=max(0,limit-own);projected=sample["projected_mib"]+unfilled;free_required=sample["actual_free_required_mib"]+unfilled\n require(projected<=gpu["total_mib"]and gpu["total_mib"]-gpu["used_mib"]>=free_required,"Preserve C3_26+H058_30+EVAL_10+4headroom plus actual external/driver occupancy")\n return dict(schema="C3_remaining_conservative_GPU0_capacity_v1",GPU0=gpu,original_unchanged_H058_capacity_sample=sample,evaluator_observed_mib=own,evaluator_cap_plus_margin_mib=limit,projected_mib=projected,actual_free_required_mib=free_required,observed_unix=time.time(),C3_slot_reused=False)\n\n\ndef held_writable_probe_v1(path):\n    with path.open(\'r+\') as probe:\n        try:\n            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)\n        except BlockingIOError:\n            return True\n        else:\n            fcntl.flock(probe, fcntl.LOCK_UN)\n            raise RuntimeError(\'Expected physical/logical exclusion lock is unheld\')\n\ndef attach_v1(spec, q):\n    require(spec[\'phase\'] in GPU_PHASES, \'Only exact remaining-cell prediction attaches GPU0\')\n    base = base_v1()\n    physical = local(\'.runtime/locks/gpu_0.lock\')\n    logical_path = local(spec[\'worker_auxiliary_scope\'] + \'/C3_remaining_GPU0.logical.lock\')\n    physical_fd = logical_fd = None\n    with base.lock_file(ROOT, \'scheduler.lock\') as mutex:\n        fcntl.flock(mutex, fcntl.LOCK_EX)\n        try:\n            (cap, registry, c3, authority) = controller_v1(q, base)\n            physical_fd = os.open(physical, os.O_RDWR | os.O_NOFOLLOW)\n            st = os.fstat(physical_fd)\n            named = physical.lstat()\n            require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and ((st.st_dev, st.st_ino) == (named.st_dev, named.st_ino)), \'Original canonical physical inode\')\n            raw_bytes = os.pread(physical_fd, 65536, 0)\n            require(len(raw_bytes) == st.st_size and hashlib.sha256(raw_bytes).hexdigest() == registry[\'lease_sha256\'], \'Original registry-bound physical lease bytes\')\n            metadata = json.loads(raw_bytes)\n            require(metadata == registry[\'lease_metadata\'] and metadata[\'launcher_pid\'] == CONTROLLER_PID and (metadata[\'gpu\'] == 0) and (metadata[\'uuid\'] == GPU0_UUID), \'Original actual physical lease identity\')\n            held_writable_probe_v1(physical)\n            logical_fd = os.open(logical_path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 384)\n            fcntl.flock(logical_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n            capacity = capacity_v1(q, base)\n            ticket = dict(schema=\'C3_remaining_borrowed_GPU0_attachment_v1\', job_id=spec[\'job_id\'], physical_gpu=0, gpu_uuid=GPU0_UUID, borrowed_physical_inode=True, physical_exclusive_lock_owner=\'unchanged_original_controller\', controller_pid=CONTROLLER_PID, controller_start_ticks=CONTROLLER_START, physical_inode=st.st_ino, physical_device=st.st_dev, physical_lease_sha256=hashlib.sha256(raw_bytes).hexdigest(), physical_lease_metadata=metadata, logical_lock_path=str(logical_path), logical_inode=os.fstat(logical_fd).st_ino, logical_device=os.fstat(logical_fd).st_dev, C3_remaining_logical_lock_owned_by_supervisor=True, capacity_at_attachment=capacity, direct_attachment_authority_ref=q[\'direct_attachment_authority_ref\'])\n            return (physical_fd, logical_fd, ticket)\n        except BaseException:\n            for fd in (physical_fd, logical_fd):\n                if fd is not None:\n                    os.close(fd)\n            raise\n\ndef install_before_torch_v1(spec, q, ticket):\n    require(not any((n in sys.modules for n in (\'torch\', \'torch_geometric\'))), \'Direct lease/title install before science\')\n    g = load(GPU0_GUARD, \'_H058_original_GPU0_guard\')\n    native = g.install_before_torch_v1()\n    original = native.validate_lease\n    base = base_v1()\n    logical_fd = int(os.environ[\'downstream_LOGICAL_LOCK_FD\'])\n    physical_fd = int(os.environ[\'ARGGNN_GPU_LEASE_FD\'])\n\n    def validate_lease(environ=None):\n        lease = original(environ)\n        require(lease[\'lease_fd\'] == physical_fd and lease[\'lease_inode\'] == ticket[\'physical_inode\'] and (lease[\'lease_device\'] == ticket[\'physical_device\']) and (lease[\'lease_metadata_sha256\'] == ticket[\'physical_lease_sha256\']) and (lease[\'launcher_metadata\'] == ticket[\'physical_lease_metadata\']), \'Stable authenticated borrowed physical inode, never controller inheritance\')\n        controller_v1(q, base)\n        st = os.fstat(logical_fd)\n        path = Path(ticket[\'logical_lock_path\'])\n        named = path.lstat()\n        require(stat.S_ISREG(st.st_mode) and (st.st_dev, st.st_ino) == (named.st_dev, named.st_ino) == (ticket[\'logical_device\'], ticket[\'logical_inode\']), \'Actual inherited H058 logical lock descriptor\')\n        held_writable_probe_v1(path)\n        capacity_v1(q, base, worker_pid=os.getpid())\n        return dict(lease, borrowed_physical_inode=True, physical_exclusive_lock_owned_by_current_controller=True, C3_remaining_logical_lock_inode=ticket[\'logical_inode\'], C3_remaining_direct_attachment_source_ref=ref(SOURCE))\n    native.validate_lease = validate_lease\n    validate_lease()\n    native.apply_and_validate_title()\n    return native\n\ndef check_attachment_v1(q, ticket, physical_fd, logical_fd):\n    controller_v1(q, base_v1())\n    for (fd, path, device, inode) in ((physical_fd, local(\'.runtime/locks/gpu_0.lock\'), ticket[\'physical_device\'], ticket[\'physical_inode\']), (logical_fd, Path(ticket[\'logical_lock_path\']), ticket[\'logical_device\'], ticket[\'logical_inode\'])):\n        opened = os.fstat(fd)\n        named = path.lstat()\n        require(stat.S_ISREG(opened.st_mode) and (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino) == (device, inode), \'Stable actual physical/logical inode identity\')\n        held_writable_probe_v1(path)\n    contents = os.pread(physical_fd, 65536, 0)\n    require(len(contents) == os.fstat(physical_fd).st_size and hashlib.sha256(contents).hexdigest() == ticket[\'physical_lease_sha256\'], \'Actual original physical lease bytes unchanged\')\n'
_DIRECT_SOURCE_V6_SHA = 'ae0b51362ce61ab1054df10ea92025a1be5de4642a5030a2f8c87bbbe2103fa2'
_DIRECT_MODULE_V6 = None
def _same_source_direct_v6():
    global _DIRECT_MODULE_V6
    if _DIRECT_MODULE_V6 is None:
        require(hashlib.sha256(_DIRECT_SOURCE_V6.encode()).hexdigest()==_DIRECT_SOURCE_V6_SHA,
            "Exact embedded unchanged direct attachment role")
        module=types.ModuleType("_C3_prepared_same_source_direct_v6")
        module.__dict__.update(__file__=__file__,__package__=__package__)
        sys.modules[module.__name__]=module
        exec(compile(_DIRECT_SOURCE_V6,__file__+"#embedded_direct", "exec",dont_inherit=True,optimize=0),module.__dict__)
        _DIRECT_MODULE_V6=module
    return _DIRECT_MODULE_V6

class _HistoryV8:
    def __init__(self):
        self.total=0;self.first=[];self.last=collections.deque(maxlen=OWNED_LIMITS_V8["history_rows"])
        self.digest=hashlib.sha256(b"H058-v8-bounded-history-v1").digest();self.nonzero=0
    def add(self,row):
        data=json.dumps(row,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        self.digest=hashlib.sha256(self.digest+len(data).to_bytes(8,"big")+data).digest();self.total+=1
        if len(self.first)<OWNED_LIMITS_V8["history_rows"]:self.first.append(dict(row))
        self.last.append(dict(row));self.nonzero+=int(row.get("returncode",0)!=0)
    def evidence(self):
        return dict(count=self.total,first=self.first,last=list(self.last),ordered_chain_sha256=self.digest.hex(),
            nonzero_returncodes=self.nonzero,full_historical_roster_serialized=False)

class _PendingV8(dict):
    def __init__(self,wrapper):self.wrapper=wrapper
    def __setitem__(self,pid,row):
        w=self.wrapper;new_identity=pid not in self
        if new_identity and (len(self)>=OWNED_LIMITS_V8["pending"] or w.discovered.total>=OWNED_LIMITS_V8["discoveries"]):
            w.complete=False;raise ValueError("Finite v8 pending/discovery identity bound; incomplete ownership")
        # Original ancestry has authenticated this row. Check the group bound
        # before touching row, group set or discovery/retirement history.
        w.remember_group_v9(row)
        if new_identity:w.discovered.add(dict(pid=pid,start=row["start"],group=row["group"]))
        super().__setitem__(pid,row)

class OwnedLifecycleV8:
    """Reuse original ancestry discovery; retire only exact identities proven gone/replaced."""
    def __init__(self,backend,leader,**kwargs):
        self.backend=backend;self.complete=True;self.groups=set();self.invalid_groups=_HistoryV8()
        self.discovered=_HistoryV8();self.retired=_HistoryV8();self.reused=_HistoryV8()
        self.retirement_reasons=dict(absent=0,replaced=0)
        self.original=backend.OwnedTree(leader,**kwargs)
        initial=self.original.known;self.original.known=_PendingV8(self)
        for pid,row in initial.items():self.original.known[pid]=row
        self._original_read_stat=self.original.read_stat
        self.original.read_stat=self.read_stat_v9
    def remember_group_v9(self,row,*,strict=True):
        group=row["group"]
        if type(group)is not int or group<=0:
            self.complete=False
            self.invalid_groups.add(dict(pid=row.get("pid"),start=row.get("start"),state=row.get("state"),
                raw_group=repr(group)[:128],raw_group_type=type(group).__name__,
                nonpositive_or_invalid_group=True,group_absence_inferred=False))
            return False
        if group not in self.groups and len(self.groups)>=OWNED_LIMITS_V8["groups"]:
            self.complete=False
            if strict:raise ValueError("Finite v9 observed owned-group bound; incomplete ownership")
            return False
        self.groups.add(group);return True
    def read_stat_v9(self,pid,*,strict=True):
        actual=self._original_read_stat(pid);expected=self.known.get(pid)
        if self.backend._same(actual,expected):self.remember_group_v9(actual,strict=strict)
        return actual
    def cleanup_read_stat_v9(self,pid):
        # Missing group capacity never blocks pidfd stop of an already-known identity.
        # Incomplete is permanent; unknown-group evidence can never authorize drain.
        return self.read_stat_v9(pid,strict=False)
    @property
    def known(self):return self.original.known
    def _retire(self):
        for pid,row in list(self.known.items()):
            actual=self.original.read_stat(pid)
            if actual is None or not self.backend._same(actual,row):
                reason="absent" if actual is None else "replaced"
                self.retired.add(dict(pid=pid,start=row["start"],group=row["group"],reason=reason))
                self.retirement_reasons[reason]+=1;del self.known[pid]
        # Zombies remain pending until a genuine wait/absence; no guessed reaping.
    def refresh(self):
        try:
            self._retire();self.original.refresh();self._retire()
            for pid in sorted(self.original.reused):self.reused.add(dict(pid=pid,excluded=True))
            self.original.reused.clear()
            return self
        except BaseException:
            self.complete=False;raise
    def live(self):
        self.refresh()
        return [pid for pid,row in self.known.items()
            if self.backend._same(actual:=self.original.read_stat(pid),row) and actual["state"] not in ("Z","X")]
    def identities_absent(self):
        self.refresh()
        return self.complete and not self.known
    def evidence(self):
        groups=sorted(self.groups);encoded=json.dumps(groups,separators=(",",":")).encode()
        return dict(owned_tree_constructed=True,complete_owned_roster_known=self.complete,
            owned_roster_truncated=not self.complete,
            current_owned_identities=[dict(pid=p,start=r["start"],group=r["group"],state=r["state"])
                for p,r in sorted(self.known.items())],
            discovered_identity_history=self.discovered.evidence(),retired_identity_history=self.retired.evidence(),
            retirement_reasons=self.retirement_reasons,reused_numeric_PID_history=self.reused.evidence(),
            historical_roster_summarized=True,retired_exact_identities_confirmed_absent_or_replaced=True,
            invalid_group_observation_history=self.invalid_groups.evidence(),
            nonpositive_group_syscalls_forbidden=True,invalid_group_makes_ownership_permanently_incomplete=True,
            group_count=len(groups),group_set_sha256=hashlib.sha256(encoded).hexdigest(),
            group_first=groups[:16],group_last=groups[-16:],limits=OWNED_LIMITS_V8,
            signals_use_pidfd_only=True,no_numeric_group_signals=True)

def observe_groups_v8(tree,deadline):
    history=_HistoryV8();absent=0;errors=[]
    for group in sorted(tree.groups):
        if time.monotonic()>=deadline:break
        gone=None
        if type(group)is not int or group<=0:
            tree.complete=False
            if len(errors)<16:errors.append("Invalid owned group rejected without syscall")
            history.add(dict(pgid=group,absent=None,method="invalid group rejected without syscall"))
            continue
        try:os.killpg(group,0)
        except ProcessLookupError:gone=True
        except OSError as exc:
            if len(errors)<16:errors.append(type(exc).__name__+": "+str(exc)[:512])
        else:gone=False
        absent+=int(gone is True)
        history.add(dict(pgid=group,absent=gone,method="targeted ESRCH within original aggregate cleanup allowance"))
    finished_within_deadline=time.monotonic()<=deadline
    return dict(history=history.evidence(),expected_group_count=len(tree.groups),observed_group_count=history.total,
        absent_group_count=absent,errors=errors,scan_finished_within_deadline=finished_within_deadline,
        all_owned_groups_observed_absent=finished_within_deadline and bool(tree.groups) and history.total==len(tree.groups)==absent)

class _RAMLog:

    def __init__(self, cap):
        self.cap = cap
        self.parts = collections.deque()
        self.kept = 0
        self.total = 0

    def add(self, data):
        self.total += len(data)
        self.parts.append(data)
        self.kept += len(data)
        while self.parts and self.kept > self.cap:
            first = self.parts.popleft()
            over = self.kept - self.cap
            if len(first) > over:
                self.parts.appendleft(first[over:])
                self.kept -= over
            else:
                self.kept -= len(first)

    def record(self):
        return dict(tail=b''.join(self.parts).decode('utf-8', errors='replace'), total_pipe_bytes=self.total, retained_raw_bytes=self.kept, cap_bytes=self.cap, discarded_earlier_bytes=self.total - self.kept, buffered_in_RAM_until_actual_drain=True)

def _subreaper():
    libc = ctypes.CDLL(None, use_errno=True)
    require(libc.prctl(36, 1, 0, 0, 0) == 0, 'Linux subreaper needed to adopt/drain native DataLoader descendants')

def _children():
    raw = Path('/proc/self/task/%d/children' % os.getpid()).read_text().strip()
    return [int(p) for p in raw.split()] if raw else []

def rss_identity_v1(owned, identity):
    before = owned._stat(identity['pid'])
    if before is None:
        return 0
    require(owned._same(before, identity), 'RSS observation identity changed')
    try:
        pages = int(Path('/proc/%d/statm' % identity['pid']).read_text().split()[1])
    except (FileNotFoundError, ProcessLookupError):
        return 0
    after = owned._stat(identity['pid'])
    if after is None:
        return 0
    require(owned._same(after, identity), 'RSS observation identity changed')
    return pages * os.sysconf('SC_PAGE_SIZE')

def worker_v1(input_ref, gate_fd, attachment_ref):
    require(os.read(gate_fd, 1) == b'G', 'Actual parent-owned launch gate')
    os.close(gate_fd)
    (s, q, cfg, storage, _) = validate_input_v1(input_ref)
    require(Path.cwd() == ROOT and os.getpid() == os.getpgrp() == os.getsid(0) and sys.dont_write_bytecode, 'Fresh owned worker session')
    paths = worker_env_paths_v1(s)
    require(all((os.environ.get(k) == v and Path(v).resolve() == Path(v) and Path(v).is_dir() for (k, v) in paths.items())), 'All own TMP/cache paths charged')
    tempfile.tempdir = paths['TMPDIR']
    from research.reviews.C3_prepared_feature_future_integration_source_20261017_v3 import worker_v1 as worker
    if s['phase'] == 'matrix':
        require(attachment_ref is None and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only matrix has no GPU attachment')
        worker.run_matrix_v1(input_ref, s, q, cfg, storage)
        return 0
    require(attachment_ref['path'] == s['worker_auxiliary_scope'] + '/attachment.json', 'Exact own attachment')
    ticket = read(attachment_ref)
    require(ticket['job_id'] == s['job_id'] and ticket['direct_attachment_authority_ref'] == q['direct_attachment_authority_ref'], 'Exact actual new attachment')
    direct = _same_source_direct_v6()
    native = direct.install_before_torch_v1(s, q, ticket)
    worker.run_prediction_v1(input_ref, s, q, cfg, storage, ticket, native)
    return 0

def execute_v1(input_ref):
    stage_started = time.monotonic()
    require(Path.cwd() == ROOT and sys.dont_write_bytecode, 'Exact ROOT cwd/no bytecode')
    (s, q, cfg, storage, protocol) = validate_input_v1(input_ref)
    from research.reviews.C3_prepared_feature_future_integration_source_20261017_v3.worker_v1 import check_external_prerequisites_v1
    check_external_prerequisites_v1(q)
    require(Path(sys.executable).resolve() == (ROOT / '.runtime/env/bin/python').resolve(), 'Original pinned runtime interpreter')
    require_work_window_v2(s)
    expires_unix = datetime.fromisoformat(s['expires_at_utc']).timestamp()
    work_deadline = min(work_deadline_monotonic_v1(), stage_started + CAPS['max_seconds'] - CAPS['work_cutoff_margin_seconds'], time.monotonic() + expires_unix - time.time() - CAPS['work_cutoff_margin_seconds'])
    g = guard_v1(q)
    g.validate_storage_v9(s, dict(output_cap_bytes=CAPS['reservation_bytes']), storage, source_ref=ref(DRIVER), config_ref=s['refs']['config'], runtime_ref=s['refs']['runtime'])
    prepare_master_parents_v1(s, g, storage)
    (out, aux) = (local(s['output_directory']), local(s['worker_auxiliary_scope']))
    require(not out.exists() and (not aux.exists()), 'Fresh exclusive science output/AUX, no replay')
    paths = worker_env_paths_v1(s)
    directories = sorted({Path(v) for v in paths.values()}, key=lambda p: (len(p.parts), str(p)))
    g.budget_check_v9(storage, boundary=True, required_write_bytes=(2 + len(directories)) * 8192)
    active_amendment_v1()
    aux.mkdir()
    for directory in directories:
        active_amendment_v1()
        directory.mkdir()
    last_census = charged_census_v1(s)
    peak_scope = sum(last_census.values())
    compact_v1(s, 'effective_execution_view.json', dict(schema='downstream_actual_explicit_execution_view_v1', original_input_ref=input_ref, actual_execution_amendment_ref=current_amendment_ref_v1(), adapter_manifest_ref=ref(MANIFEST), effective_output_directory=s['output_directory'], effective_admin_directory=s['worker_auxiliary_scope'], original_storage_ref=s['refs']['storage'], master_reservation_authenticates_execution_sources=True, separate_ROOT_phase_authority_authenticates_later_prerequisites=True, whole_original_output_root=c.STATE[0]['output_directory'], whole_original_AUX_root=c.STATE[0]['worker_auxiliary_scope']), g, storage)
    load(PIDFD, 'pidfd_backend_v1')
    owned = load(OWNED, '_downstream_original_full_parent_owned_lifecycle')
    sys.modules['pidfd_backend_v1'].self_probe_v1()
    require(not owned._children(os.getpid()), 'Supervisor has no unrelated children')
    _subreaper()
    parent_identity = owned._stat(os.getpid())
    logs = _RAMLog(CAPS['log_bytes'])
    selector = selectors.DefaultSelector()
    proc = tree = leader_seed = None
    failure = cancelled = None
    peak = 0
    adopted = _HistoryV8()
    original = {}
    pipe_nonblocking = pipe_eof = gate_released = False
    raw_status = None
    cleanup_errors = []
    identity_absence = None
    observations = []
    group_scan = None
    gate_read = gate_write = physical_fd = logical_fd = None
    attachment_ref = None
    start = stage_started
    capacity_samples = 0
    last_capacity = None
    next_capacity = 0.0

    def stop(sig, frame):
        nonlocal cancelled
        cancelled = sig
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            original[sig] = signal.signal(sig, stop)
        env = os.environ.copy()
        env.pop('PYTHONHOME', None)
        for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'NODE_RANK', 'MASTER_ADDR', 'MASTER_PORT'):
            env.pop(name, None)
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', WANDB_MODE='disabled', HF_DATASETS_OFFLINE='1', PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1', CUDA_CACHE_MAXSIZE=str(256 * 2 ** 20), VIRTUAL_ENV=str(local('.runtime/env')), BLIS_NUM_THREADS='32')
        env['PATH'] = str(local('.runtime/env') / 'bin') + os.pathsep + env.get('PATH', '')
        env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTHONPATH='', OMP_NUM_THREADS='32', MKL_NUM_THREADS='32', OPENBLAS_NUM_THREADS='32', NUMEXPR_NUM_THREADS='32', ARGGNN_LOGICAL_JOB_NAME=s['job_id'], ARGGNN_COMPLETED_CREDITS_POLICY_REF=json.dumps(q['capacity_policy_ref']), ARGGNN_BUDGET_V9_SOURCE_SHA256=storage['budget_components']['implementation']['sha256'], ARGGNN_BUDGET_V9_LAUNCHER_SHA256=storage['budget_components']['launcher']['sha256'])
        env.update(paths)
        if s['phase'] in GPU_PHASES:
            direct = _same_source_direct_v6()
            direct.install_cancellation_check_v1(lambda : require(cancelled is None and time.monotonic() < work_deadline, 'Actual cancellation or anchored work cutoff during status proof'))
            (physical_fd, logical_fd, ticket) = direct.attach_v1(s, q)
            attachment_ref = compact_v1(s, 'attachment.json', ticket, g, storage)
            env.update(CUDA_VISIBLE_DEVICES=GPU0_UUID, ARGGNN_PHYSICAL_GPU='0', ARGGNN_GPU_LEASE=str(local('.runtime/locks/gpu_0.lock')), ARGGNN_GPU_LEASE_FD=str(physical_fd), downstream_LOGICAL_LOCK_FD=str(logical_fd), ARGGNN_JOB_NAME=ticket['physical_lease_metadata']['job'])
            last_capacity = ticket['capacity_at_attachment']
            capacity_samples = 1
        else:
            env['CUDA_VISIBLE_DEVICES'] = ''
            for name in ('ARGGNN_PHYSICAL_GPU', 'ARGGNN_GPU_LEASE', 'ARGGNN_GPU_LEASE_FD', 'downstream_LOGICAL_LOCK_FD', 'ARGGNN_JOB_NAME'):
                env.pop(name, None)
        (gate_read, gate_write) = os.pipe()
        command = [str(local(NATIVE_PYTHON_REF['path'])), '-B', '-s', '-m', 'research.reviews.C3_prepared_feature_future_integration_source_20261017_v3.execution_v1', '--worker-input-ref', json.dumps(input_ref, separators=(',', ':')), '--gate-fd', str(gate_read)]
        if attachment_ref is not None:
            command += ['--attachment-ref', json.dumps(attachment_ref, separators=(',', ':'))]
        command += ['--amendment-ref', json.dumps(current_amendment_ref_v1(), separators=(',', ':'))]
        inherited = tuple((fd for fd in (gate_read, physical_fd, logical_fd) if fd is not None))
        require(time.monotonic() < work_deadline, 'No child launch inside cleanup/evidence reserve')
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0, start_new_session=True, pass_fds=inherited)
        os.close(gate_read)
        gate_read = None
        os.set_blocking(proc.stdout.fileno(), False)
        pipe_nonblocking = True
        leader_seed = owned._stat(proc.pid)
        require(leader_seed is not None and leader_seed['ppid'] == os.getpid() and (leader_seed['group'] == leader_seed['session'] == proc.pid), 'Actual direct new-session identity seed')
        tree = OwnedLifecycleV8(owned,proc.pid)
        rss_owned=types.SimpleNamespace(_stat=tree.read_stat_v9,_same=owned._same)
        compact_v1(s, 'launch.json', dict(schema='downstream_fixed_full_parent_actual_child_launch_v1', pid=proc.pid, parent_pid=os.getpid(), input_ref=input_ref, command=command, ownership=tree.evidence(), gate_still_closed=True, attachment_ref=attachment_ref, borrowed_physical_inode=attachment_ref is not None, controller_inheritance_claimed=False), g, storage)
        selector.register(proc.stdout, selectors.EVENT_READ)
        active_amendment_v1()
        require(cancelled is None and time.monotonic() < work_deadline, 'Cancelled or anchored scientific work cutoff reached before gate release')
        os.write(gate_write, b'G')
        gate_released = True
        os.close(gate_write)
        gate_write = None
        while True:
            require(cancelled is None, 'ROOT cancellation signal ' + str(cancelled))
            require(time.monotonic() < work_deadline, 'Pre-expiry scientific work cutoff; cleanup/evidence reserve retained')
            terminal = os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            tree.refresh()
            require(tree.complete, 'Complete bounded-live downstream ownership')
            peak = max(peak, rss_identity_v1(owned, parent_identity) + sum((rss_identity_v1(rss_owned, r) for r in tree.known.values())))
            require(peak <= CAPS['max_cpu_rss_bytes'], 'Combined owned RSS cap')
            last_census = charged_census_v1(s)
            peak_scope = max(peak_scope, sum(last_census.values()))
            if s['phase'] in GPU_PHASES and time.monotonic() >= next_capacity:
                last_capacity = direct.capacity_v1(q, direct.base_v1(), worker_pid=proc.pid)
                direct.check_attachment_v1(q, ticket, physical_fd, logical_fd)
                capacity_samples += 1
                next_capacity = time.monotonic() + 2
            if terminal is not None:
                if any((pid != proc.pid for pid in tree.live())):
                    failure = failure or 'Direct worker exited with live owned descendants'
                break
            for (key, _) in selector.select(timeout=0.2):
                data = os.read(key.fileobj.fileno(), 65536)
                if data:
                    logs.add(data)
                else:
                    pipe_eof = True
                    selector.unregister(key.fileobj)
    except BaseException as exc:
        failure = failure or type(exc).__name__ + ': ' + str(exc)[:2000]
    finally:
        cleanup_started = time.monotonic()
        cleanup_deadline = cleanup_started + CAPS['cleanup_seconds']
        for fd in (gate_read, gate_write):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError as exc:
                    if len(cleanup_errors) < 16:
                        cleanup_errors.append(type(exc).__name__ + ': ' + str(exc)[:1000])

        def retain_pipe():
            nonlocal pipe_eof
            if proc is None or proc.stdout is None or (not pipe_nonblocking):
                return
            for _ in range(4):
                try:
                    data = os.read(proc.stdout.fileno(), 65536)
                except BlockingIOError:
                    break
                if not data:
                    pipe_eof = True
                    break
                logs.add(data)

        def note_cleanup(exc):
            nonlocal failure
            message = type(exc).__name__ + ': ' + str(exc)[:1000]
            failure = failure or 'Owned cleanup observation failed: ' + message
            if len(cleanup_errors) < 16:
                cleanup_errors.append(message)
        if proc is not None:
            while time.monotonic() < cleanup_deadline:
                try:
                    if raw_status is None:
                        (waited, status) = os.waitpid(proc.pid, os.WNOHANG)
                        require(waited in (0, proc.pid), 'Only actual direct child OS wait')
                        if waited == proc.pid:
                            raw_status = status
                            proc.returncode = os.waitstatus_to_exitcode(status)
                    if tree is not None:
                        tree.refresh()
                except BaseException as exc:
                    note_cleanup(exc)
                rows = list(tree.known.values()) if tree is not None else [leader_seed] if leader_seed else []
                signum = signal.SIGTERM if time.monotonic() - cleanup_started < 10 else signal.SIGKILL
                for row in rows:
                    if time.monotonic() >= cleanup_deadline:
                        break
                    try:
                        owned.signal_identity(row, signum, read_stat=tree.cleanup_read_stat_v9 if tree is not None else owned._stat)
                    except BaseException as exc:
                        note_cleanup(exc)
                if raw_status is not None:
                    while time.monotonic() < cleanup_deadline:
                        try:
                            (pid, status) = os.waitpid(-1, os.WNOHANG)
                        except ChildProcessError:
                            break
                        except BaseException as exc:
                            note_cleanup(exc)
                            break
                        if pid == 0:
                            break
                        adopted.add(dict(pid=pid,status=status,returncode=os.waitstatus_to_exitcode(status)))
                        if adopted.total>OWNED_LIMITS_V8["adopted_waits"]:
                            if tree is not None:tree.complete=False
                            failure=failure or "Finite adopted-wait bound; continue genuine reaping"
                try:
                    retain_pipe()
                    if raw_status is not None and tree is None:
                        break
                    if raw_status is not None and tree is not None and (time.monotonic() < cleanup_deadline):
                        identity_absence = tree.identities_absent() and (not owned._children(os.getpid()))
                        if identity_absence and pipe_eof:
                            group_scan=observe_groups_v8(tree,cleanup_deadline)
                            observations=group_scan["history"]["first"]+group_scan["history"]["last"]
                            if group_scan["all_owned_groups_observed_absent"]:
                                break
                except BaseException as exc:
                    note_cleanup(exc)
                time.sleep(min(0.05, max(0.0, cleanup_deadline - time.monotonic())))
            if time.monotonic() < cleanup_deadline:
                try:
                    retain_pipe()
                except BaseException as exc:
                    note_cleanup(exc)
            if proc.stdout is not None:
                try:
                    proc.stdout.close()
                except OSError as exc:
                    note_cleanup(exc)
        selector.close()
        for (sig, handler) in original.items():
            signal.signal(sig, handler)
        cleanup_finished = time.monotonic()
    rc = os.waitstatus_to_exitcode(raw_status) if raw_status is not None else None
    if proc is None:
        failure = failure or 'No worker spawned'
    if raw_status is None:
        failure = failure or 'Direct OS wait unknown after finite cleanup'
    if group_scan is None:
        group_scan=dict(all_owned_groups_observed_absent=False,expected_group_count=len(tree.groups)if tree else None,
            observed_group_count=0,absent_group_count=0,method="not asserted; owned cleanup incomplete")
    roster_complete=tree is not None and tree.complete and adopted.total<=OWNED_LIMITS_V8["adopted_waits"]
    drained=raw_status is not None and roster_complete and identity_absence is True and pipe_eof and group_scan["all_owned_groups_observed_absent"]
    if not drained:
        failure = failure or 'Finite cleanup ended without genuine wait and complete owned absence'
    if rc is not None and rc != 0:
        failure = failure or 'Actual worker exit ' + str(rc)
    if adopted.nonzero:
        failure = failure or 'Nonzero actual adopted descendant wait'
    for fd in (physical_fd, logical_fd):
        if fd is not None:
            try:
                os.close(fd)
            except OSError as exc:
                failure = failure or 'Attachment descriptor close: ' + str(exc)[:1000]
    ownership=tree.evidence()if tree is not None else dict(owned_tree_constructed=False,
        actual_direct_identity_seed=leader_seed,complete_owned_roster_known=False,owned_roster_truncated=True,signals_use_pidfd_only=True)
    ownership.update(complete_owned_roster_known=roster_complete,known_identity_count=len(tree.known)if tree else None)
    observation = dict(schema='C3_remaining_cells_bounded_cleanup_observation_v2', input_ref=input_ref, actual_execution_amendment_ref=current_amendment_ref_v1(), adapter_manifest_ref=ref(MANIFEST), original_storage_ref=s['refs']['storage'], whole_original_roots_measured=True, master_reservation_authenticates_execution_sources=True, separate_ROOT_phase_authority_authenticates_later_prerequisites=True, pid=proc.pid if proc is not None else None, returncode=rc, actual_OS_wait_status=raw_status, actual_direct_worker_wait=raw_status is not None, all_owned_processes_and_groups_drained=drained, owned_identity_absence=identity_absence, group_observations=observations, ownership=ownership, adopted_waits=adopted.evidence(), group_absence_summary=group_scan, group_observations_summarized=True, operational_failure=failure, gate_released=gate_released, pipe_EOF_observed=pipe_eof, owned_roster_policy_ref=ref(DRIVER), owned_roster_policy=OWNED_LIMITS_V8, complete_owned_roster_known=roster_complete, cleanup_errors=cleanup_errors, cleanup_deadline_seconds=CAPS['cleanup_seconds'], cleanup_seconds=cleanup_finished - cleanup_started, elapsed_seconds=time.monotonic() - start, work_cutoff_margin_seconds=CAPS['work_cutoff_margin_seconds'], terminal_evidence_margin_seconds=CAPS['terminal_evidence_margin_seconds'], actual_authority_seconds_remaining=expires_unix - time.time(), no_future_stage_authorized=failure is not None or not drained, storage_close_or_credit_authorized=False, borrowed_physical_inode=attachment_ref is not None, controller_inheritance_claimed=False)
    minimal_facts = dict(schema='C3_remaining_cells_actual_minimal_wait_facts_v2', input_ref=input_ref, actual_execution_amendment_ref=current_amendment_ref_v1(), adapter_manifest_ref=ref(MANIFEST), original_storage_ref=s['refs']['storage'], pid=proc.pid if proc is not None else None, actual_OS_wait_status=raw_status, returncode=rc, actual_direct_worker_wait=raw_status is not None, all_owned_processes_and_groups_drained=drained, complete_owned_roster_known=roster_complete, known_identity_count=len(tree.known) if tree is not None else None, adopted_wait_count=adopted.total, adopted_wait_nonzero_count=adopted.nonzero, pending_owned_identity_cap=OWNED_LIMITS_V8['pending'], discovered_owned_identity_count=tree.discovered.total if tree else None, retired_owned_identity_count=tree.retired.total if tree else None, group_absence_summary=group_scan, owned_roster_policy_ref=ref(DRIVER), owned_roster_policy=OWNED_LIMITS_V8, operational_failure=failure, cleanup_deadline_seconds=CAPS['cleanup_seconds'], cleanup_seconds=cleanup_finished - cleanup_started, pipe_EOF_observed=pipe_eof, minimal_wait_only_not_scientific_completion=True, full_terminal_evidence_still_required=True, no_future_stage_authorized=True, storage_close_or_credit_authorized=False)
    require(len((json.dumps(minimal_facts, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()) <= MINIMAL_WAIT_CAP_BYTES, 'Bounded actual minimal wait facts, no ownership-array dependency')
    print(json.dumps(dict(downstream_actual_minimal_wait_facts=minimal_facts), sort_keys=True), flush=True)
    print(json.dumps(dict(downstream_bounded_cleanup_observation=observation), sort_keys=True), flush=True)
    evidence_ready = time.time() + CAPS['terminal_evidence_margin_seconds'] < expires_unix
    if not drained or raw_status is None or (not evidence_ready):
        if not evidence_ready:
            failure = failure or 'Terminal-evidence reserve no longer available'
        partial_log = logs.record()
        print(json.dumps(dict(downstream_unresolved_cleanup_log=dict(tail=partial_log['tail'][-8192:], total_pipe_bytes=partial_log['total_pipe_bytes'], retained_raw_bytes=partial_log['retained_raw_bytes'], no_filesystem_write_or_later_stage=True)), sort_keys=True), flush=True)
        return dict(status='FAILED_UNRESOLVED_CLEANUP' if not drained else 'FAILED_TERMINAL_EVIDENCE_MARGIN', completed_ref=None, actual_wait_ref=None, returncode=rc if rc not in (None, 0) else 1, actual_direct_worker_returncode=rc, actual_direct_worker_wait=raw_status is not None, all_owned_processes_and_groups_drained=drained, operational_failure=failure, no_future_stage_authorized=True, actual_outer_tool_terminal_still_required=True)
    minimal = completed_ref = wait_ref = None
    try:
        minimal = compact_v1(s, 'actual_wait_minimal.json', minimal_facts, g, storage, cap=MINIMAL_WAIT_CAP_BYTES)
        namespace = load({'path': 'research/reviews/American_C3_native_MLM_amendment_v12/terminal_namespace_v12.py', 'sha256': '51cb8950ae0f86dd0721ebd413cc9270753e8bc66408e45b437f8e2d28a8498d'}, '_downstream_postdrain_namespace')
        alias_counter = [0]

        def preserve_alias(value):
            alias_counter[0] += 1
            charged_census_v1(s, required_write_bytes=65536 + 8192)
            active_amendment_v1()
            save(aux, 'native_alias_%02d.json' % alias_counter[0], value)

        def authorized_drained_v1():
            active_amendment_v1()
            return drained
        if s['phase'] == 'contrastive':
            namespace.normalize_latest(out / 'checkpoints', prove_drained=authorized_drained_v1, preserve=preserve_alias, max_bytes=CAPS['max_checkpoint_bytes'])
        g.budget_check_v9(storage, boundary=True, required_write_bytes=8 * 2 ** 20)
        completed_path = out / 'completed.json'
        completed_ref = ref(str(completed_path.relative_to(ROOT))) if completed_path.is_file() else None
        if completed_ref is None:
            failure = failure or 'No scientific completion'
        if completed_ref is not None:
            actual = read(completed_ref)
            validate_child_completion_v1(actual, s, q)
        log_ref = compact_v1(s, 'worker_log.json', logs.record(), g, storage, cap=16 * 2 ** 20)
        final = charged_census_v1(s)
        peak_scope = max(peak_scope, sum(final.values()))
        wait_ref = compact_v1(s, 'actual_wait.json', dict(observation, schema='C3_remaining_cells_actual_wait_v2', completed_ref=completed_ref, actual_direct_child_wait=raw_status is not None, child_completion_ref=completed_ref, process_group_ESRCH=drained and group_scan['all_owned_groups_observed_absent'], saved_MLM_wait_aliases_derived_from_same_actual_wait_and_full_owned_drain=True, operational_failure=failure, no_future_stage_authorized=failure is not None, actual_wait_minimal_ref=minimal, log_ref=log_ref, peak_combined_owned_RSS_bytes=peak, sampled_peak_scope_bytes=peak_scope, final_scope_census=final, capacity_samples=capacity_samples, last_capacity_sample=last_capacity, outer_tool_terminal_consumed=False, automatic_close_or_credit=False), g, storage, cap=TERMINAL_RECEIPT_CAP_BYTES)
        g.budget_check_v9(storage, boundary=True)
    except BaseException as exc:
        failure = failure or 'Strict terminal evidence failed: ' + type(exc).__name__ + ': ' + str(exc)[:2000]
        print(json.dumps(dict(downstream_terminal_evidence_failed=observation, failure=failure, actual_wait_minimal_ref=minimal, no_future_stage_authorized=True), sort_keys=True), flush=True)
    return dict(status='PASS' if failure is None else 'FAILED', completed_ref=completed_ref, actual_wait_ref=wait_ref, actual_wait_minimal_ref=minimal, returncode=0 if failure is None else rc if rc not in (None, 0) else 1, actual_direct_worker_returncode=rc, all_owned_processes_and_groups_drained=drained, no_future_stage_authorized=failure is not None, actual_outer_tool_terminal_still_required=True)

def _put_supervisor_record_v5(s,name,value):
    directory=local(s['worker_auxiliary_scope']).parent
    path=directory/name
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    require(len(data)<=65536,'Original compact ROOT metadata cap')
    record_budget_v1(directory,len(data)+8192,worker_auxiliary_scope=local(s["worker_auxiliary_scope"]))
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:
        require(stream.write(data)==len(data),'Complete immutable caller record')
        stream.flush();os.fsync(stream.fileno());opened=os.fstat(stream.fileno())
    named=path.lstat()
    require(stat.S_ISREG(opened.st_mode) and stat.S_ISREG(named.st_mode)
        and opened.st_nlink==named.st_nlink==1 and opened.st_size==named.st_size==len(data)
        and (opened.st_dev,opened.st_ino)==(named.st_dev,named.st_ino),'Stable caller record name')
    directory_fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(directory_fd)
    finally:os.close(directory_fd)
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest())

def supervise_once_v5(input_ref):
    """One directly waitable Python supervisor; no setsid utility or detached shell."""
    require(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site
        and sys.flags.optimize==0,'Fresh nonoptimized ROOT -B -s caller')
    s,q,cfg,storage,protocol=validate_input_v1(input_ref)
    command=[str(ROOT/'.runtime/env/bin/python'),'-B','-s','-m',c.HERE.replace('/','.')+'.execution_v1',
        '--execute','--input-ref',json.dumps(input_ref,separators=(',',':')),
        '--amendment-ref',json.dumps(input_ref,separators=(',',':'))]
    # This exclusive record consumes the attempt before Popen. Failure never
    # authorizes replay, including a launch or wait whose outcome is uncertain.
    invocation_ref=_put_supervisor_record_v5(s,'caller_invocation.v5.json',dict(
        schema='C3_v5_exact_Python_supervisor_invocation_v1',input_ref=input_ref,
        command=command,root_authorizer='root_agent',created_at_utc=datetime.now(timezone.utc).isoformat(),
        direct_Python_child=True,setsid_utility_used=False,automatic_retry=False,
        scientific_execution_success_claimed=False))
    proc=None;raw_status=None;returncode=None;actual_wait=False
    try:
        proc=subprocess.Popen(command,cwd=ROOT,start_new_session=True)
        while True:
            try:waited_pid,raw_status=os.waitpid(proc.pid,0);break
            except InterruptedError:continue
        require(waited_pid==proc.pid,'Exact direct Python supervisor waitpid')
        require(os.WIFEXITED(raw_status) or os.WIFSIGNALED(raw_status),'Terminal direct Python supervisor status')
        returncode=os.waitstatus_to_exitcode(raw_status);actual_wait=True;proc.returncode=returncode
        caller_exit=returncode if returncode>=0 else 128-returncode
        result=dict(schema='C3_v5_actual_Python_supervisor_wait_v1',input_ref=input_ref,
            invocation_ref=invocation_ref,pid=proc.pid,actual_OS_wait_status=raw_status,
            actual_Python_supervisor_wait=True,actual_Python_supervisor_returncode=returncode,
            returncode=caller_exit,setsid_utility_returncode_used=False,
            worker_wait_or_scientific_completion_inferred=False,
            ROOT_outer_terminal_consumption_and_scientific_wait_audit_still_required=True,
            automatic_retry=False,storage_close_or_credit_authorized=False)
        result['wait_ref']=_put_supervisor_record_v5(s,'caller_actual_supervisor_wait.v5.json',result)
        return result
    except BaseException as error:
        failure=dict(schema='C3_v5_Python_supervisor_wait_uncertainty_v1',input_ref=input_ref,
            invocation_ref=invocation_ref,pid=proc.pid if proc is not None else None,
            actual_OS_wait_status=raw_status,actual_Python_supervisor_wait=actual_wait,
            actual_Python_supervisor_returncode=returncode,
            exception_type=type(error).__name__,message=str(error)[:2000],
            automatic_retry=False,scientific_success_claimed=False,
            ROOT_must_resolve_original_supervisor_before_any_new_action=True)
        try:_put_supervisor_record_v5(s,'caller_supervisor_failure.v5.json',failure)
        except BaseException:print(json.dumps(dict(caller_evidence_uncertain=True,automatic_retry=False)),flush=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--execute', action='store_true')
    actions.add_argument('--supervise-once', action='store_true')
    actions.add_argument('--worker-input-ref')
    parser.add_argument('--input-ref')
    parser.add_argument('--amendment-ref')
    parser.add_argument('--gate-fd', type=int)
    parser.add_argument('--attachment-ref')
    args = parser.parse_args(argv)
    if not (args.execute or args.supervise_once or args.worker_input_ref):
        print(json.dumps(dict(status='SOURCE_ONLY_INACTIVE', execution=False, admission=False)))
        return 0
    require(sys.platform.startswith('linux') and sys.dont_write_bytecode and (not any((n in sys.modules for n in ('torch', 'torch_geometric')))), 'Fresh Linux stdlib entry')
    require(args.amendment_ref is not None, 'Separate exact ROOT execution amendment required')
    set_amendment_v1(json.loads(args.amendment_ref))
    sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts')]
    if args.worker_input_ref:
        require(args.gate_fd is not None, 'Actual parent-owned launch gate')
        return worker_v1(json.loads(args.worker_input_ref), args.gate_fd, json.loads(args.attachment_ref) if args.attachment_ref is not None else None)
    require(args.input_ref is not None, 'Exact original input ref required')
    require(json.loads(args.input_ref) == json.loads(args.amendment_ref), 'ROOT phase input is the exact actual phase authority binding')
    result = supervise_once_v5(json.loads(args.input_ref)) if args.supervise_once else execute_v1(json.loads(args.input_ref))
    print(json.dumps(result, sort_keys=True), flush=True)
    return result['returncode']
if __name__ == '__main__':
    raise SystemExit(main())
