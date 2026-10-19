"""Readable candidate request, identity, resource and explicit gate boundaries."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import stat
import subprocess
import time

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
GIB = 2**30
MAX_METADATA = 2**20


def need(ok, message):
    if not ok:
        raise ValueError(message)


def contained(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT/path
    need(".." not in path.parts and path.resolve().is_relative_to(ROOT), "Project-only path required")
    return path


def read_json(path):
    path = contained(path)
    fd = os.open(path, os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        need(stat.S_ISREG(metadata.st_mode) and metadata.st_size <= MAX_METADATA,
             "Bounded ordinary metadata required")
        raw = stream.read(MAX_METADATA+1)
    need(len(raw) <= MAX_METADATA, "Metadata read limit")
    value = json.loads(raw)
    need(isinstance(value, dict), "Object metadata required")
    return value, raw


def file_ref(path):
    path = contained(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return dict(path=str(path.relative_to(ROOT)), sha256=digest.hexdigest())


def read_ref(ref):
    need(set(ref) == {"path", "sha256"}, "Exact path and SHA256 reference required")
    value, raw = read_json(ref["path"])
    need(hashlib.sha256(raw).hexdigest() == ref["sha256"], "Metadata source reference changed")
    return value


def write_json(path, value, *, replace=False):
    path = contained(path)
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    need(len(data) <= MAX_METADATA, "Bounded metadata output required")
    temporary = path.with_name(path.name+".next") if replace else path
    fd = os.open(temporary, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(temporary, path)
    return file_ref(path)


def identity(pid=None):
    pid = os.getpid() if pid is None else pid
    fields = Path('/proc')/str(pid)/'stat'
    raw = fields.read_text()
    fields = raw[raw.rfind(')')+2:].split()
    return dict(pid=pid, start_ticks=int(fields[19]))


def rss_bytes(pid=None):
    pid = os.getpid() if pid is None else pid
    for line in (Path('/proc')/str(pid)/'status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1])*1024
    return 0


def scope_bytes(path):
    total = 4096
    for directory, folders, files in os.walk(path, followlinks=False):
        for name in folders+files:
            item = Path(directory)/name
            info = item.lstat()
            need(not stat.S_ISLNK(info.st_mode), "No symlinks in candidate output scope")
            need(stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode), "Ordinary output entries only")
            total += (info.st_size if stat.S_ISREG(info.st_mode) else 0)+4096
    return total


def source_identity(refs):
    return hashlib.sha256(json.dumps(sorted(refs, key=lambda row: row['path']),
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()



def numerical_identity(refs):
    names = {'candidate_gru.py','step.py','execution.py','decoder.py','layout.py','progress.py'}
    selected = [row for row in refs if Path(row['path']).name in names
                or '/benchmark_recovery_20261020/runtime/' in row['path']]
    need(names <= {Path(row['path']).name for row in selected}, "Complete numerical source refs required")
    return source_identity(selected)


def continuation_allocation(output, request_ref, worker):
    permit, _ = read_json(output/'continue_gate.json')
    need(permit.get('schema') == 'treeloom_candidate_gate_v1'
         and permit.get('request_ref') == request_ref and permit.get('worker_identity') == worker
         and permit.get('gate') == 'continue_gate.json' and permit.get('allow') is True
         and permit.get('allow_updates_through') == 600
         and permit.get('max_rss_bytes') == 512*GIB
         and permit.get('observation_cache_bytes') == 352*GIB
         and permit.get('training_headroom_bytes') == 128*GIB,
         "Explicit 512 GiB process / 352 GiB compact pool continuation required")
    need(time.time() < permit.get('expires_unix',0), "Fresh continuation authorization required")
    qualification = read_ref(permit['qualification_ref'])
    need(qualification.get('request_ref') == request_ref
         and qualification.get('worker_identity') == worker
         and qualification.get('optimizer_updates') == 1
         and qualification.get('status') == 'completed_first_update',
         "Continuation must preserve the actual first full-parent update")
    admission = read_ref(permit['cpu_admission_ref'])
    need(admission.get('request_ref') == request_ref and admission.get('worker_identity') == worker
         and admission.get('max_rss_bytes') == 512*GIB,
         "Pinned ROOT CPU memory reservation required")
    return permit

def load_request(path):
    request, raw = read_json(path)
    ref = dict(path=str(contained(path).relative_to(ROOT)), sha256=hashlib.sha256(raw).hexdigest())
    need(request.get('schema') == 'treeloom_recovery_candidate_v1', "Candidate request schema required")
    need(request.get('phase') in ('preflight', 'fit'), "Explicit candidate phase required")
    need(request.get('max_rss_bytes') == 128*GIB
         and request.get('max_cuda_reserved_bytes') == 16*GIB
         and request.get('max_output_bytes') == 64*2**20,
         "Fixed 128 GiB RSS, 16 GiB CUDA and 64 MiB output candidate bounds")
    need(type(request.get('max_seconds')) is int and 60 <= request['max_seconds'] <= 30*86400,
         "Finite candidate lifetime required")
    need(request.get('holds_per_block') == 128 and request.get('haplotypes_per_gru') == 32
         and request.get('gru_chunk_holds') == 1024,
         "Explicit qualified execution partition proposal required")
    need(isinstance(request.get('gpu_uuid'), str) and request['gpu_uuid'].startswith('GPU-')
         and request.get('physical_gpu') == 0,
         "Exact physical GPU0 UUID required")
    need(request.get('minimum_free_gpu_bytes') >= 16*GIB, "Shared GPU free-memory floor required")
    output = contained(request['output_directory'])
    need(output.resolve() == output and output.is_relative_to(ROOT/'experiments/results'),
         "Dedicated result scope required")
    refs = request.get('source_refs')
    need(isinstance(refs,list) and refs and len({r['path'] for r in refs}) == len(refs),
         "Explicit unique source closure required")
    required = {'candidate_common.py','candidate_gru.py','candidate_worker.py','candidate_supervisor.py',
                'step.py','execution.py','decoder.py','layout.py','progress.py'}
    need(required <= {Path(r['path']).name for r in refs}, "Candidate and numerical source closure required")
    runtime_names = {'binding.py','observations.py','episodes.py','cuts.py','preparation.py'}
    need(runtime_names <= {Path(r['path']).name for r in refs if '/benchmark_recovery_20261020/runtime/' in r['path']},
         "Readable runtime source closure required")
    if request['phase'] == 'fit':
        need('candidate_fit.py' in {Path(r['path']).name for r in refs}, "Complete fit driver source required")
    for row in refs:
        need(file_ref(row['path']) == row, "Candidate source changed: "+row['path'])
    job = read_ref(request['job_ref'])
    need(job.get('task',{}).get('id') == 'AM_treeloom_fit', "Original maintained TreeLoom job required")
    barrier = read_ref(request['observation_barrier_ref'])
    if request['phase'] == 'fit':
        preflight = read_ref(request['preflight_ref'])
        waited = read_ref(request['preflight_wait_ref'])
        need(waited.get('success') is True and waited.get('actual_exit_code') == 0
             and waited.get('request_ref') == preflight.get('request_ref'),
             "Actual successful preflight supervisor wait required")
        need(preflight.get('status') == 'passed' and preflight.get('optimizer_updates') == 0
             and preflight.get('numerical_identity') == numerical_identity(refs)
             and preflight.get('gpu_uuid') == request['gpu_uuid']
             and preflight.get('gru_chunk_holds') == 1024
             and preflight.get('haplotypes') == 32,
             "Same-source full-sequence GPU compatibility receipt required before real preparation")
    return request, ref, job, barrier, output


def gpu_inventory(expected_uuid):
    command = ['nvidia-smi','-i','0','--query-gpu=uuid,memory.free,memory.total','--format=csv,noheader,nounits']
    result = subprocess.run(command, check=True, text=True, capture_output=True, timeout=15)
    rows = result.stdout.strip().splitlines()
    need(len(rows) == 1, "One physical GPU0 inventory record required")
    values = [value.strip() for value in rows[0].split(',')]
    need(len(values) == 3 and values[0] == expected_uuid, "Physical GPU0 UUID differs")
    return dict(uuid=values[0], free_bytes=int(values[1])*2**20, total_bytes=int(values[2])*2**20)


class Guard:
    def __init__(self, request, output):
        self.request, self.output = request, output
        self.started = time.monotonic()
        self.last_scope = 0.
        self.cuda = False

    def check(self, label):
        need(time.monotonic()-self.started < self.request['max_seconds'], "Candidate wall limit exceeded")
        need(rss_bytes() <= self.request['max_rss_bytes'], "Candidate RSS limit exceeded")
        now = time.monotonic()
        if now-self.last_scope > 15:
            need(scope_bytes(self.output) <= self.request['max_output_bytes'], "Candidate output cap exceeded")
            self.last_scope = now
        if self.cuda:
            import torch
            need(torch.cuda.memory_reserved(0) <= self.request['max_cuda_reserved_bytes'],
                 "Candidate CUDA allocator cap exceeded")

    def wait_gate(self, name, request_ref, *, qualification_ref=None):
        worker = identity()
        gate = self.output/name
        while not gate.exists():
            self.check('waiting_'+name)
            time.sleep(1.)
        permit, _ = read_json(gate)
        need(permit.get('schema') == 'treeloom_candidate_gate_v1'
             and permit.get('request_ref') == request_ref
             and permit.get('worker_identity') == worker
             and permit.get('gate') == name and permit.get('allow') is True,
             "Explicit same-request, same-worker ROOT gate required")
        need(isinstance(permit.get('expires_unix'), (int,float)) and time.time() < permit['expires_unix'],
             "Fresh finite gate required")
        if qualification_ref is not None:
            need(permit.get('qualification_ref') == qualification_ref
                 and permit.get('allow_updates_through') == 600,
                 "Qualification-bound full schedule continuation gate required")
            continuation_allocation(self.output,request_ref,worker)
        else:
            need(permit.get('gpu_uuid') == self.request['gpu_uuid']
                 and permit.get('max_cuda_reserved_bytes') == self.request['max_cuda_reserved_bytes'],
                 "Exact GPU allocation gate required")
            admission = read_ref(permit['admission_ref'])
            need(admission.get('request_ref') == request_ref and admission.get('worker_identity') == worker
                 and admission.get('physical_gpu') == 0 and admission.get('gpu_uuid') == self.request['gpu_uuid']
                 and admission.get('max_cuda_reserved_bytes') == self.request['max_cuda_reserved_bytes'],
                 "Pinned same-worker shared GPU0 admission required")
        return file_ref(gate)
