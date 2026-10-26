"""Bounded history restoration and exact qualified checkpoint continuation.

The original checkpoint arithmetic, model, optimizer, RNG restore and scientific
training schedule remain in the unchanged training implementation.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from arggnn.benchmarking.paper_campaign import runtime as r
from research.repairs.TreeLoom_lazy_compact_prefetch_20261024_v1 import training
from .hooks import install_tree, source_refs, verify_bindings, PARALLEL_RESERVE_BYTES

OUTPUT = 'experiments/results/paper_american_20261024/treeloom_fit/native'
MAX_HISTORY_FILES = 4096
MAX_HISTORY_BYTES = 80 * 2**20
COPY_BLOCK_BYTES = 2**20
PROOF_ROLES = frozenset(('parser_parity', 'parser_adversarial', 'tree_parity',
    'tree_adversarial', 'native_kernel_safety', 'full_observed_parent',
    'full_american_parent', 'gpu_numerical_parity', 'resume_contract', 'independent_review'))
PROCESS_PROOF_ROLES = frozenset(('process_parity', 'full_process_parent'))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _reference(reference):
    _require(type(reference) is dict and set(reference) == {'path', 'sha256'}
        and type(reference['path']) is str and type(reference['sha256']) is str
        and re.fullmatch('[0-9a-f]{64}', reference['sha256']) is not None,
        'Exact path and SHA256 artifact reference required')
    return reference


def authenticate_release(reference):
    release = r.read(_reference(reference))
    _require(type(release) is dict and release.get('schema') == 'exact_preprocessing_release_v1'
        and release.get('status') == 'passed_release' and release.get('native_competitor_changed') is False,
        'An independently reviewed exact-preparation release is required')
    _require(type(release.get('encoder_workers')) is int and release['encoder_workers'] in (1, 4),
        'An explicit qualified encoder worker count is required')
    _require(release.get('candidate_source_refs') == source_refs(), 'Candidate execution source changed')
    proofs = release.get('proof_refs')
    expected = PROOF_ROLES | (PROCESS_PROOF_ROLES if release['encoder_workers'] == 4 else frozenset())
    _require(type(proofs) is list and len(proofs) == len(expected), 'Complete explicit release proof roles required')
    seen = set()
    for row in proofs:
        _require(type(row) is dict and set(row) == {'role', 'reference'}
            and row.get('role') in expected and row['role'] not in seen, 'Unique required proof role required')
        role = row['role']
        seen.add(role)
        proof = r.read(_reference(row['reference']))
        _require(type(proof) is dict and proof.get('schema') == 'exact_preprocessing_proof_v1'
            and proof.get('role') == role and proof.get('status') == 'pass', 'Passed role-specific proof envelope required')
        refs = proof.get('evidence_refs')
        _require(type(refs) is list and bool(refs), 'Nonempty authenticated proof evidence required')
        for evidence_ref in refs:
            _reference(evidence_ref)
            path = r.path(evidence_ref['path'])
            _require(r.ref(path) == evidence_ref, 'Proof evidence changed')
            if path.suffix == '.json':
                evidence = r.read(evidence_ref)
                _require(type(evidence) is dict and evidence.get('status') == 'pass', 'Underlying qualification did not pass')
            else:
                _require(role == 'independent_review' and path.suffix == '.md', 'Qualification evidence must be passed JSON')
    _require(seen == expected, 'Every required proof role must be present')
    return release


def _ordinary_path(value, *, regular_file=False, allow_missing=False):
    """Reject symlink components before runtime.path would resolve them away."""
    _require(type(value) is str and bool(value), 'Ordinary project-contained path required')
    raw = Path(value)
    _require('..' not in raw.parts, 'Parent traversal is forbidden')
    path = raw if raw.is_absolute() else r.ROOT / raw
    _require(path.is_relative_to(r.ROOT), 'Project-contained history path required')
    cursor = r.ROOT
    for part in path.relative_to(r.ROOT).parts:
        cursor = cursor / part
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError:
            _require(allow_missing, 'Required history path is absent')
            continue
        _require(not stat.S_ISLNK(mode), 'History path must not contain symbolic links')
        if cursor != path:
            _require(stat.S_ISDIR(mode), 'History ancestors must be ordinary directories')
    if regular_file:
        _require(stat.S_ISREG(path.lstat().st_mode), 'Ordinary historical source file required')
    return path


def validate_history(reference):
    _reference(reference)
    _require(r.path(reference['path']).stat().st_size <= 8 * 2**20,
        'Preserved-history metadata exceeds its finite eight MiB bound')
    receipt = r.read(_reference(reference))
    _require(type(receipt) is dict and set(receipt) == {'schema', 'destination', 'files', 'file_count', 'total_bytes'}
        and receipt['schema'] == 'exact_preprocessing_preserved_history_v1'
        and receipt['destination'] == OUTPUT, 'Explicit original native-history manifest required')
    destination = _ordinary_path(OUTPUT, allow_missing=True)
    _require(not destination.exists(), 'Fresh original native history namespace required')
    rows = receipt['files']
    _require(type(rows) is list and type(receipt['file_count']) is int
        and len(rows) == receipt['file_count'] and 0 < len(rows) <= MAX_HISTORY_FILES,
        'History file count exceeds its finite bound')
    _require(type(receipt['total_bytes']) is int and 0 < receipt['total_bytes'] <= MAX_HISTORY_BYTES,
        'History total bytes exceed their finite bound')
    paths, total, files = set(), 0, []
    for row in rows:
        _require(type(row) is dict and set(row) == {'relative_path', 'source_ref', 'size_bytes'},
            'Exact per-file history manifest required')
        name = row['relative_path']
        _require(type(name) is str and bool(name), 'Ordinary native-history relative path required')
        relative = PurePosixPath(name)
        _require(not relative.is_absolute() and '..' not in relative.parts
            and relative.as_posix() == name and name != '.' and '\\' not in name,
            'Ordinary native-history relative path required')
        _require(name not in paths, 'Duplicate historical destination')
        paths.add(name)
        size = row['size_bytes']
        _require(type(size) is int and 0 <= size <= MAX_HISTORY_BYTES, 'Bounded exact historical file size required')
        total += size
        _require(total <= MAX_HISTORY_BYTES, 'Historical sum exceeds its finite bound')
        source_ref = _reference(row['source_ref'])
        source = _ordinary_path(source_ref['path'], regular_file=True)
        _require(not source.is_relative_to(destination), 'Archive must be outside the restored namespace')
        _require(source.stat().st_size == size and r.ref(source) == source_ref, 'Preserved native history changed')
        files.append((source, destination / name, size, source_ref))
    _require(total == receipt['total_bytes'], 'Historical manifest byte total is inconsistent')
    for name in paths:
        _require(not any(parent.as_posix() in paths for parent in PurePosixPath(name).parents),
            'Historical file and directory destinations overlap')
    _require('training_completed.json' not in paths, 'Completed training must not be restarted')
    return destination, receipt, files


def restore_history(reference, *, resource_check):
    """Prevalidate the complete archive, then copy with exclusive bounded writes."""
    _require(callable(resource_check), 'Admitted resource callback required for history restoration')
    destination, receipt, files = validate_history(reference)
    resource_check(receipt['total_bytes'])
    _ordinary_path(OUTPUT, allow_missing=True)
    destination.mkdir(parents=True, exist_ok=False)
    for source, target, size, source_ref in files:
        # A refused check precedes every output block. Partial copies are retained
        # as failure evidence and can never be silently reused by another attempt.
        _ordinary_path(str(source), regular_file=True)
        _ordinary_path(str(target), allow_missing=True)
        resource_check(size)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest, copied = hashlib.sha256(), 0
        with source.open('rb') as incoming, target.open('xb') as outgoing:
            _require(os.fstat(incoming.fileno()).st_size == size, 'Historical source changed before copying')
            while copied < size:
                block = incoming.read(min(COPY_BLOCK_BYTES, size - copied))
                _require(bool(block), 'Historical source shortened during copying')
                resource_check(len(block))
                outgoing.write(block)
                digest.update(block)
                copied += len(block)
            _require(incoming.read(1) == b'', 'Historical source grew during copying')
            outgoing.flush()
            os.fsync(outgoing.fileno())
        _require(digest.hexdigest() == source_ref['sha256'] and r.ref(target)['sha256'] == source_ref['sha256'],
            'Historical copy is not byte-identical')
    return destination


def attempt_directory(root, steps, attempt_id):
    _require(type(steps) is int and 238 <= steps < 600, 'Original durable resume update boundary required')
    _require(type(attempt_id) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', attempt_id) is not None,
        'Explicit safe immutable attempt identifier required')
    leaf = root / ('exact_preprocessing_from_update%d_%s' % (steps, attempt_id))
    _require(not leaf.exists() and not leaf.is_symlink(), 'Immutable attempt output must be new')
    return leaf


def _exclusive_json(path, value, resource_check):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    resource_check(len(raw))
    with path.open('xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return r.ref(path)


def run_task(request, *, resource_check):
    required = {'stage', 'arguments', 'release_ref', 'preserved_history_ref', 'resume_steps', 'attempt_id'}
    _require(type(request) is dict and set(request) == required and request['stage'] == 'fit',
        'Explicit exact-preparation checkpoint continuation required')
    release = authenticate_release(request['release_ref'])
    arguments = dict(request['arguments'])
    _require(arguments['output_directory'] == OUTPUT, 'Original benchmark output contract required')
    steps = request['resume_steps']
    root, history, files = validate_history(request['preserved_history_ref'])
    leaf = attempt_directory(root, steps, request['attempt_id'])
    _require(not any(target.is_relative_to(leaf) for _, target, _, _ in files), 'Attempt namespace already exists in archive')
    _require(not any(target == root / (leaf.name + '.execution.json') for _, target, _, _ in files),
        'Attempt execution receipt already exists in archive')
    arguments['output_directory'] = str(leaf)
    # This is the unchanged source/registry/observation/evidence/memory validation.
    # Full model/Adam/RNG validation remains in the original run_training entry.
    spec = training.build_spec(**arguments)
    evidence = r.read(arguments['resume_evidence_ref'])
    _require(type(evidence['complete_parent_updates']) is int and evidence['complete_parent_updates'] == steps,
        'Authenticated current durable update boundary required')
    if release['encoder_workers'] == 4:
        _require(arguments['preparation_headroom_bytes'] >= PARALLEL_RESERVE_BYTES,
            'Existing preparation reservation must include the eight GiB parallel envelope')
    resource_check('TreeLoom_before_history_restore')
    root = restore_history(request['preserved_history_ref'], resource_check=resource_check)
    bindings = install_tree(preparation_workers=release['encoder_workers'], max_rss_bytes=arguments['max_rss_bytes'])
    # Original build_spec requires an absent leaf. Keep the immutable execution
    # receipt beside it, and let the unchanged trainer create its own directory.
    receipt_path = root / (leaf.name + '.execution.json')
    receipt = _exclusive_json(receipt_path, dict(schema='exact_preprocessing_execution_v1',
        release_ref=request['release_ref'], bindings=bindings, encoder_workers=release['encoder_workers'],
        candidate_source_refs=release['candidate_source_refs'], model_or_optimizer_changed=False,
        native_competitor_changed=False, resume_steps=steps, attempt_id=request['attempt_id'],
        attempt_output_directory=str(leaf.relative_to(r.ROOT)), preserved_history_ref=request['preserved_history_ref'],
        restored_file_count=history['file_count'], restored_total_bytes=history['total_bytes'],
        resource_checked_restore=True, original_training_source_refs=spec['requirements']['source_refs']), resource_check)
    result = training.run_task(dict(stage='fit', arguments=arguments), resource_check=resource_check)
    _require(result['complete_parent_updates'] == 600 and result['resumed_from_complete_parent_updates'] == steps,
        'Original full 600-update budget required')
    _require(source_refs() == release['candidate_source_refs'], 'Candidate source changed during training')
    verify_bindings(bindings)
    completion = {key: value for key, value in result.items() if key != 'completion_ref'}
    completion.update(nested_completion_ref=result['completion_ref'], exact_preprocessing_execution_ref=receipt)
    resource_check('TreeLoom_publish_exact_preparation_completion')
    result['completion_ref'] = _exclusive_json(root / 'training_completed.json', completion, resource_check)
    result['exact_preprocessing_execution_ref'] = receipt
    return result
