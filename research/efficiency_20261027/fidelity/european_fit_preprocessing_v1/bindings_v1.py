"""Authenticated, isolated FIT bindings for the exact European observation route.

Call once in a fresh FIT interpreter before driver.run_fit. Frozen source files,
model qualification, original resource callbacks and numerical code are retained.
The caller records this returned receipt and verifies it before completion.
"""
from functools import partial
import hashlib
import importlib
import json
from pathlib import Path
import re

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
HERE = Path(__file__).parent
RELEASE_REF = dict(path='research/preprocessing_optimization_20261026/integration/release.json',
    sha256='7fe628a46be4ff61b7f955610d9959eb738d2e4aef908973b937dcfe59774944')
PARSER_REF = dict(path='research/preprocessing_optimization_20261026/observed/full_train_qualification.json',
    sha256='c5c634b189c74243d56b6cc41d492f8a6f4245807a392b275722673f0fd919a0')
TREE_REF = dict(path='research/preprocessing_optimization_20261026/integration/full_parent_qualification.json',
    sha256='1e5dfd4f2a35914acc160b1b482924bd5bdf30e04816cfc30a5b9af6cb101f0c')
JOB_REF = dict(path='.runtime/benchmarking/workflow_20261022_full_vram_v2/jobs/AP__all_train__treeloom_fit.json',
    sha256='fecd670f6f26332ab165648f4d65ad51131650d195e86df2a39a5545f2b724c1')
QUEUE = 'research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters.european_observations'
RECOVERY = 'research.repairs.benchmark_recovery_20261020.runtime'
OWNERS = ((QUEUE, 'load_relate_observation'), (RECOVERY + '.observations', 'encode_block'),
          (RECOVERY + '.episodes', 'build_episodes'), (RECOVERY + '.cuts', 'prepare_cut_plan_v1'))
PROOF_ROLES = frozenset(('parser_parity', 'parser_adversarial', 'tree_parity', 'tree_adversarial',
    'native_kernel_safety', 'full_observed_parent', 'full_american_parent', 'gpu_numerical_parity',
    'resume_contract', 'independent_review', 'process_parity', 'full_process_parent'))


def need(ok, message):
    if not ok:
        raise ValueError(message)


def file_ref(path):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    need(path.is_relative_to(ROOT) and '..' not in path.parts, 'Project-contained ordinary path required')
    cursor = ROOT
    for part in path.relative_to(ROOT).parts:
        cursor = cursor / part
        need(not cursor.is_symlink(), 'Source and evidence must not contain symlinks')
    need(path.is_file() and path.stat().st_size <= 8 * 2**20, 'Bounded source/evidence file required')
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(reference):
    need(type(reference) is dict and set(reference) == {'path', 'sha256'}
         and type(reference['path']) is str and type(reference['sha256']) is str
         and re.fullmatch('[0-9a-f]{64}', reference['sha256']) is not None,
         'Exact path and SHA256 reference required')
    need(file_ref(reference['path']) == reference, 'Source/evidence changed: ' + reference['path'])
    return json.loads((ROOT / reference['path']).read_text())


def authenticate(source_manifest_ref):
    need(source_manifest_ref['path'] == str((HERE / 'source_pins_v1.json').relative_to(ROOT)),
         'Exact bridge source manifest required')
    manifest = read(source_manifest_ref)
    need(manifest.get('schema') == 'European_fit_preprocessing_sources_v1'
         and manifest.get('release_ref') == RELEASE_REF, 'Exact bridge source contract required')
    refs = manifest['source_refs']
    need(type(refs) is list and refs and len({r['path'] for r in refs}) == len(refs),
         'Distinct nonempty source references required')
    for reference in refs:
        need(file_ref(reference['path']) == reference, 'Pinned preprocessing source changed: ' + reference['path'])
    release = read(RELEASE_REF)
    need(release['schema'] == 'exact_preprocessing_release_v1' and release['status'] == 'passed_release'
         and release['encoder_workers'] == 4 and release['native_competitor_changed'] is False,
         'Original reviewed four-process exact preprocessing release required')
    for reference in release['candidate_source_refs']:
        need(reference in refs and file_ref(reference['path']) == reference, 'Qualified source changed or omitted')
    roles = set()
    for row in release['proof_refs']:
        role = row['role']
        need(role in PROOF_ROLES and role not in roles, 'Distinct original proof roles required')
        roles.add(role)
        proof = read(row['reference'])
        need(proof['schema'] == 'exact_preprocessing_proof_v1' and proof['role'] == role
             and proof['status'] == 'pass' and proof['evidence_refs'], 'Passed original proof envelope required')
        for reference in proof['evidence_refs']:
            need(file_ref(reference['path']) == reference, 'Original proof evidence changed')
            if Path(reference['path']).suffix == '.json':
                need(read(reference)['status'] == 'pass', 'Original underlying proof must pass')
            else:
                need(role == 'independent_review' and Path(reference['path']).suffix == '.md',
                     'Only the original independent review prose may be non-JSON evidence')
    need(roles == PROOF_ROLES, 'Every original proof role required')
    parser, tree = read(PARSER_REF), read(TREE_REF)
    for path, sha256 in parser['source_sha256'].items():
        need(file_ref(path)['sha256'] == sha256, 'Full European parser proof source changed')
    for reference in tree['source_refs']:
        need(file_ref(reference['path']) == reference, 'Full American tree proof source changed')
    need(parser['parent'] == 'ancestralpaths_european_train_000'
         and parser['status'] == tree['status'] == 'pass'
         and parser['byte_identical_all_retained_calls'] is True and parser['callbacks_identical'] is True,
         'Exact prior European TRAIN000 parser proof required')
    original = read(JOB_REF)['call']['spec']['scientific_spec']
    need(all(original['train_inputs'][parser['parent']][key] == value
             for key, value in parser['source_refs'].items()), 'Parser proof must bind the original TRAIN000 inputs')
    return manifest, original


def _helper_bindings(adapter, owner):
    need(adapter.original is owner and all(getattr(adapter, name) is getattr(owner, name)
         for name in ('_implementation', 'PROTOCOLS', '_need', '_json_ref', '_bound')),
         'Exact queue adapter helper globals must be preserved')


def install(*, source_manifest_ref, max_rss_bytes, preparation_headroom_bytes, train_inputs, resource_check):
    """Return four authenticated bindings; all admission checks precede assignment."""
    need(callable(resource_check), 'Existing ROOT resource callback required')
    resource_check('European_preprocessing_before_authentication')
    manifest, original = authenticate(source_manifest_ref)
    envelope = original['memory_envelope']
    need(type(max_rss_bytes) is int and max_rss_bytes == envelope['max_rss_bytes']
         and type(preparation_headroom_bytes) is int
         and preparation_headroom_bytes == envelope['preparation_headroom_bytes']
         and train_inputs == original['train_inputs'], 'Original TRAIN inputs and memory envelope required')
    import torch
    need(not torch.cuda.is_initialized(), 'Install before observations and late GPU acquisition')
    from research.preprocessing_optimization_20261026.integration import hooks
    from research.preprocessing_optimization_20261026.tree import episodes, cut
    from . import observed_adapter_v1 as adapter
    need(preparation_headroom_bytes >= hooks.PARALLEL_RESERVE_BYTES,
         'Existing preparation headroom must contain the qualified eight GiB reserve')
    owner = importlib.import_module(QUEUE)
    _helper_bindings(adapter, owner)
    encoder = partial(hooks._process4_encode_block, max_rss_bytes=max_rss_bytes)
    replacements = (adapter.load_relate_observation, encoder, episodes.build_episodes, cut.prepare_cut_plan_v1)
    specifications = [(module, name, module, name, replacement)
                      for (module, name), replacement in zip(OWNERS, replacements)]
    resource_check('European_preprocessing_before_installation')
    installed = hooks._install(specifications)
    receipt = dict(schema='European_fit_preprocessing_bindings_v1', source_manifest_ref=source_manifest_ref,
        release_ref=RELEASE_REF, full_European_parser_proof_ref=PARSER_REF,
        full_American_tree_proof_ref=TREE_REF, bindings=installed,
        encoder_workers=4, max_rss_bytes=max_rss_bytes, preparation_headroom_bytes=preparation_headroom_bytes,
        child_reserve_bytes=hooks.PARALLEL_RESERVE_BYTES, tree_chunk_size=128, max_chunk_bytes=64*2**20,
        queue_helper_globals_preserved=True, all_original_TRAIN_inputs_preserved=True,
        scientific_model_optimizer_schedule_unchanged=True, full_European_tree_proof_claimed=False,
        frozen_source_files_changed=False, original_callbacks_retained=True)
    return receipt


def verify(receipt):
    """Check recorded source identities and exact live bindings without payload reads."""
    need(receipt['schema'] == 'European_fit_preprocessing_bindings_v1'
         and receipt['release_ref'] == RELEASE_REF, 'Exact bridge execution receipt required')
    _, original = authenticate(receipt['source_manifest_ref'])
    need(receipt['max_rss_bytes'] == original['memory_envelope']['max_rss_bytes']
         and receipt['preparation_headroom_bytes'] == original['memory_envelope']['preparation_headroom_bytes']
         and tuple((row['owner_module'], row['attribute']) for row in receipt['bindings']) == OWNERS,
         'Original envelope and exact actual call-route owners required')
    from research.preprocessing_optimization_20261026.integration import hooks
    from research.preprocessing_optimization_20261026.tree import episodes, cut
    from . import observed_adapter_v1 as adapter
    owner = importlib.import_module(QUEUE)
    _helper_bindings(adapter, owner)
    replacements = (adapter.load_relate_observation,
        partial(hooks._process4_encode_block, max_rss_bytes=receipt['max_rss_bytes']),
        episodes.build_episodes, cut.prepare_cut_plan_v1)
    need(all(row['after'] == hooks._callable_ref(replacement)
             for row, replacement in zip(receipt['bindings'], replacements)),
         'Receipt replacement functions and fixed execution parameters must be exact')
    return hooks.verify_bindings(receipt['bindings'])
