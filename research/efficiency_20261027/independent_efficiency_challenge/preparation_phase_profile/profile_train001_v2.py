"""One original TRAIN001 CPU preparation, timed without changing its algorithms.

Run only under the existing ROOT guardian. No model, optimizer, target join,
checkpoint read, GPU initialization, native retry, or large output is allowed.
"""
import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from .timing_v1 import PhaseTimers

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
BASE = ROOT / 'research/efficiency_20261027/independent_efficiency_challenge/preparation_phase_profile'
OUTPUT = ROOT / 'research/efficiency_20261027/root/runs/train001_preparation_phases_v1/data'
PARENT = 'argformer_american_train_001'
CAP = 128 * 2**30
SECONDS = 3600
COMPARISON_MANIFEST = dict(
    path='research/efficiency_20261027/execution/source_pins_two_parent_comparison_v2.json',
    sha256='0a7d4b4db78e5768dee851c1130c35760f559a84608472ca418eb5aee5c0b674')
SPEC = dict(path='research/efficiency_20261027/execution/TWO_PARENT_COMPARISON_SPEC_v1.json',
    sha256='6a20b1f61f34fed96b872d3fd57607cc695b486ce547c58a73606ff5937b09dd')
PHASES = ('load_observation', 'prepare_observation_total', 'encode_block_total',
          'build_episodes', 'prepare_cut_plan', 'validate_complete_bundle', 'pack_bundle')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read_verified(record):
    need(set(record) == {'path', 'sha256'}, 'Exact file reference required')
    path = (ROOT / record['path']).resolve()
    need(path.is_relative_to(ROOT), 'Project-contained reference required')
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == record['sha256'], 'Immutable file digest mismatch: ' + record['path'])
    return raw


def executable_source_refs(manifest):
    refs = tuple(row for row in manifest['source_refs']
                 if Path(row['path']).suffix in ('.py', '.cpp', '.so', '.sh'))
    need(refs, 'Inherited executable dependency references required')
    return refs


def verify_sources(digest):
    path = BASE / 'source_pins_v2.json'
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == digest, 'Exact isolated harness source manifest')
    pins = json.loads(raw)
    need(pins['comparison_manifest_ref'] == COMPARISON_MANIFEST and pins['spec_ref'] == SPEC,
         'Original reviewed comparison and spec required')
    for row in pins['source_refs']:
        read_verified(row)
    original = json.loads(read_verified(COMPARISON_MANIFEST))
    # The comparison manifest also pins another parent's focal archive and a
    # checkpoint. Authenticate its exact document, but read executable sources
    # only here. Original loaders authenticate the fixed TRAIN001 inputs below.
    for row in executable_source_refs(original):
        read_verified(row)
    return dict(path=str(path.relative_to(ROOT)), sha256=digest)


def write(path, value):
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b'\n'
    need(len(raw) <= 2**20, 'Metadata-only result at most one MiB')
    with path.open('xb') as handle:
        handle.write(raw)


def run(output, source_ref):
    # This executes before importing any project/torch preparation code.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from arggnn.benchmarking import contracts as c
    from arggnn.american_c3_gnn_dataset_v1 import load_observation
    from research.benchmark_rebuild_20261024.american.treeloom_speed import runtime_v2 as fresh
    from research.preprocessing_optimization_20261026.integration import hooks
    from research.reviews.workflow_progress_diagnosis_20261018.treeloom.resident_pool_v2 import pack_bundle
    from research.efficiency_20261027.execution.compare_two_parent_preparation_v2 import CPU_random_signature
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    need(not torch.cuda.is_initialized(), 'CPU-only preparation interpreter')
    begun = time.monotonic()
    def check(stage):
        rss = int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        need(rss <= CAP, 'Own main RSS exceeded128GiB at ' + stage)
        need(time.monotonic() - begun <= SECONDS, 'Finite3600s preparation lifetime')
        need(not torch.cuda.is_initialized(), 'No CUDA initialization at ' + stage)
    spec = json.loads(read_verified(SPEC))
    need(spec['parents'][0] == PARENT and spec['CPU_threads'] == 8
         and spec['encoder_workers_per_parent'] == 4
         and spec['truth_refs_included'] is False and spec['labels_or_model_or_training'] is False,
         'Fixed original TRAIN001 and reviewed CPU8/process4 preparation')
    registry = fresh.FreshParentRegistry.from_ref(spec['parent_registry_ref'])
    need(registry.split_for(PARENT) == 'train', 'One original TRAIN parent')
    barrier = c.read_ref(spec['observation_barrier_ref'])
    need(barrier['all20_observations_sealed_before_truth'] and not barrier['private_truth_opened']
         and set(barrier['parents']) == set(registry.ids('train')),
         'Original complete observation-only barrier')
    seal = spec['seals'][PARENT]
    plan = spec['plans'][PARENT]
    item = spec['observed_inputs'][PARENT]
    need(seal == barrier['parents'][PARENT]
         and plan == dict(c.read_ref(seal['plan_ref']), plan_ref=seal['plan_ref'])
         and set(item) == set(fresh.INPUT_REFS), 'Original parent seal, plan and observation-only inputs')
    bound = c.read_ref(spec['bound_ref'])
    row = next(row for row in bound['parents'] if row['parent_id'] == PARENT)
    compact_bound = row['complete_compact_cache_upper_bound_bytes'] + 160000
    need(compact_bound == spec['parent_bounds'][PARENT], 'Original compact charge bound')
    external = {'bank.genotype': c.read_ref(item['observation_ref'])['genotypes_ref']}
    runtime = fresh.FreshTreeLoom(registry)
    bindings = hooks.install_tree(preparation_workers=4, max_rss_bytes=CAP)
    hooks.verify_bindings(bindings)
    before = CPU_random_signature()
    journal_bytes = 0
    def emit(row):
        nonlocal journal_bytes
        raw = (json.dumps(row, sort_keys=True, allow_nan=False) + '\n').encode()
        journal_bytes += len(raw)
        need(journal_bytes <= 2**20, 'Bounded function-boundary journal')
        with (output / 'phases.jsonl').open('ab') as handle:
            handle.write(raw)
    timers = PhaseTimers(emit)
    write(output / 'configuration.json', dict(parent=PARENT, source_manifest_ref=source_ref,
        original_comparison_manifest_ref=COMPARISON_MANIFEST, spec_ref=SPEC,
        inherited_verification_scope='Executable dependencies only; no unrelated parent archive or checkpoint',
        observation_inputs=item, observation_barrier_ref=spec['observation_barrier_ref'],
        host_cap_bytes=CAP, output_cap_bytes=64*2**20, max_seconds=SECONDS,
        CPU_threads=8, encoder_workers=4, preparation_bindings=bindings,
        phases=list(PHASES), checkpoint_opened=False, truth_opened=False,
        model_or_optimizer_executed=False, native_preparation_calls=1,
        clock_scope='Wall and main-process/calling-thread CPU only; child CPU is excluded'))
    try:
        state = timers.call('load_observation', load_observation,
            item['observation_ref'], item['focal_registry_ref'], dict(
                native_preparation_ref=item['native_preparation_ref'], class_order=list(fresh.CLASSES),
                expected_haplotypes=1600, reference_counts=[200,200,200],
                input_reference_class_mask=None, input_reference_sample_indices=None))
        preparation = runtime.preparation
        with ExitStack() as stack:
            for attribute, phase in (('encode_block', 'encode_block_total'),
                                     ('build_episodes', 'build_episodes'),
                                     ('prepare_cut_plan_v1', 'prepare_cut_plan')):
                stack.enter_context(timers.wrap(preparation, attribute, phase))
            stack.enter_context(timers.wrap(runtime.frozen.AmericanTreeLoomWholeParentV1,
                                            'validate', 'validate_complete_bundle'))
            bundle, metadata = timers.call('prepare_observation_total', runtime.prepare_observation,
                state, split='train', resource_check=check, **item,
                max_nodes=400000000, max_holds=100000, max_episode_payload_bytes=8*2**30)
        hooks.verify_bindings(bindings)
        del state
        need(bundle.parent_id == PARENT and bundle.sequence_length_bp == 30000000
             and bundle.native_holds == plan['native_holds'] and bundle.original_sites == plan['original_sites']
             and bundle.focal_identity_sha256 == seal['focal_identity_sha256']
             and all(getattr(bundle, key) == seal[key] == item[key] for key in fresh.INPUT_REFS),
             'Same complete original native parent and focal identities')
        record = timers.call('pack_bundle', pack_bundle, bundle, metadata,
            input_refs=item, external_arrays=external, resource_check=check)
        need(record['charged_bytes'] <= compact_bound, 'Original compact bound retained')
        timers.require_once(PHASES)
        after = CPU_random_signature()
        need(before == after, 'Original global CPU RNG preserved')
        import multiprocessing
        need(not multiprocessing.active_children(), 'Original encoder children joined')
        check('completed')
        by_phase = {row['phase']: row for row in timers.rows}
        subphases = ('encode_block_total', 'build_episodes', 'prepare_cut_plan', 'validate_complete_bundle')
        residual = by_phase['prepare_observation_total']['wall_seconds'] - sum(
            by_phase[name]['wall_seconds'] for name in subphases)
        write(output / 'completed.json', dict(status='completed', parent=PARENT,
            source_manifest_ref=source_ref, spec_ref=SPEC, phases=timers.rows,
            preparation_unwrapped_residual_seconds=residual,
            residual_scope='Native tree list, reference/focal checks, wrapper journal overhead and other untimed preparation work',
            counts=dict(native_holds=bundle.native_holds, native_nodes=bundle.bank.parent.numel(),
                        sites=bundle.original_sites, haplotypes=1600, native_focals=bundle.focal_hold.numel()),
            compact_charged_bytes=record['charged_bytes'], compact_bound_bytes=compact_bound,
            CPU_RNG_sha256=after, global_CPU_RNG_unchanged=True,
            temporary_timing_bindings_restored=True, original_preparation_bindings_verified=True,
            no_added_array_hash_or_restore_pass=True, all_encoder_workers_reaped=True,
            CUDA_initialized=False, truth_opened=False, model_or_optimizer_executed=False,
            production_changed=False, limitations=[
                'One original parent, one CPU preparation; not a concurrent-producer comparison.',
                'Encoder total includes child startup, ordered consumption, join and global materialization.',
                'Nested phase times are inclusive; do not add them to preparation total.',
                'CPU clocks exclude encoder child processes; they do not measure GIL ownership or DRAM bandwidth.',
                'Metadata and exact packing guards are unchanged; no new full-signature pass is performed.']))
        del record, bundle, metadata
        gc.collect()
    except BaseException as error:
        write(output / 'failed.json', dict(status='failed', error_type=type(error).__name__,
            error=str(error)[:2000], phases=timers.rows, source_manifest_ref=source_ref,
            no_retry=True, observed_unix=time.time()))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-pins-sha256', required=True)
    args = parser.parse_args()
    need(Path.cwd() == ROOT, 'Run from the remote project root')
    output = Path(args.output).resolve()
    need(output == OUTPUT and not output.exists(), 'One fixed fresh profile output directory')
    source_ref = verify_sources(args.source_pins_sha256)
    output.mkdir(parents=False, exist_ok=False)
    run(output, source_ref)


if __name__ == '__main__':
    main()
