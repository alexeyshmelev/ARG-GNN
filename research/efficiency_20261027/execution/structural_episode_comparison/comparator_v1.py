"""Bounded CPU structural-only episode comparison. No feature bank or model calls."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import threading
import time

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
BASE = ROOT / 'research/efficiency_20261027/execution/structural_episode_comparison'
SPEC = BASE / 'spec_v1.json'
PINS = BASE / 'source_pins_v1.json'
EPISODE_MAP = {'node_flags': 'flags.node_flags',
    'member_node': 'episodes.member_node', 'member_hold': 'episodes.member_hold',
    'member_episode': 'episodes.member_episode', 'episode_size': 'episodes.episode_size',
    'hold_span': 'episodes.hold_span'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def file_ref(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            digest.update(block)
    return dict(path=str(path.relative_to(ROOT)), sha256=digest.hexdigest())


def read_ref(reference):
    need(file_ref(reference['path']) == reference, 'Exact immutable metadata reference')
    return json.loads((ROOT / reference['path']).read_text())


def write(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    need(len(raw) <= 8*2**20, 'Each comparison metadata file is at most eight MiB')
    with Path(path).open('xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return file_ref(path)


def array_record(array):
    """Exact original resident_pool_v2.array_hash contract, streaming eight MiB."""
    import numpy as np
    need(isinstance(array, np.ndarray) and array.flags.c_contiguous,
         'Owned contiguous structural array required')
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(str(array.shape).encode())
    view = memoryview(array).cast('B')
    for first in range(0, len(view), 8*2**20):
        digest.update(view[first:first+8*2**20])
    return dict(sha256=digest.hexdigest(),
        layout=dict(shape=list(array.shape), dtype=array.dtype.str, count=int(array.size)))


def authenticate_array(array, name, archived):
    actual = array_record(array)
    need(actual['sha256'] == archived['arrays'][name]
         and actual['layout'] == archived['array_metadata'][name],
         'Complete archived structural identity mismatch: ' + name)
    return actual


def structure_from_ts(ts, sample_nodes, expected, *, check, native_tree, workspace_factory,
                      layout, maximum_nodes_per_hold=65536, maximum_roots=1048576):
    """Stream native holds into final arrays in exact encoder concatenation order."""
    import numpy as np
    n, t, h = expected
    need(ts.num_trees == t and len(sample_nodes) == h, 'Exact complete observed dimensions')
    parent = np.empty(n, dtype=np.int64)
    samples = np.empty((t, h), dtype=np.int64)
    spans = np.empty(t, dtype=np.float32)
    workspace = workspace_factory(ts)
    node_time = np.asarray(ts.nodes_time, dtype=np.float64)
    age = node_time[sample_nodes]
    identity = np.arange(h, dtype=np.int64)
    offset = total_roots = 0
    isolated = []
    previous_right = 0.0
    holds_seen = 0
    for hold, tree in enumerate(ts.trees()):
        if hold % 256 == 0:
            check('structural_native_tree')
        need(hold == tree.index and hold < t, 'Exact original native hold order')
        native = native_tree(tree, sample_nodes, node_time=node_time, workspace=workspace)
        count = len(native.parent)
        need(0 < count <= maximum_nodes_per_hold and offset + count <= n,
             'Whole native tree exceeds diagnostic memory bound; never truncate')
        p, clock, sn, sid, post, depth, height, descendant, duration = layout(native)
        need(np.array_equal(sid, identity) and np.array_equal(clock[sn], age),
             'Every original sample/date retains its exact column')
        child_count = np.bincount(p[p >= 0], minlength=count)
        need(not child_count[sn].any() and np.count_nonzero(child_count == 0) == h,
             'All active leaves are the original observed samples')
        roots = int(np.count_nonzero(p == -1))
        total_roots += roots
        need(total_roots <= maximum_roots, 'Whole forest exceeds root-container memory bound')
        if roots == count:
            need(count == h and tree.num_sites == 0,
                 'Only complete all-sampled zero-site isolated holds are admitted')
            isolated.append(hold)
        else:
            need(roots == 1, 'Original connected or all-isolated native topology')
        target = parent[offset:offset+count]
        target[:] = p
        selected = p >= 0
        target[selected] += offset
        samples[hold] = sn + offset
        left, right = map(float, tree.interval)
        need(left == previous_right and right > left, 'Original contiguous half-open intervals')
        spans[hold] = np.float32(np.float64(right) - np.float64(left))
        previous_right = right
        offset += count
        holds_seen += 1
        del native, p, clock, sn, sid, post, depth, height, descendant, duration, child_count, target, selected
    need(offset == n and holds_seen == t and previous_right == ts.sequence_length,
         'All original nodes and complete sequence are retained')
    return parent, samples, spans, tuple(isolated), dict(native_nodes=n,
        native_holds=t, haplotypes=h, total_roots=total_roots,
        maximum_nodes_per_hold=maximum_nodes_per_hold, no_tree_list_retained=True)


def CPU_random_signature():
    import random
    import numpy as np
    import torch
    digest = hashlib.sha256(repr(random.getstate()).encode())
    state = np.random.get_state()
    digest.update(state[0].encode())
    digest.update(state[1].tobytes())
    digest.update(repr(state[2:]).encode())
    digest.update(torch.get_rng_state().numpy().tobytes())
    return digest.hexdigest()


def compare_builders(parent, samples, spans, isolated, archived, builders, *, check, payload_cap):
    """Authenticate before either builder; retain only hashes/metadata between arms."""
    structural = {name: authenticate_array(value, 'bank.'+name, archived)
                  for name, value in [('parent', parent), ('sample_index', samples), ('spans', spans)]}
    expected = None
    baseline_metadata = None
    records = []
    for name, builder in builders:
        check('before_' + name)
        started = time.perf_counter()
        cpu_started = time.thread_time()
        arrays, metadata = builder(parent, samples, spans, max_payload_bytes=payload_cap,
                                  isolated_hold_indices=isolated)
        timing = dict(wall_seconds=time.perf_counter()-started,
                      main_thread_CPU_seconds=time.thread_time()-cpu_started)
        check('after_' + name)
        need(set(arrays) == set(EPISODE_MAP), 'Exactly all six original episode arrays')
        actual = {key: authenticate_array(arrays[key], archived_name, archived)
                  for key, archived_name in EPISODE_MAP.items()}
        need(metadata['native_nodes'] == len(parent)
             and metadata['native_holds'] == len(spans)
             and metadata['haplotypes'] == samples.shape[1], 'Full original metadata dimensions')
        if expected is None:
            expected, baseline_metadata = actual, metadata
        else:
            need(actual == expected and metadata == baseline_metadata,
                 'Every episode array and full metadata must exactly match original')
        need(structural == {key: array_record(value) for key, value in
             [('parent', parent), ('sample_index', samples), ('spans', spans)]},
             'Original structural inputs must remain unchanged')
        records.append(dict(name=name, timing=timing, arrays=actual,
                            metadata_exact_to_original=True, archived_episode_arrays_exact=True))
        del arrays, metadata, actual
        gc.collect()
        check('released_' + name)
    need(len(records) == 2 and [r['name'] for r in records] == ['original', 'candidate'],
         'One fixed-order complete original/candidate pair only')
    return dict(structural=structural, records=records, episode_metadata=baseline_metadata,
                archived_metadata_subtree_available=False, archived_metadata_equivalence_claim=False,
                limitation='Archive contains only whole-metadata digest; exact episode metadata equality is original versus candidate, not reconstructed archived metadata.')


class OwnWorkerMonitor:
    """Own-process failstop during unmodified builders; never signal another job."""
    def __init__(self, output, *, operating_bytes, family_bytes, seconds):
        self.output = Path(output)
        self.operating = operating_bytes
        self.maximum = family_bytes
        self.seconds = seconds
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.peak_RSS = self.peak_PSS = self.samples = 0
        self.failure = None
        self.thread = threading.Thread(target=self._loop, name='episode_comparison_guard', daemon=True)

    def sample(self):
        import psutil
        process = psutil.Process()
        need(not process.children(recursive=True), 'Comparator must have no child processes')
        rss = process.memory_info().rss
        pss = getattr(process.memory_full_info(), 'pss', rss)
        elapsed = time.monotonic()-self.started
        self.peak_RSS = max(self.peak_RSS, rss)
        self.peak_PSS = max(self.peak_PSS, pss)
        self.samples += 1
        need(rss <= self.operating and pss <= self.operating, 'Twenty-four GiB operating stop reached')
        need(rss <= self.maximum and pss <= self.maximum, 'Thirty-two GiB family cap exceeded')
        need(elapsed <= self.seconds, 'Bounded structural comparison wall time reached')
        total = sum(p.stat().st_size for p in self.output.iterdir() if p.is_file())
        need(total <= 16*2**20, 'Sixteen MiB comparison output bound')
        return dict(elapsed_seconds=elapsed, RSS_bytes=rss, PSS_bytes=pss)

    def check(self, stage):
        need(self.failure is None, 'Own worker monitor failed: '+str(self.failure))
        need(time.monotonic()-self.started <= self.seconds, 'Bounded wall time at '+stage)

    def _loop(self):
        while not self.stop.wait(2.):
            try:
                self.sample()
            except BaseException as error:
                self.failure = type(error).__name__ + ': ' + str(error)
                try:
                    write(self.output/'monitor_failure.json', dict(error=self.failure,
                        peak_RSS_bytes=self.peak_RSS, peak_PSS_bytes=self.peak_PSS,
                        sampled_stop_not_OS_quota=True, own_worker_exit_code=86))
                finally:
                    os._exit(86)

    def start(self):
        self.sample()
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=10.)
        need(not self.thread.is_alive(), 'Actual comparison monitor join required')
        self.sample()
        return dict(peak_RSS_bytes=self.peak_RSS, peak_PSS_bytes=self.peak_PSS,
                    samples=self.samples, sampling_seconds=2,
                    sampled_stop_not_OS_quota=True, own_children=0)


def read_admission(path, *, source_ref, spec_ref, spec, now=None):
    path = Path(path).resolve()
    need(path.is_relative_to(ROOT), 'ROOT-contained admission metadata required')
    raw = path.read_bytes()
    need(len(raw) <= 65536, 'Bounded ROOT admission metadata')
    admission_ref = dict(path=str(path.relative_to(ROOT)),
                        sha256=hashlib.sha256(raw).hexdigest())
    admission = json.loads(raw)
    observed = admission.get('observed_unix')
    now = time.time() if now is None else now
    need(type(observed) in (int, float) and -5 <= now-observed <= 120,
         'ROOT startup admission must be at most120 seconds old')
    need(admission.get('schema') == 'TreeLoom_structural_episode_ROOT_admission_v1'
         and admission.get('approved') is True and admission.get('source_manifest_ref') == source_ref
         and admission.get('spec_ref') == spec_ref
         and admission.get('family_cap_bytes') == spec['family_cap_bytes']
         and admission.get('operating_stop_bytes') == spec['operating_stop_bytes']
         and admission.get('current_peers_checked') is True
         and admission.get('outer_guardian_required') is True
         and set(spec['required_peer_directories']) <= set(admission.get('peer_directories', []))
         and admission.get('retained_failed_v1_storage_charged') is True,
         'ROOT admission with both European peers and ongoing outer guardian required')
    return admission, admission_ref


def run(args):
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be hidden before import')
    need(file_ref(PINS)['sha256'] == args.source_sha256, 'Exact independently reviewed source manifest')
    pins = json.loads(PINS.read_text())
    need(all(file_ref(r['path']) == r for r in pins['source_refs']), 'Immutable comparator source closure')
    spec = read_ref(pins['spec_ref'])
    admission, admission_ref = read_admission(args.admission, source_ref=file_ref(PINS),
        spec_ref=pins['spec_ref'], spec=spec)
    output = Path(args.output).resolve()
    need(output.is_relative_to(ROOT/'research/efficiency_20261027/root/runs')
         and not output.exists(), 'Fresh ROOT-owned diagnostic output required')
    output.mkdir()
    monitor = OwnWorkerMonitor(output, operating_bytes=spec['operating_stop_bytes'],
                              family_bytes=spec['family_cap_bytes'], seconds=spec['max_seconds'])
    monitor.start()
    try:
        import numpy as np
        import torch
        import tskit
        from threadpoolctl import threadpool_limits, threadpool_info
        from research.repairs.benchmark_recovery_20261020.runtime.observations import native_tree, native_workspace_v1
        from research.preprocessing_optimization_20261026.tree.kernels import _layout_with_isolated_samples
        from research.preprocessing_optimization_20261026.tree.episodes import build_clade_episode_arrays_v1 as original
        from research.efficiency_20261027.independent_efficiency_challenge.episode_integer_vectorization.episodes_v1 import build_clade_episode_arrays_v1 as candidate
        from research.efficiency_20261027.execution.preparation_evidence_v2 import validate_signature
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        pool_guard = threadpool_limits(limits=1)
        need(all(p['num_threads'] <= 1 for p in threadpool_info()) and not torch.cuda.is_initialized(),
             'One-thread CPU libraries and no CUDA initialization')
        rng_before = CPU_random_signature()
        archived = read_ref(spec['archived_signature_ref'])
        archived_peer = read_ref(spec['archived_concurrent_signature_ref'])
        validate_signature(archived)
        need(archived == archived_peer, 'Qualified serial and concurrent complete signatures agree')
        manifest = read_ref(spec['observation_ref'])
        roster = read_ref(manifest['roster_ref'])
        need(manifest['parent_id'] == roster['parent_id'] == spec['parent_id']
             and manifest['schema'] == 'american_c3_observed_v1'
             and manifest['query_ancestry_truth_in_inputs'] is False
             and manifest['split'] == 'train' and manifest['sequence_length_bp'] == 30000000
             and manifest['tree_ref'] == spec['tree_ref'], 'Exact observed TRAIN001 identity')
        tree_path = ROOT / spec['tree_ref']['path']
        need(tree_path.stat().st_size == spec['tree_file_bytes']
             and file_ref(tree_path) == spec['tree_ref'], 'Authenticate only sanitized observed tree payload')
        started = time.perf_counter()
        ts = tskit.load(str(tree_path))
        need(ts.num_nodes <= spec['max_source_nodes'] and ts.nbytes <= spec['max_tree_storage_bytes'],
             'Bounded source tree storage before native workspace allocation')
        sample_nodes = np.asarray(ts.samples(), dtype=np.int64)
        rows = roster['samples']
        need([r['sample_index'] for r in rows] == list(range(spec['haplotypes']))
             and sample_nodes.tolist() == [r['observed_node_id'] for r in rows]
             and np.array_equal(ts.nodes_time[sample_nodes], np.asarray([r['age_generations'] for r in rows]))
             and ts.sequence_length == 30000000 and ts.num_samples == spec['haplotypes'],
             'Original ordered observed sample identities and dates')
        tables = ts.tables
        need(ts.num_migrations == ts.num_populations == ts.num_individuals == 0
             and not tables.metadata and not ts.num_provenances
             and (tables.nodes.population == -1).all() and (tables.nodes.individual == -1).all()
             and all(len(table.metadata) == 0 for table in [tables.nodes,tables.edges,tables.sites,tables.mutations]),
             'Same sanitized observation-only native tree contract')
        del tables
        parent, samples, spans, isolated, reconstruction = structure_from_ts(ts, sample_nodes,
            (spec['native_nodes'], spec['native_holds'], spec['haplotypes']),
            check=monitor.check, native_tree=native_tree, workspace_factory=native_workspace_v1,
            layout=_layout_with_isolated_samples, maximum_nodes_per_hold=spec['max_nodes_per_hold'],
            maximum_roots=spec['max_native_roots'])
        reconstruction['wall_seconds'] = time.perf_counter()-started
        del ts, sample_nodes, rows, roster, manifest
        gc.collect()
        result = compare_builders(parent, samples, spans, isolated, archived,
            [('original', original), ('candidate', candidate)], check=monitor.check,
            payload_cap=spec['max_episode_payload_bytes'])
        need(CPU_random_signature() == rng_before, 'Structural comparison leaves CPU RNG unchanged')
        metadata_ref = write(output/'original_episode_metadata.json', result.pop('episode_metadata'))
        result.update(schema='TreeLoom_structural_episode_comparison_result_v1', status='passed',
            source_manifest_ref=file_ref(PINS), spec_ref=pins['spec_ref'], admission_ref=admission_ref,
            archived_signature_ref=spec['archived_signature_ref'], reconstruction=reconstruction,
            episode_metadata_ref=metadata_ref, isolated_hold_indices=list(isolated),
            CUDA_initialized=torch.cuda.is_initialized(), CPU_RNG_unchanged=True,
            full_feature_bank_created=False,
            genotype_payload_opened=False, truth_or_checkpoint_opened=False,
            model_or_optimizer_executed=False, native_inference_executed=False,
            fixed_order_single_pair_not_steady_state_benchmark=True)
        del parent, samples, spans
        gc.collect()
        need(file_ref(args.admission) == admission_ref, 'ROOT admission bytes changed during comparison')
        result['monitor'] = monitor.finish()
        write(output/'completed.json', result)
        print(json.dumps(dict(status='passed', result_ref=file_ref(output/'completed.json'))))
    finally:
        monitor.stop.set()
        monitor.thread.join(timeout=10.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--admission', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
