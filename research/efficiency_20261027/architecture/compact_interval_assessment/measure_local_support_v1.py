"""Measure sparse fixed-grid node support using only observed compact tables."""
from pathlib import Path
import hashlib
import json
import math
import resource
import time

resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
import numpy as np
import tskit

BASE = Path('/workspace-SR003.nfs2/ARG-GNN')
OUT = BASE / 'research/efficiency_20261027/architecture/compact_interval_assessment'
started = time.monotonic()
counts_path = OUT / 'TRAIN000_compact_counts_v2.json'
counts = json.loads(counts_path.read_text())
tree_path = BASE / counts['tree_ref']['path']
with tree_path.open('rb') as stream:
    actual_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
assert actual_hash == counts['tree_ref']['sha256']
ts = tskit.load(tree_path)
edges = ts.tables.edges
samples = ts.samples()
node_ids = np.concatenate((edges.parent, edges.child, samples))
lefts = np.concatenate((edges.left, edges.left, np.zeros(len(samples))))
rights = np.concatenate((edges.right, edges.right, np.full(len(samples), ts.sequence_length)))
results = []
for width in (128000, 32000, 8000):
    number_of_bins = math.ceil(ts.sequence_length / width)
    starts = np.floor(lefts / width).astype(np.int64)
    ends = np.ceil(rights / width).astype(np.int64) - 1
    order = np.lexsort((starts, node_ids))
    current_node = -1
    current_end = -1
    node_bin_count = 0
    merged_bin_ranges = 0
    for index in order:
        node, first, last = int(node_ids[index]), int(starts[index]), int(ends[index])
        assert 0 <= first <= last < number_of_bins
        if node != current_node:
            current_node, current_end = node, -1
        if first > current_end:
            node_bin_count += last - first + 1
            merged_bin_ranges += 1
        elif last > current_end:
            node_bin_count += last - current_end
        current_end = max(current_end, last)
    edge_bin_count = int(np.sum(np.ceil(edges.right / width).astype(np.int64) - np.floor(edges.left / width).astype(np.int64)))
    results.append({
        'nominal_bin_width_bp': width,
        'bins': number_of_bins,
        'active_node_bin_coefficients': node_bin_count,
        'merged_node_bin_ranges': merged_bin_ranges,
        'edge_bin_intersections': edge_bin_count,
        'one_width32_FP32_coefficient_array_bytes': node_bin_count * 32 * 4,
        'two_width32_FP32_coefficient_arrays_bytes': node_bin_count * 32 * 8,
        'edge_bin_FP32_width32_message_array_if_materialized_bytes': edge_bin_count * 32 * 4,
        'one_event_segment_nonlinearity_upper_bound': 2 * edge_bin_count + node_bin_count,
        'leaf_bin_coefficients': len(samples) * number_of_bins,
        'one_leaf_width32_FP32_array_bytes': len(samples) * number_of_bins * 32 * 4,
        'fraction_of_edge_spans_shorter_than_nominal_bin': float(np.mean((edges.right - edges.left) < width)),
    })
receipt = {
    'schema': 'TRAIN000_sparse_local_support_estimates_v1',
    'passed': True,
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'tree_ref': counts['tree_ref'],
    'tree_file_sha256_verified': True,
    'count_receipt_sha256': hashlib.sha256(counts_path.read_bytes()).hexdigest(),
    'method': 'For every node, union integer grid-bin ranges touched by incident edge intervals; all samples have full-sequence support. Count without constructing node-by-bin or edge-by-bin arrays.',
    'estimates': results,
    'caveats': [
        'Logical coefficient and intersection counts only; no architecture implemented, trained, or timed.',
        'Fixed-grid projection changes position resolution and the nonlinear model; this is not exact TreeLoom compression.',
        'Memory figures exclude gradients, backward save tensors, weights, allocator and workspaces.',
        'Full observed compact site and mutation tables were loaded; genotype matrix payload and private truth were not read.',
        'Counts refer to one upward or downward incidence direction; using both directions doubles edge visits.'
    ],
    'GPU_initialized': False,
    'dense_observation_bank_materialized': False,
    'wall_seconds': time.monotonic() - started,
    'maxrss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
}
path = OUT / 'TRAIN000_local_support_v1.json'
with path.open('x') as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps(receipt, indent=2))
print('RECEIPT_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
