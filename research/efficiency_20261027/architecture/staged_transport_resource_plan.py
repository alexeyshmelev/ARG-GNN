"""Static byte inventory and schedule counts; does not import torch or run models."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
BASE = ROOT / 'research/efficiency_20261027/architecture'


def ref(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def tensor_bytes(value):
    widths = {'int64': 8, 'int32': 4, 'int16': 2, 'int8': 1, 'uint8': 1,
              'float64': 8, 'float32': 4, 'float16': 2, 'bool': 1}
    if isinstance(value, dict):
        if 'shape' in value and 'dtype' in value:
            return math.prod(value['shape']) * widths[value['dtype']]
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, list):
        return sum(tensor_bytes(item) for item in value)
    return 0


def main():
    metadata_path = ROOT / 'research/preprocessing_optimization_20261026/integration/full_candidate.json'
    static_path = ROOT / 'research/efficiency_20261027/execution/full_gpu_peak_static_accounting.json'
    metadata = json.loads(metadata_path.read_text())
    static = json.loads(static_path.read_text())
    t, h = metadata['bundle']['bank']['sample_index']['shape']
    n = metadata['bundle']['bank']['parent']['shape'][0]
    assert (t, h, n) == (77472, 1600, 205369272)
    history = t * h * 32 * 4
    tile = t * 128 * 32 * 4
    node = n * 32 * 4
    incoming = 2 * math.ceil(t / 1024) * 128 * 32 * 4
    bank_inventory = {key: tensor_bytes(value) for key, value in metadata['bundle']['bank'].items()}
    bundle_inventory = {key: tensor_bytes(value) for key, value in metadata['bundle'].items()}
    named = {}
    for cache in (False, True):
        arrays = 4 * history + 6 * tile + incoming + (2 * node if cache else 0)
        named['cached' if cache else 'uncached'] = dict(
            full_history_and_adjoint_conservative_bytes=4 * history,
            tile_scratch_conservative_bytes=6 * tile,
            copied_incoming_boundaries_bytes=incoming,
            first_node_cache_and_adjoint_bytes=2 * node if cache else 0,
            named_array_total_bytes=arrays, named_array_total_GiB=arrays / 2**30,
            plus_retained_full_prepared_bundle_bytes=arrays + sum(bundle_inventory.values()),
            plus_retained_full_prepared_bundle_GiB=(arrays + sum(bundle_inventory.values())) / 2**30)
    prior = {}
    for name in ('warm_libraries_1_monolithic.json', 'warm_libraries_1_bounded.json'):
        path = BASE / 'transport_gpu4_v1/data' / name
        row = json.loads(path.read_text())
        prior[name] = dict(ref=ref(path), **{key: row[key] for key in (
            'forward_seconds', 'backward_seconds', 'GPU_peak_allocated_bytes', 'GPU_peak_reserved_bytes')})
    receipt_path = BASE / 'staged_CPU_parity_v1.json'
    receipt = json.loads(receipt_path.read_text())
    counts = {}
    for holds in (128, t):
        old_tiles = math.ceil(h / 16)
        new_tiles = math.ceil(h / 128)
        old_scan = math.ceil(holds / 32)
        new_scan = math.ceil(holds / 1024)
        counts[str(holds)] = dict(
            old_tree_blocks=math.ceil(holds / 16), new_tree_blocks=math.ceil(holds / 128),
            old_first_history_scan_merge_calls=3 * old_tiles * old_scan,
            new_first_history_scan_merge_calls=3 * new_tiles * new_scan,
            old_haplotype_tiles=old_tiles, new_haplotype_tiles=new_tiles,
            old_scan_chunks_per_tile=old_scan, new_scan_chunks_per_tile=new_scan,
            scope='Forward first history only; excludes final-focal active tiles, VJPs, kernels, allocator and guard costs.')
    result = dict(schema='TreeLoom_Transport_staged_resource_and_overhead_plan_v1', status='prepared_not_GPU_qualified',
        sources=[ref(BASE / name) for name in ('staged_transport.py', 'test_staged_transport.py',
                 'staged_CPU_parity_v1.json', 'staged_transport_resource_plan.py')],
        full_parent_metadata_ref=ref(metadata_path), static_topology_ref=ref(static_path),
        holds=t, haplotypes=h, nodes=n, hidden_dim=32, execution_partition=dict(tree_holds=128, haplotypes=128, scan_holds=1024),
        tree_blocks=static['blocks'], maximum_native_nodes_per_tree_block=static['maximum_nodes_per_block'],
        one_history_bytes=history, one_haplotype_tile_bytes=tile, one_all_node_state_bytes=node,
        named_host_arrays=named, prepared_bank_tensor_bytes=bank_inventory,
        prepared_bundle_tensor_bytes=bundle_inventory, prepared_bundle_tensor_total_bytes=sum(bundle_inventory.values()),
        resource_limitations=[
            'These are named tensor payloads and conservative overlapping array sums, not measured RSS or an admission request.',
            'Full prepared bundle tally includes original bank, episode arrays, cut plan and focal arrays without claiming their storage cannot alias.',
            'Local topology jump indexes, one-block CPU/device copies, autograd tapes, library workspaces, allocator cache, Python objects and preparation transients are excluded.',
            'The executor accounting method reports six tile scratch arrays but omits copied incoming boundaries; this plan explicitly adds those boundaries.',
            'Six tile scratch arrays plus explicit boundaries is conservative for the reviewed code; full-parent memory still needs a guarded measurement.',
            'GPU peak cannot be inferred from one node-state size or average topology; the maximum block and scan tape must be measured.',
            'Caching avoids duplicate first-tree/SNP evaluations but adds all-node host copies and adjoints; end-to-end performance remains unknown.'
        ], schedule_counts=counts,
        cached_compute_change='Per tree block, initial/SNP/first-tree evaluation count changes from two no-grad forward evaluations plus two differentiable recomputations to one no-grad forward plus one differentiable recomputation; second-tree calls remain one forward plus one backward recomputation.',
        explicit_VJP_change='Sequential local backward calls consume cross-block cotangents; no global checkpoint/copy autograd DAG spans all histories or tree blocks.',
        previous_GPU128_records=prior,
        previous_GPU128_terminal_status='failed exit1 after scientific receipts: final json.dumps received an unsupported flush keyword; original failure and evidence preserved.',
        previous_GPU128_independent_receipt_ref=ref(BASE / 'transport_gpu128_post_result_verification.json'),
        CPU_check=dict(ref=ref(receipt_path), status=receipt['status'], trials=len(receipt['trials']),
                      maximum_logit_error=max(row['logits_max_abs_error'] for row in receipt['trials']),
                      maximum_absolute_gradient_error=max(row['gradients_max_abs_error'] for row in receipt['trials']),
                      gradient_gate='elementwise atol1e-6/rtol1e-4; maximum absolute error need not be below1e-6',
                      all_argmax_equal=all(row['argmax_equal'] for row in receipt['trials']),
                      CUDA_initialized=receipt['CUDA_initialized']),
        no_claims=['No measured GPU speedup or duty improvement.', 'No full-parent memory qualification.',
                   'No ancestry accuracy or preservation of original TreeLoom expressivity.', 'No paired-training source/config changed.'])
    output = BASE / 'staged_transport_resource_plan.json'
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(output=ref(output), named_host_arrays=named, schedule_counts=counts), indent=2))


if __name__ == '__main__':
    main()
