"""One explicit combined candidate; no automatic scientific or memory fallback."""
GIB = 2**30
FACTORY = 'research.efficiency_20261027.architecture.exact_upward_residency.large_resident_executor_v1:build_executor'
PINS = 'research/efficiency_20261027/integration_candidate/source_pins_combined_v3.json'
EXPECTED = dict(executor_kind='explicit_vjp', CPU_threads=8, holds_per_block=512,
    haplotypes_per_gru=64, history_holds_per_window=1024, context_residency='CPU',
    layout_cache_max_bytes=80*GIB, layout_cache_policy='actual_RSS_evict_unleased',
    executor_factory=FACTORY, boundary_fusion=True, upward_direct_residency=True,
    resident_scratch_bytes=16*GIB, preparation_producers=2, encoder_workers_per_producer=4,
    maximum_encoder_children=8, child_allowance_each_bytes=8*GIB,
    max_rss_bytes=768*GIB, minimum_free_gpu_bytes=76*GIB, preparation_headroom_bytes=128*GIB,
    training_headroom_bytes=128*GIB, observation_cache_bytes=352*GIB)


def configuration(value):
    if type(value) is not dict or set(value) != set(EXPECTED) or any(
            type(value[key]) is not type(wanted) or value[key] != wanted
            for key, wanted in EXPECTED.items()):
        raise ValueError('Exact combined512/64/1024 CPU-context resident adaptive2 configuration required')
    return dict(value)
