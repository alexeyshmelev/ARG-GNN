"""Install once before producer threads; reuse the exact four-process encoder."""
from research.preprocessing_optimization_20261026.integration import hooks
from research.preprocessing_optimization_20261026.tree import process_observations
from .adaptive_prefetch_v2 import EncoderCredits, CHILD_ALLOWANCE_BYTES

_CREDITS = None


def coordinated_process4_encode_block(*args, resource_check, **kwargs):
    if _CREDITS is None:
        raise ValueError('Install one shared child-credit ledger before producers')
    resource_check('TreeLoom_coordinated_process4_encoder')
    _CREDITS.before_encoder()
    # The private executor context in this original function joins its four
    # children before returning. Its credit remains owned through later packing.
    return process_observations.encode_block(*args, **kwargs, resource_check=resource_check,
        preparation_workers=4, max_parallel_bytes=CHILD_ALLOWANCE_BYTES,
        tree_chunk_size=128, max_chunk_bytes=64*2**20)


def install_tree(credits):
    global _CREDITS
    if _CREDITS is not None or not isinstance(credits, EncoderCredits) or credits.active:
        raise ValueError('One inactive shared ledger installed before any producer')
    from research.preprocessing_optimization_20261026.tree import episodes, cut
    american = 'research.benchmark_rebuild_20261024.american.treeloom_speed.static_preparation_v2'
    recovery = 'research.repairs.benchmark_recovery_20261020.runtime'
    european = 'research.reviews.TreeLoom_preparation_reuse_source_20261017_v1.optimized_observations_v1'
    compatibility = 'research.reviews.H060_isolated_hold_compatibility_source_20261017_v1'
    _CREDITS = credits
    try:
        return hooks._install([
            (american, 'encode_block', recovery+'.observations', 'encode_block', coordinated_process4_encode_block),
            (american, 'build_episodes', recovery+'.episodes', 'build_episodes', episodes.build_episodes),
            (american, 'prepare_cut_plan_v1', recovery+'.cuts', 'prepare_cut_plan_v1', cut.prepare_cut_plan_v1),
            (european, 'encode_block', european, 'encode_block', coordinated_process4_encode_block),
            (compatibility+'.episode_preparation_v1', 'build_clade_episode_arrays_v1',
             compatibility+'.episode_preparation_v1', 'build_clade_episode_arrays_v1', episodes.build_clade_episode_arrays_v1),
            (compatibility+'.cut_preparation_v1', 'prepare_cut_plan_v1',
             compatibility+'.cut_preparation_v1', 'prepare_cut_plan_v1', cut.prepare_cut_plan_v1),
        ])
    except BaseException:
        _CREDITS = None
        raise
