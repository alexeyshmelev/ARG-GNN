"""One fresh TRAIN parent's exact legacy-fast seal plus unchanged FP32 global preparation.

No runner, scheduler, admission, target resolver, writer or model step exists
here. A future guarded runner supplies a fresh original provider and its bound
fast source refs. Import only after that runner's real admission/runtime gates.
"""
from __future__ import annotations
import hashlib
import json
import os
import numpy as np
from arggnn.american_c3_common_v1 import require, bound
from arggnn.american_c3_gnn_dataset_isolated_v2 import load_observation
from arggnn.american_c3_gnn_stream_train_v1 import validate_plan_v1, json_sha
from research.reviews.H060_throughput_source_analysis_20261017_v1.observation_provider_candidate_v1 import SealedObservationProviderV1
from research.reviews.TreeLoom_observed_preparation_fast_source_20261017_v1.full_preparation_fast_v1 import (
    prepare_from_loaded_observation_v1, _bank_tensors, _episode_tensors)

HERE = 'research/reviews/H060_legacy_fast_full_training_source_20261017_v1'
REQUIRED_FAST_SOURCE_PATHS = (
    'research/reviews/TreeLoom_observed_preparation_fast_source_20261017_v1/observations_fast_v1.py',
    'research/reviews/TreeLoom_observed_preparation_fast_source_20261017_v1/full_preparation_fast_v1.py',
    'research/reviews/TreeLoom_observed_preparation_fast_source_20261017_v1/parent_startup_fast_v1.py',
    'research/reviews/level_batched_observed_preparation_source_20261017_v1/level_batched_support_v1.py',
    'research/reviews/H060_throughput_source_analysis_20261017_v1/legacy_support_exact_v1.py',
    'research/reviews/H060_throughput_source_analysis_20261017_v1/observations_legacy_exact_v1.py',
    'research/reviews/H060_throughput_source_analysis_20261017_v1/observation_provider_candidate_v1.py',
    'research/reviews/H060_legacy_fast_full_training_source_20261017_v1/parent_startup_legacy_fast_v1.py',
    'research/reviews/H060_legacy_fast_full_training_source_20261017_v1/warmstart_train_v2.py',
    'research/reviews/H060_legacy_fast_full_training_source_20261017_v1/__init__.py',
    'src/arggnn/american_c3_gnn_stream_train_isolated_v2.py',
    'src/arggnn/american_c3_gnn_dataset_isolated_v2.py',
    'src/arggnn/mutation_routed_tree_observations_v1.py',
    'research/reviews/H060_throughput_source_analysis_20261017_v1/__init__.py',
)
LEGACY_PRODUCER_SOURCE_PATHS = (
    'research/reviews/H060_throughput_source_analysis_20261017_v1/legacy_support_exact_v1.py',
    'research/reviews/H060_throughput_source_analysis_20261017_v1/observations_legacy_exact_v1.py',
    'research/reviews/H060_throughput_source_analysis_20261017_v1/observation_provider_candidate_v1.py',
    'src/arggnn/american_c3_gnn_stream_train_isolated_v2.py',
    'src/arggnn/american_c3_gnn_dataset_isolated_v2.py',
    'src/arggnn/mutation_routed_tree_observations_v1.py',
    'research/reviews/H060_legacy_fast_full_training_source_20261017_v1/parent_startup_legacy_fast_v1.py',
    'research/reviews/H060_throughput_source_analysis_20261017_v1/__init__.py',
)


def _check_fast_sources_v1(source_refs):
    require(isinstance(source_refs, (list, tuple))
        and all(isinstance(r, dict) and set(r) == {"path", "sha256"} for r in source_refs)
        and len(source_refs) == len(REQUIRED_FAST_SOURCE_PATHS)
        and {r["path"] for r in source_refs} == set(REQUIRED_FAST_SOURCE_PATHS),
        "Exact complete legacy/global preparation refs from the fresh admitted source closure")
    refs = [dict(r) for r in sorted(source_refs, key=lambda r: r["path"])]
    for record in refs:
        bound(record)
    return refs


def fast_global_identity_v1(bundle):
    """Original ordered CPU-observation tensor digest, for these new bytes."""
    tensors=(_bank_tensors(bundle.bank)+_episode_tensors(bundle.episodes)
        +bundle.cut.tensors()+(bundle.focal_hold,bundle.focal_sample_column))
    digest=hashlib.sha256();total=0
    for i,tensor in enumerate(tensors):
        require(tensor.device.type=="cpu" and not tensor.requires_grad,
            "CPU observation-only global feature seal")
        value=tensor.detach().contiguous().numpy()
        digest.update(str(i).encode());digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode());digest.update(value.tobytes())
        total+=value.nbytes
    digest.update(bundle.focal_identity_sha256.encode())
    return dict(tensor_sha256=digest.hexdigest(),tensor_bytes=total,
        focal_identity_sha256=bundle.focal_identity_sha256,
        trees=bundle.native_holds,sites=bundle.original_sites,
        all_global_structural_recipes_before_truth=True)


def seal_then_prepare_one_parent_v1(provider, cfg, runtime_guard, *, fast_source_refs):
    """Load once, run exact legacy-fast seal, then prepare/seal current global bytes.

    Only a fresh reviewed candidate provider may be used. Its original seal() dynamically
    reads the already supplied observed state and performs its usual release.
    Our local reference retains that same state for global preparation. No
    callable or class is monkeypatched; no existing provider instance migrates.
    The caller must complete/cache all20 parents before any TRAIN target join.
    """
    require(type(provider) is SealedObservationProviderV1
        and callable(runtime_guard) and provider.seal_record is None
        and not provider.cache and provider.used == 0
        and provider.dataset._state is None and provider.dataset._pid is None
        and provider.dataset.direction == "forward",
        "Fresh candidate legacy provider, not a live/migrated or already sealed provider")
    plan = provider.plan
    validate_plan_v1(plan, parent_id=plan["parent_id"], length_bp=2000000)
    checked_sources = _check_fast_sources_v1(fast_source_refs)
    state = None
    try:
        runtime_guard("before_single_observed_state_load")
        state = load_observation(plan["observation_ref"], plan["focal_registry_ref"], plan["config"])
        # The independent component pass covers the actual original loader domain.
        require(all(isinstance(state[key], np.ndarray) and state[key].dtype == np.int8
            and (state[key].size == 0 or all(stride > 0 for stride in state[key].strides))
            for key in ("genotype", "allele_count")),
            "Reviewed positive-stride NumPy int8 original genotype/allele axes only")
        # This explicit handoff is solely to this fresh zero-worker provider.
        # Its original __getitem__ sees the exact current PID and does not load
        # another state. Original seal/read/hash/coverage logic remains intact.
        provider.dataset._state = state
        provider.dataset._pid = os.getpid()
        legacy_seal = provider.seal(resource_check=runtime_guard)
        # The original inherited coverage/hash method genuinely produced this
        # fresh record through the newly bound encoder. Never relabel old seals.
        legacy_seal = dict(legacy_seal,
            legacy_block_seal_producer_unchanged=False,
            legacy_block_seal_producer_route="level_int64_counts_original_ordered_summary_v1",
            legacy_block_seal_producer_source_refs=[dict(r) for r in checked_sources
                if r["path"] in LEGACY_PRODUCER_SOURCE_PATHS],
            historical_full_parent_byte_equivalence_claimed=False)
        require(provider.dataset._state is None and provider.dataset._pid is None,
            "Original legacy seal performed its genuine normal release")
        runtime_guard("before_fast_global_observation_preparation")
        bundle, metadata = prepare_from_loaded_observation_v1(state, split="train",
            observation_ref=plan["observation_ref"], focal_registry_ref=plan["focal_registry_ref"],
            native_preparation_ref=plan["config"]["native_preparation_ref"],
            max_nodes=cfg["max_native_nodes"], max_holds=cfg["max_native_holds"],
            max_episode_payload_bytes=cfg["max_episode_payload_bytes"])
        require(bundle.native_holds == plan["trees"] and bundle.original_sites == plan["sites"]
            and bundle.focal_identity_sha256 == legacy_seal["focal_identity_sha256"],
            "Same complete original trees/sites/native focals across legacy and fast global preparation")
        runtime_guard("before_source_bound_fast_global_seal")
        require(_check_fast_sources_v1(fast_source_refs) == checked_sources,
            "Exact bound fast preparation sources remain unchanged")
        identity = fast_global_identity_v1(bundle)
        global_seal = dict(
            schema="TreeLoom_fast_global_observed_preparation_seal_v1",
            parent_id=plan["parent_id"], observation_ref=plan["observation_ref"],
            focal_registry_ref=plan["focal_registry_ref"],
            native_preparation_ref=plan["config"]["native_preparation_ref"],
            identity=identity, preparation_metadata=metadata,
            fast_preparation_source_refs=checked_sources,
            enclosing_scientific_source_closure_binding_required=True,
            legacy_block_seal_sha256=json_sha(legacy_seal),
            legacy_block_seal_producer_unchanged=False,
            legacy_block_seal_producer_source_refs=legacy_seal["legacy_block_seal_producer_source_refs"],
            global_FP32_preparation_implementation_unchanged=True,
            historical_global_byte_identity_claimed=False,
            private_truth_opened=False, model_executed=False)
        require(_check_fast_sources_v1(fast_source_refs) == checked_sources,
            "Bound source identity retained through actual global hashing")
        runtime_guard("after_source_bound_fast_global_seal")
        return legacy_seal, bundle, global_seal
    finally:
        # The original provider remains available for its plan/seal metadata,
        # target joins and normal later releases, but retains no observed state.
        provider.release()
        state = None
