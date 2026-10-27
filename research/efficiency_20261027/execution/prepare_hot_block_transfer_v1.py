"""Authenticated observed-only 1153-hold prefix for isolated transfer measurement."""
import gc
import itertools
import json
import time
import numpy as np
import torch
from research.efficiency_20261027.root.compare_host_parallelism import ROOT, JOB, GIB, need
from .prepare_large_slice_v2 import plain_scope_metadata


def prepare_prefix(saved, check):
    from arggnn.benchmarking import contracts
    from arggnn.american_c3_gnn_dataset_v1 import load_observation
    from research.benchmark_rebuild_20261024.american.treeloom_speed import runtime_v2 as fresh
    from research.preprocessing_optimization_20261026.tree import process_observations as observations
    from research.repairs.benchmark_recovery_20261020.runtime.episodes import build_episodes
    from research.efficiency_20261027.execution.large_layout_plan import LargeLayoutPlan
    from research.efficiency_20261027.execution.large_layout_cache import LargeBlockLayoutCache
    need(not torch.cuda.is_initialized(), "Observed prefix preparation must precede CUDA")
    original = json.loads((ROOT / JOB).read_text())["call"]["spec"]["arguments"]
    parent = "argformer_american_train_000"
    item = original["train_inputs"][parent]
    registry = fresh.FreshParentRegistry.from_ref(saved["parent_registry_ref"])
    need(registry.split_for(parent) == "train", "Registered TRAIN observation required")
    barrier = contracts.read_ref(saved["observation_barrier_ref"])
    seal = barrier["parents"][parent]
    for key in fresh.INPUT_REFS:
        need(item[key] == seal[key], "Exact checkpoint observation barrier binding")
    started = time.perf_counter()
    state = load_observation(item["observation_ref"], item["focal_registry_ref"], dict(
        native_preparation_ref=item["native_preparation_ref"], class_order=list(fresh.CLASSES),
        expected_haplotypes=1600, reference_counts=[200, 200, 200],
        input_reference_class_mask=None, input_reference_sample_indices=None))
    check("transfer_authenticated_observations_loaded")
    ts = state["ts"]
    workspace = observations.native_workspace_v1(ts)
    node_time = np.asarray(ts.nodes_time, dtype=np.float64)
    trees, intervals = [], []
    for tree in itertools.islice(ts.trees(), 1153):
        trees.append(observations.native_tree(tree, state["nodes"], node_time=node_time, workspace=workspace))
        intervals.append(tuple(tree.interval))
    intervals = np.asarray(intervals, dtype=np.float64)
    need(len(trees) == 1153 and intervals[0, 0] == 0
         and np.array_equal(intervals[:-1, 1], intervals[1:, 0]), "Exact contiguous native prefix required")
    counts = [len(tree.parent) for tree in trees]
    lo, hi = map(int, np.searchsorted(state["positions"], [intervals[0, 0], intervals[-1, 1]]))
    need(lo == 0, "Original global site origin required")
    events = []
    check("transfer_preparation_parallel_reserve", additional_bytes=8*GIB)
    bank, metadata = observations.encode_block(trees, intervals,
        np.asarray(state["genotype"][lo:hi]), state["allele_count"][lo:hi], state["positions"][lo:hi],
        state["labels"], state["ages"], num_classes=3, age_split=350., query_columns=state["query_columns"],
        resource_check=check, preparation_workers=4, max_parallel_bytes=8*GIB,
        preparation_observer=lambda event, detail: events.append(dict(event=event, detail=detail)))
    isolated = tuple(metadata.get("isolated_zero_site_holds", ()))
    del trees, state, ts, workspace, node_time
    gc.collect()
    episodes, episode_metadata = build_episodes(bank, max_payload_bytes=2*GIB, isolated_hold_indices=isolated)
    check("transfer_prefix_before_layout")
    retained, working = LargeBlockLayoutCache.estimate(bank, counts, 512)
    check("transfer_layout_working_reserve", additional_bytes=working)
    layout_started = time.perf_counter()
    plan = LargeLayoutPlan.build(bank, counts, episodes.member_hold, holds_per_block=512)
    layout = plan.executor_layout()
    need([(z[0], z[1]) for z in layout] == [(0, 512), (512, 1024), (1024, 1153)], "Exact three-block prefix")
    scope = dict(parent_id=parent, native_holds=1153, native_nodes=sum(counts), haplotypes=1600,
        sites=hi, native_nodes_per_tree=counts, physical_support_bp=[0, int(intervals[-1, 1])],
        observation_ref=item["observation_ref"], focal_registry_ref=item["focal_registry_ref"],
        native_preparation_ref=item["native_preparation_ref"], observation_barrier_ref=saved["observation_barrier_ref"],
        preparation_seconds=time.perf_counter()-started, layout_seconds=time.perf_counter()-layout_started,
        layout_array_bytes=plan.array_bytes, layout_retained_estimate_bytes=retained,
        layout_working_reserve_bytes=working, preparation_workers=4, preparation_parallel_reserve_bytes=8*GIB,
        preparation_events=events, truth_opened=False, learned_model_created=False,
        full_parent_claim=False, history_computed=False,
        episode_metadata=episode_metadata,
        note="Reference labels are observed inputs. TRAIN targets are never joined. Episodes only supply the original layout member boundaries.")
    provenance = json.dumps(dict(parent_id=parent, observation_ref=item["observation_ref"],
        focal_registry_ref=item["focal_registry_ref"], native_preparation_ref=item["native_preparation_ref"],
        holds=[0, 1153], holds_per_block=512), sort_keys=True, separators=(",", ":"))
    del episodes
    check("transfer_prefix_prepared")
    return bank, layout, provenance, plain_scope_metadata(scope)
