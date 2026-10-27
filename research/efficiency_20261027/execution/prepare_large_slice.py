"""Authenticated observed TRAIN prefix preparation for larger-block diagnostics.

The selected prefix is the complete history domain of this diagnostic only.
No targets are opened until observations, global slice episodes and cut plan
have been prepared. Global site indexing starts at zero, preserving chunks.
"""
from dataclasses import dataclass
import gc
import itertools
import json
from types import SimpleNamespace
import time
import numpy as np
import torch
from research.efficiency_20261027.root.compare_host_parallelism import (
    ROOT, JOB, GIB, need, ref, DiagnosticSliceBundle)


@dataclass
class LargeDiagnosticSliceBundle(DiagnosticSliceBundle):
    def validate(self):
        from research.benchmark_rebuild_20261024.american.treeloom_speed import static_model_v2 as frozen
        from research.reviews.order2_cut_memory_implementation_20261015_v1 import order2_cut_memory_v1 as core
        need(self.native_holds in (1024,1153)
             and self.bank.sample_index.shape == (self.native_holds,1600),
             "Exact declared larger diagnostic prefix and all haplotypes")
        need(self.bank_stamps == core._stamps(frozen._bank_tensors(self.bank)),
             "Immutable complete diagnostic bank")
        need(self.episode_stamps == core._stamps(frozen._episode_tensors(self.episodes)),
             "Immutable global diagnostic episodes")
        need(self.focal_stamps == core._stamps((self.focal_hold,self.focal_sample_column)),
             "Immutable original selected focal coordinates")
        self.cut.validate(self.bank)
        return self.bank


def focal_prefix_selection(positions, columns, intervals):
    need(intervals.ndim == 2 and intervals.shape[1] == 2
         and intervals[0,0] == 0 and np.array_equal(intervals[:-1,1],intervals[1:,0]),
         "Contiguous original zero-origin prefix required for unchanged global site chunks")
    selected=(positions>=intervals[0,0]) & (positions<intervals[-1,1])
    need(bool(selected.any()),"Actual original TRAIN targets required")
    hold=np.searchsorted(intervals[:,1],positions[selected],side="right").astype(np.int64)
    return selected,hold,np.asarray(columns[selected],dtype=np.int64)


def prepare_large_prefix(args,rt,saved,check):
    from arggnn.benchmarking import contracts as contracts
    from arggnn.american_c3_gnn_dataset_v1 import load_observation
    from research.benchmark_rebuild_20261024.american.treeloom_speed import runtime_v2 as fresh
    from research.preprocessing_optimization_20261026.tree import process_observations as observations
    from research.repairs.benchmark_recovery_20261020.runtime.episodes import build_episodes
    from research.repairs.benchmark_recovery_20261020.runtime.cuts import prepare_cut_plan_v1
    from research.reviews.order2_cut_memory_implementation_20261015_v1 import order2_cut_memory_v1 as core
    need(not torch.cuda.is_initialized(),"Observation preparation must precede CUDA initialization")
    need(args.slice_holds in (1024,1153),"Explicit original 1024/1153-hold prefix only")
    original=json.loads((ROOT/JOB).read_text())["call"]["spec"]["arguments"]
    parent="argformer_american_train_000"
    item=original["train_inputs"][parent]
    need(rt.registry.split_for(parent) == "train","Only registered TRAIN parent")
    barrier=contracts.read_ref(saved["observation_barrier_ref"])
    seal=barrier["parents"][parent]
    plan=dict(contracts.read_ref(seal["plan_ref"]),plan_ref=seal["plan_ref"])
    for key in fresh.INPUT_REFS:
        need(item[key] == seal[key],"Exact checkpoint observation barrier binding")
    started=time.perf_counter()
    state=load_observation(item["observation_ref"],item["focal_registry_ref"],dict(
        native_preparation_ref=item["native_preparation_ref"],class_order=list(fresh.CLASSES),
        expected_haplotypes=1600,reference_counts=[200,200,200],
        input_reference_class_mask=None,input_reference_sample_indices=None))
    check("large_prefix_authenticated_observation_loaded")
    ts=state["ts"]
    workspace=observations.native_workspace_v1(ts)
    node_time=np.asarray(ts.nodes_time,dtype=np.float64)
    trees=[];intervals=[]
    for tree in itertools.islice(ts.trees(),args.slice_holds):
        trees.append(observations.native_tree(tree,state["nodes"],node_time=node_time,workspace=workspace))
        intervals.append(tuple(tree.interval))
    intervals=np.asarray(intervals,dtype=np.float64)
    need(len(trees) == args.slice_holds and intervals[0,0] == 0,
         "Every declared native prefix hold retained")
    counts=[len(tree.parent) for tree in trees]
    lo,hi=map(int,np.searchsorted(state["positions"],[intervals[0,0],intervals[-1,1]]))
    need(lo == 0,"Global site rows start at zero; original 256-site chunk origin retained")
    preparation_events=[]
    bank,observation_metadata=observations.encode_block(trees,intervals,
        np.asarray(state["genotype"][lo:hi]),state["allele_count"][lo:hi],
        state["positions"][lo:hi],state["labels"],state["ages"],
        num_classes=3,age_split=350.,query_columns=state["query_columns"],
        resource_check=check,preparation_workers=4,max_parallel_bytes=8*GIB,
        preparation_observer=lambda event,detail:preparation_events.append(dict(event=event,detail=detail)))
    selected,hold,column=focal_prefix_selection(state["focals"]["positions_bp0"],
        state["focals"]["sample_indices"],intervals)
    hold=torch.from_numpy(hold);column=torch.from_numpy(column)
    isolated=tuple(observation_metadata.get("isolated_zero_site_holds",()))
    del trees,state,ts,workspace,node_time
    gc.collect()
    episodes,episode_metadata=build_episodes(bank,max_payload_bytes=2*GIB,
        isolated_hold_indices=isolated)
    cut,cut_metadata=prepare_cut_plan_v1(bank,expected_haplotypes=1600,max_nodes=4000000,
        max_holds=1153,isolated_hold_indices=isolated)
    bundle=LargeDiagnosticSliceBundle(bank,episodes,cut,parent,"train",
        int(intervals[-1,1]),args.slice_holds,hi,hold,column,seal["focal_identity_sha256"],
        item["observation_ref"],item["focal_registry_ref"],item["native_preparation_ref"],
        core._stamps(rt.frozen._bank_tensors(bank)),core._stamps(rt.frozen._episode_tensors(episodes)),
        core._stamps((hold,column)))
    bundle.validate()
    metadata=dict(parent_id=parent,declared_split="train",parent_registry_ref=rt.registry.reference,
        episode_metadata=dict(episode_metadata,native_nodes_per_tree=counts),
        observation_metadata=observation_metadata,cut_metadata=cut_metadata,
        layout_cache_scope=dict(kind="diagnostic_larger_prefix",parent_id=parent,
            holds=[0,args.slice_holds],source_site_range=[lo,hi]),
        observation_ref=item["observation_ref"],focal_registry_ref=item["focal_registry_ref"],
        native_preparation_ref=item["native_preparation_ref"],
        diagnostic_slice=True,private_truth_opened=False)
    frozen=SimpleNamespace(AmericanTreeLoomC3FullV1=rt.frozen.AmericanTreeLoomC3FullV1,
        AmericanTreeLoomWholeParentV1=LargeDiagnosticSliceBundle,
        _bank_tensors=rt.frozen._bank_tensors,_episode_tensors=rt.frozen._episode_tensors)
    check("large_prefix_prepared_before_truth")
    preparation_seconds=time.perf_counter()-started
    print(json.dumps(dict(event="large_prefix_prepared_before_truth",holds=args.slice_holds,
        nodes=sum(counts),sites=hi,seconds=preparation_seconds)),flush=True)
    joined_at=time.perf_counter()
    join=fresh.FreshTrainTargetJoin(rt.registry,lambda p:original["train_inputs"][p]["truth_ref"],
        train_inputs=original["train_inputs"],max_truth_uncompressed_bytes=original["max_truth_uncompressed_bytes"])
    target=join.join(parent,split="train",plan=plan,observation_seal=seal)
    labels=target.labels_cpu[torch.from_numpy(selected)].clone()
    need(len(labels) == len(hold) and len(labels)>0,"Every selected actual TRAIN focal retained once")
    scope=dict(mode="larger_observed_prefix_diagnostic",parent_id=parent,
        native_holds=args.slice_holds,native_nodes=sum(counts),haplotypes=1600,sites=hi,
        global_site_start=0,supervised_original_focals=len(labels),full_original_focals=10000,
        preparation_seconds=preparation_seconds,true_TRAIN_label_join_seconds=time.perf_counter()-joined_at,
        full_parent_claim=False,heldout_truth_opened=False,
        observation_ref=item["observation_ref"],focal_registry_ref=item["focal_registry_ref"],
        truth_ref=item["truth_ref"],focal_identity_sha256=seal["focal_identity_sha256"],
        physical_support_bp=[0,int(intervals[-1,1])],
        history_window_boundary="Complete bidirectional restricted prefix; original history outside prefix absent.",
        block_partition_is_not_history_boundary=True,
        preparation_workers=4,preparation_child_and_IPC_reserve_bytes=observations.parallel_admission(4,64*2**20)["additional_bytes"],
        preparation_events=preparation_events)
    return bundle,metadata,labels,frozen,scope

