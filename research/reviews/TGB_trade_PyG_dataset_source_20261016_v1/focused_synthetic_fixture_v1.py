"""ROOT-only synthetic data/schedule fixture; default imports no NumPy/Torch."""
from pathlib import Path
import argparse,hashlib,json,os,sys

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=Path(__file__).resolve().parent

def execute_fixture(output):
    # No worker, encoder instance/forward, fitting, GPU API or real data reader.
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    import numpy as np
    import torch
    from torch_geometric.data import Batch
    sys.path.insert(0,str(ROOT))
    from research.reviews.TGB_trade_PyG_dataset_source_20261016_v1 import trade_tree_core_v1 as core
    from research.reviews.TGB_trade_PyG_dataset_source_20261016_v1.trade_tree_pyg_v1 import (
        _TradeTreeDatasetCoreV1,graph_from_snapshot_v1,aligned_country_cuts_v1,make_trade_tree_loader_v1)
    from research.reviews.TGB_trade_PyG_dataset_source_20261016_v1.whole_country_tree_encoder_v1 import make_full_tree_schedule_v1
    checks=[]
    def ck(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    def reject(name,fn):
        try:fn()
        except ValueError:checks.append(name);return
        raise AssertionError(name)
    def complete(tree,n):
        nodes=len(tree.parent);child=[[] for unused in range(nodes)]
        for i,p in enumerate(tree.parent):
            if p>=0:child[p].append(i)
        ck("all_and_only_catalogue_leaves_retained",set(i for i,c in enumerate(child) if not c)==set(range(n)))
        ck("one_connected_complete_tree_no_unary_artifact",sum(tree.parent<0)==1 and n+1<=nodes<=2*n-1 and
            all(len(c)>=2 for c in child[n:]) and tree.leaf_count[tree.root]==n)
        ck("all_edges_retained_and_acyclic",sum(len(c) for c in child)==nodes-1 and
            all(p<0 or p>i for i,p in enumerate(tree.parent)))
        ck("equal_height_internal_chains_absent",all(p<0 or i<n or tree.merge_height[p]>tree.merge_height[i]
            for i,p in enumerate(tree.parent)))
        ck("upward_downward_dependency_levels",all(p<0 or
            (tree.upward_level[p]>tree.upward_level[i] and tree.downward_level[i]==tree.downward_level[p]+1)
            for i,p in enumerate(tree.parent)))
    profile=np.array([[1,0,0,0],[.75,.25,0,0],[.5,.5,0,0],[0,0,0,1]],dtype=np.float32)
    tree=core.complete_single_linkage_v1(profile,np.zeros(4,dtype=bool));complete(tree,4)
    merge=np.flatnonzero((tree.merge_height==.25)&(tree.child_count==3))
    ck("equal_height_chain_collapses_to_true_three_way_polytomy",len(merge)==1 and all(tree.parent[:3]==merge[0]))
    ck("fixed_cut_exactly_matches_simultaneous_component",tree.cut_node_index[0]==tree.cut_node_index[1]==tree.cut_node_index[2]
        and tree.cut_node_index[3]!=tree.cut_node_index[0])
    identical=np.repeat(profile[:1],4,axis=0)
    zero=core.complete_single_linkage_v1(identical,np.zeros(4,dtype=bool));complete(zero,4)
    ck("zero_distance_ties_form_single_polytomy_without_artificial_height_one_root",
        len(zero.parent)==5 and zero.child_count[zero.root]==4 and zero.merge_height[zero.root]==0.)
    all_missing=core.complete_single_linkage_v1(np.zeros((255,255),dtype=np.float32),np.ones(255,dtype=bool))
    complete(all_missing,255)
    ck("all_255_synthetic_countries_survive_in_true_maximal_polytomy",len(all_missing.parent)==256
        and all_missing.child_count[all_missing.root]==255 and all_missing.merge_height[all_missing.root]==1.)
    ck("all_255_missing_histories_remain_distinct_below_cut",len(set(all_missing.cut_node_index.tolist()))==255)
    def clades(t,stable_ids):
        members=[{int(i)} for i in stable_ids]
        out=[]
        for node in range(len(stable_ids),len(t.parent)):
            group=set().union(*(members[i] for i in range(node) if t.parent[i]==node))
            members.append(group);out.append((tuple(sorted(group)),float(t.merge_height[node])))
        return sorted(out)
    permutation=np.array([2,0,3,1])
    permuted=core.complete_single_linkage_v1(profile[permutation][:,permutation],np.zeros(4,dtype=bool))
    ck("hierarchy_equivariant_to_joint_country_destination_permutation",
        clades(tree,np.arange(4))==clades(permuted,permutation))
    cat=core.CountryCatalogueV1.synthetic_for_fixture(tuple("C%d"%i for i in range(5)))
    rows=[(0,0,2.),(0,1,2.),(1,4,3.)]
    first=core.observed_year_from_edges_v1(1986,rows,cat)
    reordered=core.observed_year_from_edges_v1(1986,list(reversed(rows)),cat)
    ck("complete_year_row_permutation_cannot_change_profiles",
        np.array_equal(first.profile,reordered.profile) and np.array_equal(first.present,reordered.present))
    reject("duplicate_directed_year_pair_rejected",lambda:core.observed_year_from_edges_v1(1986,rows+rows[:1],cat))
    reject("nonofficial_catalogue_cannot_pass_official_factory",lambda:core.CountryCatalogueV1.from_official_mapping(dict(zip(cat.names,range(5)))))
    reject("absent_sources_cannot_hide_profile_mass",lambda:core.observed_year_v1(1986,np.eye(5),np.zeros(5,dtype=bool)))
    observation={
        1986:core.observed_year_from_edges_v1(1986,[(0,0,1.)],cat),
        1987:core.observed_year_from_edges_v1(1987,[(1,1,1.)],cat),
        1988:core.observed_year_from_edges_v1(1988,[(2,2,0.)],cat)}
    calls=[]
    def load(year):
        calls.append(year)
        if year>=1989:raise AssertionError("Current/future edge year was accessed")
        return observation[year]
    snapshot=core.snapshot_before_v1(cat,1989,tuple(range(1986,1991)),load)
    ck("causal_loader_reads_only_strict_complete_prefix",calls==[1986,1987,1988] and snapshot.released_years==(1986,1987,1988))
    ck("most_recent_observed_profile_is_retained",np.array_equal(snapshot.profile[0],observation[1986].profile[0]))
    ck("last_observed_year_and_staleness_are_past_only",snapshot.last_observed_year.tolist()==[1986,1987,1988,-1,-1]
        and snapshot.staleness_years.tolist()==[2,1,0,0,0])
    ck("never_observed_and_observed_zero_are_distinct",snapshot.missing.tolist()==[False,False,False,True,True]
        and snapshot.zero_mass.tolist()==[False,False,True,False,False])
    missing_tree=core.complete_single_linkage_v1(snapshot.profile,snapshot.missing);complete(missing_tree,5)
    ck("missing_countries_have_distance_one_to_every_other_country",
        all(missing_tree.distance[i,j]==1. for i in (3,4) for j in range(5) if i!=j))
    ck("every_missing_history_is_singleton_below_fixed_cut",
        all(np.count_nonzero(missing_tree.cut_node_index==missing_tree.cut_node_index[i])==1 for i in (3,4)))
    calls.clear()
    dataset=_TradeTreeDatasetCoreV1(cat,load,available_edge_years=tuple(range(1986,1991)),forecast_years=(1987,1988,1989))
    ck("dataset_constructor_does_not_read_observations",not calls)
    graphs=[dataset[i] for i in range(3)]
    ck("lazy_observed_year_memo_loads_each_released_year_once",calls==[1986,1987,1988])
    ck("next_year_target_fields_absent_from_graph",all(not any(k in graph for k in ("y","target","target_profile","query_mask","labels")) for graph in graphs))
    ck("all_countries_predicted_without_target_source_selection",all(torch.equal(g.country_witness_index,torch.arange(5)) for g in graphs))
    old=dataset[0].leaf_profile.clone();graphs[0].leaf_profile.fill_(123.)
    ck("returned_graph_mutation_cannot_poison_dataset_cache",torch.equal(dataset[0].leaf_profile,old))
    graphs[0]=dataset[0]
    aligned=aligned_country_cuts_v1(*graphs)
    ck("H051_cut_vectors_align_by_country_witness_not_internal_ID",len(aligned)==3 and all(a.shape==(5,) for a in aligned))
    batch=Batch.from_data_list([graphs[0],graphs[2]]);offset=graphs[0].num_nodes
    ck("PyG_Batch_offsets_leaf_and_root_node_pointers",
        torch.equal(batch.leaf_node_index,torch.cat((graphs[0].leaf_node_index,graphs[2].leaf_node_index+offset))) and
        torch.equal(batch.root_node_index,torch.cat((graphs[0].root_node_index,graphs[2].root_node_index+offset))))
    ck("PyG_Batch_offsets_edges_and_cut_node_pointers",
        torch.equal(batch.edge_index,torch.cat((graphs[0].edge_index,graphs[2].edge_index+offset),dim=1)) and
        torch.equal(batch.cut_node_index,torch.cat((graphs[0].cut_node_index,graphs[2].cut_node_index+offset))))
    ck("PyG_Batch_preserves_country_witness_IDs_without_node_offsets",
        torch.equal(batch.country_witness_index,torch.arange(5).repeat(2)))
    ck("batched_complete_prefix_metadata_has_fixed_shape",batch.released_year_mask.shape==(2,31)
        and batch.released_year_mask.sum(1).tolist()==[1,3])
    ck("batched_edges_never_cross_graphs",torch.equal(batch.batch[batch.edge_index[0]],batch.batch[batch.edge_index[1]]))
    schedule=make_full_tree_schedule_v1(batch)
    ck("ragged_batch_schedules_visit_every_node_once_each_direction",
        torch.equal(torch.cat(schedule.up_nodes).sort().values,torch.arange(batch.num_nodes)) and
        torch.equal(torch.cat(schedule.down_nodes).sort().values,torch.arange(batch.num_nodes)))
    ck("ragged_upward_schedule_contains_every_tree_edge_once",sum(c.numel() for c in schedule.up_children)==batch.edge_index.shape[1])
    state=torch.zeros(batch.num_nodes,dtype=torch.long)
    for nodes,children,rank in zip(schedule.up_nodes,schedule.up_children,schedule.up_parent_rank):
        value=torch.zeros(nodes.numel(),dtype=torch.long);value.index_add_(0,rank,state[children])
        value+=batch.node_is_leaf[nodes].to(torch.long);state[nodes]=value
    ck("model_free_upward_sum_recovers_all_descendant_leaf_counts",torch.equal(state,batch.node_leaf_count))
    depth=torch.zeros(batch.num_nodes,dtype=torch.long)
    for level,(nodes,parents) in enumerate(zip(schedule.down_nodes,schedule.down_parents)):
        if level:depth[nodes]=depth[parents]+1
    ck("model_free_downward_schedule_recovers_every_node_depth",torch.equal(depth,batch.downward_level))
    loader=make_trade_tree_loader_v1(dataset,batch_size=2,num_workers=0)
    ck("ordered_loader_retains_all_forecast_periods_without_workers",
        [int(y) for batch_item in loader for y in batch_item.forecast_year]==[1987,1988,1989])
    ck("final_forecast_period_is_not_dropped",dataset.forecast_years[-1]==1989)
    receipt=dict(schema="TGB_trade_whole_country_tree_synthetic_fixture_v1",status="PASS",
        checks=checks,check_count=len(checks),distinct_checks=sorted(set(checks)),
        numpy_version=np.__version__,torch_version=torch.__version__,synthetic_country_count=5,
        source_refs=[dict(path=str(HERE/name),sha256=hashlib.sha256((HERE/name).read_bytes()).hexdigest())
            for name in ("trade_tree_core_v1.py","trade_tree_pyg_v1.py","whole_country_tree_encoder_v1.py",Path(__file__).name)],
        real_dataset_or_network_read=False,encoder_model_constructed_or_forwarded=False,
        fitting_or_predictive_metric=False,worker_process_created=False,GPU_API_call=False,guard_or_admission=False)
    with Path(output).open("x") as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write("\n")
    print(json.dumps(dict(status="PASS",check_count=len(checks),distinct_checks=len(set(checks)))))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute-fixture-root",action="store_true");p.add_argument("--output")
    args=p.parse_args()
    if not args.execute_fixture_root:
        print(json.dumps(dict(status="source_only_synthetic_fixture_root_deferred",Torch_imported=False)));return 0
    if not args.output:raise ValueError("New explicit root fixture receipt path required")
    execute_fixture(args.output);return 0

if __name__=="__main__":raise SystemExit(main())
