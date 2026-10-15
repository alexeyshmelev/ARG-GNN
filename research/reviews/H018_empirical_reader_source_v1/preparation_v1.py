"""Production H018 H40/R16/Q24 array preparation; pure CPU, no file/model I/O."""
from dataclasses import fields
import math
import numpy as np
from arggnn import mutation_routed_tree_observations_v1 as frozen
from research.reviews.H018_variable_panel_adapter_source_v1.reader_contract_proposal import (
    panel_metadata_v1,verify_native_tip_dates_v1,need)

FEATURE_SEMANTICS=dict(
    haplotypes=40,references=16,queries=24,classes=["ANA","CHG","WHG","EHG"],
    site_tuple_width=19,leaf_width=6,node_width=3,edge_width=4,support_width=5,gate_width=9,
    age_normalization="log1p(generations)/log(10001)",
    genomic_normalization="log1p(bp)/log(2000001)",
    donor_age_split_generations=350,pseudocount=.5,
    donor_axes="youngerANA/CHG/WHG/EHG then olderANA/CHG/WHG/EHG",
    absent_donor_policy="masked_no_synthetic_donors",categorical_allele_embedding=False,
    native_ancestral_polarization=False,query_input_labels=-1,
    allowed_control=["aligned"],sham_field="identical_unused_structural_placeholder")

def native_hold_readout_v1(intervals):
    a=np.asarray(intervals,dtype=np.float64)
    need(a.ndim==2 and a.shape[1]==2 and len(a)>0 and np.isfinite(a).all()
         and a[0,0]>=0 and np.all(a[:,1]>a[:,0])
         and np.array_equal(a[:-1,1],a[1:,0]),"All contiguous genuine intervals")
    centers=a[:,0]+.5*(a[:,1]-a[:,0])
    need(np.all((centers>=a[:,0])&(centers<a[:,1])),"Finite native-hold midpoints")
    return centers

def prepare_arrays_v1(trees,intervals,positions,genotype,allele_count,roster):
    """Complete H40 production contract; no fixture bypass and no site filtering."""
    meta=panel_metadata_v1(roster)
    need(all(row.get("query_ancestry_truth") is None for row in roster),"No query truth input")
    g,k=frozen._validate_alleles(genotype,allele_count)
    x=np.asarray(positions,dtype=np.float64)
    a=np.asarray(intervals,dtype=np.float64)
    readout=native_hold_readout_v1(a)
    h=40;sid=np.arange(h,dtype=np.int64)
    ref=np.asarray(meta["reference_class"],dtype=np.int64)
    age=np.asarray(meta["source_sample_age_generations"],dtype=np.float64)
    qc=np.arange(16,40,dtype=np.int64)
    need(len(trees)==len(a) and g.shape[1]==40 and len(g)>0 and x.shape==(len(g),)
         and np.isfinite(x).all() and np.all(np.diff(x)>0)
         and np.all(x>=a[0,0]) and np.all(x<a[-1,1]),
         "Every original observed site inside genuine coverage; zero drops")
    site_hold=np.searchsorted(a[:,1],x,side="right").astype(np.int64)
    readout_hold=np.searchsorted(a[:,1],readout,side="right").astype(np.int64)
    need(np.array_equal(readout_hold,np.arange(len(trees))),"One output per genuine native hold")
    parents=[];times=[];samples=[];support=[];heights=[];depths=[];holds=[];counts=[];durations=[]
    offset=0;unary=0;rounded_ties=0
    for hold,tree in enumerate(trees):
        p,t,sn,tsid,_,depth,height,descendant,duration=frozen._layout(tree)
        need(set(tsid.tolist())==set(sid.tolist()),"Every hold retains exact original40 identities")
        at=dict(zip(tsid.tolist(),sn.tolist()))
        row=np.asarray([at[int(i)] for i in sid],dtype=np.int64)
        verify_native_tip_dates_v1(age.tolist(),t[row].tolist())
        children=np.bincount(p[p>=0],minlength=len(p))
        need(set(np.flatnonzero(children==0).tolist())==set(row.tolist()),
             "Every leaf is an original sampled haplotype")
        unary+=int(np.count_nonzero(children==1))
        nonroot=p>=0;clock32=t.astype(np.float32)
        rounded_ties+=int(np.count_nonzero(clock32[p[nonroot]]==clock32[nonroot]))
        pp=p.copy();pp[nonroot]+=offset
        parents.append(pp);times.append(t.copy());samples.append(row+offset)
        lo,hi=np.searchsorted(site_hold,[hold,hold+1])
        support.append(frozen.branch_concordance_support_v1(tree,g[lo:hi],k[lo:hi],sid))
        heights.append(height);depths.append(depth);holds.append(np.full(len(p),hold,dtype=np.int64))
        counts.append(descendant);durations.append(duration);offset+=len(p)
    parent=np.concatenate(parents);node_time=np.concatenate(times)
    height=np.concatenate(heights);depth=np.concatenate(depths)
    node_hold=np.concatenate(holds);descendant=np.concatenate(counts);duration=np.concatenate(durations)
    true=np.concatenate(support);spans=a[:,1]-a[:,0]
    geometry=np.zeros((len(parent),4),dtype=np.float32);child=np.flatnonzero(parent>=0)
    geometry[child,:3]=np.stack((np.log1p(node_time[child]),
        np.log1p(node_time[parent[child]]),np.log1p(duration[child])),axis=1)/math.log(10001.)
    geometry[child,3]=descendant[child]/h
    levels=frozen._level_arrays(height);rank=np.empty(len(parent),dtype=np.int64)
    for nodes in levels:rank[nodes]=np.arange(len(nodes))
    groups=[[] for _ in levels]
    for node in child:groups[int(height[parent[node]])].append(int(node))
    upward_children=tuple(np.asarray(nodes,dtype=np.int64) for nodes in groups)
    upward_parent_rank=tuple(rank[parent[nodes]] for nodes in upward_children)
    downward=frozen._level_arrays(depth)[1:]
    logfreq,mask,panel,entropy,donor_count=frozen.observed_allele_frequency_arrays_v1(g,k,ref,age)
    metadata=dict(schema="H018_empirical_aligned_observations_v1",input_only=True,
        feature_semantics=FEATURE_SEMANTICS,targets_read=False,sites_dropped=0,
        original_sites=len(g),constant_sites=int(sum(len(np.unique(row[row>=0]))==1 for row in g)),
        native_holds=len(trees),native_vertex_occurrences=len(parent),native_nonroot_branches=len(child),
        raw_node_times_preserved_before_FP32_cast=True,source_sample_age_generations=age.tolist(),
        native_tip_clock_generations=times[0][np.asarray(samples[0])].tolist(),
        raw_strict_chronology_verified=True,rounded_FP32_parent_child_ties=rounded_ties,
        unary_vertices_preserved=unary,source_ages_and_native_clocks_kept_separate=True,
        sample_age_native_clock_exact_FP32_equality=True,
        source_allele_polarization="original_VCF_REF_ALT_codes",
        branch_support="unchanged_original_SNP_vs_native_descendant_concordance_including_discrepancy",
        sham_field_usable_as_control=False,sham_bins_fitted=False,allowed_controls=["aligned"],
        donor_group_counts=donor_count.tolist(),intervals_regional_bp0=a.tolist(),
        readout_policy="one_midpoint_per_genuine_native_hold_no_scoring_bins",
        readout_positions_regional_bp0=readout.tolist(),query_columns=qc.tolist(),
        no_span_resets=True,full_10Mb_coverage_claim=False,no_accuracy_claim=True)
    return frozen.MutationRoutedTreeObservationsV1(parent,node_time,np.stack(samples),levels,
        upward_children,upward_parent_rank,downward,geometry,true,true.copy(),g.astype(np.int16,copy=True),
        logfreq,mask,panel,entropy,site_hold,spans,readout_hold,qc,ref.copy(),age.copy(),metadata)

def to_torch_bank_v1(observations):
    """Construct and validate the unchanged tensor dataclass only; no learned model."""
    from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1,validate_inputs_v1
    bank=MutationRoutedWholeTreeInputsV1.from_arrays(observations.model_arrays(),device="cpu")
    report=validate_inputs_v1(bank,expected_haplotypes=40)
    need(report["haplotypes"]==40 and report["queries"]==24
         and tuple(bank.query_column.tolist())==tuple(range(16,40)),"Exact H018 tensor interface")
    return bank,report

def tensor_array_bytes_v1(observations):
    return sum(int(x.nbytes) for value in observations.model_arrays().values()
        for x in (value if isinstance(value,tuple) else (value,)))

def preparation_plan_v1(trees,site_count):
    """Conservative payload/scratch formula, not an observed RSS claim."""
    need(type(site_count) is int and site_count>0 and len(trees)>0,"Complete observation dimensions")
    n=sum(len(tree.parent) for tree in trees)
    tensor_bound=256*n+16*site_count*40+65536
    flag_scratch=2058*n+128*40
    return dict(native_vertex_occurrences=n,native_holds=len(trees),sites=site_count,
        complete_tensor_payload_upper_bound_bytes=tensor_bound,
        flag_preparation_allowance_bytes=flag_scratch,
        retained_payload_plus_flag_preparation_upper_bound_bytes=tensor_bound+flag_scratch,
        estimate_is_measured_RSS=False)

def aligned_inputs_v1(result,*,control="aligned"):
    """Future model caller must use this guard for this aligned-only bank."""
    need(control=="aligned" and result["metadata"]["feature_semantics"]["allowed_control"]==["aligned"],
         "Empirical placeholder sham field is never a permitted model control")
    return result["bank"],result["flags"]
