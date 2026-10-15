"""Tiny synthetic CPU reader/preparer checks; no empirical bank/model/checkpoint execution."""
from pathlib import Path
import ast,copy,gzip,hashlib,json,sys
import numpy as np
import tskit,torch
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from research.reviews.H018_empirical_reader_source_v1 import reader_v1 as r,preparation_v1 as p
from arggnn import mutation_routed_tree_observations_v1 as old
from arggnn.anonymous_clade_flag_control_v1 import build_clade_flag_arrays_v1
D=ROOT/"research/reviews/H018_empirical_reader_source_v1"
torch.set_num_threads(1)
checks=[]
def check(name,ok):
    if not ok:raise AssertionError(name)
    checks.append(name)
def reject(name,fn):
    try:fn()
    except (ValueError,TypeError,KeyError,IndexError):checks.append(name)
    else:raise AssertionError(name)
ages=[288.25,359.92857142857144,347.14285714285717,473.39285714285717,
      287.5,488.0357142857143,400.25,500.75]+[220.25+i for i in range(12)]
roster=[]
for i in range(40):
    mean=ages[i//2]*28;age=mean/28
    cls=i//4 if i<16 else None
    roster.append(dict(sample_index=i,subject_id="synthetic_"+str(i//2),haplotype=i%2,
        role="reference" if i<16 else "query",observed_reference_class=cls,
        observed_cohort=r.preparation.FEATURE_SEMANTICS["classes"][cls] if i<16 else "Denmark_Neolithic",
        age_generations=age,age_mean_cal_BP=mean,date_zero="calBP1950",years_per_generation=28,
        query_ancestry_truth=None,native_tree_node_id=None))
ids=[["22",20000002,"A","C"],["22",20000010,"G","T"],
     ["22",20000060,"C","G"],["22",20000100,"T","A"]]
g=np.zeros((4,40),dtype=np.int16);g[0,1::2]=1;g[2,:]=1;g[3,::3]=1
names=[roster[i]["subject_id"] for i in range(0,40,2)]
header="#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"+"\t".join(names)+"\n"
def vcfline(row,codes):
    return "\t".join([row[0],str(row[1]),".",row[2],row[3],".","PASS",".","GT"]+
        [str(codes[i])+"|"+str(codes[i+1]) for i in range(0,40,2)])+"\n"
records=[(row[1],vcfline(row,codes)) for row,codes in zip(ids,g)]
records.append((20000030,vcfline(["22",20000030,"A","G"],np.zeros(40,dtype=np.int16))))
text=header+"".join(line for _,line in sorted(records))
raw=gzip.compress(text.encode(),mtime=0)
sites=r.parse_original_sites_v1(raw,roster,ids)
check("frozen_site_join_preserves_every_declared_site_and_order",sites["identities"]==ids)
check("candidate_sites_are_not_re_QCed_or_used_as_extra_SNPs",sites["candidate_records"]==5 and len(sites["genotype"])==4)
check("original_phased_categorical_codes_exact",np.array_equal(sites["genotype"],g))
check("both_constant_reference_and_constant_alternate_sites_retained",
      sites["constant_absolute_POS_bp1"]==[20000010,20000060] and np.array_equal(sites["allele_count"],[2]*4))
check("absolute_bp1_to_regional_bp0_minus1",sites["positions_regional_bp0"].tolist()==[1,9,59,99])
bad=copy.deepcopy(ids);bad[-1][2]="A"
reject("frozen_identity_allele_mismatch_rejected",lambda:r.parse_original_sites_v1(raw,roster,bad))
reject("frozen_missing_site_rejected",lambda:r.parse_original_sites_v1(gzip.compress((header+records[0][1]).encode()),roster,ids))
reject("candidate_read_bound_rejected",lambda:r.parse_original_sites_v1(raw,roster,ids,max_candidates=1))
reject("decompression_bound_rejected",lambda:r.parse_original_sites_v1(raw,roster,ids,max_decompressed_bytes=10))
badraw=gzip.compress(text.replace(names[0],names[1],1).encode())
reject("wrong_subject_order_rejected",lambda:r.parse_original_sites_v1(badraw,roster,ids))
badraw=gzip.compress(text.replace("0|1","0/1",1).encode())
reject("unphased_original_call_rejected",lambda:r.parse_original_sites_v1(badraw,roster,ids))
bad=copy.deepcopy(roster);bad[16]["observed_reference_class"]=0
reject("query_input_label_rejected",lambda:r.parse_original_sites_v1(raw,bad,ids))
haps=[]
for i,(row,codes) in enumerate(zip(ids,g)):
    chrom,pos,ref,alt=row;flip=i%2==0
    aa,da=(alt,ref) if flip else (ref,alt)
    vals=1-codes if flip else codes
    haps.append(" ".join(["22","H018_22_"+str(pos)+"_"+ref+"_"+alt,str(pos-20000000),aa,da]+list(map(str,vals))))
haps_raw=("\n".join(haps)+"\n").encode()
sample_raw=("ID_1 ID_2 missing\n0 0 0\n"+"".join(n+" "+n+" 0\n" for n in names)).encode()
ages_raw="".join(format(row["age_generations"],".17g")+"\n" for row in roster).encode()
readback={k:sites[k] for k in ("original_nucleotide_vectors_sha256","monomorphic_source_POS_bp1_sha256")}
readback["retained_sites"]=4
proof=r.verify_bridge_v1(sites,roster,haps_raw,sample_raw,ages_raw,readback)
check("bridge_opposite_polarization_preserves_original_biological_phase",proof["biological_nucleotide_and_phase_equivalence"])
bad=haps.copy();f=bad[0].split();f[5]=str(1-int(f[5]));bad[0]=" ".join(f)
reject("single_bridge_biological_call_change_rejected",lambda:r.verify_bridge_v1(sites,roster,("\n".join(bad)+"\n").encode(),sample_raw,ages_raw,readback))
reject("changed_fractional_source_age_text_rejected",lambda:r.verify_bridge_v1(sites,roster,haps_raw,sample_raw,ages_raw+b"0\n",readback))
perm=np.roll(np.arange(40),7)
native_ages=np.array([float(format(row["age_generations"],".6f")) for row in roster])
tables=tskit.TableCollection(101.)
tip_times=np.zeros(40);tip_times[perm]=native_ages
for t in tip_times:tables.nodes.add_row(flags=1,time=float(t))
local_parent=np.full(79,-1,dtype=np.int64);active=list(range(40));node=40
while len(active)>1:
    next_level=[]
    for i in range(0,len(active),2):
        if i+1==len(active):next_level.append(active[i]);continue
        local_parent[active[i]]=local_parent[active[i+1]]=node
        next_level.append(node);node+=1
    active=next_level
for hold,(left,right) in enumerate(((0.,50.),(50.,101.))):
    internal=[tables.nodes.add_row(time=2000.+i) for i in range(39)]
    native=list(map(int,perm))+internal
    parent=local_parent.copy()
    if hold:parent[0],parent[2]=parent[2],parent[0]
    for child,par in enumerate(parent):
        if par>=0:tables.edges.add_row(left,right,native[int(par)],native[child])
tables.sort();ts=tables.tree_sequence()
mapping=[dict(native_sample_node_id=int(perm[i]),subject_id=row["subject_id"],haplotype=row["haplotype"],
    source_sample_index=i,role=row["role"],source_age_generations=row["age_generations"],
    anc_age_generations=float(native_ages[i]),native_node_time=float(native_ages[i])) for i,row in enumerate(roster)]
expected=dict(native_nodes=118,native_edges=156,native_trees=2)
class GraphOnly:
    allowed={"nodes_time","nodes_flags","edges_parent","edges_child","edges_left","edges_right",
             "num_nodes","num_edges","num_trees","sequence_length","trees"}
    def __getattr__(self,name):
        if name not in self.allowed:raise AssertionError("Forbidden native surface: "+name)
        return getattr(ts,name)
trees,intervals,graph=r.extract_graph_v1(GraphOnly(),roster,mapping,[2,10,60,100],expected_counts=expected)
check("native_read_uses_only_node_edge_topology_surfaces",len(trees)==2)
check("explicit_native_sample_mapping_preserved_without_sort_guess",
      graph["source_sample_native_node_ids"]==perm.tolist() and perm.tolist()!=list(range(40)))
check("minus1_first_last_intervals_clipped_to_genuine_support",
      intervals.tolist()==[[1.,49.],[49.,100.]] and graph["coordinate_view"]["sites_dropped"]==0)
check("all_native_holds_retained_without_coalescing",graph["raw_interval_count"]==2 and len(trees)==2)
check("raw_native_clock_remains_exact_separate_from_fractional_source",
      graph["date_proof"]["native_times_edited"] is False and native_ages[2]!=roster[2]["age_generations"])
bad=copy.deepcopy(mapping);bad[0]["native_sample_node_id"]=bad[1]["native_sample_node_id"]
reject("nonbijective_or_wrong_native_mapping_rejected",lambda:r.extract_graph_v1(GraphOnly(),roster,bad,[2,10,60,100],expected_counts=expected))
bad=copy.deepcopy(mapping);bad[0]["native_node_time"]+=.000001
reject("native_clock_change_rejected_without_tolerance",lambda:r.extract_graph_v1(GraphOnly(),roster,bad,[2,10,60,100],expected_counts=expected))
prepared=p.prepare_arrays_v1(trees,intervals,sites["positions_regional_bp0"],sites["genotype"],sites["allele_count"],roster)
bank,summary=p.to_torch_bank_v1(prepared)
check("production_H40_Q24_preparer_uses_no_fixture_bypass",summary==dict(nodes=158,trees=2,haplotypes=40,sites=4,queries=24,readouts=2))
check("all_original_site_codes_and_constants_survive_tensor_conversion",
      torch.equal(bank.genotype,torch.from_numpy(g)) and prepared.metadata["constant_sites"]==2)
check("native_hold_midpoints_are_observation_outputs_without_scoring_grid",
      prepared.metadata["readout_positions_regional_bp0"]==[25.,74.5] and bank.readout_hold.tolist()==[0,1])
check("Q24_output_order_exact_original_columns16_to39",bank.query_column.tolist()==list(range(16,40)))
check("all24_query_reference_labels_unknown",torch.equal(bank.reference_class[bank.query_column],torch.full((24,),-1,dtype=torch.long)))
check("original_fractional_source_ages_and_native_clocks_equal_only_after_FP32_cast",
      np.array_equal(prepared.sample_age,np.array([row["age_generations"] for row in roster]))
      and torch.equal(bank.node_time[bank.sample_index[0]],bank.sample_age))
lf,mask,panel,entropy,counts=old.observed_allele_frequency_arrays_v1(g,np.full(4,2,dtype=np.int64),
    prepared.reference_class,prepared.sample_age)
check("donor_frequency_and_common_callability_exact_frozen_helper",
      all(np.array_equal(a,b) for a,b in ((lf,prepared.log_donor_allele_frequency),(mask,prepared.donor_call_mask),
      (panel,prepared.panel_allele_frequency),(entropy,prepared.site_entropy))))
check("absent_donor_channels_masked_without_synthetic_donor",all(not prepared.donor_call_mask[:,i].any() for i,c in enumerate(counts) if c==0))
check("fixed_feature_widths_site19_leaf6_node3_gate9",p.FEATURE_SEMANTICS["site_tuple_width"]==19
      and bank.edge_geometry.shape[1]==4 and bank.branch_support_true.shape[1]==5
      and p.FEATURE_SEMANTICS["leaf_width"]==6 and p.FEATURE_SEMANTICS["node_width"]==3)
check("no_empirical_sham_bins_or_sham_control_invented",
      prepared.metadata["allowed_controls"]==["aligned"] and prepared.metadata["sham_field_usable_as_control"] is False
      and np.array_equal(prepared.branch_support_true,prepared.branch_support_sham))
lo,hi=np.searchsorted(prepared.site_hold,[0,1])
check("branch_support_exact_original_GT_vs_topology_discrepancy_helper",
      np.array_equal(prepared.branch_support_true[:79],old.branch_concordance_support_v1(trees[0],g[lo:hi],np.full(hi-lo,2,dtype=np.int64),np.arange(40))))
reject("outside_support_site_is_rejected_not_dropped",lambda:p.prepare_arrays_v1(trees,intervals,np.array([1,9,59,100]),g,np.full(4,2,dtype=np.int64),roster))
reject("missing_haplotype_column_rejected",lambda:p.prepare_arrays_v1(trees,intervals,np.array([1,9,59,99]),g[:,:39],np.full(4,2,dtype=np.int64),roster))
flags,flagmeta=build_clade_flag_arrays_v1(bank.parent.numpy(),bank.sample_index.numpy())
check("complete_H40_exact_clade_arrays_prepare_without_learned_model",
      isinstance(flags,np.ndarray) and flags.shape==(158,2) and flags.dtype==np.uint8)
cfg=json.loads((D/"config_inactive.json").read_text());r.validate_config_v1(cfg)
reject("inactive_root_context_denied_before_reader_or_data_access",lambda:r.prepare_from_root_context_v1(dict(cfg=cfg,admitted=False)))
check("inactive_config_has_no_model_checkpoint_GPU_or_native_authority",
      all(cfg[k] is False for k in ("preparation_authorized","model_application_authorized","checkpoint_read_authorized","GPU_authorized","native_inference_authorized")))
source_paths=[D/"reader_v1.py",D/"preparation_v1.py"]
for path in source_paths:compile(path.read_text(),str(path),"exec")
attrs={n.attr for path in source_paths for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.Attribute)}
check("sources_never_access_native_variants_mutations_populations_or_ancestral_states",
      not attrs.intersection({"variants","mutations","populations","ancestral_state","genotype_matrix","sites"}))
check("sources_never_load_or_execute_learned_weights",not any(s in path.read_text()
      for path in source_paths for s in ("torch.load(","torch.save(",".forward(",".forward_holds(","optimizer.step(")))
check("small_synthetic_payload_fits_declared_CPU_budgets",p.tensor_array_bytes_v1(prepared)<cfg["CPU_tensor_cap_bytes"]
      and p.tensor_array_bytes_v1(prepared)+2058*158+128*40<cfg["CPU_preparation_cap_bytes"])
planning=p.preparation_plan_v1(trees,4)
check("actual_graph_dimensions_preflight_complete_payload_and_flag_scratch",
      planning["native_vertex_occurrences"]==158
      and planning["complete_tensor_payload_upper_bound_bytes"]>p.tensor_array_bytes_v1(prepared)
      and planning["retained_payload_plus_flag_preparation_upper_bound_bytes"]<cfg["CPU_preparation_cap_bytes"])
reject("aligned_model_input_boundary_rejects_sham_control",lambda:p.aligned_inputs_v1({},control="shuffled"))
manifest=r.bank_array_manifest_v1(bank)
check("complete_CPU_bank_tensor_manifest_binds_all_declared_model_fields",
      set(manifest)==set(prepared.model_arrays()) and manifest["genotype"]==r.array_identity(g)
      and manifest["query_column"]==r.array_identity(np.arange(16,40,dtype=np.int64)))
receipt=dict(schema="H018_empirical_reader_synthetic_CPU_checks_v1",status="passed",check_count=len(checks),checks=checks,
    source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
    empirical_GT_graph_or_checkpoint_payload_read=False,actual_empirical_bank_prepared=False,
    model_or_checkpoint_executed=False,GPU_guard_native_inference_or_admission=False,
    runtime_versions=dict(numpy=np.__version__,torch=torch.__version__,tskit=tskit.__version__),
    synthetic_trees=2,synthetic_haplotypes=40,synthetic_sites=4,synthetic_constants=2,
    actual_interface_tensor_dtype="float32",raw_native_source_clocks="float64",source_only=True)
(D/"synthetic_checks_receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(dict(status="passed",check_count=len(checks))))
