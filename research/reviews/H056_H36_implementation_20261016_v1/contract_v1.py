"""H056 pure metadata/constants. Import performs no IO or numerical import."""
from pathlib import Path
import hashlib, json, math
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H056_H36_implementation_20261016_v1"
H,R,Q,PEOPLE=36,16,20,18
BEGIN,END=20_000_000,30_000_000
CHROMOSOMES=("20","21","22")
CLASSES=("ANA","CHG","WHG","EHG")
SUBJECTS=("Bar8","Bon002","KK1","SATP","Loschbour","Bichon","Sidelkino","NEO88","NEO866","NEO945","NEO28","NEO942","NEO790","NEO739","Gjerrild5","NEO735","NEO92","NEO870")
EARLIER=SUBJECTS[8:13]
LATER=SUBJECTS[13:18]
AGES_CALBP=(8071,10078,9720,13255,8050,13665,11258.5,7858,5457,5444,5461,5495,5668,3971,4110,3995,4218,4255)
MIN_SITES=1000
PROTOCOL_REF={"path":"research/reviews/H056_frozen_H045_Danish_application_registration_20261016_v1/protocol.root_frozen_v1.json","sha256":"9544bdbd499bddafeb0774c67e4e86ef25535d41dd2409e710c29b679619cd6e"}
ROSTER_REF={"path":"research/reviews/Danish_genotype_corroboration_source_20261016_v2/candidate_roster.metadata.v1.tsv","sha256":"eefcb18bb4c59d959b26d1e6b8a4a62c4eb0c9bbcfa89a7602797716e6d3cf46"}
CHECKPOINT_REF={"path":"experiments/results/H045_GPU5_episode_train_seed101_v1/foreign_episode.pass_30.pt","sha256":"bfd8a87632015a900d9da6009d4b0ee037958460038d7949d0998e80b770796b"}
CHECKPOINT_DESCRIPTOR_REF={"path":"research/reviews/H018_H045_empirical_application_source_v1/checkpoint_descriptor.json","sha256":"fd8bbd7e61af5d93316e3f4c1d31275d1ee991f6f734991f0091ec069e5a1f12"}
COUNT_BASE=("candidate_sites","sites","constants")
COUNT_GRAPH=("native_trees","native_nodes","native_edges","vertex_occurrences")
NATIVE_PROFILE=dict(effective_haploid_N=30000,mutation_rate="1.25e-8",years_per_generation=28,
    seed=414855340,population_size_iterations_executed=0,pre_size_only=True)
FEATURE_SEMANTICS=dict(haplotypes=H,references=R,queries=Q,classes=list(CLASSES),
    site_tuple_width=19,leaf_width=6,node_width=3,edge_width=4,support_width=5,gate_width=9,
    age_normalization="log1p(generations)/log(10001)",
    genomic_normalization="log1p(bp)/log(2000001)",donor_age_split_generations=350,pseudocount=.5,
    donor_axes="youngerANA/CHG/WHG/EHG then olderANA/CHG/WHG/EHG",
    absent_donor_policy="masked_no_synthetic_donors",categorical_allele_embedding=False,
    native_ancestral_polarization=False,query_input_labels=-1,
    allowed_control=["aligned"],sham_field="identical_unused_structural_placeholder")
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
def validate_chromosome_v1(chromosome):
    require(type(chromosome)is str and chromosome in CHROMOSOMES,"Exact H056 chromosome20/21/22")
    return chromosome
def validate_counts_v1(counts,*,graph=False):
    keys=set(COUNT_BASE+COUNT_GRAPH) if graph else set(COUNT_BASE)
    require(isinstance(counts,dict) and set(counts)==keys and all(type(v)is int for v in counts.values()),
        "Exact actual integer count seal, no historical or predicted dimensions")
    require(counts["candidate_sites"]>=counts["sites"]>=MIN_SITES and 0<=counts["constants"]<=counts["sites"],
        "Original minimum1000 qualified sites and complete candidate/constant counts")
    if graph:
        require(0<counts["native_trees"]<=counts["sites"] and counts["native_nodes"]>=H
            and counts["native_edges"]>0 and counts["vertex_occurrences"]>=H*counts["native_trees"],
            "Complete actual native graph/forest dimensions")
    return dict(counts)
def validate_roster_identity_v1(rows):
    require(isinstance(rows,list) and len(rows)==H,"Exact36 source haplotypes")
    for i,row in enumerate(rows):
        j=i//2;label=j//2 if i<R else None
        require(type(row["sample_index"])is int and row["sample_index"]==i
            and row["subject_id"]==SUBJECTS[j] and type(row["haplotype"])is int and row["haplotype"]==i%2
            and row["role"]==("reference" if i<R else "query")
            and row["observed_reference_class"]==label and (label is None or type(row["observed_reference_class"])is int)
            and row.get("query_ancestry_truth")is None and row.get("native_tree_node_id")is None,
            "Frozen subject/pair/reference/query identity; no truth or inferred node input")
        age=row["age_generations"]
        require(row["date_zero"]=="calBP1950" and row["years_per_generation"]==28
            and type(row["age_mean_cal_BP"])in(int,float) and row["age_mean_cal_BP"]==AGES_CALBP[j]
            and type(age)in(int,float) and math.isfinite(age) and age==AGES_CALBP[j]/28.,
            "Exact frozen source mean calBP1950/28 dates")
        if label is not None:require(row["observed_cohort"]==CLASSES[label],"Frozen observed donor class")
    return rows
def planned_roster_v1():
    """Pure deterministic metadata for root sealing; writes no file/authority."""
    rows=[]
    for i in range(H):
        j=i//2;label=j//2 if i<R else None
        rows.append(dict(sample_index=i,subject_id=SUBJECTS[j],haplotype=i%2,
            role="reference" if i<R else "query",observed_reference_class=label,
            observed_cohort=CLASSES[label] if label is not None else None,
            age_mean_cal_BP=AGES_CALBP[j],age_generations=AGES_CALBP[j]/28.,
            date_zero="calBP1950",years_per_generation=28,query_ancestry_truth=None,native_tree_node_id=None))
    return rows

def variant_id_v1(chromosome,pos,ref,alt):
    validate_chromosome_v1(chromosome)
    require(type(pos)is int and BEGIN<=pos-1<END and ref in ("A","C","G","T") and alt in ("A","C","G","T") and ref!=alt,"Exact source variant identity")
    return "H056_"+chromosome+"_"+str(pos)+"_"+ref+"_"+alt

FROZEN_SOURCE_SHA256={'research/reviews/H018_H045_empirical_application_source_v1/inference_v2.py': 'a8223979ee775d0aae83f576d48d3cbec482eb03022397acf1d684a08d50f227', 'research/reviews/H018_H045_empirical_application_source_v1/variable_panel_model_v1.py': '2c86da54ddd61883fc313f658dc725d9a203048cdf11d87e2edec488044c8ef6', 'research/reviews/H018_empirical_reader_source_v1/preparation_v1.py': '8bb6b9f3b77360d9eae31350337f85dd595dbfcbb0d69f95f9a41d035d96bc3c', 'research/reviews/H018_empirical_reader_source_v1/reader_v1.py': '326f2fd83e9c71c32fc230d6b55aeec0652322331e2f3d5aeb7ca9f66acc0246', 'research/reviews/H018_variable_panel_adapter_source_v1/flag_interface_proposal.py': 'fc1f533190652df8205406d2b026ca00415eb4b76e503350b4b0137288abf69b', 'research/reviews/H018_variable_panel_adapter_source_v1/reader_contract_proposal.py': '862fe3255b16a2369d87eb2b19da351eb762e11567d81bfcccc1674857407fca', 'src/arggnn/H040_latent_clade_execution_v1.py': '28973fe0a9dae2c950c07dd463a23edcb6b45ab6659813f5ef31b9dc766bf84d', 'src/arggnn/anonymous_clade_flag_control_v1.py': 'd88a4639a53e63e2bff91e20d98e2af1f5cf8abc71e5fde2e02ac35e831eb8ce', 'src/arggnn/anonymous_clade_gnn_v1.py': '2f878e6334079cff083f67b6da8eda129dd30e7beebc62ae2a93d51796c3f137', 'src/arggnn/anonymous_clade_witness_v1.py': '0cdb438132e11c86fb61306a02244af0471f468c2ff2a57324644cafb97b6310', 'src/arggnn/anonymous_exact_clade_transport_v1.py': '441751a252f111fdf0d8af4e70e93e5a923b7bad7829f3aed7d5060d65727304', 'src/arggnn/biological_gst_control_v1.py': '66788b41cabcb852c739505200a641bdc93fea3c9fbb642c291d52cefc77350f', 'src/arggnn/mutation_routed_full_panel_fast_reader_v1.py': 'd6b963dc14e1f6752e310e627faf2e4f9a2e408fcb7dab17e3439771ab227e03', 'src/arggnn/mutation_routed_full_panel_primary_v1.py': 'a94fd5af1cd10b1605f75621c36a130faa3f9d5ad71068f8b187d8e73c7e5374', 'src/arggnn/mutation_routed_full_panel_pyg_model_v1.py': '786864bbf47dadbd9668abf96119ac7a0d227d91b10a7831374b18a6e63cdefa', 'src/arggnn/mutation_routed_pyg_gnn_v1.py': '53866aa4616d12e7d6cbdd34160ee7ab8fb00999444c348ee4d251b8fdfd4337', 'src/arggnn/mutation_routed_tree_observations_v1.py': '922df286e595370ade23595420a325014e4df64aac88060e8a1a846684eab068', 'src/arggnn/mutation_routed_whole_tree_gnn_v1.py': '254f5155e31f81ae675e2bd08d38c97c9bf130d47e98efa2deea4c7344d9e073', 'src/arggnn/ordered_clade_episode_gnn_v1.py': '71d93e2131858474bb7f6d0856459545dce1e6889de831b0092682f19dcbc6e6', 'src/arggnn/train_anonymous_exact_clade_transport_v1.py': 'adaec4acf070c53a894c71138674ec14a215dc4d4df580e861c2725750627189', 'src/arggnn/train_mutation_routed_full_panel_pyg_v1.py': 'c01c4d9506faae7bf537282e4d722b115b3c69cc5de1d4570758136de292dae3'}
NEW_RUNTIME_FILES=("contract_v1.py","panel_h36_v1.py","preparation_h36_v1.py","reader_h36_v1.py","model_h36_v1.py","apply_h36_v1.py")
def validate_source_manifest_v1(manifest):
    source=manifest["source_sha256"]
    require(isinstance(source,dict) and all(source.get(p)==h for p,h in FROZEN_SOURCE_SHA256.items())
        and all(HERE+"/"+name in source for name in NEW_RUNTIME_FILES)
        and all(isinstance(h,str) and len(h)==64 and all(c in "0123456789abcdef" for c in h) for h in source.values()),
        "Complete unchanged frozen operator/source family plus all H056 runtime sources")
    return source
