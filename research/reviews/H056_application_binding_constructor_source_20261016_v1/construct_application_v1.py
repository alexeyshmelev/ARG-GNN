"""Pure inactive all-three-bank H056 frozen-application metadata construction.
No IO, current clock, numerical/project import, guard, admission or dispatcher.
Root supplies authenticated JSON bytes and owns every actual authority and run.
"""
from pathlib import PurePosixPath
from datetime import datetime
import hashlib,json,math,re
ROOT='/workspace-SR003.nfs2/ARG-GNN'
IO_RUNTIME_REF={'path': 'research/reviews/H056_bank_application_runtime_source_20261016_v1/runtime_v2.py', 'sha256': '647e7ce205a19b7cf5847faf45b70e18825e6d28fdc7e00907157dd3a0b0436d'}
PROTOCOL_REF={'path': 'research/reviews/H056_frozen_H045_Danish_application_registration_20261016_v1/protocol.root_frozen_v1.json', 'sha256': '9544bdbd499bddafeb0774c67e4e86ef25535d41dd2409e710c29b679619cd6e'}
CHECKPOINT_REF={'path': 'experiments/results/H045_GPU5_episode_train_seed101_v1/foreign_episode.pass_30.pt', 'sha256': 'bfd8a87632015a900d9da6009d4b0ee037958460038d7949d0998e80b770796b'}
CHECKPOINT_DESCRIPTOR_REF={'path': 'research/reviews/H018_H045_empirical_application_source_v1/checkpoint_descriptor.json', 'sha256': 'fd8bbd7e61af5d93316e3f4c1d31275d1ee991f6f734991f0091ec069e5a1f12'}
RAW_ROSTER_REF={'path': 'research/reviews/Danish_genotype_corroboration_source_20261016_v2/candidate_roster.metadata.v1.tsv', 'sha256': 'eefcb18bb4c59d959b26d1e6b8a4a62c4eb0c9bbcfa89a7602797716e6d3cf46'}
FROZEN_ROSTER_SHA256='23e298c586bfcd8b668429552bbeb49bffd148a407d5c53d0000f6ece6b92fc1'
FIXED_REFS={'guard': {'path': 'research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py', 'sha256': 'fc71030865e0a3d5da31120d4daa2caa0b2d39ec7dd212b8c46c476b5a866d56'}, 'policy': {'path': 'research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json', 'sha256': 'eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018'}, 'repair_authority': {'path': 'research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json', 'sha256': 'f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db'}, 'renewal': {'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py', 'sha256': '491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468'}, 'renewal_certificate': {'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json', 'sha256': '66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7'}, 'memo': {'path': 'research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py', 'sha256': 'c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364'}, 'v4': {'path': 'src/arggnn/project_budget_v4.py', 'sha256': 'a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47'}, 'monitor': {'path': 'research/reviews/FLARE_existing_JDK_source_v1/root_build_driver.py', 'sha256': 'd7afaf5c6c6976a5c4804fe41d320f8609fd346d3f4ed0bca29235001d7fe561'}, 'contract': {'path': 'research/reviews/H056_H36_implementation_20261016_v1/contract_v1.py', 'sha256': '4c7727e27be2dddeb698dc78ff6d7ca143d85b47187a1fdd23dc7c17217d8662'}}
SCIENCE_SOURCE_SHA256={'research/reviews/H018_H045_empirical_application_source_v1/inference_v2.py': 'a8223979ee775d0aae83f576d48d3cbec482eb03022397acf1d684a08d50f227', 'research/reviews/H018_H045_empirical_application_source_v1/variable_panel_model_v1.py': '2c86da54ddd61883fc313f658dc725d9a203048cdf11d87e2edec488044c8ef6', 'research/reviews/H018_empirical_reader_source_v1/preparation_v1.py': '8bb6b9f3b77360d9eae31350337f85dd595dbfcbb0d69f95f9a41d035d96bc3c', 'research/reviews/H018_empirical_reader_source_v1/reader_v1.py': '326f2fd83e9c71c32fc230d6b55aeec0652322331e2f3d5aeb7ca9f66acc0246', 'research/reviews/H018_variable_panel_adapter_source_v1/flag_interface_proposal.py': 'fc1f533190652df8205406d2b026ca00415eb4b76e503350b4b0137288abf69b', 'research/reviews/H018_variable_panel_adapter_source_v1/reader_contract_proposal.py': '862fe3255b16a2369d87eb2b19da351eb762e11567d81bfcccc1674857407fca', 'research/reviews/H056_H36_implementation_20261016_v1/apply_h36_v1.py': '789c16842862200f8e609dedc09bbf6da93e6b2bb443ae732708923e6344304e', 'research/reviews/H056_H36_implementation_20261016_v1/contract_v1.py': '4c7727e27be2dddeb698dc78ff6d7ca143d85b47187a1fdd23dc7c17217d8662', 'research/reviews/H056_H36_implementation_20261016_v1/model_h36_v1.py': 'ecb1426f1b405937a3be6dd8771a9c226252d4fecf0ba73a2cdcab63b6fbf1aa', 'research/reviews/H056_H36_implementation_20261016_v1/panel_h36_v1.py': '2ed90dc7c951aaa4e928ca3b05f6e7b582db8a1c4fdb6afc03d56e2f9e52d39e', 'research/reviews/H056_H36_implementation_20261016_v1/preparation_h36_v1.py': 'f37f615fb649a7569e086178a66e77cdbaacbddf33df63cac214236a77bc2efc', 'research/reviews/H056_H36_implementation_20261016_v1/reader_h36_v1.py': '817ec90cdca781ba56115d7bf56ed013122b9642b147c0c43d3249fc18b7125c', 'src/arggnn/H040_latent_clade_execution_v1.py': '28973fe0a9dae2c950c07dd463a23edcb6b45ab6659813f5ef31b9dc766bf84d', 'src/arggnn/anonymous_clade_flag_control_v1.py': 'd88a4639a53e63e2bff91e20d98e2af1f5cf8abc71e5fde2e02ac35e831eb8ce', 'src/arggnn/anonymous_clade_gnn_v1.py': '2f878e6334079cff083f67b6da8eda129dd30e7beebc62ae2a93d51796c3f137', 'src/arggnn/anonymous_clade_witness_v1.py': '0cdb438132e11c86fb61306a02244af0471f468c2ff2a57324644cafb97b6310', 'src/arggnn/anonymous_exact_clade_transport_v1.py': '441751a252f111fdf0d8af4e70e93e5a923b7bad7829f3aed7d5060d65727304', 'src/arggnn/biological_gst_control_v1.py': '66788b41cabcb852c739505200a641bdc93fea3c9fbb642c291d52cefc77350f', 'src/arggnn/mutation_routed_full_panel_fast_reader_v1.py': 'd6b963dc14e1f6752e310e627faf2e4f9a2e408fcb7dab17e3439771ab227e03', 'src/arggnn/mutation_routed_full_panel_primary_v1.py': 'a94fd5af1cd10b1605f75621c36a130faa3f9d5ad71068f8b187d8e73c7e5374', 'src/arggnn/mutation_routed_full_panel_pyg_model_v1.py': '786864bbf47dadbd9668abf96119ac7a0d227d91b10a7831374b18a6e63cdefa', 'src/arggnn/mutation_routed_pyg_gnn_v1.py': '53866aa4616d12e7d6cbdd34160ee7ab8fb00999444c348ee4d251b8fdfd4337', 'src/arggnn/mutation_routed_tree_observations_v1.py': '922df286e595370ade23595420a325014e4df64aac88060e8a1a846684eab068', 'src/arggnn/mutation_routed_whole_tree_gnn_v1.py': '254f5155e31f81ae675e2bd08d38c97c9bf130d47e98efa2deea4c7344d9e073', 'src/arggnn/ordered_clade_episode_gnn_v1.py': '71d93e2131858474bb7f6d0856459545dce1e6889de831b0092682f19dcbc6e6', 'src/arggnn/train_anonymous_exact_clade_transport_v1.py': 'adaec4acf070c53a894c71138674ec14a215dc4d4df580e861c2725750627189', 'src/arggnn/train_mutation_routed_full_panel_pyg_v1.py': 'c01c4d9506faae7bf537282e4d722b115b3c69cc5de1d4570758136de292dae3'}
LINEAGE_REFS={'checkpoint_descriptor': {'path': 'research/reviews/H018_H045_empirical_application_source_v1/checkpoint_descriptor.json', 'sha256': 'fd8bbd7e61af5d93316e3f4c1d31275d1ee991f6f734991f0091ec069e5a1f12'}, 'selection_summary': {'path': 'research/reviews/H045_completed_comparison_root_20261015_v1/summary.json', 'sha256': 'd845b864d8ab0c9632e1b55f55b9db9fac172cd007457d6c37a06d778058f6c1'}, 'TRAIN_completion': {'path': 'experiments/results/H045_GPU5_episode_train_seed101_v1/completed.json', 'sha256': '258c7da76aa1f693c684094569100e19a3430566c46fe9452e841eb5127f955e'}, 'TRAIN_config': {'path': 'research/reviews/H045_GPU45_episode_root_source_v1/config.json', 'sha256': '9e778ec9b45a01d4cb233edb9586d3d44e24b998bcf191da18bbfb371ffcd0cf'}}
H,R,Q,PEOPLE=36,16,20,18
CHROMOSOMES=("20","21","22")
CLASSES=("ANA","CHG","WHG","EHG")
SUBJECTS=("Bar8","Bon002","KK1","SATP","Loschbour","Bichon","Sidelkino","NEO88","NEO866","NEO945","NEO28","NEO942","NEO790","NEO739","Gjerrild5","NEO735","NEO92","NEO870")
AGES_CALBP=(8071,10078,9720,13255,8050,13665,11258.5,7858,5457,5444,5461,5495,5668,3971,4110,3995,4218,4255)
FEATURE_SEMANTICS=dict(haplotypes=H,references=R,queries=Q,classes=list(CLASSES),
    site_tuple_width=19,leaf_width=6,node_width=3,edge_width=4,support_width=5,gate_width=9,
    age_normalization="log1p(generations)/log(10001)",
    genomic_normalization="log1p(bp)/log(2000001)",donor_age_split_generations=350,pseudocount=.5,
    donor_axes="youngerANA/CHG/WHG/EHG then olderANA/CHG/WHG/EHG",
    absent_donor_policy="masked_no_synthetic_donors",categorical_allele_embedding=False,
    native_ancestral_polarization=False,query_input_labels=-1,
    allowed_control=["aligned"],sham_field="identical_unused_structural_placeholder")
def need(ok,message):
    if not ok:raise ValueError(message)

def ceil_mib(n):return ((n+2**20-1)//2**20)*2**20

def valid_counts(counts):
    need(set(counts)=={"candidate_sites","sites","constants","native_trees","native_nodes","native_edges","vertex_occurrences"}
      and all(type(v)is int for v in counts.values()) and counts["candidate_sites"]>=counts["sites"]>=1000
      and 0<=counts["constants"]<=counts["sites"] and 0<counts["native_trees"]<=counts["sites"]
      and counts["native_nodes"]>=36 and counts["native_edges"]>0
      and counts["vertex_occurrences"]>=36*counts["native_trees"],"Actual seven native counts, not historical or forecast")
    return counts

def preparation_caps_v1(counts):
    """Inactive until actual counts exist. Abort ceilings are not measured fits."""
    c=valid_counts(counts);V,S,T=c["vertex_occurrences"],c["sites"],c["native_trees"]
    tensor=256*V+640*S+65536
    flags=2*V
    meta=2**20+1024*T
    bank=ceil_mib(tensor+flags+meta+2*2**20)
    complete=ceil_mib(meta+2**20)
    science=bank+complete
    total=ceil_mib(science+2**20+AUX+2**20)
    roles=dict(original_VCF=24*2**20,bridge_haps=16*2**20,bridge_sample=65536,bridge_ages=65536,
      frozen_sites=16*2**20,roster=65536,serialized_readback=2**20,graph_contract=2**20,native_mapping=2**20,
      native_graph=ceil_mib(8*2**20+8192*S+128*(c["native_nodes"]+c["native_edges"])))
    return dict(bank_serialization_cap_bytes=bank,completion_serialization_cap_bytes=complete,
      scientific_output_cap_bytes=science,output_cap_bytes=total,root_worker_receipt_cap_bytes=2**20,
      native_facts_read_cap_bytes=ceil_mib(2**20+3072*S),observation_role_caps=roles,
      input_read_cap_bytes=sum(roles.values()),VCF_decompressed_cap_bytes=200_000_000,
      CPU_tensor_cap_bytes=tensor,flag_cap_bytes=flags,CPU_preparation_cap_bytes=tensor+2058*V+5120)

def application_caps_v1(chromosomes):
    need(set(chromosomes)=={"20","21","22"},"All three actual chromosomes")
    cs={k:valid_counts(v) for k,v in chromosomes.items()}
    probability=sum(ceil_mib(384*c["native_trees"]+65536) for c in cs.values())
    provenance=sum(2**20+512*c["native_trees"] for c in cs.values())+2**20
    payload=ceil_mib(probability+provenance)
    total=ceil_mib(payload+2**20+AUX+2**20)
    return dict(max_serialized_artifact_payload_bytes=payload,output_cap_bytes=total,
      bank_read_cap_bytes=max(preparation_caps_v1(c)["bank_serialization_cap_bytes"] for c in cs.values()),
      checkpoint_read_cap_bytes=1048576,episode_cap_bytes=max(18*c["vertex_occurrences"]+4*c["native_trees"] for c in cs.values()),
      root_worker_receipt_cap_bytes=2**20)

AUX=524288
RSS,DEADLINE=2_000_000_000,1800
DIRECT_SUFFIXES={".py",".sh",".json"}

def encoded(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()

def relative_path(value):
    need(isinstance(value,str) and value and "\\" not in value,"Project POSIX relative path")
    p=PurePosixPath(value)
    need(not p.is_absolute() and ".." not in p.parts and "." not in p.parts
         and p.as_posix()==value and value!=".","Normalized contained relative path")
    return value

def exact_ref(value):
    need(isinstance(value,dict) and set(value)=={"path","sha256"}
         and isinstance(value["sha256"],str) and re.fullmatch("[0-9a-f]{64}",value["sha256"]),"Exact path/SHA ref")
    relative_path(value["path"])
    return dict(value)

def metadata_record(record):
    need(isinstance(record,dict) and set(record)=={"ref","raw"},"Exact supplied JSON ref/raw record")
    r=exact_ref(record["ref"]);raw=record["raw"]
    need(PurePosixPath(r["path"]).suffix==".json" and isinstance(raw,bytes)
         and 0<len(raw)<=32*2**20 and hashlib.sha256(raw).hexdigest()==r["sha256"],"Bounded authenticated JSON metadata")
    def pairs(items):
        out={}
        for k,v in items:
            need(k not in out,"Unique JSON keys");out[k]=v
        return out
    def invalid(value):raise ValueError("Nonfinite JSON constant "+value)
    return r,json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)

def unique(records):
    values={}
    for r in records:
        exact_ref(r)
        need(PurePosixPath(r["path"]).suffix in DIRECT_SUFFIXES,"Direct PY/SH/JSON refs only")
        need(r["path"] not in values or values[r["path"]]==r["sha256"],"No conflicting ref hashes")
        values[r["path"]]=r["sha256"]
    result=[dict(path=p,sha256=h) for p,h in sorted(values.items())]
    need(len(result)<=64,"Complete actual direct union<=64 without omitted authority")
    return result

def validate_roster_v1(roster,record):
    need(record["sha256"]==FROZEN_ROSTER_SHA256 and isinstance(roster,list) and len(roster)==36,"Exact frozen H36 roster")
    for i,row in enumerate(roster):
        person=i//2;label=person//2 if i<16 else None
        need(type(row["sample_index"])is int and row["sample_index"]==i
             and row["subject_id"]==SUBJECTS[person] and type(row["haplotype"])is int and row["haplotype"]==i%2
             and row["role"]==("reference" if i<16 else "query")
             and row["observed_reference_class"]==label
             and row["observed_cohort"]==(CLASSES[label] if label is not None else None)
             and row["query_ancestry_truth"] is None and row["native_tree_node_id"] is None
             and row["age_mean_cal_BP"]==AGES_CALBP[person] and row["age_generations"]==AGES_CALBP[person]/28.
             and row["date_zero"]=="calBP1950" and row["years_per_generation"]==28,"Frozen source subject/phase/role/age identity")

def validate_chromosome_bank_v1(chromosome,item):
    need(chromosome in CHROMOSOMES and isinstance(item,dict)
         and set(item)=={"bank_completion","preparation_wait","counts_ref","roster"},"Exact actual bank/wait/count-seal/roster fields")
    done_ref,done=metadata_record(item["bank_completion"])
    wait_ref,wait=metadata_record(item["preparation_wait"])
    roster_ref,roster=metadata_record(item["roster"])
    # The newly completed bank itself seals all seven counts. No extra native-count
    # authority is omitted or silently read by the application to fit the64ref limit.
    need(exact_ref(item["counts_ref"])==done_ref,"Explicit seven-count seal is the exact bank completion")
    counts=dict(valid_counts(done["counts"]))
    need(done["schema"]=="H056_empirical_CPU_preparation_completed_v1"
         and done["status"]=="completed_observation_preparation" and done["protocol_ref"]==PROTOCOL_REF
         and done["chromosome"]==chromosome
         and done["producer_ref"]==dict(path="research/reviews/H056_H36_implementation_20261016_v1/reader_h36_v1.py",
              sha256=SCIENCE_SOURCE_SHA256["research/reviews/H056_H36_implementation_20261016_v1/reader_h36_v1.py"])
         and all(done[k] is True for k in ("no_query_ancestry_truth","no_model_or_checkpoint_executed","no_GPU_used")),"New completed observation-only bank")
    bank_ref=exact_ref(done["bank_ref"]);input_ref=exact_ref(done["input_ref"])
    need(PurePosixPath(bank_ref["path"]).name=="observation_bank.pt"
         and PurePosixPath(done_ref["path"]).name=="completed.json"
         and PurePosixPath(bank_ref["path"]).parent==PurePosixPath(done_ref["path"]).parent,"Exact new same-scope bank/completion")
    need(wait["schema"]=="H056_root_genuine_preparation_wait_v1" and wait["protocol_ref"]==PROTOCOL_REF
         and type(wait["actual_exit_code"])is int and wait["actual_exit_code"]==0
         and wait["actual_direct_wait_observed"] is True and wait["process_group_drained"] is True
         and wait["completed_ref"]==done_ref and wait["input_ref"]==input_ref
         and wait["source_ref"]==IO_RUNTIME_REF,"Original genuine root-sealed preparation wait0/drain")
    for key in ("original_worker_minimal_wait_ref","original_worker_outer_wait_ref","actual_root_tool_wait_ref"):
        exact_ref(wait[key])
    meta=done["metadata"]
    need(meta["protocol_ref"]==PROTOCOL_REF and meta["chromosome"]==chromosome
         and meta["expected_counts"]==counts and meta["feature_semantics"]==FEATURE_SEMANTICS
         and meta["observation_refs"]["roster"]==roster_ref
         and meta["no_model_or_checkpoint_executed"] is True and meta["no_query_truth_read"] is True,"Original prepared model-input metadata")
    coordinate=meta["graph_projection"]["coordinate_view"]
    intervals=coordinate["intervals_regional_bp0"]
    need(isinstance(intervals,list) and len(intervals)==counts["native_trees"],"All genuine native holds")
    for i,pair in enumerate(intervals):
        need(isinstance(pair,list) and len(pair)==2 and all(type(v)in(int,float) and math.isfinite(v) for v in pair)
             and 0<=pair[0]<pair[1]<=10_000_000 and (i==0 or intervals[i-1][1]==pair[0]),"Contiguous finite genuine support")
    support=[intervals[0][0],intervals[-1][1]]
    need(coordinate["genuine_regional_bp0"]==support and coordinate["interval_count_preserved"] is True
         and coordinate["sites_dropped"]==0 and coordinate["native_tables_modified"] is False
         and coordinate["coordinate_transform"]=="native_minus1","Original genuine coverage, no fill/drop/rescale")
    validate_roster_v1(roster,roster_ref)
    return dict(bank_completion_ref=done_ref,bank_ref=bank_ref,counts=counts,
                genuine_regional_bp0=support,preparation_wait_ref=wait_ref,roster_ref=roster_ref)

def validate_target_v1(target):
    keys={"metadata_directory","job_id","reservation_key","output_directory","auxiliary_directory"}
    need(isinstance(target,dict) and set(target)==keys,"Exact root-selected target fields")
    for key in ("metadata_directory","output_directory","auxiliary_directory"):relative_path(target[key])
    need(target["metadata_directory"].startswith("research/reviews/") and target["metadata_directory"]!="research/reviews/","Dedicated root metadata folder")
    need(all(target[k].startswith("experiments/results/") and target[k]!="experiments/results/"
             for k in ("output_directory","auxiliary_directory")),"Dedicated output/aux scopes")
    need(isinstance(target["job_id"],str) and re.fullmatch(r"H056[-A-Za-z0-9_.]{1,59}",target["job_id"])
         and isinstance(target["reservation_key"],str) and re.fullmatch(r"H056/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",target["reservation_key"]),"Safe dedicated H056 job/key")
    return tuple(target[k] for k in ("metadata_directory","job_id","reservation_key","output_directory","auxiliary_directory"))

def construct_application_v1(chromosomes,actual_runtime_identity,original_bindings,*,target,created_at_utc,expires_at_utc):
    """Return a prospective config/closure and inactive root input, with no authority."""
    need(isinstance(chromosomes,dict) and set(chromosomes)==set(CHROMOSOMES),"All three actual chromosome banks")
    items={chrom:validate_chromosome_bank_v1(chrom,chromosomes[chrom]) for chrom in CHROMOSOMES}
    need(len({json.dumps(v["roster_ref"],sort_keys=True) for v in items.values()})==1,"One identical shared H36 roster ref")
    need(len({v["bank_ref"]["path"] for v in items.values()})==3
         and len({v["bank_completion_ref"]["path"] for v in items.values()})==3
         and len({v["preparation_wait_ref"]["path"] for v in items.values()})==3,"Three distinct new banks/completions/waits")
    META,JOB,KEY,OUTPUT,AUXILIARY=validate_target_v1(target)
    scopes=[PurePosixPath(p) for p in (META,OUTPUT,AUXILIARY)]
    need(all(a!=b and a not in b.parents and b not in a.parents for i,a in enumerate(scopes) for b in scopes[i+1:]),"Disjoint future metadata/output/aux")
    for item in items.values():
        immutable=PurePosixPath(item["bank_ref"]["path"]).parent
        need(all(p!=immutable and p not in immutable.parents and immutable not in p.parents for p in scopes),"Future scopes do not overlap immutable banks")
    issued=datetime.fromisoformat(created_at_utc.replace("Z","+00:00"));end=datetime.fromisoformat(expires_at_utc.replace("Z","+00:00"))
    need(issued.tzinfo is not None and end.tzinfo is not None and 0<(end-issued).total_seconds()<=DEADLINE,"Original<=1800s absolute authority interval")
    need(isinstance(original_bindings,dict) and original_bindings and all(isinstance(k,str) and k for k in original_bindings),"Complete actual original guard binding map")
    for r in original_bindings.values():exact_ref(r)
    ri=actual_runtime_identity
    need(isinstance(ri,dict) and set(ri)=={"prefix","executable","python","CUDA_VISIBLE_DEVICES","CPU_threads","packages"}
         and ri["prefix"]==ROOT+"/.runtime/env" and ri["executable"].startswith(ROOT+"/.runtime/env/bin/")
         and isinstance(ri["python"],str) and ri["python"] and ri["CUDA_VISIBLE_DEVICES"]=="" and ri["CPU_threads"]==4
         and set(ri["packages"])=={"numpy","torch","torch_geometric","tskit"}
         and all(isinstance(v,str) and v and len(v)<=256 for v in ri["packages"].values()),"Actual existing CPU runtime identity subset")
    runtime=dict(ri,created_at_utc=created_at_utc,expires_at_utc=expires_at_utc)
    caps=application_caps_v1({k:v["counts"] for k,v in items.items()})
    lineage=list(LINEAGE_REFS.values())
    stage_metadata=unique([*lineage,*[v[k] for v in items.values() for k in ("bank_completion_ref","preparation_wait_ref","roster_ref")]])
    manifest=dict(schema="H056_H36_application_source_manifest_v1",protocol_ref=dict(PROTOCOL_REF),
                  source_sha256=dict(SCIENCE_SOURCE_SHA256),required_metadata_refs=stage_metadata)
    manifests_bytes=encoded(manifest)
    manifest_ref=dict(path=META+"/source_manifest.json",sha256=hashlib.sha256(manifests_bytes).hexdigest())
    # This future config readiness flag is not execution authority. The only root
    # input returned below is inactive and has no storage or enforcement authority.
    cfg=dict(schema="H056_H36_CPU_application_config_v1",execution_enabled=True,protocol_ref=dict(PROTOCOL_REF),
         checkpoint_ref=dict(CHECKPOINT_REF),checkpoint_descriptor_ref=dict(CHECKPOINT_DESCRIPTOR_REF),
         chromosomes=items,CPU_threads=4,optimization_steps=0,max_host_RSS_bytes=RSS,wall_deadline_seconds=DEADLINE,
         source_manifest_ref=manifest_ref,**{k:caps[k] for k in ("output_cap_bytes","max_serialized_artifact_payload_bytes",
          "bank_read_cap_bytes","checkpoint_read_cap_bytes","episode_cap_bytes")})
    documents={"config.json":cfg,"runtime.json":runtime,"source_manifest.json":manifest}
    rawdocs={name:encoded(value) for name,value in documents.items()}
    docrefs={name:dict(path=META+"/"+name,sha256=hashlib.sha256(raw).hexdigest()) for name,raw in rawdocs.items()}
    resources=unique([*original_bindings.values(),*FIXED_REFS.values()])
    allsources=unique([*[dict(path=p,sha256=h) for p,h in SCIENCE_SOURCE_SHA256.items()],IO_RUNTIME_REF,
                       *[r for r in resources if PurePosixPath(r["path"]).suffix==".py"]])
    io_metadata=unique([PROTOCOL_REF,*stage_metadata,*docrefs.values(),
                        *[r for r in resources if PurePosixPath(r["path"]).suffix==".json"]])
    io_manifest=dict(schema="H056_specific_CPU_IO_closure_v1",operation="apply",protocol_ref=dict(PROTOCOL_REF),
         scientific_manifest_ref=manifest_ref,source_sha256={r["path"]:r["sha256"] for r in allsources},
         required_metadata_refs=io_metadata,original_bindings=json.loads(json.dumps(original_bindings)))
    documents["io_source_manifest.json"]=io_manifest
    rawdocs["io_source_manifest.json"]=encoded(io_manifest)
    docrefs["io_source_manifest.json"]=dict(path=META+"/io_source_manifest.json",sha256=hashlib.sha256(rawdocs["io_source_manifest.json"]).hexdigest())
    direct=unique([IO_RUNTIME_REF,dict(path="research/reviews/H056_H36_implementation_20261016_v1/apply_h36_v1.py",
             sha256=SCIENCE_SOURCE_SHA256["research/reviews/H056_H36_implementation_20261016_v1/apply_h36_v1.py"]),
             *docrefs.values(),PROTOCOL_REF,*resources,*allsources,*io_metadata])
    inactive=dict(schema="H056_H36_CPU_application_root_v1",operation="apply",root_authorizer="root_agent",
         hypothesis_id="H056",registered=True,execution_authorized=False,protocol_ref=dict(PROTOCOL_REF),
         GPU_authorized=False,optimization_authorized=False,root_host_resource_enforcement_active=False,
         producer_ref=dict(path="research/reviews/H056_H36_implementation_20261016_v1/apply_h36_v1.py",
             sha256=SCIENCE_SOURCE_SHA256["research/reviews/H056_H36_implementation_20261016_v1/apply_h36_v1.py"]),
         config_ref=docrefs["config.json"],runtime_ref=docrefs["runtime.json"],storage_ref=None,
         storage_adapter_ref=dict(FIXED_REFS["guard"]),io_context_ref=dict(IO_RUNTIME_REF),
         io_source_manifest_ref=docrefs["io_source_manifest.json"],source_manifest_ref=manifest_ref,
         resource_binding_refs=resources,caps=caps,job_id=JOB,output_directory=OUTPUT,
         auxiliary_directory=AUXILIARY,expires_at_utc=expires_at_utc)
    documents["input.inactive.json"]=inactive;rawdocs["input.inactive.json"]=encoded(inactive)
    docrefs["input.inactive.json"]=dict(path=META+"/input.inactive.json",sha256=hashlib.sha256(rawdocs["input.inactive.json"]).hexdigest())
    need(all(len(raw)<=65536 for raw in rawdocs.values()),"Finite root metadata documents<=64KiB each")
    return dict(schema="H056_frozen_all_three_bank_application_pure_construction_v1",execution_authorized=False,
         documents=documents,encoded_documents=rawdocs,document_refs=docrefs,consumer_bindings=direct,
         consumer_binding_count=len(direct),planned_complete_union_count=64,caps=caps,target=dict(target),
         reservation_expectations=dict(key=KEY,output=ROOT+"/"+OUTPUT,auxiliary_scopes=[ROOT+"/"+AUXILIARY],
             reserved_bytes=caps["output_cap_bytes"],consumer_bindings=direct),
         storage_record_expectations=dict(job_id=JOB,output_directory=OUTPUT,expires_at_utc=expires_at_utc,
             reserved_new_bytes=caps["output_cap_bytes"],completed_overlay_policy_ref=dict(FIXED_REFS["policy"])),
         frozen_claims=dict(checkpoint_ref=dict(CHECKPOINT_REF),parameter_count=31812,selected_arm="foreign_episode",
             selected_pass=30,optimization_steps=0,chromosomes=list(CHROMOSOMES),haplotypes=36,references=16,queries=20,
             registered_contrasts=["EHG","CHG"],report_both_regardless_of_sign=True,accuracy_available=False),
         root_pending=["Independently review exact constructor/source and actual supplied metadata",
             "Verify actual runtime/source hashes, genuine bank wait provenance and complete original guard binding map",
             "Write exact four future config/runtime/scientific/IO document bytes under existing root metadata guard",
             "Fresh original prospective980GB admission of exact disjoint output/aux and complete actual union<=64",
             "Only then derive active input: fill actual storage_ref and set execution_authorized/root_host_resource_enforcement_active true",
             "Hash fresh actual active input and invoke exact reviewed runtime_v2 with --launch",
             "Consume genuine external application toolterminal0, current owned drain and final original scope guard before publication",
             "Report both prespecified contrasts regardless of sign as descriptive probabilities without ancestry truth/accuracy"],
         future_active_input_path=META+"/input.json",
         argv_template=[".runtime/env/bin/python","-B",IO_RUNTIME_REF["path"],"--launch","--input",META+"/input.json","--expected-sha256","<actual-active-input-SHA256>"],
         no_IO_clock_project_import_model_or_admission=True)
