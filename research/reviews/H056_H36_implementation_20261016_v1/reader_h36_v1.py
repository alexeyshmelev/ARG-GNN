"""Inactive H056 H36 original phased-GT plus native-graph reader. No CLI/writes."""
from pathlib import Path
from dataclasses import fields
import gzip,hashlib,io,json,math
import numpy as np
from arggnn.mutation_routed_tree_observations_v1 import NativeObservedTreeV1
from research.reviews.H056_H36_implementation_20261016_v1.panel_h36_v1 import (
    panel_metadata_v1,verify_native_tip_dates_v1,translated_interval_view_v1,need)
from research.reviews.H056_H36_implementation_20261016_v1 import preparation_h36_v1 as preparation
from research.reviews.H056_H36_implementation_20261016_v1.contract_v1 import (H,R,Q,PEOPLE,BEGIN,END,PROTOCOL_REF,COUNT_BASE,validate_counts_v1,validate_chromosome_v1,variant_id_v1,validate_source_manifest_v1)

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/H056_H36_implementation_20261016_v1/reader_h36_v1.py"
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return json.dumps(value,separators=(",",":"),allow_nan=False).encode()
def array_identity(value):
    a=np.ascontiguousarray(value)
    return dict(shape=list(a.shape),dtype=a.dtype.str,sha256=sha(a.tobytes()))

def bank_array_manifest_v1(bank):
    """One-time complete CPU tensor identity manifest for a future input seal."""
    result={}
    for f in fields(bank):
        value=getattr(bank,f.name)
        if isinstance(value,tuple):
            result[f.name]=[array_identity(x.detach().numpy()) for x in value]
        else:
            need(value.device.type=="cpu","Preparation seal requires CPU observation tensors")
            result[f.name]=array_identity(value.detach().numpy())
    return result

def parse_original_sites_v1(vcf_gzip,roster,frozen_identities,*,chromosome,counts,max_decompressed_bytes):
    validate_chromosome_v1(chromosome)
    validate_counts_v1({k:counts[k] for k in COUNT_BASE})
    max_candidates=counts["candidate_sites"]
    panel_metadata_v1(roster)
    need(all(r.get("query_ancestry_truth") is None and r.get("native_tree_node_id") is None for r in roster),
         "Original roster only; no query truth or inferred labels")
    ids=frozen_identities
    need(isinstance(ids,list) and len(ids)>0 and all(isinstance(r,list) and len(r)==4
         and r[0]==chromosome and type(r[1]) is int and 20000000<r[1]<=30000000
         and r[2] in ("A","C","G","T") and r[3] in ("A","C","G","T") and r[2]!=r[3] for r in ids)
         and all(a[1]<b[1] for a,b in zip(ids,ids[1:])),"Exact frozen unique ordered chromosome SNP identities")
    wanted={r[1]:r for r in ids};rows=[];genotypes=[];candidates=0;decoded=0;header=None
    expected_names=[roster[i]["subject_id"] for i in range(0,H,2)]
    with gzip.GzipFile(fileobj=io.BytesIO(vcf_gzip),mode="rb") as stream:
        for raw in stream:
            decoded+=len(raw);need(decoded<=max_decompressed_bytes,"VCF decompression bound")
            line=raw.decode("ascii").rstrip("\r\n")
            if line.startswith("#CHROM\t"):
                need(header is None,"Single VCF header")
                header=line.split("\t")[9:]
                need(header==expected_names,"Exact frozen18 subject order; no sorting")
                continue
            if line.startswith("#"):continue
            need(header is not None,"VCF header precedes all records")
            fields=line.split("\t");need(len(fields)==9+PEOPLE,"Exact frozen18 diploid VCF columns")
            candidates+=1;need(candidates<=max_candidates,"VCF candidate record bound")
            pos=int(fields[1])
            if pos not in wanted:continue
            identity=[fields[0],pos,fields[3],fields[4]]
            need(identity==wanted[pos] and (not rows or pos>rows[-1][1]),"Frozen identity exact once, original order")
            formats=fields[8].split(":");need(formats.count("GT")==1,"One original observed GT field")
            at=formats.index("GT");codes=[]
            for value in fields[9:]:
                values=value.split(":");need(at<len(values),"Original phased GT cell")
                gt=values[at].split("|")
                need(len(gt)==2 and all(x in ("0","1") for x in gt),"Complete original phased biallelic calls")
                codes.extend(int(x) for x in gt)
            rows.append(identity);genotypes.append(codes)
    need(rows==ids and len(ids)==counts["sites"] and candidates==counts["candidate_sites"],"Exact sealed candidate/site identities; no re-QC or new site mask")
    g=np.asarray(genotypes,dtype=np.int16);positions=np.asarray([r[1]-20000001 for r in rows],dtype=np.int64)
    vectors=hashlib.sha256();constants=[]
    for r,codes in zip(rows,g):
        nt="".join((r[2],r[3])[int(v)] for v in codes)
        vectors.update((str(r[1])+":"+nt+"\n").encode())
        if np.all(codes==codes[0]):constants.append(r[1])
    return dict(identities=rows,positions_regional_bp0=positions,genotype=g,
        allele_count=np.full(len(g),2,dtype=np.int64),candidate_records=candidates,
        original_nucleotide_vectors_sha256=vectors.hexdigest(),constant_absolute_POS_bp1=constants,
        monomorphic_source_POS_bp1_sha256=sha(encoded(constants)),decompressed_bytes=decoded,
        source_header_order=header,VCF_fields_consumed=["CHROM","POS","REF","ALT","FORMAT/GT"])

def verify_bridge_v1(sites,roster,haps_raw,sample_raw,ages_raw,readback,*,chromosome):
    names=[roster[i]["subject_id"] for i in range(0,H,2)]
    need(sample_raw==("ID_1 ID_2 missing\n0 0 0\n"+"".join(n+" "+n+" 0\n" for n in names)).encode(),
         "Immutable bridge sample order")
    need(ages_raw=="".join(format(r["age_generations"],".17g")+"\n" for r in roster).encode(),
         "Exact original fractional source age file")
    lines=haps_raw.decode("ascii").splitlines()
    need(len(lines)==len(sites["identities"]),"Every bridge site retained")
    for line,r,codes in zip(lines,sites["identities"],sites["genotype"]):
        f=line.split();chrom,pos,ref,alt=r
        need(len(f)==5+H and f[:3]==[chromosome,variant_id_v1(chromosome,pos,ref,alt),str(pos-BEGIN)]
             and f[3] in (ref,alt) and f[4] in (ref,alt) and f[3]!=f[4]
             and all(v in ("0","1") for v in f[5:]),"Exact original bridge site identity")
        original="".join((ref,alt)[int(v)] for v in codes)
        bridge="".join((f[3],f[4])[int(v)] for v in f[5:])
        need(original==bridge,"All original biological nucleotides and phase equal immutable bridge")
    need(readback["original_nucleotide_vectors_sha256"]==sites["original_nucleotide_vectors_sha256"]
         and readback["monomorphic_source_POS_bp1_sha256"]==sites["monomorphic_source_POS_bp1_sha256"]
         and readback["retained_sites"]==len(lines),"Independent serialized readback digests")
    return dict(biological_nucleotide_and_phase_equivalence=True,sites_checked=len(lines),
                original_VCF_codes_used=True,native_inference_alleles_used=False)

def extract_graph_v1(ts,roster,mapping,local_POS_bp1,*,expected_counts):
    """Only node flags/time, edge relations/support and marginal parent relations."""
    panel=panel_metadata_v1(roster);need(len(mapping)==H,"Complete explicit reviewed native mapping")
    native=[];ages=panel["source_sample_age_generations"];raw_times=np.asarray(ts.nodes_time,dtype=np.float64)
    for i,(m,row) in enumerate(zip(mapping,roster)):
        node=m["native_sample_node_id"]
        need(type(node) is int and 0<=node<len(raw_times) and m["source_sample_index"]==i
             and m["subject_id"]==row["subject_id"] and m["haplotype"]==row["haplotype"]
             and m["role"]==row["role"] and m["source_age_generations"]==ages[i]
             and m["native_node_time"]==m["anc_age_generations"]==float(raw_times[node]),
             "Exact reviewed source haplotype/native-node/date mapping")
        native.append(node)
    sample_flags=np.asarray(ts.nodes_flags,dtype=np.uint32)
    need(len(set(native))==H and set(np.flatnonzero(sample_flags&1).tolist())==set(native),
         "All and only explicitly mapped native sample nodes")
    date_proof=verify_native_tip_dates_v1(ages,raw_times[native].tolist())
    ep=np.asarray(ts.edges_parent,dtype=np.int64);ec=np.asarray(ts.edges_child,dtype=np.int64)
    el=np.asarray(ts.edges_left,dtype=np.float64);er=np.asarray(ts.edges_right,dtype=np.float64)
    need(ts.num_nodes==expected_counts["native_nodes"] and ts.num_edges==expected_counts["native_edges"]
         and ts.num_trees==expected_counts["native_trees"] and len(ep)==len(ec)==len(el)==len(er),
         "Exact declared native graph dimensions")
    need(np.isfinite(raw_times).all() and np.all(raw_times>=0)
         and np.all((ep>=0)&(ep<len(raw_times))) and np.all((ec>=0)&(ec<len(raw_times)))
         and np.isfinite(el).all() and np.isfinite(er).all() and np.all(er>el)
         and np.all(el>=0) and np.all(er<=float(ts.sequence_length))
         and np.all(raw_times[ep]>raw_times[ec]),"Finite strict untouched native edge chronology/support")
    trees=[];intervals=[];raw_node_rows=[]
    for marginal in ts.trees():
        retained=set()
        for sample in native:
            node=sample
            while node>=0 and node not in retained:
                retained.add(node);node=int(marginal.parent(node))
        rows=np.asarray(sorted(retained),dtype=np.int64);at={int(v):i for i,v in enumerate(rows)}
        p=np.asarray([at[int(marginal.parent(int(v)))] if marginal.parent(int(v))>=0 else -1 for v in rows],dtype=np.int64)
        sample_rows=np.asarray([at[v] for v in native],dtype=np.int64)
        need(np.count_nonzero(p==-1)==1,"Every hold contains one complete36-sample tree")
        trees.append(NativeObservedTreeV1(p,raw_times[rows].copy(),sample_rows,np.arange(H,dtype=np.int64)))
        intervals.append([float(marginal.interval.left),float(marginal.interval.right)])
        raw_node_rows.append(rows)
    need(len(trees)==expected_counts["native_trees"] and sum(len(t.parent) for t in trees)==expected_counts["vertex_occurrences"],"All sealed native holds and sampled-ancestor-union vertices")
    view=translated_interval_view_v1(intervals,list(map(int,local_POS_bp1)))
    proof=dict(schema="H056_node_edge_only_graph_projection_v1",source_sample_native_node_ids=native,
        date_proof=date_proof,raw_interval_count=len(trees),native_tables_modified=False,
        nodes_flags=array_identity(sample_flags),nodes_time=array_identity(raw_times),
        edges_parent=array_identity(ep),edges_child=array_identity(ec),
        edges_left=array_identity(el),edges_right=array_identity(er),
        per_hold_native_node_rows=[array_identity(x) for x in raw_node_rows],
        native_variants_mutations_populations_or_ancestral_alleles_used=False,
        sample_mapping_inferred_from_genotypes_or_sort=False,coordinate_view=view)
    return trees,np.asarray(view["intervals_regional_bp0"],dtype=np.float64),proof

def validate_config_v1(cfg):
    validate_chromosome_v1(cfg["chromosome"]);validate_counts_v1(cfg["expected_counts"],graph=True)
    coverage=cfg["genuine_regional_bp0"]
    need(cfg["schema"]=="H056_H36_reader_config_v1" and cfg["protocol_ref"]==PROTOCOL_REF
         and cfg["feature_semantics"]==preparation.FEATURE_SEMANTICS
         and cfg["readout_policy"]=="one_midpoint_per_genuine_native_hold_no_scoring_bins"
         and cfg["coordinate_policy"]=="native_minus1_genuine_support_only"
         and isinstance(coverage,list) and len(coverage)==2 and all(type(v)in(int,float) and math.isfinite(v) for v in coverage)
         and 0<=coverage[0]<coverage[1]<=END-BEGIN
         and cfg["no_query_truth_or_accuracy_claim"] is True
         and cfg["native_GT_population_ancestral_allele_inputs"] is False,
         "Exact reviewed H056 source-GT and genuine native support semantics")
    for k in ("input_read_cap_bytes","VCF_decompressed_cap_bytes","CPU_preparation_cap_bytes","CPU_tensor_cap_bytes","flag_cap_bytes"):
        need(type(cfg[k])is int and cfg[k]>0,"Positive root-reviewed bound: "+k)

def prepare_from_root_context_v1(ctx):
    """Future admitted CPU preparation API; returns RAM objects, performs no write."""
    cfg=ctx["cfg"];validate_config_v1(cfg);counts=cfg["expected_counts"];chromosome=cfg["chromosome"]
    need(ctx.get("root_authorizer")=="root_agent" and ctx.get("admitted") is True
         and cfg.get("status")=="root_admitted_observation_preparation"
         and cfg.get("preparation_authorized") is True
         and callable(ctx.get("before_stage")),"Inactive until separate root resource admission")
    reader=ctx["reader"]
    need(Path(reader.root).resolve()==ROOT and reader.decode_json(ctx["config_ref"])==cfg,
         "Exact root reader and hash-bound config")
    source_manifest=reader.decode_json(cfg["source_manifest_ref"])
    need(source_manifest["schema"]=="H056_H36_reader_source_manifest_v1"
         and source_manifest["protocol_ref"]==PROTOCOL_REF
         and source_manifest["observation_refs"]==cfg["observation_refs"],
         "Exact source/data transitive closure")
    validate_source_manifest_v1(source_manifest)
    for path,digest in source_manifest["source_sha256"].items():
        need(sha(reader.read(dict(path=path,sha256=digest)))==digest,"Frozen reader/preparer semantics")
    refs=cfg["observation_refs"];total=0
    ctx["before_stage"]("observation_read",dict(cap_bytes=cfg["input_read_cap_bytes"]))
    def read(name):
        nonlocal total
        r=refs[name];raw=reader.read(r)
        need(isinstance(raw,bytes) and sha(raw)==r["sha256"],"Exact immutable observation bytes")
        total+=len(raw);need(total<=cfg["input_read_cap_bytes"],"Total observation input read cap")
        return raw
    roster=json.loads(read("roster"));identities=json.loads(read("frozen_sites"))
    sites=parse_original_sites_v1(read("original_VCF"),roster,identities,
        chromosome=chromosome,counts=counts,max_decompressed_bytes=cfg["VCF_decompressed_cap_bytes"])
    readback=json.loads(read("serialized_readback"))
    need(readback["root_preparation_wait_verified"] is True and readback["root_owned_group_absent"] is True
         and all(readback[k] is True for k in ("actual_serialized_site_nucleotide_phase_order_verified",
             "actual_serialized_sample_haplotype_date_join_verified","actual_qualified_map_bytes_and_local_site_join_verified")),
         "Actual root serialization/map/sample/date readback and genuine preparation wait/drain")
    bridge=verify_bridge_v1(sites,roster,read("bridge_haps"),read("bridge_sample"),read("bridge_ages"),readback,chromosome=chromosome)
    need(len(sites["identities"])==counts["sites"] and len(sites["constant_absolute_POS_bp1"])==counts["constants"]
         and sites["candidate_records"]==counts["candidate_sites"],"Every sealed actual site/constant/candidate")
    mapping_doc=json.loads(read("native_mapping"));contract=json.loads(read("graph_contract"))
    need(contract["schema"]=="H056_graph_structure_plus_original_GT_v1"
         and contract["protocol_ref"]==PROTOCOL_REF and contract["chromosome"]==chromosome
         and contract["counts"]==counts and contract["graph_only_ready"] is True
         and contract["immutable_original_GT_features"] is True
         and contract["native_graph_ref"]==refs["native_graph"]
         and contract["coverage"]["genuine_regional_bp0"]==cfg["genuine_regional_bp0"]
         and mapping_doc["source_native_facts_ref"]==contract["native_facts_ref"],
         "Actual new graph-only proof plus immutable original phased GT")
    facts=reader.decode_json(contract["native_facts_ref"])
    need(facts["schema"]=="H056_H36_actual_native_facts_v1"
         and facts["graph_only_ready"] is True and facts["protocol_ref"]==PROTOCOL_REF
         and facts["chromosome"]==chromosome and facts["counts"]==counts
         and facts["coverage"]["genuine_regional_bp0"]==cfg["genuine_regional_bp0"]
         and contract["strict_GT_actual_status"]==facts["strict_GT_actual_status"]
         and mapping_doc["rows"]==facts["native_sample_haplotype_date_mapping"],
         "Actual qualified new native graph/sample/date/support/strict-status facts")
    graph_raw=read("native_graph");graph_ref=refs["native_graph"]
    path=ROOT/graph_ref["path"]
    need(path.resolve()==path and path.is_relative_to(ROOT),"Canonical bound existing native graph")
    ctx["before_stage"]("native_graph_projection",dict(native_trees=counts["native_trees"],native_nodes=counts["native_nodes"],native_edges=counts["native_edges"]))
    import tskit
    ts=tskit.load(path)
    need(sha(path.read_bytes())==graph_ref["sha256"],"Native graph remained immutable across load")
    trees,intervals,graph_proof=extract_graph_v1(ts,roster,mapping_doc["rows"],
        sites["positions_regional_bp0"]+1,expected_counts=counts)
    del ts,graph_raw
    need(intervals[[0,-1],[0,1]].tolist()==cfg["genuine_regional_bp0"],"Exact genuine support")
    planning=preparation.preparation_plan_v1(trees,counts["sites"])
    need(planning["complete_tensor_payload_upper_bound_bytes"]<=cfg["CPU_tensor_cap_bytes"]
         and planning["retained_payload_plus_flag_preparation_upper_bound_bytes"]<=cfg["CPU_preparation_cap_bytes"],
         "Complete actual graph-dimension preflight before empirical feature preparation")
    ctx["before_stage"]("prepare_arrays",dict(planning,cap_bytes=cfg["CPU_preparation_cap_bytes"]))
    prepared=preparation.prepare_arrays_v1(trees,intervals,sites["positions_regional_bp0"],
        sites["genotype"],sites["allele_count"],roster)
    need(preparation.tensor_array_bytes_v1(prepared)<=cfg["CPU_tensor_cap_bytes"],"Complete tensor payload cap")
    bank,report=preparation.to_torch_bank_v1(prepared)
    del trees
    n=int(bank.parent.numel())
    flag_plan=2058*n+128*40
    need(preparation.tensor_array_bytes_v1(prepared)+flag_plan<=cfg["CPU_preparation_cap_bytes"],
         "Retained tensor arrays plus original exact-clade preparation allowance")
    ctx["before_stage"]("prepare_exact_clade_flags",dict(native_vertex_occurrences=n,
        preparation_allowance_bytes=flag_plan,retained_tensor_payload_bytes=preparation.tensor_array_bytes_v1(prepared)))
    from arggnn.anonymous_clade_flag_control_v1 import build_clade_flag_arrays_v1,AnonymousCladeFlagsV1
    flag_arrays,flag_meta=build_clade_flag_arrays_v1(bank.parent.numpy(),bank.sample_index.numpy())
    flags=AnonymousCladeFlagsV1.from_verified_arrays(flag_arrays,flag_meta)
    need(flags.tensor_bytes()<=cfg["flag_cap_bytes"],"Complete CPU flag payload cap")
    result=dict(schema="H056_empirical_reader_RAM_result_v1",observation_refs=refs,
        feature_semantics=preparation.FEATURE_SEMANTICS,counts=report,sites_dropped=0,
        constant_sites=counts["constants"],protocol_ref=PROTOCOL_REF,chromosome=chromosome,expected_counts=counts,input_bytes_read=total,historical_H018_strict_export_gate_unchanged=True,native_strict_GT_actual_status=contract["strict_GT_actual_status"],
        no_model_or_checkpoint_executed=True,no_query_truth_read=True,no_accuracy_claim=True,
        biological_phase_proof=bridge,graph_projection=graph_proof,preparation=prepared.metadata,preparation_plan=planning,
        original_site_join=dict(positions_regional_bp0=array_identity(sites["positions_regional_bp0"]),
            original_VCF_genotype=array_identity(sites["genotype"]),allele_count=array_identity(sites["allele_count"]),
            original_nucleotide_vectors_sha256=sites["original_nucleotide_vectors_sha256"],
            constant_source_positions_sha256=sites["monomorphic_source_POS_bp1_sha256"]),
        model_input_tensor_manifest=bank_array_manifest_v1(bank),
        output_design=dict(native_holds=counts["native_trees"],queries=Q,classes=4,probabilities_produced=False),
        exact_clade_flags=flag_meta,flag_tensor_bytes=flags.tensor_bytes(),durable_output_written=False)
    return dict(bank=bank,flags=flags,metadata=result)
