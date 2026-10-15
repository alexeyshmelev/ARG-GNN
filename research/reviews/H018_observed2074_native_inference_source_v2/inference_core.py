"""Pure observed bridge/native verification API; no I/O or execution authority."""
import csv,hashlib,io,json,math,struct
from decimal import Decimal,ROUND_HALF_EVEN
from collections import Counter

def need(ok, text):
    if not ok:
        raise ValueError(text)

def encoded(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def rounded_age_token(age):
    # Frozen reference is binary64 source age, not an invented exact true date.
    return format(Decimal.from_float(age).quantize(Decimal("0.000001"),rounding=ROUND_HALF_EVEN),"f")

def parse_anc(raw):
    lines=raw.decode("ascii").splitlines()
    first=lines[0].split()
    need(first[:2] == ["NUM_HAPLOTYPES","40"] and len(first)==42, "Exactly40 native fractional ages, no zero-age fallback")
    need(lines[1].startswith("NUM_TREES "), "Native .anc tree count header")
    ages=[float(x) for x in first[2:]]
    ntrees=int(lines[1].split()[1])
    need(len(lines)==ntrees+2 and ntrees>0, "All native .anc trees present")
    trees=[]
    for line in lines[2:]:
        at,tail=line.split(": ",1)
        fields=tail.split(") ")
        if fields[-1].endswith(")"):fields[-1]=fields[-1][:-1]
        if fields[-1]=="":fields.pop()
        need(len(fields)==79, "Forty-leaf native79-node tree")
        parents=[];lengths=[]
        for text in fields:
            parent,values=text.split(":(",1)
            values=values.split()
            need(len(values)==4, "Native branch record shape")
            parents.append(int(parent));lengths.append(float(values[0]))
        need(parents[-1]==-1 and all(0 <= p < 79 for p in parents[:-1]), "Native root/parent references")
        children=[[] for _ in range(79)]
        for child,parent in enumerate(parents[:-1]):children[parent].append(child)
        coordinates=[None]*79
        def coord(node,active):
            need(node not in active, "Native ancestry cycle")
            if node < 40:
                need(not children[node],"Native sample is a leaf")
                value=ages[node]
            else:
                need(len(children[node])==2, "Native binary internal topology")
                active=active|{node}
                estimates=[coord(c,active)+lengths[c] for c in children[node]]
                need(all(math.isfinite(x) for x in estimates),"Finite native branch chronology")
                value=max(estimates)
            coordinates[node]=value
            return value
        coord(78,set())
        need(all(x is not None for x in coordinates),"All native nodes reached")
        # Reconstruct source float32 arithmetic to detect converter chronology
        # intervention without changing any native tree or date.
        f32=lambda x:struct.unpack("f",struct.pack("f",x))[0]
        coordinates32=[None]*79
        def coord32(node):
            if node<40:value=f32(ages[node])
            else:value=f32(max(coord32(c)+lengths[c] for c in children[node]))
            coordinates32[node]=value
            return value
        coord32(78)
        nonpositive=[child for child,parent in enumerate(parents[:-1]) if coordinates32[parent] <= coordinates32[child]]
        trees.append(dict(start_snp0=int(at),parents=parents,lengths=lengths,coordinates32=coordinates32,
                          nonpositive_float32_branches=nonpositive))
    return first[2:],ages,trees

def expected_observations(inputs, roster_raw, readback):
    """Parse only the root-read frozen bridge bytes; no new file or source-VCF I/O."""
    roster=json.loads(roster_raw)
    need(isinstance(roster,list) and len(roster)==40,"Exact original40-row roster")
    names=[];ages=[]
    for i,row in enumerate(roster):
        need(row["sample_index"]==i and row["haplotype"]==i%2
          and row["role"]==("reference" if i<16 else "query"),"Original R16/Q24 order")
        need(row["date_zero"]=="calBP1950" and row["years_per_generation"]==28
          and row["age_generations"]==row["age_mean_cal_BP"]/28.
          and row["query_ancestry_truth"] is None and row["native_tree_node_id"] is None,
          "Original fractional calBP1950/28, no query truth or inferred node IDs")
        ages.append(row["age_generations"])
        if i%2:need(row["subject_id"]==roster[i-1]["subject_id"],"Original adjacent diploid pair")
        else:names.append(row["subject_id"])
    need(len(set(names))==20 and inputs["baseline.sample"]==
      ("ID_1 ID_2 missing\n0 0 0\n"+"".join(n+" "+n+" 0\n" for n in names)).encode(),"Exact sample text")
    need(inputs["sample_ages.generations"]=="".join(format(a,".17g")+"\n" for a in ages).encode(),
      "Exact fractional source .17g age file")
    identities=json.loads(inputs["frozen_site_identities.json"])
    audits=json.loads(inputs["polarization_equivalence.json"])
    lines=inputs["baseline.haps"].decode("ascii").splitlines()
    need(len(lines)==len(identities)==len(audits)==2074,"All original2074 sites")
    rows=[];source_vectors=hashlib.sha256();monomorphic=[]
    for index,(line,identity,audit) in enumerate(zip(lines,identities,audits)):
        fields=line.split();chrom,pos,ref,alt=identity;local=pos-20000000
        need(len(fields)==45 and chrom=="22" and 1<=local<=10000000
          and ref in "ACGT" and alt in "ACGT" and ref!=alt,"Original chr22 SNP identity")
        aa,da=fields[3:5]
        need(all(v in ("0","1") for v in fields[5:]),"Binary original phased haps")
        codes=[int(v) for v in fields[5:]];original=audit["source_GT_haplotype_codes"]
        need(len(original)==40 and all(v in (0,1) for v in original),"Original40 source codes")
        nt="".join((ref,alt)[v] for v in original)
        need(fields[:3]==["22","H018_22_"+str(pos)+"_"+ref+"_"+alt,str(local)]
          and aa in (ref,alt) and da in (ref,alt) and aa!=da,"Original AA/DA identity and position")
        need(nt=="".join((aa,da)[v] for v in codes)
          and audit["source_POS_bp1"]==pos and audit["haps_POS_bp1"]==local
          and audit["regional_bp0"]==local-1 and audit["source_REF"]==ref and audit["source_ALT"]==alt
          and audit["native_haps_codes"]==codes and audit["nucleotide_equivalence_sha256"]==sha(nt.encode())
          and audit["exact_nucleotide_and_phase_order_equivalence"] is True,
          "Every original biological nucleotide and phase")
        mono=len(set(codes))==1
        need(audit["monomorphic_among_actual40"]==mono,"Original monomorphic identity")
        source_vectors.update((str(pos)+":"+nt+"\n").encode())
        if mono:monomorphic.append(pos)
        rows.append(dict(index0=index,name=fields[1],local_bp1=local,local_bp0=local-1,
          absolute_POS_bp1=pos,native_AA=aa,native_DA=da,biological_nucleotides=nt,monomorphic=mono))
    need(all(a["local_bp1"]<b["local_bp1"] for a,b in zip(rows,rows[1:])),"Strict unique original SNP order")
    need(len(monomorphic)==277 and source_vectors.hexdigest()==readback["original_nucleotide_vectors_sha256"]
      and sha(json.dumps(monomorphic,separators=(",",":")).encode())==readback["monomorphic_source_POS_bp1_sha256"],
      "Independent readback vector and277 monomorphic identity digests")
    return dict(rows=rows,ages=ages,names=names,roster=roster,monomorphic_absolute_POS=monomorphic,
      source_vectors_sha256=source_vectors.hexdigest())


def verify_native(inputs, observed, anc_raw, mut_raw, tree_sequence):
    # Accepts a loaded tree sequence object; does not read files or invoke native.
    details=dict(schema="H018_observed2074_native_facts_v1",observed_GT_only=True,expected_sites=2074,haplotypes=40,
                 biological_or_accuracy_claim=False,failures=[],all_original_sites_preserved=None,
                 qualifier_modified_native_genotypes_or_times=False)
    def fact(ok,name):
        details[name]=bool(ok)
        if not ok:details["failures"].append(name)
    try:
        tokens,anc_ages,anc_trees=parse_anc(anc_raw)
        details["native_tree_count"]=len(anc_trees)
        fact(all(tokens[i]==rounded_age_token(a) for i,a in enumerate(observed["ages"])),"exact_six_decimal_age_tokens")
        age_errors=[];bounds=[]
        for i,(a,b) in enumerate(zip(observed["ages"],anc_ages)):
            fact(float(format(a,".17g"))==a,"age_input_roundtrip_"+str(i))
            spacing=max(b-math.nextafter(b,-math.inf),math.nextafter(b,math.inf)-b)
            bound=0.5e-6+0.5*spacing
            age_errors.append(abs(b-a));bounds.append(bound)
        fact(all(e<=b for e,b in zip(age_errors,bounds)),"combined_binary64_sixdecimal_age_bound")
        details["source_to_native_tip_errors_generations"]=age_errors
        details["source_derived_tip_error_bounds_generations"]=bounds
        reader=csv.DictReader(io.StringIO(mut_raw.decode("ascii")),delimiter=";")
        muts=list(reader)
        need(reader.fieldnames is not None,"Native .mut header")
        details["native_mut_rows"]=len(muts)
        expected=observed["rows"]
        fact(len(muts)==2074,"native_mut_exact_row_count")
        mut_identities=[]
        for index,row in enumerate(muts):
            mut_identities.append(dict(index0=int(row["snp"]),position=int(row["pos_of_snp"]),name=row["rs-id"],
              alleles=row["ancestral_allele/alternative_allele"],branches=row["branch_indices"],flipped=row["is_flipped"],
              is_not_mapping=row["is_not_mapping"],tree_index=int(row["tree_index"])))
        fact(len(muts)==2074 and all(r["index0"]==i and r["position"]==e["local_bp1"] and r["name"]==e["name"]
             and r["alleles"]==e["native_AA"]+"/"+e["native_DA"] for i,(r,e) in enumerate(zip(mut_identities,expected))),
             "native_mut_every_identity_position_allele_order")
        details["native_mut_identity_sha256"]=sha(encoded(mut_identities))
        details["native_mut_identity_facts"]=mut_identities
        details["native_mut_flipped_rows0"]=[i for i,r in enumerate(mut_identities) if r["flipped"]=="1"]
        details["native_mut_multi_branch_rows0"]=[i for i,r in enumerate(mut_identities) if len(r["branches"].split())>1]
        details["native_mut_constant_branch_facts"]=[mut_identities[i] for i,e in enumerate(expected) if e["monomorphic"] and i<len(mut_identities)]
        ts=tree_sequence
        samples=list(map(int,ts.samples()))
        fact(samples==list(range(40)) and ts.num_samples==40,"sample_node_insertion_and_count")
        fact(all(float(ts.node(node).time)==anc_ages[i] for i,node in enumerate(samples)),"exact_parsed_anc_to_binary_sample_time")
        fact(ts.num_sites==2074,"tree_sequence_exact_site_count")
        positions=[float(x) for x in ts.tables.sites.position]
        fact(positions==[e["local_bp1"] for e in expected],"all_exact_local_bp1_sites_no_jitter")
        fact(float(ts.sequence_length)==expected[-1]["local_bp1"]+1,"actual_last_site_plus_one_raw_length")
        vector_errors=[];constant_facts=[];vector_facts=[];hasher=hashlib.sha256()
        for index,variant in enumerate(ts.variants(samples=samples,isolated_as_missing=True)):
            if index>=2074:
                vector_errors.append(dict(index0=index,reason="extra_site"));continue
            codes=list(map(int,variant.genotypes))
            missing=any(g<0 or g>=len(variant.alleles) or variant.alleles[g] is None for g in codes)
            actual="" if missing else "".join(variant.alleles[g] for g in codes)
            e=expected[index]
            vector_facts.append(dict(index0=index,position=float(variant.site.position),ancestral_state=variant.site.ancestral_state,biological_nucleotides=actual,missing=missing))
            if missing or actual!=e["biological_nucleotides"]:
                vector_errors.append(dict(index0=index,missing=missing,actual=actual,expected=e["biological_nucleotides"]))
            hasher.update((str(e["local_bp1"])+":"+actual+"\n").encode())
            if e["monomorphic"]:constant_facts.append(dict(index0=index,actual=actual,expected=e["biological_nucleotides"],equal=actual==e["biological_nucleotides"]))
        fact(not vector_errors and ts.num_sites==2074,"all82960_biological_nucleotide_and_phase_calls")
        details["all_site_vector_sha256"]=hasher.hexdigest()
        details["vector_errors"]=vector_errors
        details["all2074_native_vector_facts"]=vector_facts
        details["all277_constant_vector_facts"]=constant_facts
        details["native_anc_float32_nonpositive_branches"]=[dict(tree_index=i,branches=t["nonpositive_float32_branches"]) for i,t in enumerate(anc_trees) if t["nonpositive_float32_branches"]]
        details["native_converter_chronology_adjustment_triggered"]=bool(details["native_anc_float32_nonpositive_branches"])
        fact(not details["native_anc_float32_nonpositive_branches"],"no_converter_chronology_adjustment_trigger")
        # The converter creates39 internal nodes per native tree. Compare exact
        # float-derived coordinates, not a broad chronology tolerance.
        expected_nodes=40+39*len(anc_trees)
        fact(ts.num_nodes==expected_nodes,"native_tree_local_internal_node_layout")
        coordinate_errors=[]
        if ts.num_nodes==expected_nodes:
            for k,t in enumerate(anc_trees):
                for node in range(40,79):
                    ident=node+k*39
                    if float(ts.node(ident).time)!=t["coordinates32"][node]:
                        coordinate_errors.append(dict(tree_index=k,native_node=node,exported_node=ident))
        fact(not coordinate_errors,"no_unreviewed_internal_time_changes")
        details["internal_time_mismatch_facts"]=coordinate_errors
        edge_bad=[];edge_facts=[]
        for edge in ts.edges():
            edge_facts.append([float(edge.left),float(edge.right),int(edge.parent),int(edge.child),float(ts.node(edge.parent).time),float(ts.node(edge.child).time)])
            if not (math.isfinite(edge.left) and math.isfinite(edge.right) and 0<=edge.left<edge.right<=ts.sequence_length
                    and math.isfinite(ts.node(edge.parent).time) and ts.node(edge.parent).time>ts.node(edge.child).time):
                edge_bad.append(edge.id)
        fact(not edge_bad,"finite_strict_native_edge_chronology")
        details["invalid_edges"]=edge_bad
        details["native_edge_count"]=len(edge_facts)
        details["native_edge_facts_sha256"]=sha(encoded(edge_facts))
        intervals=[]
        for tr in ts.trees():
            intervals.append(dict(left=float(tr.interval.left),right=float(tr.interval.right),roots=int(tr.num_roots),
              all40_connected=tr.num_roots==1 and tr.num_samples(tr.root)==40))
        details["raw_tree_interval_facts"]=intervals
        fact(all(t["all40_connected"] for t in intervals),"all_raw_intervals_carry_all40_samples")
        fact(all(ts.at(e["local_bp1"]).num_roots==1 and ts.at(e["local_bp1"]).num_samples(ts.at(e["local_bp1"]).root)==40 for e in expected),"every_original_site_has_full_tree_support")
        first=expected[0]["local_bp1"];last=expected[-1]["local_bp1"]
        details["coverage"]=dict(raw_native_domain=[0,float(ts.sequence_length)],
          requested_regional_bp0=[0,10000000],genuine_native_domain=[first,last+1],
          genuine_regional_bp0=[first-1,last],genuine_absolute_bp0=[20000000+first-1,20000000+last],
          covered_bp=last-first+1,unrepresented_left_bp=first-1,unrepresented_right_bp=10000000-last,
          affine_join="regional_bp0=native_position-1;absolute_bp0=20000000+regional_bp0",
          full_10000000bp_coverage_claim=False,no_flanking_tree_extrapolation=True,
          native_output_not_clipped_or_modified=True)
        starts=[t["start_snp0"] for t in anc_trees]
        fact(bool(starts) and starts[0]==0 and all(a<b for a,b in zip(starts,starts[1:]))
          and starts[-1]<2074,"strict_native_tree_SNP_starts")
        edge_expected=[]
        if details["strict_native_tree_SNP_starts"]:
            for k,t in enumerate(anc_trees):
                stop=starts[k+1] if k+1<len(starts) else 2074
                left=0. if starts[k]==0 else .5*(expected[starts[k]-1]["local_bp1"]+expected[starts[k]]["local_bp1"])
                right=float(last+1) if stop==2074 else .5*(expected[stop-1]["local_bp1"]+expected[stop]["local_bp1"])
                for i in range(starts[k],stop):
                    fact(mut_identities[i]["tree_index"]==k,"mut_tree_index_"+str(i))
                for child,parent in enumerate(t["parents"][:-1]):
                    edge_expected.append((left,right,parent+39*k,child if child<40 else child+39*k))
            actual_edges=[(float(e.left),float(e.right),int(e.parent),int(e.child)) for e in ts.edges()]
            fact(Counter(edge_expected)==Counter(actual_edges),"exact_native_marginal_topology_and_edge_support")
        else:
            fact(False,"exact_native_marginal_topology_and_edge_support")
        details["native_sample_haplotype_date_mapping"]=[dict(
          native_sample_node_id=node,subject_id=row["subject_id"],haplotype=row["haplotype"],
          source_sample_index=i,role=row["role"],source_age_generations=observed["ages"][i],
          anc_age_generations=anc_ages[i],native_node_time=float(ts.node(node).time))
          for i,(node,row) in enumerate(zip(samples,observed["roster"]))]
        details["all_original_sites_preserved"]=all(details[key] for key in (
            "native_mut_exact_row_count","native_mut_every_identity_position_allele_order",
            "tree_sequence_exact_site_count","all_exact_local_bp1_sites_no_jitter",
            "all82960_biological_nucleotide_and_phase_calls","every_original_site_has_full_tree_support"))
    except BaseException as exc:
        details["failures"].append("verification_exception")
        details["verification_error"]=dict(type=type(exc).__name__,message=str(exc)[:2000])
    details["status"]="verified_original2074site_GT_pre_size_native_genealogy" if not details["failures"] else "unsupported_native_facts_preserved"
    details["model_training_or_application_performed"]=False
    details["ancestry_truth_available"]=False
    details["main_post_five_size_condition_executed"]=False
    return details

def parse_headers(global_raw, chunks):
    """Pure bounded native x86_64 header decoding; supports overlapping chunks."""
    need(isinstance(global_raw,bytes) and 28<=len(global_raw)<=84,"Bounded global native header")
    n,l,c=struct.unpack("<iii",global_raw[:12]);memory=struct.unpack("<d",global_raw[12:20])[0]
    need(n==40 and l==2074 and 1<=c<=8 and len(global_raw)==20+8*c
      and math.isfinite(memory) and 0<memory<8.,"Exact original40/2074 and bounded chunk count")
    starts=list(struct.unpack("<"+"i"*c,global_raw[20:20+4*c]))
    ends=list(struct.unpack("<"+"i"*c,global_raw[20+4*c:]))
    need(starts[0]==0 and ends[-1]==2074 and all(0<=a<b<=2074 for a,b in zip(starts,ends))
      and all(a<b for a,b in zip(starts,starts[1:])) and all(a<b for a,b in zip(ends,ends[1:]))
      and all(starts[i]<=ends[i-1] for i in range(1,c)),"Complete original overlapping SNP support")
    need(set(chunks)==set(range(c)),"Exactly bounded original chunk header set")
    result=[]
    for k in range(c):
        raw=chunks[k];need(isinstance(raw,bytes) and 20<=len(raw)<=8312,"Bounded chunk metadata bytes")
        cn,cl,b=struct.unpack("<iii",raw[:12])
        need(cn==40 and cl==ends[k]-starts[k] and 2<=b<=min(cl+1,2075)
          and len(raw)==12+4*b,"Original40 haplotypes and complete section header")
        bounds=list(struct.unpack("<"+"i"*b,raw[12:]))
        need(bounds[0]==0 and bounds[-1]==cl and all(a<z for a,z in zip(bounds,bounds[1:])),
          "Strict complete native section boundaries")
        result.append(dict(chunk_index=k,sites=cl,first_section=0,last_section=b-2,
          boundaries=bounds,header_sha256=sha(raw),header_hex=raw.hex()))
    return dict(schema="H018_observed2074_native_headers_v1",haplotypes=n,sites=l,num_chunks=c,
      native_memory_GB=memory,global_starts=starts,global_ends=ends,
      global_header_sha256=sha(global_raw),global_header_hex=global_raw.hex(),chunks=result)


def cli_stages(binaries,inputs,profile,headers=None):
    """One full section range per chunk preserves the --mode All RNG resets."""
    common=["--haps",str(inputs/"baseline.haps"),"--sample",str(inputs/"baseline.sample"),
      "--sample_ages",str(inputs/"sample_ages.generations"),"--map",str(inputs/"qualified.regional.map"),
      "--mutation_rate",profile["mutation_rate"],"--effectiveN",str(profile["effective_haploid_N"]),
      "--memory",profile["native_memory_GB"],"--seed",str(profile["seed"]),"--output","native"]
    def stage(name,mode,extra=()):
        return name,[binaries["Relate"],"--mode",mode,*common,*extra],"relate"
    if headers is None:return [stage("make_chunks","MakeChunks")]
    stages=[]
    for item in headers["chunks"]:
        c=item["chunk_index"];chunk=["--chunk_index",str(c)]
        full=chunk+["--first_section","0","--last_section",str(item["last_section"])]
        for name,mode,extra in [("paint","Paint",chunk),("build_topology","BuildTopology",full),
          ("find_equivalent_branches","FindEquivalentBranches",chunk),
          ("infer_branch_lengths","InferBranchLengths",full),("combine_sections","CombineSections",chunk)]:
            stages.append(stage(name+"_c"+str(c),mode,extra))
    stages.append(stage("finalize","Finalize"))
    stages.append(("convert",[binaries["RelateFileFormats"],"--mode","ConvertToTreeSequence",
      "--input","native","--output","native"],"convert"))
    return stages
