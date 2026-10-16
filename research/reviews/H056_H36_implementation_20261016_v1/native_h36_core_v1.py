"""Pure H056 H36 native parsing/verification, inherited from qualified H018.
No file/native/model IO, controller, admission or execution authority.
Actual strict native GT export and graph-only structural facts remain separate.
"""
import csv,hashlib,io,json,math,struct
from decimal import Decimal,ROUND_HALF_EVEN
from collections import Counter
from research.reviews.H056_H36_implementation_20261016_v1.contract_v1 import (
 H,R,Q,PEOPLE,BEGIN,END,PROTOCOL_REF,require as need,sha,encoded,
 validate_chromosome_v1,validate_counts_v1,validate_roster_identity_v1,variant_id_v1)
NATIVE_PROFILE=dict(effective_haploid_N=30000,mutation_rate="1.25e-8",years_per_generation=28,
 population_size_iterations_executed=0,pre_size_only=True,seed=414855340,native_memory_GB="2")

def checked_counts_v1(counts):
    validate_counts_v1(counts,graph="native_trees" in counts)
    return counts

def validate_profile_v1(profile):
    for k,v in NATIVE_PROFILE.items():
        need(k in profile and type(profile[k]) is type(v) and profile[k]==v,"Exact inherited pre-size native option: "+k)
    return profile

def rounded_age_token(age):
    # Frozen reference is binary64 source age, not an invented exact true date.
    return format(Decimal.from_float(age).quantize(Decimal("0.000001"),rounding=ROUND_HALF_EVEN),"f")

def parse_anc_v1(raw,*,counts):
    checked_counts_v1(counts);S=counts["sites"]
    need(isinstance(raw,bytes),"Root-read ANC bytes")
    lines=raw.decode("ascii").splitlines()
    first=lines[0].split()
    need(first[:2] == ["NUM_HAPLOTYPES",str(H)] and len(first)==(H+2), "Exactly36 native fractional ages, no zero-age fallback")
    need(lines[1].startswith("NUM_TREES "), "Native .anc tree count header")
    ages=[float(x) for x in first[2:]]
    ntrees=int(lines[1].split()[1])
    need(len(lines)==ntrees+2 and 0<ntrees<=S, "All native .anc trees present")
    if "native_trees" in counts:need(ntrees==counts["native_trees"],"Exact independently sealed native tree count")
    trees=[]
    for line in lines[2:]:
        at,tail=line.split(": ",1)
        fields=tail.split(") ")
        if fields[-1].endswith(")"):fields[-1]=fields[-1][:-1]
        if fields[-1]=="":fields.pop()
        need(len(fields)==(2*H-1), "Thirty-six-leaf native71-node tree")
        parents=[];lengths=[]
        for text in fields:
            parent,values=text.split(":(",1)
            values=values.split()
            need(len(values)==4, "Native branch record shape")
            parents.append(int(parent));lengths.append(float(values[0]))
        need(parents[-1]==-1 and all(0 <= p < (2*H-1) for p in parents[:-1]), "Native root/parent references")
        children=[[] for _ in range((2*H-1))]
        for child,parent in enumerate(parents[:-1]):children[parent].append(child)
        coordinates=[None]*(2*H-1)
        def coord(node,active):
            need(node not in active, "Native ancestry cycle")
            if node < H:
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
        coord((2*H-2),set())
        need(all(x is not None for x in coordinates),"All native nodes reached")
        # Reconstruct source float32 arithmetic to detect converter chronology
        # intervention without changing any native tree or date.
        f32=lambda x:struct.unpack("f",struct.pack("f",x))[0]
        coordinates32=[None]*(2*H-1)
        def coord32(node):
            if node<H:value=f32(ages[node])
            else:value=f32(max(coord32(c)+lengths[c] for c in children[node]))
            coordinates32[node]=value
            return value
        coord32((2*H-2))
        nonpositive=[child for child,parent in enumerate(parents[:-1]) if coordinates32[parent] <= coordinates32[child]]
        trees.append(dict(start_snp0=int(at),parents=parents,lengths=lengths,coordinates32=coordinates32,
                          nonpositive_float32_branches=nonpositive))
    return first[2:],ages,trees

def expected_observations_v1(inputs,roster_raw,readback,*,chromosome,counts):
    validate_chromosome_v1(chromosome);checked_counts_v1(counts);S,K=counts["sites"],counts["constants"]
    """Parse only the root-read frozen bridge bytes; no new file or source-VCF I/O."""
    roster=json.loads(roster_raw);validate_roster_identity_v1(roster)
    need(readback["retained_sites"]==S,"New independent serialized readback site count")
    need(isinstance(roster,list) and len(roster)==H,"Exact frozen36-row roster")
    names=[];ages=[]
    for i,row in enumerate(roster):
        need(row["sample_index"]==i and row["haplotype"]==i%2
          and row["role"]==("reference" if i<16 else "query"),"Frozen H36/R16/Q20 order")
        need(row["date_zero"]=="calBP1950" and row["years_per_generation"]==28
          and row["age_generations"]==row["age_mean_cal_BP"]/28.
          and row["query_ancestry_truth"] is None and row["native_tree_node_id"] is None,
          "Original fractional calBP1950/28, no query truth or inferred node IDs")
        ages.append(row["age_generations"])
        if i%2:need(row["subject_id"]==roster[i-1]["subject_id"],"Original adjacent diploid pair")
        else:names.append(row["subject_id"])
    need(len(set(names))==PEOPLE and inputs["baseline.sample"]==
      ("ID_1 ID_2 missing\n0 0 0\n"+"".join(n+" "+n+" 0\n" for n in names)).encode(),"Exact sample text")
    need(inputs["sample_ages.generations"]=="".join(format(a,".17g")+"\n" for a in ages).encode(),
      "Exact fractional source .17g age file")
    identities=json.loads(inputs["frozen_site_identities.json"])
    audits=json.loads(inputs["polarization_equivalence.json"])
    lines=inputs["baseline.haps"].decode("ascii").splitlines()
    need(len(lines)==len(identities)==len(audits)==S,"Every sealed qualified site")
    rows=[];source_vectors=hashlib.sha256();monomorphic=[]
    for index,(line,identity,audit) in enumerate(zip(lines,identities,audits)):
        fields=line.split();chrom,pos,ref,alt=identity;local=pos-BEGIN
        need(len(fields)==(5+H) and chrom==chromosome and 1<=local<=(END-BEGIN)
          and ref in "ACGT" and alt in "ACGT" and ref!=alt,"Fixed chromosome SNP identity")
        aa,da=fields[3:5]
        need(all(v in ("0","1") for v in fields[5:]),"Binary original phased haps")
        codes=[int(v) for v in fields[5:]];original=audit["source_GT_haplotype_codes"]
        need(len(original)==H and all(v in (0,1) for v in original),"Original36 source codes")
        nt="".join((ref,alt)[v] for v in original)
        need(fields[:3]==[chromosome,variant_id_v1(chromosome,pos,ref,alt),str(local)]
          and aa in (ref,alt) and da in (ref,alt) and aa!=da,"Original AA/DA identity and position")
        need(nt=="".join((aa,da)[v] for v in codes)
          and audit["source_POS_bp1"]==pos and audit["haps_POS_bp1"]==local
          and audit["regional_bp0"]==local-1 and audit["source_REF"]==ref and audit["source_ALT"]==alt
          and audit["native_haps_codes"]==codes and audit["nucleotide_equivalence_sha256"]==sha(nt.encode())
          and audit["exact_nucleotide_and_phase_order_equivalence"] is True,
          "Every original biological nucleotide and phase")
        mono=len(set(codes))==1
        need(audit["monomorphic_among_panel"]==mono,"Original monomorphic identity")
        source_vectors.update((str(pos)+":"+nt+"\n").encode())
        if mono:monomorphic.append(pos)
        rows.append(dict(index0=index,name=fields[1],local_bp1=local,local_bp0=local-1,
          absolute_POS_bp1=pos,native_AA=aa,native_DA=da,biological_nucleotides=nt,monomorphic=mono))
    need(all(a["local_bp1"]<b["local_bp1"] for a,b in zip(rows,rows[1:])),"Strict unique original SNP order")
    need(len(monomorphic)==K and source_vectors.hexdigest()==readback["original_nucleotide_vectors_sha256"]
      and sha(json.dumps(monomorphic,separators=(",",":")).encode())==readback["monomorphic_source_POS_bp1_sha256"],
      "Independent readback vector and all sealed monomorphic identity digests")
    return dict(protocol_ref=PROTOCOL_REF,chromosome=chromosome,counts=dict(counts),rows=rows,ages=ages,names=names,roster=roster,monomorphic_absolute_POS=monomorphic,
      source_vectors_sha256=source_vectors.hexdigest())

def _verify_native_strict_v1(inputs,observed,anc_raw,mut_raw,tree_sequence,*,chromosome,counts):
    validate_chromosome_v1(chromosome);validate_counts_v1(counts,graph=True);S,K=counts["sites"],counts["constants"]
    need(observed["protocol_ref"]==PROTOCOL_REF and observed["chromosome"]==chromosome and observed["counts"]==counts,"Exact new observations/dimensions")
    # Accepts a loaded tree sequence object; does not read files or invoke native.
    details=dict(schema="H056_H36_strict_native_facts_v1",observed_GT_only=True,expected_sites=S,haplotypes=H,
                 biological_or_accuracy_claim=False,failures=[],all_original_sites_preserved=None,
                 qualifier_modified_native_genotypes_or_times=False)
    def fact(ok,name):
        details[name]=bool(ok)
        if not ok:details["failures"].append(name)
    try:
        tokens,anc_ages,anc_trees=parse_anc_v1(anc_raw,counts=counts)
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
        fact(len(muts)==S,"native_mut_exact_row_count")
        mut_identities=[]
        for index,row in enumerate(muts):
            mut_identities.append(dict(index0=int(row["snp"]),position=int(row["pos_of_snp"]),name=row["rs-id"],
              alleles=row["ancestral_allele/alternative_allele"],branches=row["branch_indices"],flipped=row["is_flipped"],
              is_not_mapping=row["is_not_mapping"],tree_index=int(row["tree_index"])))
        fact(len(muts)==S and all(r["index0"]==i and r["position"]==e["local_bp1"] and r["name"]==e["name"]
             and r["alleles"]==e["native_AA"]+"/"+e["native_DA"] for i,(r,e) in enumerate(zip(mut_identities,expected))),
             "native_mut_every_identity_position_allele_order")
        details["native_mut_identity_sha256"]=sha(encoded(mut_identities))
        details["native_mut_identity_facts"]=mut_identities
        details["native_mut_flipped_rows0"]=[i for i,r in enumerate(mut_identities) if r["flipped"]=="1"]
        details["native_mut_multi_branch_rows0"]=[i for i,r in enumerate(mut_identities) if len(r["branches"].split())>1]
        details["native_mut_constant_branch_facts"]=[mut_identities[i] for i,e in enumerate(expected) if e["monomorphic"] and i<len(mut_identities)]
        ts=tree_sequence
        samples=list(map(int,ts.samples()))
        fact(samples==list(range(H)) and ts.num_samples==H,"sample_node_insertion_and_count")
        fact(all(float(ts.node(node).time)==anc_ages[i] for i,node in enumerate(samples)),"exact_parsed_anc_to_binary_sample_time")
        fact(ts.num_sites==S,"tree_sequence_exact_site_count")
        positions=[float(x) for x in ts.tables.sites.position]
        fact(positions==[e["local_bp1"] for e in expected],"all_exact_local_bp1_sites_no_jitter")
        fact(float(ts.sequence_length)==expected[-1]["local_bp1"]+1,"actual_last_site_plus_one_raw_length")
        vector_errors=[];constant_facts=[];vector_facts=[];hasher=hashlib.sha256()
        for index,variant in enumerate(ts.variants(samples=samples,isolated_as_missing=True)):
            if index>=S:
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
        fact(not vector_errors and ts.num_sites==S,"all_biological_nucleotide_and_phase_calls")
        details["all_site_vector_sha256"]=hasher.hexdigest()
        details["vector_errors"]=vector_errors
        details["all_native_vector_facts"]=vector_facts
        details["all_constant_vector_facts"]=constant_facts
        details["native_anc_float32_nonpositive_branches"]=[dict(tree_index=i,branches=t["nonpositive_float32_branches"]) for i,t in enumerate(anc_trees) if t["nonpositive_float32_branches"]]
        details["native_converter_chronology_adjustment_triggered"]=bool(details["native_anc_float32_nonpositive_branches"])
        fact(not details["native_anc_float32_nonpositive_branches"],"no_converter_chronology_adjustment_trigger")
        # The converter createsH-1 internal nodes per native tree. Compare exact
        # float-derived coordinates, not a broad chronology tolerance.
        expected_nodes=H+(H-1)*len(anc_trees)
        fact(ts.num_nodes==expected_nodes,"native_tree_local_internal_node_layout")
        coordinate_errors=[]
        if ts.num_nodes==expected_nodes:
            for k,t in enumerate(anc_trees):
                for node in range(H,(2*H-1)):
                    ident=node+k*(H-1)
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
              all_panel_connected=tr.num_roots==1 and tr.num_samples(tr.root)==H))
        details["raw_tree_interval_facts"]=intervals
        fact(all(t["all_panel_connected"] for t in intervals),"all_raw_intervals_carry_all_panel_samples")
        fact(all(ts.at(e["local_bp1"]).num_roots==1 and ts.at(e["local_bp1"]).num_samples(ts.at(e["local_bp1"]).root)==H for e in expected),"every_original_site_has_full_tree_support")
        first=expected[0]["local_bp1"];last=expected[-1]["local_bp1"]
        details["coverage"]=dict(raw_native_domain=[0,float(ts.sequence_length)],
          requested_regional_bp0=[0,(END-BEGIN)],genuine_native_domain=[first,last+1],
          genuine_regional_bp0=[first-1,last],genuine_absolute_bp0=[BEGIN+first-1,BEGIN+last],
          covered_bp=last-first+1,unrepresented_left_bp=first-1,unrepresented_right_bp=(END-BEGIN)-last,
          affine_join="regional_bp0=native_position-1;absolute_bp0=20000000+regional_bp0",
          full_10000000bp_coverage_claim=False,no_flanking_tree_extrapolation=True,
          native_output_not_clipped_or_modified=True)
        starts=[t["start_snp0"] for t in anc_trees]
        fact(bool(starts) and starts[0]==0 and all(a<b for a,b in zip(starts,starts[1:]))
          and starts[-1]<S,"strict_native_tree_SNP_starts")
        edge_expected=[]
        if details["strict_native_tree_SNP_starts"]:
            for k,t in enumerate(anc_trees):
                stop=starts[k+1] if k+1<len(starts) else S
                left=0. if starts[k]==0 else .5*(expected[starts[k]-1]["local_bp1"]+expected[starts[k]]["local_bp1"])
                right=float(last+1) if stop==S else .5*(expected[stop-1]["local_bp1"]+expected[stop]["local_bp1"])
                for i in range(starts[k],stop):
                    fact(mut_identities[i]["tree_index"]==k,"mut_tree_index_"+str(i))
                for child,parent in enumerate(t["parents"][:-1]):
                    edge_expected.append((left,right,parent+(H-1)*k,child if child<H else child+(H-1)*k))
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
            "all_biological_nucleotide_and_phase_calls","every_original_site_has_full_tree_support"))
    except BaseException as exc:
        details["failures"].append("verification_exception")
        details["verification_error"]=dict(type=type(exc).__name__,message=str(exc)[:2000])
    details["status"]="verified_original_GT_pre_size_native_genealogy" if not details["failures"] else "unsupported_native_facts_preserved"
    details["model_training_or_application_performed"]=False
    details["ancestry_truth_available"]=False
    details["main_post_five_size_condition_executed"]=False
    return details

def verify_graph_only_v1(observed,anc_raw,tree_sequence,*,chromosome,counts):
    """Independent structural proof. Never reads native variants/mutations/populations.
    This may pass when actual strict native GT export fails; it does not repair
    that failure. Root must additionally bind new map/readback/wait authority.
    """
    validate_chromosome_v1(chromosome);validate_counts_v1(counts,graph=True)
    need(observed["protocol_ref"]==PROTOCOL_REF and observed["chromosome"]==chromosome and observed["counts"]==counts,"Exact independently sealed H056 observations/counts")
    S=counts["sites"];expected=observed["rows"];ts=tree_sequence
    result=dict(schema="H056_H36_graph_only_native_facts_v1",protocol_ref=PROTOCOL_REF,
        chromosome=chromosome,counts=dict(counts),failures=[],graph_only_ready=False,
        native_variants_mutations_populations_or_ancestral_alleles_used=False,
        immutable_original_GT_features=True,new_root_map_serialized_readback_wait_and_drain_proofs_required=True)
    def fact(ok,name):
        result[name]=bool(ok)
        if not ok:result["failures"].append(name)
    try:
        need(len(expected)==S and len(observed["ages"])==H and len(observed["roster"])==H,"Complete original observed sites/people")
        tokens,anc_ages,trees=parse_anc_v1(anc_raw,counts=counts)
        fact(all(tokens[i]==rounded_age_token(a) for i,a in enumerate(observed["ages"])),"exact_six_decimal_age_tokens")
        f32=lambda a:struct.unpack("f",struct.pack("f",a))[0]
        fact(all(f32(a)==f32(b) for a,b in zip(observed["ages"],anc_ages)),"source_native_tip_exact_FP32_bank_equality")
        samples=list(map(int,ts.samples()))
        fact(samples==list(range(H)) and ts.num_samples==H,"sample_node_insertion_and_count")
        need(len(samples)==H,"All H36 sample nodes")
        fact(all(float(ts.node(n).time)==anc_ages[i] for i,n in enumerate(samples)),"exact_parsed_anc_to_binary_sample_time")
        fact(ts.num_nodes==counts["native_nodes"] and ts.num_edges==counts["native_edges"] and ts.num_trees==counts["native_trees"],"all_independently_sealed_native_dimensions")
        expected_nodes=H+(H-1)*len(trees)
        fact(ts.num_nodes==expected_nodes,"native_tree_local_internal_node_layout")
        coordinate_errors=[]
        if ts.num_nodes==expected_nodes:
            for k,t in enumerate(trees):
                for node in range(H,2*H-1):
                    ident=node+k*(H-1)
                    if float(ts.node(ident).time)!=t["coordinates32"][node]:
                        coordinate_errors.append(dict(tree_index=k,native_node=node,exported_node=ident))
        else:coordinate_errors.append(dict(reason="unexpected_internal_node_layout"))
        result["internal_time_mismatch_facts"]=coordinate_errors
        fact(not coordinate_errors,"no_unreviewed_internal_time_changes")
        fact(not any(t["nonpositive_float32_branches"] for t in trees),"no_converter_chronology_adjustment_trigger")
        edges=[];bad=[]
        for e in ts.edges():
            row=(float(e.left),float(e.right),int(e.parent),int(e.child));edges.append(row)
            pt=float(ts.node(e.parent).time);ct=float(ts.node(e.child).time)
            if not(math.isfinite(e.left) and math.isfinite(e.right) and 0<=e.left<e.right<=ts.sequence_length and math.isfinite(pt) and math.isfinite(ct) and pt>ct):bad.append(int(e.id))
        fact(not bad,"finite_strict_native_edge_chronology");result["invalid_edges"]=bad
        first=expected[0]["local_bp1"];last=expected[-1]["local_bp1"]
        fact(float(ts.sequence_length)==last+1,"actual_last_source_site_plus_one_raw_length")
        starts=[t["start_snp0"] for t in trees]
        start_ok=bool(starts) and starts[0]==0 and all(a<b for a,b in zip(starts,starts[1:])) and starts[-1]<S
        fact(start_ok,"strict_native_tree_SNP_starts")
        expected_edges=[]
        if start_ok:
            for k,t in enumerate(trees):
                stop=starts[k+1] if k+1<len(starts) else S
                left=0. if starts[k]==0 else .5*(expected[starts[k]-1]["local_bp1"]+expected[starts[k]]["local_bp1"])
                right=float(last+1) if stop==S else .5*(expected[stop-1]["local_bp1"]+expected[stop]["local_bp1"])
                for child,parent in enumerate(t["parents"][:-1]):
                    expected_edges.append((left,right,parent+(H-1)*k,child if child<H else child+(H-1)*k))
        fact(start_ok and Counter(expected_edges)==Counter(edges),"exact_native_marginal_topology_and_edge_support")
        intervals=[];vertex_occurrences=0
        for tr in ts.trees():
            connected=tr.num_roots==1 and tr.num_samples(tr.root)==H
            fact(connected,"complete_connected_native_hold_"+str(len(intervals)))
            reached=set()
            for sample in samples:
                node=sample;steps=0
                while node!=-1:
                    reached.add(node);steps+=1;need(steps<=ts.num_nodes,"Acyclic sampled ancestor union")
                    node=int(tr.parent(node))
            vertex_occurrences+=len(reached)
            left=max(first-1,0.,float(tr.interval.left)-1.)
            right=min(float(last),float(END-BEGIN),float(tr.interval.right)-1.)
            need(left<right,"No empty/dropped hold in genuine native support")
            intervals.append([left,right])
        fact(vertex_occurrences==counts["vertex_occurrences"],"exact_sampled_ancestor_vertex_occurrences")
        fact(len(intervals)==counts["native_trees"] and intervals[0][0]==first-1 and intervals[-1][1]==last and all(a[1]==b[0] for a,b in zip(intervals,intervals[1:])),"complete_contiguous_genuine_support")
        fact(all(ts.at(e["local_bp1"]).num_roots==1 and ts.at(e["local_bp1"]).num_samples(ts.at(e["local_bp1"]).root)==H for e in expected),"every_original_site_has_full_tree_support")
        result["coverage"]=dict(raw_native_domain=[0,float(ts.sequence_length)],requested_regional_bp0=[0,END-BEGIN],
            genuine_native_domain=[first,last+1],genuine_regional_bp0=[first-1,last],genuine_absolute_bp0=[BEGIN+first-1,BEGIN+last],
            intervals_regional_bp0=intervals,covered_bp=last-first+1,unrepresented_left_bp=first-1,unrepresented_right_bp=END-BEGIN-last,
            coordinate_transform="native_minus1",full_10000000bp_coverage_claim=False,no_flanking_tree_extrapolation=True,native_output_not_clipped_or_modified=True)
        result["native_sample_haplotype_date_mapping"]=[dict(native_sample_node_id=node,subject_id=row["subject_id"],haplotype=row["haplotype"],
            source_sample_index=i,role=row["role"],source_age_generations=observed["ages"][i],anc_age_generations=anc_ages[i],
            native_node_time=float(ts.node(node).time)) for i,(node,row) in enumerate(zip(samples,observed["roster"]))]
        result["actual_vertex_occurrences"]=vertex_occurrences
        result["graph_only_ready"]=not result["failures"]
    except Exception as exc:
        result["failures"].append("graph_verification_exception")
        result["verification_error"]=dict(type=type(exc).__name__,message=str(exc)[:2000])
    result["status"]="new_graph_only_structural_proofs_passed" if result["graph_only_ready"] else "new_graph_only_structural_proofs_failed"
    return result

def verify_native_v1(inputs,observed,anc_raw,mut_raw,tree_sequence,*,chromosome,counts):
    """Actual strict GT facts and independent graph-only facts; no forced pass."""
    graph=verify_graph_only_v1(observed,anc_raw,tree_sequence,chromosome=chromosome,counts=counts)
    strict=_verify_native_strict_v1(inputs,observed,anc_raw,mut_raw,tree_sequence,chromosome=chromosome,counts=counts)
    result=dict(strict)
    result.update(schema="H056_H36_actual_native_facts_v1",protocol_ref=PROTOCOL_REF,chromosome=chromosome,counts=dict(counts),
        strict_GT_actual_status=strict["status"],strict_GT_actual_passed=not strict["failures"],graph_only_ready=graph["graph_only_ready"],
        graph_only=graph,coverage=graph.get("coverage"),native_sample_haplotype_date_mapping=graph.get("native_sample_haplotype_date_mapping",[]),
        no_historical_failure_or_recovery_reused=True,source_GT_features_required=True)
    return result


def parse_headers_v1(global_raw,chunks,*,counts,max_chunks):
    checked_counts_v1(counts);S=counts["sites"]
    need(type(max_chunks)is int and 1<=max_chunks<=S,"Root count-derived native chunk bound")
    """Pure bounded native x86_64 header decoding; supports overlapping chunks."""
    need(isinstance(global_raw,bytes) and 28<=len(global_raw)<=(20+8*max_chunks),"Bounded global native header")
    n,l,c=struct.unpack("<iii",global_raw[:12]);memory=struct.unpack("<d",global_raw[12:20])[0]
    need(n==H and l==S and 1<=c<=max_chunks and len(global_raw)==20+8*c
      and math.isfinite(memory) and 0<memory<8.,"Exact H36/sealedS and bounded chunk count")
    starts=list(struct.unpack("<"+"i"*c,global_raw[20:20+4*c]))
    ends=list(struct.unpack("<"+"i"*c,global_raw[20+4*c:]))
    need(starts[0]==0 and ends[-1]==S and all(0<=a<b<=S for a,b in zip(starts,ends))
      and all(a<b for a,b in zip(starts,starts[1:])) and all(a<b for a,b in zip(ends,ends[1:]))
      and all(starts[i]<=ends[i-1] for i in range(1,c)),"Complete original overlapping SNP support")
    need(set(chunks)==set(range(c)),"Exactly bounded original chunk header set")
    result=[]
    for k in range(c):
        raw=chunks[k];need(isinstance(raw,bytes) and 20<=len(raw)<=(12+4*(S+1)),"Bounded chunk metadata bytes")
        cn,cl,b=struct.unpack("<iii",raw[:12])
        need(cn==H and cl==ends[k]-starts[k] and 2<=b<=min(cl+1,(S+1))
          and len(raw)==12+4*b,"Exact36 haplotypes and complete section header")
        bounds=list(struct.unpack("<"+"i"*b,raw[12:]))
        need(bounds[0]==0 and bounds[-1]==cl and all(a<z for a,z in zip(bounds,bounds[1:])),
          "Strict complete native section boundaries")
        result.append(dict(chunk_index=k,sites=cl,first_section=0,last_section=b-2,
          boundaries=bounds,header_sha256=sha(raw),header_hex=raw.hex()))
    return dict(schema="H056_H36_native_headers_v1",haplotypes=n,sites=l,num_chunks=c,
      native_memory_GB=memory,global_starts=starts,global_ends=ends,
      global_header_sha256=sha(global_raw),global_header_hex=global_raw.hex(),chunks=result)

def cli_stages_v1(binaries,inputs,profile,headers=None):
    validate_profile_v1(profile)
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
