"""Tiny deterministic synthetic fixture and independent native readback.
No native command, process, guard or real observation I/O is defined here.
tskit is imported only inside verify_native(), after root runtime pin checks.
"""
import csv
import hashlib
import io
import json
import math
import struct
from decimal import Decimal, ROUND_HALF_EVEN


def need(ok, text):
    if not ok:
        raise ValueError(text)


def encoded(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def make_fixture(profile):
    need(profile["sites"] == 512 and profile["haplotypes"] == 40 and profile["subjects"] == 20, "Fixed tiny40/512 profile")
    names = ["SYNTH_"+str(i).zfill(2) for i in range(20)]
    ages = [float(Decimal(profile["sample_age_calBP_base"])+i*Decimal(profile["sample_age_calBP_step"]))/28.0 for i in range(20) for _ in (0,1)]
    rows, haps = [], []
    clades = [(0,1),(0,3),(0,4),(4,8),(0,8),(8,16),(0,16),(16,24),(24,32),(16,32),(0,32),(32,40)]
    for i in range(512):
        pos = 1+100*i
        if str(i) in profile["constant_rows0"]:
            vector = [int(profile["constant_rows0"][str(i)] == "all_derived")]*40
        else:
            left,right = clades[i % len(clades)]
            vector = [int(left <= j < right) for j in range(40)]
        ref,alt = (("A","C"),("G","T"))[i%2]
        flip = i%7 in (1,3)
        aa,da = (alt,ref) if flip else (ref,alt)
        original_codes = [v ^ int(flip) for v in vector]
        original_nt = "".join((ref,alt)[v] for v in original_codes)
        need(original_nt == "".join((aa,da)[v] for v in vector), "Synthetic phase/AA-DA equivalence")
        name = "SYNTH_22_"+str(pos)+"_"+ref+"_"+alt
        haps.append(" ".join(["22",name,str(pos),aa,da]+[str(v) for v in vector])+"\n")
        rows.append(dict(index0=i,name=name,local_bp1=pos,local_bp0=pos-1,absolute_POS_bp1=20_000_000+pos,
                         original_REF=ref,original_ALT=alt,original_GT_codes=original_codes,
                         original_GT=["%d|%d"%tuple(original_codes[j:j+2]) for j in range(0,40,2)],
                         native_AA=aa,native_DA=da,native_codes=vector,AA_is_ALT=flip,
                         biological_nucleotides=original_nt,monomorphic=len(set(vector))==1))
    need(sum(x["monomorphic"] for x in rows) == 6, "Six explicit boundary/interior constant rows")
    truth=dict(schema="H018_Relate_synthetic_truth_v1",synthetic_only=True,names=names,ages=ages,rows=rows,
               constant_ancestral_rows0=[0,256,510],constant_derived_rows0=[1,257,511],no_query_truth_or_real_GT=True)
    payloads={"input.haps":"".join(haps).encode(),
      "input.sample":("ID_1 ID_2 missing\n0 0 0\n"+"".join(n+" "+n+" 0\n" for n in names)).encode(),
      "sample_ages.generations":"".join(format(v,".17g")+"\n" for v in ages).encode(),
      "input.map":b"Position(bp) Rate(cM/Mb) Map(cM)\n1 1 0\n10001 2 0.01\n30001 0.5 0.05\n60001 0.5 0.065\n",
      "fixture_truth.json":encoded(truth)}
    need(sum(map(len,payloads.values())) < 1_000_000, "Tiny fixture byte ceiling")
    return payloads,truth


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
            else:value=max(f32(coord32(c)+f32(lengths[c])) for c in children[node])
            coordinates32[node]=value
            return value
        coord32(78)
        nonpositive=[child for child,parent in enumerate(parents[:-1]) if coordinates32[parent] <= coordinates32[child]]
        trees.append(dict(start_snp0=int(at),parents=parents,lengths=lengths,coordinates32=coordinates32,
                          nonpositive_float32_branches=nonpositive))
    return first[2:],ages,trees


def verify_native(inputs, truth, anc_raw, mut_raw, tree_sequence):
    # Accepts a loaded tree sequence object; does not read files or invoke native.
    details=dict(schema="H018_Relate_synthetic_native_facts_v1",synthetic_only=True,expected_sites=512,haplotypes=40,
                 biological_or_accuracy_claim=False,failures=[],all_fixture_sites_preserved=None,
                 qualifier_modified_native_genotypes_or_times=False)
    def fact(ok,name):
        details[name]=bool(ok)
        if not ok:details["failures"].append(name)
    try:
        tokens,anc_ages,anc_trees=parse_anc(anc_raw)
        details["native_tree_count"]=len(anc_trees)
        fact(all(tokens[i]==rounded_age_token(a) for i,a in enumerate(truth["ages"])),"exact_six_decimal_age_tokens")
        age_errors=[];bounds=[]
        for i,(a,b) in enumerate(zip(truth["ages"],anc_ages)):
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
        expected=truth["rows"]
        fact(len(muts)==512,"native_mut_exact_row_count")
        mut_identities=[]
        for index,row in enumerate(muts):
            mut_identities.append(dict(index0=int(row["snp"]),position=int(row["pos_of_snp"]),name=row["rs-id"],
              alleles=row["ancestral_allele/alternative_allele"],branches=row["branch_indices"],flipped=row["is_flipped"],
              is_not_mapping=row["is_not_mapping"],tree_index=int(row["tree_index"])))
        fact(len(muts)==512 and all(r["index0"]==i and r["position"]==e["local_bp1"] and r["name"]==e["name"]
             and r["alleles"]==e["native_AA"]+"/"+e["native_DA"] for i,(r,e) in enumerate(zip(mut_identities,expected))),
             "native_mut_every_identity_position_allele_order")
        details["native_mut_identity_sha256"]=sha(encoded(mut_identities))
        details["native_mut_identity_facts"]=mut_identities
        details["native_mut_flipped_rows0"]=[i for i,r in enumerate(mut_identities) if r["flipped"]=="1"]
        details["native_mut_multi_branch_rows0"]=[i for i,r in enumerate(mut_identities) if len(r["branches"].split())>1]
        details["native_mut_constant_branch_facts"]=[mut_identities[i] for i in [0,1,256,257,510,511] if i<len(mut_identities)]
        ts=tree_sequence
        samples=list(map(int,ts.samples()))
        fact(samples==list(range(40)) and ts.num_samples==40,"sample_node_insertion_and_count")
        fact(all(float(ts.node(node).time)==anc_ages[i] for i,node in enumerate(samples)),"exact_parsed_anc_to_binary_sample_time")
        fact(ts.num_sites==512,"tree_sequence_exact_site_count")
        positions=[float(x) for x in ts.tables.sites.position]
        fact(positions==[e["local_bp1"] for e in expected],"all_exact_local_bp1_sites_no_jitter")
        fact(float(ts.sequence_length)==expected[-1]["local_bp1"]+1,"actual_last_site_plus_one_raw_length")
        vector_errors=[];constant_facts=[];vector_facts=[];hasher=hashlib.sha256()
        for index,variant in enumerate(ts.variants(samples=samples,isolated_as_missing=True)):
            if index>=512:
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
        fact(not vector_errors and ts.num_sites==512,"all20480_biological_nucleotide_and_phase_calls")
        details["all_site_vector_sha256"]=hasher.hexdigest()
        details["vector_errors"]=vector_errors
        details["all512_native_vector_facts"]=vector_facts
        details["six_constant_vector_facts"]=constant_facts
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
        fact(all(ts.at(e["local_bp1"]).num_roots==1 and ts.at(e["local_bp1"]).num_samples(ts.at(e["local_bp1"]).root)==40 for e in expected),"every_fixture_site_has_full_tree_support")
        details["coverage"]=dict(raw_native_domain=[0,float(ts.sequence_length)],requested_raw_domain=[1,60001],
          affine_join="local_bp0=native_site_position-1=absolute_POS_bp1-20000001",
          artificial_left_span_bp=1,unsupported_terminal_span_bp=60001-float(ts.sequence_length),
          full_declared_60000bp_coverage=False,no_flanking_tree_extrapolation=True)
        details["all_fixture_sites_preserved"]=all(details[key] for key in (
            "native_mut_exact_row_count","native_mut_every_identity_position_allele_order",
            "tree_sequence_exact_site_count","all_exact_local_bp1_sites_no_jitter",
            "all20480_biological_nucleotide_and_phase_calls","every_fixture_site_has_full_tree_support"))
    except BaseException as exc:
        details["failures"].append("verification_exception")
        details["verification_error"]=dict(type=type(exc).__name__,message=str(exc)[:2000])
    details["status"]="qualified_tiny_synthetic_native_path_only" if not details["failures"] else "unsupported_native_facts_preserved"
    details["observed2074_baseline_authorized"]=False
    return details
