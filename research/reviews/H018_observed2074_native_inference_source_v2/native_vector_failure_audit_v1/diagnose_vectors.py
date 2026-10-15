"""Read-only stdlib diagnosis of retained H018 observed native vector failure."""
from pathlib import Path
import csv,io,json,hashlib,collections
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/H018_observed2074_native_inference_source_v2/native_vector_failure_audit_v1"
P=ROOT/"research/reviews/H018_observed2074_native_inference_source_v2"
O=ROOT/"data/H018_observed2074_native_Relate_pre_size_v2"
B=ROOT/"data/H018_R16_observed_bridge_v2"
checks=[]
def check(name,ok):
    if not ok:raise AssertionError(name)
    checks.append(name)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(p):return json.loads(p.read_text())
def ref(p):return dict(path=str(p.relative_to(ROOT)),sha256=sha(p.read_bytes()),bytes=p.stat().st_size)
def save(p,a):p.write_text(json.dumps(a,sort_keys=True,indent=2,allow_nan=False)+"\n")
facts=read(O/"native_facts.json")
check("actual observed facts exact recorded hash",sha((O/"native_facts.json").read_bytes())=="335d8aa9e5446a0c013f95d8fa9d7b76287851e53cb2a372539e6084112a66ec")
check("only failed fact is full biological phase nucleotide preservation",facts["failures"]==["all82960_biological_nucleotide_and_phase_calls"])
manifest=read(P/"source_manifest.json")
for name in ("baseline.haps","polarization_equivalence.json","frozen_site_identities.json"):
    check("frozen observed input hash "+name,sha((B/name).read_bytes())==manifest["data_refs"][name]["sha256"])
haps=[s.split() for s in (B/"baseline.haps").read_text().splitlines()]
audits=read(B/"polarization_equivalence.json");identities=read(B/"frozen_site_identities.json")
check("all2074 observed input rows",len(haps)==len(audits)==len(identities)==2074)
native=(O/"work/native.anc").read_text().splitlines()
check("exact40 ancestral leaves and355 retained trees",native[0].split()[:2]==["NUM_HAPLOTYPES","40"] and native[1]=="NUM_TREES 355" and len(native)==357)
descendants=[];starts=[]
for line in native[2:]:
    at,tail=line.split(": ",1);fields=tail.strip().rstrip(")").split(") ")
    parents=[int(x.split(":(",1)[0]) for x in fields]
    check("79node binary tree "+at,len(parents)==79 and parents[78]==-1)
    children=[[] for _ in range(79)]
    for child,parent in enumerate(parents[:-1]):children[parent].append(child)
    memo={}
    def leaves(n,active=frozenset()):
        if n in memo:return memo[n]
        if n in active:raise ValueError("ancestry cycle")
        if n<40:value={n}
        else:
            if len(children[n])!=2:raise ValueError("nonbinary native topology")
            value=set().union(*(leaves(c,active|{n}) for c in children[n]))
        memo[n]=value
        return value
    check("full40 native root "+at,leaves(78)==set(range(40)))
    descendants.append([leaves(n) for n in range(79)]);starts.append(int(at))
muts=list(csv.DictReader(io.StringIO((O/"work/native.mut").read_text()),delimiter=";"))
vectors=facts["all2074_native_vector_facts"]
check("full2074 native rows and vectors",len(muts)==len(vectors)==2074)
rows=[];converter_errors=[];native_errors=[];export_errors=[];input_hasher=hashlib.sha256()
native_semantic_hasher=hashlib.sha256();export_hasher=hashlib.sha256()
for i,(m,h,v,a,identity) in enumerate(zip(muts,haps,vectors,audits,identities)):
    aa,da=h[3:5];pos=int(h[2]);refbase,alt=identity[2:4];absolute=identity[1]
    expected="".join((aa,da)[int(x)] for x in h[5:])
    source="".join((refbase,alt)[int(x)] for x in a["source_GT_haplotype_codes"])
    check("original biological nucleotide and phase vector "+str(i),len(expected)==40 and source==expected and sha(source.encode())==a["nucleotide_equivalence_sha256"])
    check("native exact site allele identity "+str(i),int(m["snp"])==i and int(m["pos_of_snp"])==pos and m["rs-id"]==h[1] and m["ancestral_allele/alternative_allele"]==aa+"/"+da)
    ti=int(m["tree_index"])
    check("native mutation assigned ancestral tree "+str(i),starts[ti]<=i and (ti==len(starts)-1 or i<starts[ti+1]))
    branch=list(map(int,m["branch_indices"].split()))
    union=set().union(*(descendants[ti][b] for b in branch))
    converter_carriers=union if len(branch)==1 else set()
    predicted_export="".join(da if hindex in converter_carriers else aa for hindex in range(40))
    carriers=set(range(40))-union if m["is_flipped"]=="1" else union
    semantic="".join(da if hindex in carriers else aa for hindex in range(40))
    actual=v["biological_nucleotides"]
    if predicted_export!=actual:converter_errors.append(i)
    em=[hindex for hindex,(x,y) in enumerate(zip(expected,actual)) if x!=y]
    nm=[hindex for hindex,(x,y) in enumerate(zip(expected,semantic)) if x!=y]
    if em:export_errors.append(i)
    if nm:native_errors.append(i)
    mono=len(set(expected))==1
    pairs_changed=sum(sorted(expected[hindex:hindex+2])!=sorted(actual[hindex:hindex+2]) for hindex in range(0,40,2))
    rows.append(dict(index0=i,absolute_POS_bp1=absolute,local_POS_bp1=pos,monomorphic=mono,
      native_AA=aa,native_DA=da,tree_index=ti,branches=m["branch_indices"],branch_count=len(branch),
      flipped=int(m["is_flipped"]),is_not_mapping=int(m["is_not_mapping"]),
      export_hamming=len(em),native_branch_semantic_hamming=len(nm),
      export_mismatch_haplotype_indices=" ".join(map(str,em)),native_semantic_mismatch_haplotype_indices=" ".join(map(str,nm)),
      export_changed_unphased_diploid_genotypes=pairs_changed,
      reference_export_mismatch_cells=sum(j<16 for j in em),query_export_mismatch_cells=sum(j>=16 for j in em)))
    input_hasher.update((str(absolute)+":"+expected+"\n").encode())
    native_semantic_hasher.update((str(pos)+":"+semantic+"\n").encode())
    export_hasher.update((str(pos)+":"+actual+"\n").encode())
check("independent converter source rule predicts all2074 recorded export vectors",converter_errors==[])
check("exact reported198 export site identities",export_errors==[x["index0"] for x in facts["vector_errors"]])
check("all277 constants exact",sum(r["monomorphic"] for r in rows)==277 and all(r["export_hamming"]==0 for r in rows if r["monomorphic"]))
check("all692 exported mismatch cells",sum(r["export_hamming"] for r in rows)==692)
check("180 native singlebranch semantic mismatch cells",len(native_errors)==180 and sum(r["native_branch_semantic_hamming"] for r in rows)==180 and all(r["branch_count"]==1 and r["native_branch_semantic_hamming"]==1 for r in rows if r["native_branch_semantic_hamming"]))
check("all13 full multibranch semantic vectors exact",sum(r["branch_count"]>1 for r in rows)==13 and all(r["native_branch_semantic_hamming"]==0 for r in rows if r["branch_count"]>1))
check("export vector digest agrees",export_hasher.hexdigest()==facts["all_site_vector_sha256"])
readback=read(ROOT/"research/reviews/H018_QC_boolean_serialization_repair_source_v1/worker_aux_v2/serialized_readback_root_v2.json")
check("independent original input vector digest agrees",input_hasher.hexdigest()==readback["original_nucleotide_vectors_sha256"])
source_cpp=ROOT/"research/reviews/relate_ancestral_identity_source_audit_v1/ConvertToTreeSequence.pinned.cpp"
builder=ROOT/"research/reviews/relate_ancestral_identity_source_audit_v1/anc_builder.cpp"
mutcpp=ROOT/"research/reviews/H018_native_Relate_support_readiness_v1/sources/mutations.cpp"
check("pinned converter source exact",sha(source_cpp.read_bytes())=="452298c9a5ad60159d9f86dd402f71798e8a88108f2bb217fcb941d28778bfc4")
check("pinned builder source exact",sha(builder.read_bytes())=="2b929b31de232d5ab0bde0fc0346b76a07822d1f4de23d3dd8872bd4933f53b5")
bs=builder.read_text()
check("author source approximate threshold is1 at40haplotypes","thr = (int) (0.03 * N);" in bs and int(.03*40)==1 and "report.min <= thr" in bs)
categories=[]
for kind in sorted({(r["monomorphic"],r["branch_count"],r["flipped"],r["is_not_mapping"]) for r in rows}):
    selected=[r for r in rows if (r["monomorphic"],r["branch_count"],r["flipped"],r["is_not_mapping"])==kind]
    categories.append(dict(monomorphic=kind[0],branch_count=kind[1],flipped=kind[2],is_not_mapping=kind[3],
      sites=len(selected),export_mismatch_sites=sum(r["export_hamming"]>0 for r in selected),export_mismatch_cells=sum(r["export_hamming"] for r in selected),
      native_semantic_mismatch_sites=sum(r["native_branch_semantic_hamming"]>0 for r in selected),native_semantic_mismatch_cells=sum(r["native_branch_semantic_hamming"] for r in selected)))
result=dict(schema="H018_observed2074_native_vector_failure_diagnosis_v1",
 strict_original_gate_status="failed_no_promotion",site_count=2074,haplotype_count=40,nucleotide_cells=82960,
 export_mismatch_sites=len(export_errors),export_mismatch_nucleotide_cells=sum(r["export_hamming"] for r in rows),
 export_exact_sites=2074-len(export_errors),export_missing_sites=sum(v["missing"] for v in vectors),
 monomorphic_sites=277,monomorphic_export_mismatch_sites=0,polymorphic_sites=1797,
 source_rule_prediction_disagreements_with_recorded_export=converter_errors,
 native_branch_semantic_mismatch_sites=len(native_errors),native_branch_semantic_mismatch_cells=180,
 native_branch_semantic_single_flip_mismatch_indices=[r["index0"] for r in rows if r["flipped"]==1 and r["native_branch_semantic_hamming"]>0],
 flipped_site_indices=[r["index0"] for r in rows if r["flipped"]],
 multibranch_site_indices=[r["index0"] for r in rows if r["branch_count"]>1],
 all_multibranch_native_semantic_vectors_exact=True,
 export_only_failure_sites=sum(r["export_hamming"]>0 and r["native_branch_semantic_hamming"]==0 for r in rows),
 reference_export_mismatch_cells=sum(r["reference_export_mismatch_cells"] for r in rows),
 query_export_mismatch_cells=sum(r["query_export_mismatch_cells"] for r in rows),
 changed_unphased_diploid_genotype_cells=sum(r["export_changed_unphased_diploid_genotypes"] for r in rows),
 approximate_author_threshold_calls_at_N40=1,categories=categories,
 source_explanations=dict(approximate_mapping="anc_builder.cpp lines365/383 set thr=int(.03*N), MapMutation lines1019/1032 or1099/1112 accept minimum carrier mismatches <=thr. Native .mut retains180 one-call approximate clade fits.",
 converter_export="ConvertToTreeSequence.pinned.cpp lines290/303 use mutation_type[0] as ancestral state; lines389-400 insert mutation_type[2] only for one branch. Flipped flag is ignored and multibranch mutations omitted.",
 flag_meaning="mutations.cpp lines484-489 writes is_not_mapping solely from branch_count>1; zero is not proof of exact genotype matching."),
 interpretation="The verifier uses biological AA/DA nucleotides in fixed haplotype order correctly. Approximate inferred topology and lossless mutation/genotype export are distinct. Successful native topology/date/support facts do not imply lossless genotype export. Strictoldgate remains failed.",
 source_build_identity_proven=False,checks=len(checks),
 native_rerun=False,native_or_observed_inputs_or_outputs_modified=False,mutation_reattachment=False,sites_filtered=False,
 runtime_model_guard_GPU_accounting_executed=False,original_controller_exit_preserved=1)
save(HERE/"diagnosis.json",result)
with (HERE/"site_diagnostics.csv").open("w",newline="") as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
save(HERE/"checks.json",dict(status="passed",checks=len(checks),check_names=checks))
paths=[O/"native_facts.json",O/"work/native.anc",O/"work/native.mut",O/"work/native.trees",
 B/"baseline.haps",B/"polarization_equivalence.json",B/"frozen_site_identities.json",
 P/"inference_core.py",P/"source_manifest.json",P/"admission_v1/controller_result.json",
 P/"admission_v1/actual_controller_wait_root_v1.json",source_cpp,builder,mutcpp]
save(HERE/"input_source_refs.json",dict(schema="H018_native_vector_audit_input_refs_v1",files=[ref(p) for p in paths]))
print(json.dumps({k:result[k] for k in ["export_mismatch_sites","export_mismatch_nucleotide_cells","native_branch_semantic_mismatch_sites","native_branch_semantic_mismatch_cells","export_only_failure_sites","checks"]}))
