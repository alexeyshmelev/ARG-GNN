"""Pure stdlib source/rounding checks; no retained native tree or data loaded."""
from pathlib import Path
import ast,hashlib,json,math,struct
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/H018_native_Relate_qualification_source_v1/v4_internal_coordinate_verifier_audit_v1"
ORIGINAL=ROOT/"research/reviews/H018_native_Relate_qualification_source_v1/fixture_and_verify.py"
old=ORIGINAL.read_text();new=(HERE/"fixture_and_verify_corrected_v1.py").read_text()
checks=[]
def check(name,ok):
    if not ok:raise AssertionError(name)
    checks.append(name)
def node(source,name):
    return next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
def pure(source,name,scope):
    exec(compile(ast.Module(body=[node(source,name)],type_ignores=[]),"<pure-"+name+">","exec"),scope)
    return scope[name]
before="else:value=max(f32(coord32(c)+f32(lengths[c])) for c in children[node])"
after="else:value=f32(max(coord32(c)+lengths[c] for c in children[node]))"
check("exactly one arithmetic-line correction",old.count(before)==1 and new==old.replace(before,after))
compile(new,"corrected_candidate","exec");checks.append("candidate AST compile")
for name in ("make_fixture","rounded_age_token","verify_native","need","encoded","sha"):
    check("unchanged "+name,ast.dump(node(old,name),include_attributes=False)==ast.dump(node(new,name),include_attributes=False))
header=(ROOT/"research/reviews/relate_ancestral_identity_source_audit_v1/anc.hpp").read_text()
src=(ROOT/"research/reviews/relate_ancestral_identity_source_audit_v1/anc.cpp").read_text()
check("source branch is binary64 double","double branch_length = 0.0;" in header)
check("source text branch uses percentlf",'sscanf(&line[i], "%d:(%lf %f %d %d)"' in src)
check("source child-float plus branch-double then float assignment","std::vector<float>& coordinates" in src and 'coordinates[n.label] = std::max(coordinates[(*n.child_right).label] + (*n.child_right).branch_length, coordinates[(*n.child_left).label] + (*n.child_left).branch_length);' in src)
f32=lambda x:struct.unpack("<f",struct.pack("<f",x))[0]
cases=json.loads((HERE/"rounding_counterexamples.json").read_text())["cases"]
for i,row in enumerate(cases):
    check("premature rounding counterexample "+str(i),f32(row["child_coordinate"]+f32(row["branch_length"]))==row["old_parent"] and f32(row["child_coordinate"]+row["branch_length"])==row["source_typed_parent"] and row["old_parent"]!=row["source_typed_parent"])
def need(ok,message):
    if not ok:raise ValueError(message)
# Full79-node synthetic comb is the smallest format accepted by the fixed40-tip parser.
parents=[-1]*79
parents[0]=parents[1]=40
for leaf in range(2,40):parents[leaf]=39+leaf
for internal in range(40,78):parents[internal]=internal+1
records=[]
for i,parent in enumerate(parents):
    branch="1234.6" if i==0 else "0" if i==78 else "1.25"
    records.append(str(parent)+":("+branch+" 0 0 0)")
age=36.123451
raw=("NUM_HAPLOTYPES 40 "+" ".join([format(age,".6f")]*40)+"\nNUM_TREES 1\n0: "+" ".join(records)+"\n").encode()
scope_old=dict(math=math,struct=struct,need=need);scope_new=dict(scope_old)
prior=pure(old,"parse_anc",scope_old)(raw)
actual=pure(new,"parse_anc",scope_new)(raw)
check("all forty exact parsed tip ages unchanged",prior[0]==actual[0] and prior[1]==actual[1]==[age]*40)
expected=f32(f32(age)+float("1234.6"))
check("native source typed synthetic parent exact",actual[2][0]["coordinates32"][40]==expected)
check("old parser demonstrably differs on same anc bytes",prior[2][0]["coordinates32"][40]!=actual[2][0]["coordinates32"][40])
check("positive synthetic chronology not rewritten",not actual[2][0]["nonpositive_float32_branches"])
check("no date tolerance or node comparison relaxed",'float(ts.node(ident).time)!=t["coordinates32"][node]' in new and 'float(ts.node(node).time)==anc_ages[i]' in new)
check("no source time tree genotype modifications","qualifier_modified_native_genotypes_or_times=False" in new and ".set_columns(" not in new)
refs=json.loads((HERE/"source_and_evidence_refs.json").read_text())["refs"]
check("every originally read source/evidence pin preserved",all(hashlib.sha256((ROOT/r["path"]).read_bytes()).hexdigest()==r["sha256"] for r in refs))
result=dict(schema="H018_v4_internal_coordinate_verifier_source_checks_v1",status="passed_source_and_synthetic_only",
 checks=len(checks),check_names=checks,retained_native_trees_or_empirical_arrays_loaded=False,
 native_model_GPU_process_guard_accounting_or_Git_actions_executed=False,
 corrected_candidate_executed_on_retained_outputs=False)
print(json.dumps(result,sort_keys=True,indent=2))
