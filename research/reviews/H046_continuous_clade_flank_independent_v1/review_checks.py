"""Independent H046 source/stdlib review. No model/runtime/data imports."""
from pathlib import Path
import ast,hashlib,json,math
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OWN=ROOT/"research/reviews/H046_continuous_clade_flank_independent_v1"
MODEL=ROOT/"src/arggnn/continuous_clade_flank_gnn_v1.py"
WORKER=ROOT/"src/arggnn/train_continuous_clade_flank_v1.py"
checks=[]
def check(name,ok):
    if not ok:raise AssertionError(name)
    checks.append(name)
def function(source,name):
    return next(n for n in ast.walk(ast.parse(source)) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
def text(source,name):return ast.get_source_segment(source,function(source,name))
model=MODEL.read_text();worker=WORKER.read_text()
compile(model,str(MODEL),"exec");compile(worker,str(WORKER),"exec")
checks.append("both current source files compile without imports")
h44=(ROOT/"src/arggnn/train_marker_resolution_readout_v1.py").read_text()
for a,b in [("H046PhaseReaderV1","H044PhaseReaderV1"),("DenseRAMMatrixV1","DenseRAMMatrixV1")]:
    actual=text(worker,a).replace("H046","H044")
    check("inherited complete barrier "+a,ast.dump(ast.parse(actual),include_attributes=False)==ast.dump(ast.parse(text(h44,b)),include_attributes=False))
prepare=text(model,"prepare_continuous_clade_bank_v1")
check("exact original interval identity binding",'tensor_identity_v1(intervals_bp)==markers.original_binding["array_identities"]["intervals_bp"]' in prepare)
builder=text(model,"build_clade_flank_arrays_v1")
check("original physical half-open windows retained","centers-1000." in builder and "intervals[grid, 0]" in builder and "centers+1000." in builder and 'side="left"' in builder)
check("reference excludes roots and singletons via existing exact clades","_tree_clades(root, row, parent, head, sibling)" in builder)
check("deepest eligible proper clade map","sizes[episode] >= 2" in builder and "stack = [(root, -1)]" in builder and "representative[node] = selected" in builder)
scan=text(model,"segmented_flank_integrals_v1")
check("stable normalized dimensionless exponential weights","torch.exp(-widths/tau)" in scan and "-torch.expm1(-widths/tau)" in scan)
check("strict exclusive left and right endpoints","inclusive[:-1]" in scan and "masked_fill(boundary[:,None],0.)" in scan and "ends.flip(0)" in scan)
check("no cancellation of full and focal pools","forward - " not in scan and "backward - " not in scan)
prefix=text(model,"LocalReferencePrefixV1")
check("all-reference-class callability and query call", "(q>=0)&markers.reference_called.all(-1)" in prefix)
check("reference-only inherited log evidence","markers.log_evidence.gather" in prefix and "values.mean(-1,keepdim=True)" in prefix)
check("FP32 block prefix rebase","joined.reshape(count,SITE_BLOCK,b,5)" in prefix and "block.cumsum(1)" in prefix and "torch.float64" not in prefix)
check("missing-side gate zero and no learned gate","((nl>0)&(nr>0))" in prefix and "norm/(norm+1.)" in prefix)
readout=text(model,"_readout_block")
check("retain distance attenuation both sides","torch.exp(-distance_left/tau)" in readout and "torch.exp(-distance_right/tau)" in readout)
check("own control retains same mass geometry and zero foreign gradient","cl*current+0.*vl,cr*current+0.*vr" in readout)
check("shared swapped head and bounded gate correction",".5*(self.H(value).tanh()+self.H(swapped).tanh())" in readout and "gate*self.O(hidden).tanh()*eligible" in readout)
body=text(model,"backbone_state_v1")
check("full two sweeps and original BiGRU",body.count("sweep_tree_state_v1(")==2 and "self.backbone.genomic_gru(sequence)" in body and ")),upward" in body)
forward=text(model,"forward")
check("context from pregenomic first upward state","self.A(upward.index_select" in forward)
check("complete grid logits remain materialized","return torch.cat(blocks,1)" in forward)
train=text(worker,"run_train_v1")
check("complete observation barrier before TRAIN truth",train.index("load_observations_v1")<train.index("reader.truth_gate=True")<train.index("h44.original_mass_v1"))
check("one original-grid target per parent shared by arms",train.count("h44.original_mass_v1(")==1 and train.index("h44.original_mass_v1")<train.index("for arm in ARMS:") and "del bundle,mass,mass_cpu,cpu_bundle" in train)
update=text(worker,"update_v1")
check("one complete-parent backward and step",update.count("loss.backward()")==1 and update.count("optimizer.step()")==1 and "original_grid_span_nll_v1" in update)
evaluate=text(worker,"run_evaluate_v1")
check("all96 before VAL truth and selected diagnostics",evaluate.index("ram.verify();reader.truth_gate=True")<evaluate.index("validation.score(")<evaluate.index("validation.full_final("))
check("no optimization in VAL","optimizer.step" not in evaluate and "optimization_performed=False" in evaluate)
criterion=text(worker,"practical_criteria_v1")
check("equal parent mean criterion explicit","equal_parent_mean_Present182_across8VAL" in criterion and "worst_parent_boundary_loss_bound=False" in criterion)
strata=next(n.value for n in ast.parse(worker).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="LOCAL_CALLABILITY_STRATA" for t in n.targets))
values=ast.literal_eval(strata)
check("fixed original-callability strata",[(lo,hi) for _,lo,hi in values]==[(0,1),(1,3),(3,6),(6,None)])
callability=text(worker,"local_callability_diagnostics_v1")
check("strata membership only original observed usable calls","core.LocalReferencePrefixV1(bundle.markers,q0,q1)" in callability and "torch.minimum(left,right)" in callability)
check("strata reuse primary exact-span mass","primary_cache.mass[:,selected]" in callability and "FullPanelExternalTarget" not in callability)
check("selected strata only after complete seal barrier",evaluate.index("ram.verify();reader.truth_gate=True")<evaluate.index("local_callability_diagnostics_v1(") and 'ram.records[arm,best["pass_index"],pid]' in evaluate)
score=text(worker,"callability_block_sums_v1")
check("zero probability keeps infinite NLL and no probability floor","(p==0)&(m>0)" in score and "np.log(np.where(p>0,p,np.float32(1)))" in score)
aggregate=text(worker,"aggregate_callability_diagnostics_v1")
check("missing-support and finite/infinite parent counts exposed","parents_without_support" in aggregate and "finite_parent_count" in aggregate and "infinite_parent_count" in aggregate)
ids=tuple("synthetic_%d"%i for i in range(8));arms=("foreign_flank","own_state")
scope=dict(VAL=ids,ARMS=arms)
exec(compile(ast.Module(body=[function(worker,"practical_criteria_v1")],type_ignores=[]),"<pure-criterion>","exec"),scope)
full={arm:dict(cells=[dict(parent_id=pid,groups={"primary_present_day_182":dict(accuracy=.6+(.002 if arm==arms[0] else 0.),boundaries=dict(f1=.43 if arm==arms[0] and j==0 else .5))}) for j,pid in enumerate(ids)]) for arm in arms}
result=scope["practical_criteria_v1"](full)
check("criterion allows one parent loss when equal mean satisfies root rule",result["meets_registered_practical_criteria"] and result["worst_parent_boundary_loss_bound"] is False)
full[arms[0]]["cells"][0]["groups"]["primary_present_day_182"]["boundaries"]["f1"]=None
result=scope["practical_criteria_v1"](full)
check("undefined parent boundary F1 never hidden as practical success",not result["meets_registered_practical_criteria"] and result["mean_boundary_F1_delta"] is None)

# Independent scalar derivation of the intended operator. This validates the
# source-to-math interpretation, not Torch execution or a GPU precision claim.
def literal(cells, focal, x, tau):
    lo,hi=focal
    left=right=0.
    for a,b,f in cells:
        if b<=lo:left+=f*(math.exp(-(x-b)/tau)-math.exp(-(x-a)/tau))
        elif a>=hi:right+=f*(math.exp(-(a-x)/tau)-math.exp(-(b-x)/tau))
        else:check_overlap=(a>=lo and b<=hi);assert check_overlap
    return left,right
def endpoints(cells,focal,x,tau):
    lo,hi=focal;left=right=0.
    for a,b,f in cells:
        if b<=lo:
            w=b-a;left=math.exp(-w/tau)*left-math.expm1(-w/tau)*f
    for a,b,f in reversed(cells):
        if a>=hi:
            w=b-a;right=math.exp(-w/tau)*right-math.expm1(-w/tau)*f
    return left*math.exp(-(x-lo)/tau),right*math.exp(-(hi-x)/tau)
cells=[(0.,250.,2.),(250.,650.,-1.),(650.,1350.,.5)]
split=[(0.,100.,2.),(100.,250.,2.),(250.,650.,-1.),(650.,1350.,.5)]
focal=(250.,650.);x=350.;tau=5000.
a=literal(cells,focal,x,tau);b=endpoints(cells,focal,x,tau);c=endpoints(split,focal,x,tau)
check("independent literal integral matches endpoint recurrence",max(abs(v-w) for v,w in zip(a,b))<1e-14)
check("fixed source support neighbor-mesh refinement invariant",max(abs(v-w) for v,w in zip(b,c))<1e-14)
delta=.01
da=[(v-w)/(2*delta) for v,w in zip(endpoints(cells,focal,x,tau+delta),endpoints(cells,focal,x,tau-delta))]
db=[(v-w)/(2*delta) for v,w in zip(endpoints(split,focal,x,tau+delta),endpoints(split,focal,x,tau-delta))]
check("fixed support tau derivative matches under refinement",max(abs(v-w) for v,w in zip(da,db))<1e-13)
changed=[cells[0],(250.,650.,1234.),cells[-1]]
check("strict focal field excluded from context",endpoints(changed,focal,x,tau)==b)
check("foreign support mass retained not renormalized",0.<endpoints([(0.,250.,1.),(250.,650.,0.),(650.,1350.,1.)],focal,x,tau)[0]<1.)
result=dict(schema="H046_independent_source_stdlib_review_checks_v1",status="passed",
 checks=len(checks),check_names=checks,model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
 worker_sha256=hashlib.sha256(WORKER.read_bytes()).hexdigest(),
 actual_model_or_data_or_guard_or_process_execution=False,
 scalar_oracle_is_exact_arithmetic_interpretation_not_FP32_runtime_verification=True)
print(json.dumps(result,sort_keys=True,indent=2))
