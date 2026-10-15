"""Independent bounded source and synthetic FP32 loss audit; no project imports."""
from pathlib import Path
import ast,hashlib,json,copy,types
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/H047_cohort_risk_independent_v1"
checks=[]
def check(n,ok):
    if not ok:raise AssertionError(n)
    checks.append(n)
def reject(n,f):
    try:f()
    except (ValueError,TypeError):checks.append(n);return
    raise AssertionError(n+" accepted")
def fn(s,n):return next(x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name==n)
def segment(s,n):return ast.get_source_segment(s,fn(s,n))
def extract(s,n,ns):
    exec(compile(ast.Module(body=[fn(s,n)],type_ignores=[]),"<source-extracted-"+n+">","exec"),ns)
    return ns[n]
core=(ROOT/"src/arggnn/cohort_risk_continuation_v1.py").read_text()
worker=(ROOT/"src/arggnn/train_cohort_risk_continuation_v1.py").read_text()
check("exact reviewed core",hashlib.sha256(core.encode()).hexdigest()=="d76094d9363ef2c4452b497b0949ea56a8c731f82d0f02aa32d607d889ffb9ef")
check("exact reviewed worker",hashlib.sha256(worker.encode()).hexdigest()=="d542dddc574e93b0bec7801ace9a15d0cc9bd28fb1774a773520f3e29cbd7557")
def require(ok,msg):
    if not ok:raise ValueError(msg)
old=(ROOT/"src/arggnn/biological_gst_control_v1.py").read_text()
ns=dict(np=np,torch=torch,F=F,require=require,ARMS=("original_risk","balanced_risk"),
 COHORT_COUNTS={"present_day":182,"Bronze_Age":162,"BAA":14,"Neo":302,"Yam":26})
exact=extract(old,"exact_span_loss",ns)
objective=extract(core,"cohort_objective_v1",ns);columns=extract(core,"cohort_columns_v1",ns)
cohorts=np.array(["present_day"]*182+["Bronze_Age"]*162+["BAA"]*14+["Neo"]*302+["Yam"]*26)
p,a=columns(cohorts);check("observed exact182/504 columns",np.array_equal(p,np.arange(182)) and len(a)==504)
bad=cohorts.copy();bad[0]="Neo";reject("wrong observed group count rejected",lambda:columns(bad))
torch.set_num_threads(1);torch.manual_seed(4747)
logits=torch.randn(3,686,4,dtype=torch.float32,requires_grad=True)
mass=torch.rand(3,686,4,dtype=torch.float32)
mass[:,p]*=2.
loss,metrics=objective(logits,mass,p,arm="original_risk")
reference=exact(logits,mass)
check("original control FP32 loss bitwise exact",torch.equal(loss,reference))
g1=torch.autograd.grad(loss,logits,retain_graph=True)[0];g2=torch.autograd.grad(reference,logits,retain_graph=True)[0]
check("original control FP32 gradient bitwise exact",torch.equal(g1,g2))
balanced,_=objective(logits,mass,p,arm="balanced_risk")
logp=F.log_softmax(logits,-1)
lp=-(logp[:,p]*mass[:,p]).sum()/mass[:,p].sum()
la=-(logp[:,a]*mass[:,a]).sum()/mass[:,a].sum()
expected=.5*lp+.5*la
check("balanced group BP loss correct",torch.allclose(balanced,expected,atol=1e-6,rtol=1e-6))
b1=torch.autograd.grad(balanced,logits,retain_graph=True)[0];b2=torch.autograd.grad(expected,logits)[0]
check("balanced group BP gradients correct",torch.allclose(b1,b2,atol=1e-7,rtol=1e-6))
check("actual support weights used",torch.allclose(metrics["actual_present_query_bp_weight"],mass[:,p].sum()/mass.sum())
 and not torch.isclose(metrics["actual_present_query_bp_weight"],torch.tensor(182/686,dtype=torch.float32)))
badmass=mass.clone();badmass[:,0]=0
reject("zero represented query rejected",lambda:objective(logits,badmass,p,arm="balanced_risk"))
reject("duplicate observed columns rejected",lambda:objective(logits,mass,[0,0],arm="balanced_risk"))
reject("noninteger observed columns rejected",lambda:objective(logits,mass,[0.,1.],arm="balanced_risk"))
# Exercise the ACTUAL unchanged original-grid to hold mass function synthetically.
h038=(ROOT/"src/arggnn/train_mutation_routed_full_panel_pyg_v1.py").read_text()
aggregate=extract(h038,"native_hold_train_mass_v1",dict(np=np))
edges=np.array([0.,100.,200.,300.,400.]);intervals=np.array([[0.,60.],[60.,150.],[150.,400.]])
binids=[];rowids=[];widths=[]
for i in range(4):
 for j,(lo,hi) in enumerate(intervals):
    width=min(edges[i+1],hi)-max(edges[i],lo)
    if width>0:binids.append(i);rowids.append(j);widths.append(width)
binids=np.array(binids);rowids=np.array(rowids);widths=np.array(widths)
target=types.SimpleNamespace(intervals=intervals,c4=np.array([[(i+j)%4 for j in range(686)] for i in range(3)]))
helpers=types.SimpleNamespace(intersections=lambda bins,truth:(binids,rowids,widths))
readout=torch.tensor([0,0,1,2],dtype=torch.long)
bank=types.SimpleNamespace(readout_hold=readout,query_column=torch.arange(686),spans=torch.tensor([200.,100.,100.]))
held=aggregate(dict(positions=np.array([50.,150.,250.,350.]),edges=edges),bank,target,helpers)
grid=np.zeros((4,686,4),dtype=np.float32)
for i,j,w in zip(binids,rowids,widths):grid[i,np.arange(686),target.c4[j]]+=w
independent=np.zeros_like(held)
for i,hold in enumerate(readout):independent[int(hold)]+=grid[i]
check("actual mass helper preserves every original grid class span",np.array_equal(held,independent))
check("actual mass helper preserves400bp eachquery",np.array_equal(held.sum((0,2)),np.full(686,400.,dtype=np.float32)))
for arm in ["original_risk","balanced_risk"]:
 x=torch.randn(3,686,4,dtype=torch.float32,requires_grad=True)
 holdloss,_=objective(x,torch.from_numpy(held),p,arm=arm)
 gridloss,_=objective(x.index_select(0,readout),torch.from_numpy(grid),p,arm=arm)
 check("grid-to-hold loss equivalent "+arm,torch.allclose(holdloss,gridloss,rtol=1e-6,atol=1e-6))
 gh=torch.autograd.grad(holdloss,x,retain_graph=True)[0];gg=torch.autograd.grad(gridloss,x)[0]
 check("grid-to-hold gradient equivalent "+arm,torch.allclose(gh,gg,rtol=2e-6,atol=1e-7))
check("no new learned class in core",not any(isinstance(n,ast.ClassDef) for n in ast.parse(core).body))
family=segment(core,"make_matched_family_v1")
check("identical existing H042 class used","model=AnonymousCladeFlagControlGNNV1(**MODEL)" in family and "model.load_state_dict(state,strict=True)" in family and "model.requires_grad_(True)" in family)
train=segment(worker,"run_train_v1");evaluate=segment(worker,"run_evaluate_v1")
check("original mass helper direct call","mass=h038.native_hold_train_mass_v1(" in train)
check("all32 mass built before passes",train.index("for pid in TRAIN:")<train.index("for at in range(1,31):") and train.index("targets[pid]=torch.from_numpy(mass)")<train.index("for at in range(1,31):"))
check("mass targets shared retained acrossarms",train.count("native_hold_train_mass_v1(")==1 and "FullParentResidencyV1(banks,targets" in train and "for arm in ARMS:" in train)
check("same parent order and960perarm","random.Random(101+at).shuffle(order)" in train and "steps=={a:32*at for a in ARMS}" in train and "steps=={a:960 for a in ARMS}" in train)
check("first qualification is actualfirstupdate","qualify=first,after_optimizer_step=" in train and "no_extra_qualification_updates=True" in train)
load=segment(worker,"load_observations_v1")
check("cohort masks only observed metadata",'core.cohort_columns_v1(parent["cohorts"])' in load)
check("allphase flags retained before target reads","set(cache.values)==set(ids)" in load and train.index("load_observations_v1")<train.index("reader.truth_gate=True"))
update=segment(worker,"update_v1")
check("only risk arm changes optimizer path","model.forward_train_holds(bank,flags=flags,control=\"aligned\")" in update and "core.cohort_objective_v1(logits,mass,present,arm=arm)" in update)
check("no payload hash or read in update",all(s not in update for s in ["hashlib",".read(","read_bytes","array_identity","state_identity"]))
cache=next(n for n in ast.parse(worker).body if isinstance(n,ast.ClassDef) and n.name=="FlagCacheV1")
get=ast.get_source_segment(worker,next(n for n in cache.body if isinstance(n,ast.FunctionDef) and n.name=="get"))
check("cached hit returns before contenthash",get.index("self.hits+=1;return flag")<get.index("h42.shared.array_identity("))
check("no VAL optimization calls",all(s not in evaluate for s in ["optimizer",".backward(","update_v1(","cohort_objective_v1("]))
check("all96 sealbarrier precedes VALtruth",evaluate.index("ram.verify()")<evaluate.index("reader.truth_gate=True"))
ram=next(n for n in ast.parse(worker).body if isinstance(n,ast.ClassDef) and n.name=="DenseRAMMatrixV1")
rv=ast.get_source_segment(worker,next(n for n in ram.body if isinstance(n,ast.FunctionDef) and n.name=="verify"))
check("complete2x6x8 products required",'set(self.values)=={(a,at,p) for a in ARMS for at in PASSES for p in VAL}' in rv)
h40=(ROOT/"src/arggnn/H040_latent_clade_execution_v1.py").read_text()
check("exact32TRAIN8VALsixcheckpoint schedule","range(32)" in h40 and "range(8)" in h40 and "PASSES = (5, 10, 15, 20, 25, 30)" in h40)
print(json.dumps(dict(schema="H047_independent_source_FP32_loss_review_v1",status="PASS",checks=len(checks),check_names=checks,
 source_only=True,synthetic_CPU_FP32_only=True,project_modules_imported=False,real_data_model_checkpoint_native_GPU_guard_admission_Git=False,
 source_refs=[dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in [
 "src/arggnn/cohort_risk_continuation_v1.py","src/arggnn/train_cohort_risk_continuation_v1.py",
 "src/arggnn/anonymous_clade_flag_control_v1.py","src/arggnn/train_mutation_routed_full_panel_pyg_v1.py",
 "src/arggnn/biological_gst_control_v1.py"]]),sort_keys=True,indent=2))
