"""Tiny synthetic H40/Q24 CPU/FP32 interface checks. No empirical or checkpoint reads."""
from pathlib import Path
import ast,json,hashlib,importlib.util,dataclasses
import numpy as np
import torch
from arggnn import anonymous_clade_flag_control_v1 as original
from arggnn import mutation_routed_tree_observations_v1 as prep
from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1,validate_inputs_v1
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/H018_variable_panel_adapter_source_v1"
checks=[]
def check(name,ok):
    if not ok:raise AssertionError(name)
    checks.append(name)
def reject(name,call):
    try:call()
    except (ValueError,TypeError):checks.append(name);return
    raise AssertionError(name+" accepted")
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
adapter=load("_H018_synthetic_count_interface",HERE/"flag_interface_proposal.py")
contract=load("_H018_synthetic_metadata_interface",HERE/"reader_contract_proposal.py")
torch.set_num_threads(1);torch.manual_seed(101)
# Two complete synthetic40-leaf binary trees, with a changed proper clade.
parents=np.full(79,-1,dtype=np.int64);times=np.full(79,100.,dtype=np.float64)
active=list(range(40));nextnode=40
while len(active)>1:
    following=[]
    for at in range(0,len(active),2):
        if at+1==len(active):following.append(active[at]);continue
        left,right=active[at:at+2];parents[left]=parents[right]=nextnode
        times[nextnode]=max(times[left],times[right])+10.
        following.append(nextnode);nextnode+=1
    active=following
check("synthetic complete40-leaf79node tree",nextnode==79 and active==[78])
p2=parents.copy();p2[0],p2[2]=p2[2],p2[0]
trees=[prep.NativeObservedTreeV1(p,times.copy(),np.arange(40,dtype=np.int64),np.arange(40,dtype=np.int64)) for p in [parents,p2]]
g=np.zeros((4,40),dtype=np.int16)
g[0,::2]=1;g[1,:]=1;g[2,1::3]=1;g[3,4::4]=1
labels=np.array([c for c in range(4) for _ in range(4)]+[-1]*24,dtype=np.int64)
ages=np.full(40,100.,dtype=np.float64)
geometry=prep.observed_sham_geometry_v1(trees)
bins=prep.fit_sham_bin_edges_v1(*geometry,split="TRAIN")
obs=prep.prepare_mutation_routed_tree_observations_v1(trees,np.array([[0.,1000.],[1000.,2000.]]),
 np.array([100.,200.,1100.,1200.]),g,np.full(4,2,dtype=np.int64),np.arange(40,dtype=np.int64),
 labels,ages,np.arange(16,40,dtype=np.int64),np.array([500.,1500.]),sham_bin_edges=bins,fixture=True)
arrays={f.name:getattr(obs,f.name) for f in dataclasses.fields(MutationRoutedWholeTreeInputsV1)}
bank=MutationRoutedWholeTreeInputsV1.from_arrays(arrays,device="cpu")
summary=validate_inputs_v1(bank,expected_haplotypes=40)
check("generic original bank validator accepts synthetic40/Q24",summary["queries"]==24)
flags,meta=adapter.build_variable_panel_flags_v1(bank)
check("generic exact-clade flag arrays retain40 tips",flags.haplotypes==40 and flags.native_holds==2 and tuple(flags.node_flags.shape)==(158,2))
check("changed synthetic clade flags present",bool(flags.node_flags.count_nonzero()))
model=adapter.VariablePanelFlagInterfaceV1(checkpoint_sites=False).eval()
legacy=original.AnonymousCladeFlagControlGNNV1(checkpoint_sites=False).eval()
check("identical30788 learned parameters",sum(p.numel() for p in model.parameters())==sum(p.numel() for p in legacy.parameters())==30788)
check("identical ordered state keys and shapes",[(k,tuple(v.shape)) for k,v in model.state_dict().items()]==[(k,tuple(v.shape)) for k,v in legacy.state_dict().items()])
loaded=model.load_state_dict(legacy.state_dict(),strict=True)
check("strict synthetic state transfer needs no reshaping",not loaded.missing_keys and not loaded.unexpected_keys)
check("exact inherited learned H042 operations",model.encode_trees.__func__ is legacy.encode_trees.__func__
 and model.prepare_tree_base_v1.__func__ is legacy.prepare_tree_base_v1.__func__
 and model.sweep_tree_state_v1.__func__ is legacy.sweep_tree_state_v1.__func__
 and model.leaf_context_v1.__func__ is legacy.leaf_context_v1.__func__
 and model.encode_sites.__func__ is legacy.encode_sites.__func__)
with torch.no_grad():
 allholds=model.forward_holds(bank,flags=flags)
 queryholds=model.forward_train_holds(bank,flags=flags)
 readouts=model(bank,flags=flags)
check("synthetic allhold shape2x40x4",tuple(allholds.shape)==(2,40,4))
check("synthetic query/readout shapes2x24x4",tuple(queryholds.shape)==tuple(readouts.shape)==(2,24,4))
check("FP32 finite CPU outputs",all(x.dtype==torch.float32 and x.device.type=="cpu" and bool(torch.isfinite(x).all()) for x in [allholds,queryholds,readouts]))
check("query order exactly16through39",torch.equal(queryholds,allholds.index_select(1,bank.query_column)))
reject("original H042 full952 forward gate remains intact",lambda:legacy.forward_holds(bank,flags=flags))
reject("original H042 full952 flag wrapper remains intact",lambda:original.build_clade_flags_v1(bank))
reject("adapter rejects undeclared panel",lambda:adapter.VariablePanelFlagInterfaceV1(expected_haplotypes=41))
# Verify the copied hold function differs only at declared count gates/message.
def clsfn(s,cls,name):
 c=next(n for n in ast.parse(s).body if isinstance(n,ast.ClassDef) and n.name==cls)
 return next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
newfn=clsfn((HERE/"flag_interface_proposal.py").read_text(),"VariablePanelFlagInterfaceV1","forward_holds")
oldfn=clsfn((ROOT/"src/arggnn/anonymous_clade_flag_control_v1.py").read_text(),"AnonymousCladeFlagControlGNNV1","forward_holds")
class Restore(ast.NodeTransformer):
 def visit_Attribute(self,n):
  if isinstance(n.value,ast.Name) and n.value.id=="self" and n.attr in ("expected_haplotypes","expected_queries"):
   return ast.Constant(952 if n.attr=="expected_haplotypes" else 686)
  return self.generic_visit(n)
 def visit_Constant(self,n):
  if n.value=="Complete FP32 declared-panel bank and matching flags required":
   n.value="Complete original FP32 native bank and matching flags required"
  return n
check("hold computation AST unchanged except count gate",ast.dump(Restore().visit(newfn),include_attributes=False)==ast.dump(oldfn,include_attributes=False))
rows=[]
for i in range(40):
 rows.append(dict(sample_index=i,haplotype=i%2,role="reference" if i<16 else "query",
  observed_reference_class=i//4 if i<16 else None,observed_cohort=contract.CLASSES[i//4] if i<16 else "synthetic_query",
  date_zero="calBP1950",years_per_generation=28,age_generations=100.,age_mean_cal_BP=2800.,
  query_ancestry_truth=None,subject_id="SYNTH_"+str(i//2)))
panel=contract.panel_metadata_v1(rows)
check("synthetic roster exact class and query order",panel["reference_class"]==labels.tolist() and panel["query_column"]==list(range(16,40)))
check("eight donor channel order and absent group masks",panel["donor_group_counts"]==[4,4,4,4,0,0,0,0]
 and np.all(obs.donor_call_mask[:,:4]) and not np.any(obs.donor_call_mask[:,4:]))
dates=contract.verify_native_tip_dates_v1([100.]*40,[100.]*40)
check("date contract preserves source and native times",dates["native_times_edited"] is False)
reject("date contract rejects silent native time change",lambda:contract.verify_native_tip_dates_v1([100.]*40,[0.]*40))
view=contract.translated_interval_view_v1([[0.,1000.],[1000.,2001.]],[501,1501,2000])
check("native minus1 then genuine view correct",view["intervals_regional_bp0"]==[[500,999.],[999.,2000.]]
 and view["positions_regional_bp0"]==[500,1500,1999] and view["sites_dropped"]==0)
reject("coordinate contract forbids dropping a native hold",lambda:contract.translated_interval_view_v1([[0.,100.],[100.,2001.]],[501,1501,2000]))
packet=json.loads((HERE/"reader_view.inactive.json").read_text())
check("all empirical execution authorities inactive",all(packet[k] is False for k in
 ["raw_empirical_read_authorized","native_execution_authorized","model_checkpoint_read_authorized",
 "model_application_authorized","training_authorized","GPU_authorized","admission_authorized","actual_empirical_bank_prepared"]))
check("all source pins exact",all(hashlib.sha256((ROOT/r["path"]).read_bytes()).hexdigest()==r["sha256"] for r in packet["source_refs"].values()))
print(json.dumps(dict(schema="H018_variable_panel_synthetic_interface_checks_v1",status="passed",checks=len(checks),
 check_names=checks,synthetic_only=True,CPU_FP32_only=True,torch_version=torch.__version__,
 actual_empirical_arrays_or_native_graph_read=False,real_model_checkpoint_loaded=False,model_training_performed=False,
 native_execution=False,GPU_execution=False,guard_or_admission=False,synthetic_output_shapes=[[2,40,4],[2,24,4]]),
 sort_keys=True,indent=2))
